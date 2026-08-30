"""
La rete del capitolo 1 di Nielsen, ma scritta con quello che sappiamo adesso.

Il problema è quello del libro — 784 pixel in ingresso, dieci cifre in uscita,
una MLP con un layer nascosto, ma seguiamo le best practices che conosciamo.

    784 ingressi     i pixel dell'immagine, 28x28 appiattiti
    800 nascosti     tanh
     10 uscite       logits, nessuna attivazione sopra

Il libro arriva al 95%, la stessa rete scritta bene al 98%, e questa qui al 99%.

Ma da 98% a 99% l'architettura non cambia: quello che si aggiunge sono le *distorsioni
elastiche* di Simard, Steinkraus e Platt (ICDAR 2003), rigenerate a ogni epoca
come in Cireșan et al. (2010). Si prende un campo di spostamenti casuali, lo si
sfoca con una gaussiana, e lo si usa per riscrivere l'immagine: viene fuori una
cifra deformata come se fosse stata scritta da una mano che tremava in modo
diverso.

L'alternativa e' aggiungere una rete di convoluzione. Simard li aveva provati
entrambi:

    niente distorsioni     1.6% di errore     98.4%
    distorsioni affini     1.1%               98.9%
    distorsioni elastiche  0.7%               99.3%
    la sua convnet         0.4%               99.6%

Visto che sono simili, non avere la convnet è un guadagno perchè sono molti meno
calcoli a runtime durante la classificazione.

**Cosa cambia rispetto alla versione senza deformazioni**, e sono tre cose che
si tengono insieme.

Gli 800 neuroni nascosti al posto di 200: con i dati fissi non servivano a
niente, perché la rete già imparava le 50.000 immagini a memoria (overfitting) e
allargarla peggiorava solo. Con dati sempre nuovi la capacità torna a essere il
vincolo. Misurato: con 200 nascosti le stesse deformazioni si fermano a 98.96%
di validation, sotto la soglia.

Le 60 epoche al posto di 30: la rete non vede mai due volte la stessa immagine,
quindi non c'è il momento in cui smette di imparare e comincia a memorizzare.

E l'overfitting che sparisce. La versione precedente di questo file finiva con
train accuracy a 100.00%, loss di training a 0.0001, e la loss di validation che
risaliva dopo l'epoca 9 — overfitting da manuale. Qui non succede: le due curve
restano appaiate per tutte le 60 epoche.
"""

import base64
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mnist

SEED = 1
HERE = Path(__file__).parent

HIDDEN = 800
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-3

# I due numeri delle deformazioni elastiche, presi da Simard 2003: sigma e' la
# larghezza della gaussiana che sfoca il campo casuale ("coefficiente di
# elasticita'"), alpha di quanti pixel si sposta la roba. Il paper li tara su
# immagini 29x29 e trova sigma=4, alpha=34.
SIGMA = 4.0
ALPHA = 34.0

torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# 0. i dati, e il patto sul test
# ---------------------------------------------------------------------------

(xtr, ytr), (xva, yva), (xte, yte) = mnist.load()

print("=== 0. i dati ===\n")
print(f"  train       {tuple(xtr.shape)}   ci si allena")
print(f"  validation  {tuple(xva.shape)}   ci si decide tutto")
print(f"  test        {tuple(xte.shape)}   si guarda una volta sola, alla fine")
print(
    """
  Il capitolo valuta sul test a ogni epoca e riporta la migliore. Quello e'
  scegliere sul test, e il numero che ne esce non stima piu' niente: il test ha
  partecipato alla decisione. Il validation set e' nel file da sempre, il
  capitolo semplicemente non lo usa."""
)


# ---------------------------------------------------------------------------
# 1. le deformazioni, che sono il punto di questo file
# ---------------------------------------------------------------------------


def gaussian_kernel(sigma, radius):
    """Una gaussiana normalizzata, lunga 2*radius+1, per sfocare in 1D."""
    t = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(t**2) / (2 * sigma**2))
    return k / k.sum()


def distort(x, sigma=SIGMA, alpha=ALPHA, chunk=10000):
    """(N, 784) -> (N, 784), ogni immagine deformata a modo suo.

    La ricetta di Simard 2003, in quattro passaggi:

      1. per ogni pixel si estrae uno spostamento casuale uniforme in [-1, 1],
         uno orizzontale e uno verticale. Da solo e' rumore bianco: applicarlo
         farebbe a pezzi l'immagine;
      2. si sfoca il campo con una gaussiana di deviazione `sigma`. E' qui che
         nasce l'elasticita': pixel vicini finiscono per spostarsi in direzioni
         simili, e il tratto si piega invece di sbriciolarsi. Sigma piccolo da'
         di nuovo rumore, sigma grande da' una traslazione rigida;
      3. si moltiplica per `alpha`, che decide di quanti pixel si sposta;
      4. si legge l'immagine originale nei punti spostati, interpolando
         bilinearmente (`grid_sample`).

    Il campo casuale si genera con un margine di `radius` per lato e lo si
    convolve in "valid": e' il trucco dello zero-padding di Cireșan 2010, e
    serve perche' altrimenti la sfocatura ai bordi somma degli zeri e tira gli
    spostamenti verso zero proprio dove il tratto e' piu' fragile.
    """
    radius = int(math.ceil(3 * sigma))
    kernel = gaussian_kernel(sigma, radius)
    margin = 2 * radius
    out = torch.empty_like(x)

    # a blocchi, perche' il campo casuale con margine e' molto piu' grosso
    # delle immagini: 50.000 x 2 x 52 x 52 sarebbe mezzo giga in una volta
    for start in range(0, len(x), chunk):
        images = x[start : start + chunk].view(-1, 1, 28, 28)
        n = len(images)

        field = torch.rand(n, 2, 28 + margin, 28 + margin) * 2 - 1
        field = field.view(n * 2, 1, 28 + margin, 28 + margin)
        field = F.conv2d(field, kernel.view(1, 1, 1, -1))  # sfoca in orizzontale
        field = F.conv2d(field, kernel.view(1, 1, -1, 1))  # e poi in verticale
        field = field.view(n, 2, 28, 28).permute(0, 2, 3, 1) * alpha

        # grid_sample vuole coordinate normalizzate in [-1, 1], non pixel
        field = field * (2.0 / 28.0)

        identity = torch.zeros(n, 2, 3)
        identity[:, 0, 0] = 1.0
        identity[:, 1, 1] = 1.0
        grid = F.affine_grid(identity, (n, 1, 28, 28), align_corners=False)

        out[start : start + chunk] = F.grid_sample(
            images, grid + field, mode="bilinear", padding_mode="zeros",
            align_corners=False,
        ).view(-1, 784)

    return out


print("\n=== 1. le deformazioni ===\n")

sample = distort(xtr[:1].expand(8, 784).contiguous())
print(f"  sigma={SIGMA} (quanto e' elastica), alpha={ALPHA} (di quanti pixel si sposta)")
print(f"\n  la stessa immagine, l'originale e una delle infinite versioni deformate:\n")
print(mnist.draw(xtr[0]))
print()
print(mnist.draw(sample[0]))

deform_time = time.time()
distort(xtr)
deform_time = time.time() - deform_time

print(
    f"""
  Deformare tutte le 50.000 costa {deform_time:.1f} secondi, e si rifa' *a ogni epoca*:
  in {EPOCHS} epoche la rete non vede mai due volte la stessa immagine. E' la
  risposta di Ciresan 2010 alla domanda di come facciano reti enormi a
  generalizzare su cinquantamila esempi — non generalizzano da cinquantamila
  esempi, ne vedono un milione e mezzo."""
)

# la figura: la stessa cifra deformata otto volte, per far vedere che restano
# tutte leggibili e nessuna e' uguale all'altra
fig, axes = plt.subplots(2, 9, figsize=(15, 3.6))
for row, digit in enumerate([0, 1]):
    versions = distort(xtr[digit : digit + 1].expand(8, 784).contiguous())
    axes[row, 0].imshow(xtr[digit].reshape(28, 28), cmap="gray_r")
    axes[row, 0].set_title("originale", fontsize=9)
    for i in range(8):
        axes[row, i + 1].imshow(versions[i].reshape(28, 28), cmap="gray_r")
    for ax in axes[row]:
        ax.set_xticks([])
        ax.set_yticks([])
fig.suptitle(f"la stessa cifra, deformata (sigma={SIGMA}, alpha={ALPHA}): questo e' cio' su cui la rete si allena")
plt.tight_layout()
plt.savefig(HERE / "01_out_distortions.png", bbox_inches="tight", dpi=100)
plt.close()
print("\n  figura in 01_out_distortions.png")


# ---------------------------------------------------------------------------
# 2. la rete
# ---------------------------------------------------------------------------


def build():
    """784 -> 800 con tanh -> 10 logits.
    """
    model = nn.Sequential(
        nn.Linear(784, HIDDEN),
        nn.Tanh(),
        nn.Linear(HIDDEN, 10),
    )

    hidden_layer, output_layer = model[0], model[2]

    # Qua tutto è come 03_mlp/02_optimizations.py

    # Moltiplicare un vettore (gli input) per una matrice *allarga* la sua
    # distribuzione, proporzionalmente a quanti ingressi si sommano in ogni
    # neurone ("fan in"). In particolare, 784 contributi indipendenti moltiplica
    # la deviazione standard per sqrt(784), quindi inizializziamo con gaussiana
    # divisa per lo stesso valore. Ma dopo la moltiplicazione c'e' il tanh, e
    # lui invece stringe da 1.0 a 0.63, quindi moltiplichiamo per l'inverso
    # (5/3).
    nn.init.kaiming_normal_(hidden_layer.weight, nonlinearity="tanh")

    # Azzeriamo i bias iniziali Perché zero: X @ W1 e' gia' tarato per non
    # saturare la tanh (std 0.55), e un bias grande sballerebbe la taratura. Ma
    # zero e' solo il *punto di partenza*: b1 resta un parametro che impara.
    nn.init.zeros_(hidden_layer.bias)

    # L'alternativa a queste inizializzazioni degli hidden layer e' un batch
    # normalization layer pre-tanh, ma con un solo hidden layer è una scelta
    # eccessiva.

    # Ora, riguardo all'ultimo layer che sputa i logits. Niente Kaiming: il gain
    # serve a compensare lo schiacciamento di una tanh, e sopra i logits non
    # c'e' nessuna tanh.

    # Il discorso sull'ultimo layer è solo che vogliamo evitare di perdere i
    # primi N cicli di training per valori iniziali completamente sballati. Ha
    # senso che questo layer parta dando valori di logits distribuiti
    # uniformemente ("non so niente"). Quindi logits tutti piccoli e intorno a
    # 0 -> pari distribuzione. Non tutti a zero perchè vogliamo un po' di entropia.
    nn.init.normal_(output_layer.weight, std=0.01)
    nn.init.zeros_(output_layer.bias)

    return model


model = build()
n_params = sum(p.numel() for p in model.parameters())

print("\n=== 2. la rete ===\n")
print(f"  {model}\n")
print(f"  {n_params:,} parametri")
print(
    """
  800 nascosti e non 200: e' la larghezza di Simard, ed e' anche l'unico modo
  di sfruttare i dati nuovi. Senza deformazioni allargare la rete non serviva a
  niente — imparava le 50.000 a memoria comunque, solo prima."""
)


# ---------------------------------------------------------------------------
# 3. l'inizializzazione, verificata invece che sperata
# ---------------------------------------------------------------------------

print("\n=== 3. come parte ===\n")

with torch.no_grad():
    logits = model(xtr[:1000])
    hidden = torch.tanh(model[0](xtr[:1000]))

expected = -torch.tensor(1 / 10).log().item()
measured = F.cross_entropy(logits, ytr[:1000]).item()
saturated = (hidden.abs() > 0.97).float().mean().item() * 100

print(f"  loss attesa al passo 0:  -log(1/10) = {expected:.4f}")
print(f"  loss misurata:                        {measured:.4f}")

print(f"\n  attivazioni nascoste sature (|h| > 0.97): {saturated:.1f}%")
print(f"  deviazione standard delle pre-attivazioni: {model[0](xtr[:1000]).std():.2f}")

# ---------------------------------------------------------------------------
# 4. il training
# ---------------------------------------------------------------------------

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

weight_matrices = [m.weight for m in model if isinstance(m, nn.Linear)]


@torch.no_grad()
def evaluate(model, x, y, chunk=5000):
    """(loss media, accuratezza %) su un intero split.

    `model.eval()` qui non cambia niente — non ci sono dropout ne' batchnorm —
    ma la valutazione va scritta cosi' comunque: il giorno che si aggiunge uno
    dei due, dimenticarselo vuol dire misurare un'altra rete.

    Si valuta sempre sulle immagini *non* deformate: le deformazioni servono ad
    allenare, non a misurare.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    for k in range(0, len(x), chunk):
        logits = model(x[k : k + chunk])
        total_loss += F.cross_entropy(logits, y[k : k + chunk], reduction="sum").item()
        correct += (logits.argmax(dim=1) == y[k : k + chunk]).sum().item()
    model.train()
    return total_loss / len(x), correct / len(x) * 100


@torch.no_grad()
def update_data(before):
    """log10 di quanto si e' spostato ogni tensore, in frazione di se' stesso.

        update:data = std(passo appena fatto) / std(pesi)

    La regola pratica di 03_mlp/09_learning_speed.py e' che stia intorno a -3,
    un millesimo per passo. Il passo si misura *dopo* `optimizer.step()`,
    diffando i pesi: con AdamW non coincide con `lr * grad`, perche' in mezzo
    ci sono le due medie mobili.
    """
    return [
        ((w - w_before).std() / w.std()).log10().item()
        for w, w_before in zip(weight_matrices, before)
    ]


steps_per_epoch = len(xtr) // BATCH_SIZE

print("\n=== 4. il training ===\n")
print(f"  {EPOCHS} epoche, minibatch da {BATCH_SIZE}, AdamW lr={LEARNING_RATE} con cosine annealing")
print(f"  {steps_per_epoch} passi per epoca, {EPOCHS * steps_per_epoch:,} in tutto")
print(f"  il training set si rideforma all'inizio di ogni epoca\n")
print(f"  {'epoca':>6}  {'loss train':>11}  {'loss val':>9}  {'acc train':>10}  {'acc val':>8}")

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
ratios = []
best_val_acc = 0.0
best_epoch = 0
best_state = None
start = time.time()

for epoch in range(EPOCHS):
    # i dati nuovi dell'epoca: stesse etichette, immagini mai viste prima
    loader = DataLoader(
        TensorDataset(distort(xtr), ytr), batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb), yb).backward()

        before = [w.detach().clone() for w in weight_matrices]
        optimizer.step()
        ratios.append(update_data(before))

    scheduler.step()

    train_loss, train_acc = evaluate(model, xtr, ytr)
    val_loss, val_acc = evaluate(model, xva, yva)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    # i pesi che si tengono sono quelli con la validation migliore, non gli
    # ultimi: e' la stessa regola di 05_gpt/06_gpt.py
    if val_acc > best_val_acc:
        best_val_acc, best_epoch = val_acc, epoch
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    marker = "  <- migliore finora" if epoch == best_epoch else ""
    print(f"  {epoch:>6}  {train_loss:>11.4f}  {val_loss:>9.4f}  {train_acc:>9.2f}%  {val_acc:>7.2f}%{marker}")

elapsed = time.time() - start
model.load_state_dict(best_state)

print(f"\n  {elapsed / 60:.1f} minuti.")
print(f"  ripresi i pesi dell'epoca {best_epoch}, validation {best_val_acc:.2f}%")


# --- l'overfitting, che adesso non c'e' piu' ---

best_val_loss = min(history["val_loss"])
print(
    f"""
  La train accuracy si ferma a {history['train_acc'][-1]:.2f}% e la loss di training a
  {history['train_loss'][-1]:.4f}. La versione senza deformazioni di questo file finiva a
  100.00% e 0.0001: imparava le 50.000 immagini a memoria, e la loss di
  validation risaliva dopo l'epoca 9. Qui la loss di validation tocca il minimo
  di {best_val_loss:.4f} all'epoca {history['val_loss'].index(best_val_loss)} su {EPOCHS}, e le due curve restano appaiate.

  Non e' che abbiamo regolarizzato meglio: la rete e' quattro volte piu' grossa
  di prima e non c'e' ne' dropout ne' weight decay in piu'. E' che non esiste
  piu' un training set da imparare a memoria. Il dropout, provato sulla
  versione precedente, non recuperava niente (97.96% contro 98.14%), e la
  ragione si vede solo adesso: la regolarizzazione mette vincoli sulle stesse
  50.000 immagini, le deformazioni aggiungono informazione che nei dati non
  c'era — che un 5 storto e' ancora un 5."""
)


# --- quanto si sono mossi i pesi ---

ratios_t = torch.tensor(ratios)

print("\n  update:data in log10, media sulla prima e sull'ultima epoca:\n")
print(f"    {'':>22}  {'1a epoca':>9}  {'ultima':>8}")
for i, w in enumerate(weight_matrices):
    print(
        f"    layer {i} {str(tuple(w.shape)):>14}  "
        f"{ratios_t[:steps_per_epoch, i].mean():>9.2f}  {ratios_t[-steps_per_epoch:, i].mean():>8.2f}"
    )

print(
    """
  La regola pratica di 03_mlp/09_learning_speed.py e' che questo numero stia
  intorno a -3: un millesimo di se' stesso per passo. Nella prima epoca ci
  siamo. Alla fine e' piu' in basso perche' il cosine annealing ha portato il
  learning rate quasi a zero.

  Un update:data che *restasse* a -5 fin dalla prima epoca sarebbe tutt'altra
  cosa — il sintomo di un learning rate troppo basso — ed e' per questo che i
  due numeri si guardano insieme, mai solo l'ultimo."""
)


# ---------------------------------------------------------------------------
# 5. il test, una volta sola
# ---------------------------------------------------------------------------

test_loss, test_acc = evaluate(model, xte, yte)
errors = int(round((100 - test_acc) * len(xte) / 100))

print("\n=== 5. il test ===\n")
print(f"  loss {test_loss:.4f}, accuratezza {test_acc:.2f}%  ({errors} errori su {len(xte)})")
print(
    f"""
  Prima riga di codice che tocca il test, e l'ultima. La rete valutata qui e'
  quella dell'epoca {best_epoch}, tenuta perche' aveva la validation migliore
  ({best_val_acc:.2f}%) — non perche' avesse il test migliore, che non lo abbiamo mai
  guardato. Che validation e test cadano cosi' vicini e' il segno che scegliere
  su validation non ha barato.

  Il capitolo, con la stessa MLP a un layer nascosto, riporta 95.42%. La stessa
  rete scritta bene ma senza deformazioni fa 98.11%. Simard, con questa
  identica architettura e queste identiche deformazioni, riporta 0.7% di
  errore, cioe' 99.3%."""
)


# ---------------------------------------------------------------------------
# 6. i grafici
# ---------------------------------------------------------------------------

print("\n=== 6. i grafici ===")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

axes[0].plot(history["train_loss"], color="#2b6cb0", label="train")
axes[0].plot(history["val_loss"], color="#c05621", label="validation")
axes[0].set_xlabel("epoca")
axes[0].set_ylabel("cross-entropy")
axes[0].set_title("la loss (le due curve non si separano: niente overfitting)")
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].plot(history["train_acc"], color="#2b6cb0", label="train")
axes[1].plot(history["val_acc"], color="#c05621", label="validation")
axes[1].axhline(99, color="#38a169", linestyle=":", linewidth=1.2)
axes[1].annotate("99%", xy=(0, 99), xytext=(4, 3), textcoords="offset points",
                 color="#38a169", fontsize=9)
axes[1].set_xlabel("epoca")
axes[1].set_ylabel("accuratezza (%)")
axes[1].set_title("l'accuratezza")
axes[1].axvline(best_epoch, color="#4a5568", linestyle="--", linewidth=1)
axes[1].annotate(
    f"epoca {best_epoch}: i pesi che teniamo",
    xy=(best_epoch, history["val_acc"][best_epoch]),
    xytext=(6, -16), textcoords="offset points", color="#4a5568", fontsize=9,
)
axes[1].legend(loc="lower right")
axes[1].grid(alpha=0.25)

for i, w in enumerate(weight_matrices):
    axes[2].plot(ratios_t[:, i], linewidth=0.6, alpha=0.75, label=f"layer {i} {tuple(w.shape)}")
axes[2].axhline(-3, color="#4a5568", linestyle="--", linewidth=1)
axes[2].annotate("-3, la regola pratica", xy=(0, -3), xytext=(4, 4),
                 textcoords="offset points", color="#4a5568", fontsize=9)
axes[2].set_xlabel("passo")
axes[2].set_ylabel("log10(update : data)")
axes[2].set_title("quanto si muovono i pesi")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.25)

plt.tight_layout()
plt.savefig(HERE / "01_out_training.png", bbox_inches="tight", dpi=100)
plt.close()
print("\n  grafici in 01_out_training.png")

with torch.no_grad():
    weights = model[0].weight[:40].clone()

fig, axes = plt.subplots(4, 10, figsize=(14, 6))
limit = weights.abs().max().item()
for i, ax in enumerate(axes.flat):
    ax.imshow(weights[i].reshape(28, 28), cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(f"i 784 pesi di 40 dei {HIDDEN} neuroni nascosti, rimessi in forma 28x28")
plt.tight_layout()
plt.savefig(HERE / "01_out_hidden_neurons.png", bbox_inches="tight", dpi=100)
plt.close()
print("  immagine in 01_out_hidden_neurons.png")


# ---------------------------------------------------------------------------
# 7. i pesi, per la rete che gira nel browser
# ---------------------------------------------------------------------------

# 02_demo.html rifa' questo stesso forward in quindici righe di JavaScript, e
# per farlo gli servono i pesi. I quattro tensori si mettono in fila e si
# scrivono in base64: W1 (HIDDEN, 784) per righe, b1, W2 (10, HIDDEN) per
# righe, b2, tutti float32 little-endian.
#
# Esce un .js e non un .json per un motivo pratico: un <script> si carica anche
# con file://, un fetch() no, lo blocca la CORS policy. Cosi' la pagina si apre
# col doppio click, senza tirare su un server.

with torch.no_grad():
    flat = torch.cat([
        model[0].weight.flatten(),
        model[0].bias,
        model[2].weight.flatten(),
        model[2].bias,
    ])

raw = flat.numpy().astype("<f4").tobytes()
payload = base64.b64encode(raw).decode("ascii")

(HERE / "01_out_weights.js").write_text(
    f"""// generato da 01_network.py - non si modifica a mano.
// W1 ({HIDDEN}, 784), b1 ({HIDDEN}), W2 (10, {HIDDEN}), b2 (10) in fila, float32 little-endian.
const WEIGHTS = {{
  hidden: {HIDDEN},
  params: {n_params},
  test_accuracy: {test_acc:.2f},
  data: "{payload}",
}};
""",
    encoding="ascii",
)

print("\n=== 7. i pesi per la demo ===\n")
print(f"  {len(raw):,} byte di float32, {len(payload):,} caratteri in base64")
print("  scritti in 01_out_weights.js, che 02_demo.html carica con un <script>")


# ---------------------------------------------------------------------------
# 8. quelle che sbaglia
# ---------------------------------------------------------------------------

with torch.no_grad():
    model.eval()
    predicted = torch.cat([model(xte[k : k + 5000]).argmax(dim=1) for k in range(0, len(xte), 5000)])
wrong = (predicted != yte).nonzero().flatten()

print(f"\n=== 8. le {len(wrong)} che sbaglia ===\n")
for i in wrong[:2].tolist():
    print(f"  vera: {yte[i].item()}, la rete dice: {predicted[i].item()}\n")
    print(mnist.draw(xte[i]))
    print()
print(
    """  Parecchie sono discutibili anche per noi. Il record su MNIST e' 99.79%, e a
  quel livello si e' oltre il punto in cui due persone sono d'accordo su cosa
  ci sia scritto."""
)
