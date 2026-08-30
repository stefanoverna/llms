"""
La rete del capitolo 1 di Nielsen, ma scritta con quello che sappiamo adesso.

Il problema è quello del libro — 784 pixel in ingresso, dieci cifre in uscita,
una MLP con un layer nascosto, ma seguiamo le best practices che conosciamo.

    784 ingressi     i pixel dell'immagine, 28x28 appiattiti
    200 nascosti     tanh
     10 uscite       logits, nessuna attivazione sopra

Il libro arriva al 95% di accuratezza, noi 98%.

**I pesi che si tengono** sono quelli dell'epoca con la validation migliore,
non quelli dell'ultima epoca: è la regola di `05_gpt/06_gpt.py`, e qui serve
davvero, perché la rete overfitta. La train accuracy arriva a 100% e la loss di
validation, dopo il minimo, risale — mentre l'accuratezza di validation, nello
stesso intervallo, continua a *salire*. Le due misure divergono perché la
cross-entropy è `-log p(giusta)`: limitata a zero quando indovini, illimitata
quando sbagli. A fine training le poche immagini sbagliate sono il 2% del set e
pesano per il 93% della loss, quindi la media racconta soprattutto quanto male
fallisce su quelle — e la rete, continuando a spingere i margini, diventa più
sicura di sé anche dove ha torto. L'accuratezza guarda solo l'argmax e non se
ne accorge.

**Due cose che abbiamo provato e non messo**, perché misurate non pagano.
`nn.BatchNorm1d` fra il layer lineare e la tanh porta la validation da 96.52% a
96.11%, ed è la conclusione a cui era già arrivato `03_mlp/03_batchnorm.py`: con
*un* layer nascosto la scala giusta si calcola a mano, il problema che la
batchnorm risolve nasce a cinquanta. E il dropout a 0.2 abbassa la validation,
97.96% contro 98.14% — che è più sorprendente, visto che la rete
overfitta eccome, e vuol dire che su MNIST quel divario fra train e validation è
quasi tutto irriducibile. Metterli lo stesso sarebbe stato culto del cargo.
"""

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

HIDDEN = 200
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3

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

train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)


# ---------------------------------------------------------------------------
# 1. la rete
# ---------------------------------------------------------------------------


def build():
    """784 -> 200 con tanh -> 10 logits.
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

print("\n=== 1. la rete ===\n")
print(f"  {model}\n")
print(f"  {n_params:,} parametri")


# ---------------------------------------------------------------------------
# 2. l'inizializzazione, verificata invece che sperata
# ---------------------------------------------------------------------------

print("\n=== 2. come parte ===\n")

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
# 3. il training
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


print("\n=== 3. il training ===\n")
print(f"  {EPOCHS} epoche, minibatch da {BATCH_SIZE}, AdamW lr={LEARNING_RATE} con cosine annealing")
print(f"  {len(train_loader)} passi per epoca, {EPOCHS * len(train_loader):,} in tutto\n")
print(f"  {'epoca':>6}  {'loss train':>11}  {'loss val':>9}  {'acc train':>10}  {'acc val':>8}")

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
ratios = []
best_val_acc = 0.0
best_epoch = 0
best_state = None
start = time.time()

for epoch in range(EPOCHS):
    for xb, yb in train_loader:
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

print(f"\n  {elapsed:.0f} secondi.")
print(f"  ripresi i pesi dell'epoca {best_epoch}, validation {best_val_acc:.2f}%")


# --- l'overfitting, che c'e' e si vede ---

best_val_loss = min(history["val_loss"])
print(
    f"""
  La train accuracy arriva a {history['train_acc'][-1]:.2f}% e la loss di training a
  {history['train_loss'][-1]:.4f}: la rete impara le 50.000 immagini a memoria. Intanto la
  loss di validation tocca il minimo di {best_val_loss:.4f} all'epoca
  {history['val_loss'].index(best_val_loss)}, e poi *risale* fino a {history['val_loss'][-1]:.4f}. E' overfitting da manuale.

  Ma l'accuratezza di validation nello stesso intervallo non peggiora affatto:
  continua a salire, fino al massimo dell'epoca {best_epoch}. Le due misure non
  dicono la stessa cosa. La cross-entropy e' -log p(giusta), limitata a zero
  quando indovini e illimitata quando sbagli, quindi la media e' quasi tutta un
  resoconto di *quanto male* fallisce sulle poche che sbaglia — e quelle
  rincarano, perche' la rete diventa piu' sicura di se' anche dove ha torto.
  L'argmax non se ne accorge. Il dropout a 0.2, provato a parte, non recupera
  niente
  (97.96% contro 98.14%): quel divario fra train e validation e' quasi tutto
  irriducibile, non e' rumore da regolarizzare via.

  Il costo pratico e' comunque zero, perche' i pesi che teniamo sono quelli
  dell'epoca migliore su validation. Le epoche dopo la {best_epoch} sono tempo di
  calcolo buttato, non un danno."""
)


# --- quanto si sono mossi i pesi ---

ratios_t = torch.tensor(ratios)
steps_per_epoch = len(train_loader)

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
  siamo. Alla fine e' molto piu' in basso, e non e' un problema, e' quello che
  deve succedere: il cosine annealing ha portato il learning rate quasi a zero
  e la loss di training e' a 0.0001, quindi i gradienti sono minuscoli.

  Un update:data che *restasse* a -5 fin dalla prima epoca sarebbe tutt'altra
  cosa — il sintomo di un learning rate troppo basso — ed e' per questo che i
  due numeri si guardano insieme, mai solo l'ultimo."""
)


# ---------------------------------------------------------------------------
# 4. il test, una volta sola
# ---------------------------------------------------------------------------

test_loss, test_acc = evaluate(model, xte, yte)

print("\n=== 4. il test ===\n")
print(f"  loss {test_loss:.4f}, accuratezza {test_acc:.2f}%  ({int(round((100 - test_acc) * len(xte) / 100))} errori su {len(xte)})")
print(
    f"""
  Prima riga di codice che tocca il test, e l'ultima. La rete valutata qui e'
  quella dell'epoca {best_epoch}, tenuta perche' aveva la validation migliore
  ({best_val_acc:.2f}%) — non perche' avesse il test migliore, che non lo abbiamo mai
  guardato. Che validation e test cadano cosi' vicini e' il segno che scegliere
  su validation non ha barato.

  Il capitolo, con la stessa MLP a un layer nascosto, riporta 95.42%."""
)


# ---------------------------------------------------------------------------
# 5. i grafici
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

axes[0].plot(history["train_loss"], color="#2b6cb0", label="train")
axes[0].plot(history["val_loss"], color="#c05621", label="validation")
axes[0].set_xlabel("epoca")
axes[0].set_ylabel("cross-entropy")
axes[0].set_title("la loss (quella di validation risale: overfitting)")
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].plot(history["train_acc"], color="#2b6cb0", label="train")
axes[1].plot(history["val_acc"], color="#c05621", label="validation")
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
# 6. quelle che sbaglia
# ---------------------------------------------------------------------------

with torch.no_grad():
    model.eval()
    predicted = torch.cat([model(xte[k : k + 5000]).argmax(dim=1) for k in range(0, len(xte), 5000)])
wrong = (predicted != yte).nonzero().flatten()

print(f"\n=== 5. le {len(wrong)} che sbaglia ===\n")
for i in wrong[:2].tolist():
    print(f"  vera: {yte[i].item()}, la rete dice: {predicted[i].item()}\n")
    print(mnist.draw(xte[i]))
    print()
print(
    """  Parecchie sono discutibili anche per noi. Il record su MNIST e' 99.79%, e a
  quel livello si e' oltre il punto in cui due persone sono d'accordo su cosa
  ci sia scritto."""
)
