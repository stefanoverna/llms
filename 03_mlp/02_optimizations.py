"""
makemore, parte 3 (prima metà): l'inizializzazione.

`01_mlp.py` si chiude dicendo che la rete parte da loss 26 invece che da 3.30, e
che quella è la cosa più costosa del file. Qui la sistemiamo, seguendo la
lezione 3 (`02_out_lecture.srt`), che affronta il problema in due tempi:

  1. i logits partono troppo grandi -> il modello è sicuro di sé a caso ->
     la loss iniziale è 26 invece di 3.30
  2. le pre-attivazioni partono troppo grandi -> la tanh è satura -> i
     gradienti non passano più

I due problemi hanno la stessa forma (un tensore troppo largo) e la stessa cura
(rimpicciolire i pesi che lo producono), ma sintomi molto diversi: il primo si
legge nella loss, il secondo no. Una rete con solo il primo problema risolto
sembra a posto, e non lo è. Il secondo si vede solo andando a guardare *dentro*
la rete, che è poi il tema della lezione.

E alla fine: quel "numero piccolo" per cui moltiplicare non si sceglie a occhio.
C'è una formula, e si chiama inizializzazione di Kaiming.

Le tre reti allenate qui sono identiche a quella di `01_mlp.py` — stessi dati,
stessi iperparametri, stesso seed, stessi 200k passi. Cambia solo come partono i
pesi. Per questo il primo numero della tabella finale coincide con quello di
`01_mlp.py`: è la stessa rete.
"""

import random
from pathlib import Path

import torch
from torch.nn import functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 2147483647
HERE = Path(__file__).parent
NAMES = HERE.parent / "02_bigram" / "names.txt"

BLOCK_SIZE = 3
EMB_DIM = 10
HIDDEN = 200
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# 0. dataset e rete: gli stessi di 01_mlp.py, in versione compatta
# ---------------------------------------------------------------------------

words = open(NAMES, "r").read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
VOCAB = len(itos)

random.seed(42)
random.shuffle(words)
n_train, n_dev = int(0.8 * len(words)), int(0.9 * len(words))


def build_dataset(word_list):
    X, Y = [], []
    for word in word_list:
        context = [0] * BLOCK_SIZE
        for ch in word + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


Xtr, Ytr = build_dataset(words[:n_train])
Xdev, Ydev = build_dataset(words[n_train:n_dev])


def make_params(fix_logits=False, fix_tanh=False, kaiming=False):
    """I soliti C, W1, b1, W2, b2, con le correzioni della lezione opzionali."""
    g = torch.Generator().manual_seed(SEED)
    C = torch.randn((VOCAB, EMB_DIM), generator=g)
    W1 = torch.randn((BLOCK_SIZE * EMB_DIM, HIDDEN), generator=g)
    b1 = torch.randn(HIDDEN, generator=g)
    W2 = torch.randn((HIDDEN, VOCAB), generator=g)
    b2 = torch.randn(VOCAB, generator=g)

    if fix_logits:
        W2 = W2 * 0.01  # logits piccoli, ma non zero: serve un po' di entropia
        b2 = b2 * 0.0  # il bias invece a zero e basta
    if fix_tanh:
        W1 = W1 * 0.2  # il valore trovato a occhio nella lezione
        b1 = b1 * 0.01
    if kaiming:
        W1 = W1 * (5 / 3) / (BLOCK_SIZE * EMB_DIM) ** 0.5
        b1 = b1 * 0.01

    params = [C, W1, b1, W2, b2]
    for p in params:
        p.requires_grad = True
    return params


def forward(params, X):
    """Come in 01_mlp.py, ma restituisce anche hpreact: è quello che vogliamo guardare."""
    C, W1, b1, W2, b2 = params
    emb = C[X]
    hpreact = emb.view(emb.shape[0], -1) @ W1 + b1
    h = torch.tanh(hpreact)
    return h @ W2 + b2, hpreact, h


@torch.no_grad()
def evaluate(params, X, Y):
    return F.cross_entropy(forward(params, X)[0], Y).item()


def train(params, steps=200_000):
    g = torch.Generator().manual_seed(SEED)
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
        loss = F.cross_entropy(forward(params, Xtr[ix])[0], Ytr[ix])
        for p in params:
            p.grad = None
        loss.backward()
        with torch.no_grad():
            for p in params:
                p -= (0.1 if i < steps // 2 else 0.01) * p.grad


# ---------------------------------------------------------------------------
# 1. primo problema: la loss iniziale è 26, e dovrebbe essere 3.30
# ---------------------------------------------------------------------------

print("=== 1. la loss iniziale ===\n")

net_base = make_params()
batch = Xtr[:BATCH_SIZE]
with torch.no_grad():
    logits, hpreact, h = forward(net_base, batch)

print(f"  loss al passo 0:                  {F.cross_entropy(logits, Ytr[:BATCH_SIZE]).item():.2f}")
print(f"  quella che ci aspettiamo, -log(1/27): {-torch.tensor(1 / VOCAB).log().item():.2f}")
print(f"  logits della prima riga: da {logits[0].min():.1f} a {logits[0].max():.1f}")

# Perché ci aspettiamo 3.30: al passo 0 la rete non ha nessun motivo per
# credere che una lettera sia più probabile di un'altra, quindi la distribuzione
# onesta è uniforme, 1/27 ciascuna, e la loss è -log(1/27) = 3.2958.
#
# Il meccanismo si vede meglio su un giocattolo con 4 sole classi, dove
# l'uniforme vale -log(1/4) = 1.386:

print("\n  lo stesso problema con 4 classi, su 1000 estrazioni a caso:\n")
print(f"    {'logits':<16} {'loss media':>11} {'la migliore':>12} {'la peggiore':>12}")
for scale, label in ((0.0, "tutti zero"), (1.0, "casuali"), (10.0, "casuali x10")):
    gen = torch.Generator().manual_seed(SEED)
    toy_logits = torch.randn(1000, 4, generator=gen) * scale
    toy_targets = torch.randint(0, 4, (1000,), generator=gen)
    toy_losses = F.cross_entropy(toy_logits, toy_targets, reduction="none")
    # clamp perché una loss quasi zero stampata a due decimali esce come "-0.00"
    print(
        f"    {label:<16} {toy_losses.mean():>11.2f}"
        f" {toy_losses.min().clamp(min=0):>12.2f} {toy_losses.max():>12.2f}"
    )

print(
    """
  Con i logits a zero la softmax dà 1/4 a ciascuna classe e la loss è 1.386,
  sempre, qualunque sia la risposta giusta: è l'ignoranza onesta. Man mano che i
  logits si allargano la media peggiora, e in fretta.

  Guarda però la colonna "la migliore": allargando i logits ogni tanto si va
  *meglio* di 1.386. Succede quando per caso il logit più alto capita proprio
  sulla risposta giusta — una volta su quattro, qui. Ma è una scommessa, e le
  altre tre volte si paga carissimo: è la colonna "la peggiore" a spiegare la
  media. Ecco perché non basta guardare una riga di logits per convincersi che
  l'inizializzazione sia buona.

  Nota che non conta che i logits siano zero, conta che siano tutti *uguali*:
  la softmax normalizza, quindi una costante aggiunta a tutti non cambia niente.
  Zero è semplicemente la scelta simmetrica.
"""
)


# ---------------------------------------------------------------------------
# 2. la cura: rimpicciolire W2 e azzerare b2
# ---------------------------------------------------------------------------

print("=== 2. logits sistemati ===\n")

# I logits sono h @ W2 + b2. Per farli venire piccoli: b2 non deve aggiungere
# rumore (quindi zero), e W2 va moltiplicata per un numero piccolo.
#
# Perché 0.01 e non 0? Con W2 a zero i logits sarebbero esattamente uguali e la
# loss esattamente 3.2958, ma tutti i neuroni dell'ultimo layer partirebbero
# identici. Un pizzico di rumore rompe la simmetria, che è quello che permette
# a neuroni diversi di specializzarsi in cose diverse.

net_fix_logits = make_params(fix_logits=True)
with torch.no_grad():
    logits_small, hpreact_saturated, h_saturated = forward(net_fix_logits, batch)

print(f"  loss al passo 0: {F.cross_entropy(logits_small, Ytr[:BATCH_SIZE]).item():.2f}   (era 26 circa)")
print(f"  logits della prima riga: da {logits_small[0].min():.2f} a {logits_small[0].max():.2f}")


# ---------------------------------------------------------------------------
# 3. secondo problema: la tanh è satura, e la loss non lo dice
# ---------------------------------------------------------------------------

print("\n=== 3. dentro la rete: le attivazioni ===\n")

# La loss iniziale adesso è giusta, quindi sembrerebbe tutto a posto. Ma la loss
# guarda solo l'uscita. Andiamo a vedere h, cioè cosa esce dalla tanh.

saturated_pct = (h_saturated.abs() > 0.99).float().mean().item() * 100
dead_neurons = (h_saturated.abs() > 0.99).all(dim=0).sum().item()

print(f"  pre-attivazioni (prima della tanh): da {hpreact_saturated.min():.1f} a {hpreact_saturated.max():.1f}")
print(f"  attivazioni con |h| > 0.99:         {saturated_pct:.0f}% di tutte")
print(f"  neuroni morti (saturi su ogni esempio del batch): {dead_neurons} su {HIDDEN}")

print(
    """
  La tanh schiaccia qualsiasi numero dentro (-1, 1). Se le pre-attivazioni
  arrivano a ±15, quello che esce è ±1 quasi sempre: la tanh non sta
  trasformando niente, sta solo decidendo un segno.

  Il guaio non è nel forward, è nel backward. In micrograd la derivata della
  tanh era

      dx = (1 - t**2) * dout        con t = tanh(x)

  e con t = ±1 quel fattore è zero. Il gradiente non viene attenuato: viene
  *azzerato*. Ogni neurone saturo, su ogni esempio in cui è saturo, non riceve
  nessuna informazione e non impara niente da quell'esempio. È coerente con
  l'intuizione: se sei nella parte piatta della tanh, cambiare i pesi non cambia
  l'uscita, quindi non cambia la loss, quindi il gradiente è zero.

  Il caso peggiore è il neurone *morto*: saturo per ogni esempio del dataset.
  Quello non imparerà mai niente, per sempre. Nella mappa qui sotto sarebbe una
  colonna tutta bianca.
"""
)

tanh_grad_factor = (1 - h_saturated**2)
print(f"  fattore (1 - t^2) che moltiplica il gradiente: mediana {tanh_grad_factor.median():.4f}")
print(f"  quante attivazioni lo hanno sotto 0.01: {(tanh_grad_factor < 0.01).float().mean().item() * 100:.0f}%")


# ---------------------------------------------------------------------------
# 4. la cura: rimpicciolire anche W1
# ---------------------------------------------------------------------------

print("\n=== 4. tanh sistemata ===\n")

net_fix_tanh = make_params(fix_logits=True, fix_tanh=True)
with torch.no_grad():
    _, hpreact_ok, h_ok = forward(net_fix_tanh, batch)

print(f"  pre-attivazioni: da {hpreact_ok.min():.1f} a {hpreact_ok.max():.1f}   (erano ±15)")
print(f"  attivazioni con |h| > 0.99: {(h_ok.abs() > 0.99).float().mean().item() * 100:.0f}%")
print(f"  neuroni morti: {(h_ok.abs() > 0.99).all(dim=0).sum().item()}")


# ---------------------------------------------------------------------------
# 5. i grafici: prima e dopo
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for row, (pre_tanh, post_tanh, label) in enumerate(
    ((hpreact_saturated, h_saturated, "come in 01_mlp.py"), (hpreact_ok, h_ok, "con W1 * 0.2"))
):
    axes[row][0].hist(pre_tanh.flatten().tolist(), bins=50)
    axes[row][0].set_title(f"pre-attivazioni, {label}")
    axes[row][1].hist(post_tanh.flatten().tolist(), bins=50)
    axes[row][1].set_title(f"dopo la tanh, {label}")
    axes[row][1].set_xlim(-1, 1)
plt.tight_layout()
plt.savefig(HERE / "02_out_init_activations.png", bbox_inches="tight", dpi=100)
plt.close()

fig, axes = plt.subplots(2, 1, figsize=(11, 5))
for ax, activations, label in ((axes[0], h_saturated, "come in 01_mlp.py"), (axes[1], h_ok, "con W1 * 0.2")):
    ax.imshow((activations.abs() > 0.99).T.numpy(), cmap="gray", aspect="auto", interpolation="nearest")
    ax.set_title(f"|h| > 0.99, {label}   (bianco = saturo)")
    ax.set_xlabel("esempio del batch")
    ax.set_ylabel("neurone")
plt.tight_layout()
plt.savefig(HERE / "02_out_init_saturation.png", bbox_inches="tight", dpi=100)
plt.close()

print(f"\n  istogrammi in {HERE / '02_out_init_activations.png'}")
print(f"  mappa della saturazione in {HERE / '02_out_init_saturation.png'}")


# ---------------------------------------------------------------------------
# 6. da dove viene 0.2: l'inizializzazione di Kaiming
# ---------------------------------------------------------------------------

print("\n=== 6. il numero giusto, invece che a occhio ===\n")

# 0.2 l'abbiamo trovato guardando l'istogramma finché non sembrava ragionevole.
# Ma esiste il numero giusto, e si ricava chiedendosi: se l'ingresso di un layer
# è una gaussiana con deviazione standard 1, quanto devono essere grandi i pesi
# perché anche l'uscita abbia deviazione standard 1?
#
# Moltiplicare per una matrice *allarga* la distribuzione, e di quanto dipende
# da quanti ingressi si sommano in ogni neurone (il "fan in").

fan_in = BLOCK_SIZE * EMB_DIM
g = torch.Generator().manual_seed(SEED)
x = torch.randn(1000, fan_in, generator=g)
print(f"  ingresso: deviazione standard {x.std():.2f}\n")
print(f"  {'pesi':<24} {'std prima della tanh':>21} {'dopo la tanh':>14}")
for scale, label in (
    (1.0, "W come randn"),
    (1 / fan_in**0.5, "W / sqrt(30)"),
    (0.2, "W * 0.2 (a occhio)"),
    ((5 / 3) / fan_in**0.5, "W * 5/3/sqrt(30)"),
):
    W = torch.randn(fan_in, HIDDEN, generator=torch.Generator().manual_seed(SEED)) * scale
    y = x @ W
    print(f"  {label:<24} {y.std():>21.2f} {torch.tanh(y).std():>14.2f}")

print(
    """
  Le righe si leggono così. Con W presa da randn la moltiplicazione allarga di
  cinque volte e mezza, e la tanh non può che saturare. Dividendo per la radice
  del fan in la deviazione standard torna esattamente a 1: è il conto sulla
  varianza, ogni neurone somma 30 contributi indipendenti e sommare 30 cose
  moltiplica la deviazione per sqrt(30), quindi si divide per quella.

  Ma guarda la colonna dopo: la tanh *stringe*, e da 1.00 si scende a 0.63.
  Layer dopo layer quel restringimento si accumula finché il segnale sparisce.
  Quindi si compensa allargando un po' in partenza, e quel "un po'" è il gain,
  che per la tanh vale 5/3. È la formula di Kaiming (dal paper "Delving Deep
  into Rectifiers"), in PyTorch torch.nn.init.kaiming_normal_:

      std = gain / sqrt(fan_in)

  Per noi 5/3 / sqrt(30) = 0.304, e infatti dopo la tanh si sta molto più vicini
  a 1 che con lo 0.2 scelto a occhio. Non ci si arriva esattamente, perché di
  quanto la tanh stringa dipende dalla distribuzione che le arriva: il gain è
  una correzione tarata bene, non un'identità esatta.

  Il punto vero è un altro. Qui c'è un layer solo, e sbagliare la scala costa
  qualche centesimo di loss. In una rete profonda l'errore si compone a ogni
  layer: un fattore 1.5 ripetuto cinquanta volte fa esplodere le attivazioni,
  un fattore 0.7 le fa sparire. È il motivo per cui questa formula esiste, ed è
  il motivo per cui prima del 2015 le reti profonde erano difficili da allenare.
"""
)


# ---------------------------------------------------------------------------
# 7. quanto vale, in loss
# ---------------------------------------------------------------------------

print("=== 7. le quattro reti, allenate ===\n")
print("  (200k passi ciascuna come in 01_mlp.py, ci vogliono un paio di minuti)\n")

print(f"  {'inizializzazione':<34} {'loss0':>7} {'train':>8} {'dev':>8}")
for label, kwargs in (
    ("come in 01_mlp.py", {}),
    ("+ logits piccoli (W2, b2)", {"fix_logits": True}),
    ("+ tanh non satura (W1, b1)", {"fix_logits": True, "fix_tanh": True}),
    ("+ kaiming invece che a occhio", {"fix_logits": True, "kaiming": True}),
):
    params = make_params(**kwargs)
    loss0 = evaluate(params, Xtr, Ytr)
    train(params)
    print(f"  {label:<34} {loss0:>7.2f} {evaluate(params, Xtr, Ytr):>8.4f} {evaluate(params, Xdev, Ydev):>8.4f}")

# Quello che esce, e che ricalca la lezione quasi cifra per cifra:
#
#     inizializzazione                  dev     Karpathy
#     come in 01_mlp.py                 2.1731         2.17
#     + logits piccoli               2.1296         2.13
#     + tanh non satura              2.1043         2.10
#     + kaiming invece che a occhio  2.1070           --
#
# Il primo numero è identico a quello di 01_mlp.py, come deve: è la stessa rete.
# I due guadagni veri sono i primi due, -0.044 e -0.025, presi senza aggiungere
# un parametro né un secondo di training.
#
# L'ultima riga invece è un pareggio, anzi Kaiming perde di 0.0027 contro lo 0.2
# scelto guardando l'istogramma. Non è un fallimento della formula: con un layer
# solo qualsiasi scala ragionevole va bene, e la differenza fra 0.2 e 0.304 si
# perde nel rumore. Il valore di gain / sqrt(fan_in) non è battere lo 0.2 qui,
# è non dover guardare nessun istogramma quando i layer sono cinquanta e gli
# errori si moltiplicano fra loro.


# ---------------------------------------------------------------------------
# cosa ci portiamo dietro
# ---------------------------------------------------------------------------
#
# 1. Prima di allenare, calcola che loss ti aspetti al passo 0. Per una
#    classificazione a N classi è log(N). Se il numero che vedi è molto più
#    grande, l'inizializzazione è sbagliata e lo sai prima di sprecare ore.
#
# 2. La loss è un riassunto dell'uscita e non dice niente su cosa succede in
#    mezzo. La rete con i soli logits sistemati ha la loss iniziale giusta e la
#    tanh completamente satura: due problemi indipendenti, un sintomo solo.
#
# 3. Un neurone saturo non impara, perché il fattore (1 - t^2) del backward
#    azzera il suo gradiente. Se è saturo su *tutti* gli esempi è morto, e non
#    imparerà mai più. Guardare l'istogramma delle attivazioni è il modo più
#    veloce per accorgersene.
#
# 4. La scala giusta dei pesi non è a occhio: è gain / sqrt(fan_in). Su un layer
#    solo cambia poco, su cinquanta è la differenza fra allenare e non allenare.
#
# 5. Quello che resta da fare, ed è la seconda metà della lezione 3: invece di
#    azzeccare l'inizializzazione perché le attivazioni restino ragionevoli, si
#    può normalizzarle a mano a ogni passo. Si chiama batch normalization.
