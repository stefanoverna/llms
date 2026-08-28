"""
Cosa la batchnorm ti toglie dalle mani, e cosa ti lascia.

`07_gain_depth.py`, `08_gain_gradients.py` e `09_learning_speed.py` mostrano
tre modi diversi in cui una rete profonda si rompe se il gain è sbagliato.
Questo file li mette tutti e quattro uno accanto all'altro, la stessa rete con
e senza batchnorm, e quattro gain diversi, per rispondere a una domanda sola
(`02_out_lecture.srt`, dal minuto 1:47):

    con una normalizzazione in mezzo, che cosa resta da calibrare a mano?

La risposta della lezione è "il learning rate", e qui si vede perché. La
batchnorm rende l'uscita indipendente dalla scala dei pesi che la precedono —
se moltiplichi W per k, media e deviazione standard del batch si moltiplicano
per k anche loro, e la divisione le cancella. Ma quella scala non sparisce: si
trasferisce tutta sulla velocità con cui i pesi si aggiornano.

Le quattro diagnostiche, misurate layer per layer su un singolo forward +
backward all'inizializzazione:

    attivazioni       std dell'uscita di ogni tanh
    grad attivazioni  std del gradiente su quelle stesse uscite
    grad pesi         std del gradiente su ogni matrice di pesi
    update:data       log10( std(lr * grad) / std(W) ), il grafico di 09

Il grafico è in 10_out_batchnorm_robustness.png: due righe (senza e con
batchnorm), quattro colonne, e in ogni pannello una curva per gain.
"""

import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 2147483647
HERE = Path(__file__).parent
NAMES = HERE.parent / "02_bigram" / "names.txt"

BLOCK_SIZE = 3
EMB_DIM = 10
HIDDEN = 100
DEPTH = 5
BATCH = 1000
LR = 0.1

GAINS = [0.5, 1.0, 5 / 3, 3.0]


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

words = open(NAMES, "r").read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
VOCAB = len(stoi)

random.seed(42)
random.shuffle(words)

X, Y = [], []
for word in words[: int(0.8 * len(words))]:
    context = [0] * BLOCK_SIZE
    for ch in word + ".":
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]
Xtr, Ytr = torch.tensor(X), torch.tensor(Y)


def build(gain, batchnorm):
    """La rete di 09_learning_speed.py, con o senza una batchnorm prima di ogni tanh.

    L'ultimo layer è inizializzato sempre allo stesso modo, indipendentemente
    dal gain: non ha una batchnorm dopo di sé, quindi la sua scala arriverebbe
    dritta sui logits e sporcherebbe il confronto. Vogliamo che l'unica cosa a
    cambiare sia la scala dei layer *normalizzati*.
    """
    torch.manual_seed(SEED)
    layers = [nn.Embedding(VOCAB, EMB_DIM), nn.Flatten()]
    fan_in = BLOCK_SIZE * EMB_DIM
    for _ in range(DEPTH):
        linear = nn.Linear(fan_in, HIDDEN, bias=not batchnorm)
        nn.init.normal_(linear.weight, std=gain / fan_in**0.5)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)
        layers.append(linear)
        if batchnorm:
            layers.append(nn.BatchNorm1d(HIDDEN))
        layers.append(nn.Tanh())
        fan_in = HIDDEN
    last = nn.Linear(HIDDEN, VOCAB)
    nn.init.normal_(last.weight, std=0.01)
    nn.init.zeros_(last.bias)
    return nn.Sequential(*layers, last)


def diagnose(net, x, y):
    """Un forward e un backward, e le quattro misure per ogni layer nascosto."""
    acts, hidden_weights = [], []
    h = x
    for layer in net:
        h = layer(h)
        if isinstance(layer, nn.Tanh):
            h.retain_grad()
            acts.append(h)
        elif isinstance(layer, nn.Linear):
            hidden_weights.append(layer.weight)
    hidden_weights = hidden_weights[:-1]  # l'ultimo non è un layer nascosto

    F.cross_entropy(h, y).backward()

    last = acts[-1].detach()
    counts, edges = torch.histogram(last, bins=60, range=(-1.0, 1.0))

    return {
        "istogramma": (edges[:-1].tolist(), counts.tolist()),
        "saturi": (last.abs() > 0.97).float().mean().item() * 100,
        "attivazioni": [a.std().item() for a in acts],
        "grad attivazioni": [a.grad.std().item() for a in acts],
        "grad pesi": [w.grad.std().item() for w in hidden_weights],
        "update:data": [((LR * w.grad).std() / w.std()).log10().item() for w in hidden_weights],
    }


torch.manual_seed(SEED)
ix = torch.randint(0, Xtr.shape[0], (BATCH,))
xb, yb = Xtr[ix], Ytr[ix]

METRICS = ["attivazioni", "grad attivazioni", "grad pesi", "update:data"]

# la stessa cosa scritta per esteso: le prime due misurano valori che passano
# per la rete, le altre due i parametri e quanto si spostano
SUBTITLE = {
    "attivazioni": "std dell'uscita di ogni tanh",
    "grad attivazioni": "std di dL/dh, sulle stesse uscite",
    "grad pesi": "std di dL/dW, una matrice per layer",
    "update:data": "log10( std(lr · dL/dW) / std(W) )",
}

results = {}

for batchnorm in (False, True):
    for gain in GAINS:
        results[(batchnorm, gain)] = diagnose(build(gain, batchnorm), xb, yb)

print("=== le quattro misure ===\n")
for m in METRICS:
    print(f"  {m:>18}   {SUBTITLE[m]}")

for batchnorm in (False, True):
    label = "CON batchnorm" if batchnorm else "SENZA batchnorm"
    lo_g, hi_g = GAINS[0], GAINS[-1]
    print(f"\n=== {label} ===\n")
    print(f"  di quanto si sposta ogni misura passando da gain {lo_g} a gain {hi_g}"
          f" (un fattore {hi_g / lo_g:.0f}), layer per layer:\n")
    print(f"  {'':>18}" + "".join(f"{'layer ' + str(i):>10}" for i in range(1, DEPTH + 1)))
    for m in METRICS:
        lo = results[(batchnorm, lo_g)][m]
        hi = results[(batchnorm, hi_g)][m]
        if m == "update:data":  # è già in log10: la differenza sono decadi
            cells = "".join(f"{b - a:>+10.2f}" for a, b in zip(lo, hi))
            print(f"  {m:>18}" + cells + "   decadi")
        else:
            cells = "".join(f"{'x%.2f' % (b / a):>10}" for a, b in zip(lo, hi))
            print(f"  {m:>18}" + cells)

    sat = "  ".join(f"gain {g:.2f}: {results[(batchnorm, g)]['saturi']:.1f}%" for g in GAINS)
    print(f"\n  attivazioni sature (|h| > 0.97) all'ultimo layer nascosto:\n    {sat}")

k = GAINS[-1] / GAINS[0]
no_bn, bn = results[(False, GAINS[0])], results[(False, GAINS[-1])]
act_drift = bn["attivazioni"][-1] / no_bn["attivazioni"][-1]
grad_drift = bn["grad attivazioni"][0] / no_bn["grad attivazioni"][0]
ud_no_bn = bn["update:data"][0] - no_bn["update:data"][0]
w_bn = results[(True, GAINS[-1])]["grad pesi"][0] / results[(True, GAINS[0])]["grad pesi"][0]
ud_bn = results[(True, GAINS[-1])]["update:data"][0] - results[(True, GAINS[0])]["update:data"][0]

print(
    f"""
  Le due tabelle vanno lette come una sola, riga per riga.

  Le prime due misure sono quelle che la batchnorm azzera come problema, e lo
  fa in modo totale: x1.00 a ogni profondità, contro il x{act_drift:.0f} delle attivazioni
  in fondo e il x{grad_drift:.0f} del gradiente in cima che si prendono senza. Non è
  "molto meglio", è esattamente invariante. Le attivazioni perché vengono
  normalizzate per costruzione; il gradiente su di esse perché il fattore 1/k
  che la normalizzazione introduce nel backward si semplifica con il k della
  matrice che sta subito prima.

  Le altre due invece si muovono, e non in modo confuso: seguono una legge.
  Con la batchnorm il gradiente sui pesi scala come x{w_bn:.2f}, cioè esattamente 1/{k:.0f} —
  W è cresciuto di k ma il segnale che gli arriva no. E siccome anche i pesi
  sono cresciuti di k, il rapporto update:data va come 1/k^2: {ud_bn:+.2f} decadi,
  contro le {-2 * math.log10(k):+.2f} previste. Due decadi ogni volta che moltiplichi il gain
  per dieci.

  Il ribaltamento è nella colonna update:data della prima tabella: {ud_no_bn:+.2f} decadi.
  Senza batchnorm quel rapporto è quasi insensibile al gain — cioè su questa
  misura la rete *senza* normalizzazione sembra più robusta di quella con. Non
  lo è: è la stessa trappola di 08_gain_gradients.py. Il rapporto è il prodotto
  di due errori che si compensano, i pesi crescono e il gradiente cala, e la
  compensazione nasconde una rete che intanto ha le attivazioni morte e i
  gradienti sbilanciati di un fattore cinquanta. La batchnorm rimuove i due
  errori, e con loro sparisce anche la compensazione: quello che resta è la
  dipendenza vera, pulita, di 1/k^2.

  Ed è la risposta alla domanda in cima al file. La batchnorm ti toglie la
  calibrazione delle scale *interne* — il gain, la divisione per sqrt(fan_in),
  perfino l'inizializzazione dei layer nascosti, che a questo punto puoi
  sbagliare di un fattore sei senza che il forward e il backward se ne
  accorgano. Non ti toglie la scala *globale*: pesi più larghi vogliono dire
  training più lento a parità di learning rate, e quel numero resta tuo da
  scegliere. È esattamente quello che dice la lezione — "you may have to retune
  your learning rate if you are changing sufficiently the scale of the
  activations that are coming into the batch norms".

  E il corollario pratico: delle quattro diagnostiche, dopo aver messo le
  normalizzazioni solo l'ultima continua a dirti qualcosa. Le prime due sono
  garantite a posto per costruzione, la terza si muove ma non ti serve
  guardarla, la quarta è quella su cui si legge se il learning rate è giusto."""
)


# ---------------------------------------------------------------------------
# il grafico
# ---------------------------------------------------------------------------

COLUMNS = [METRICS[0], "istogramma"] + METRICS[1:]
SUBTITLE["istogramma"] = "distribuzione, ultimo layer nascosto"

fig, axes = plt.subplots(2, len(COLUMNS), figsize=(22, 8))
depths = range(1, DEPTH + 1)

for row, batchnorm in enumerate((False, True)):
    for col, metric in enumerate(COLUMNS):
        ax = axes[row][col]

        if metric == "istogramma":
            for i, gain in enumerate(GAINS):
                edges, counts = results[(batchnorm, gain)]["istogramma"]
                ax.plot(edges, counts, linewidth=4.5 - 1.1 * i, label=f"gain {gain:.2f}")
            ax.set_xlim(-1, 1)
            # log sui conteggi: il picco di gain 0.5 è venti volte più alto di
            # tutto il resto, e in scala lineare schiaccerebbe la riga di sotto
            ax.set_yscale("log")
            ax.set_xlabel("valore dopo la tanh")
            if row == 0:
                ax.set_title(f"{metric}\n{SUBTITLE[metric]}", fontsize=10)
            ax.grid(alpha=0.3)
            continue

        # spessore decrescente: dove le curve coincidono esattamente (la riga
        # con batchnorm, prime due colonne) si vedono comunque tutte e quattro
        for i, gain in enumerate(GAINS):
            ax.plot(depths, results[(batchnorm, gain)][metric], marker="o",
                    linewidth=4.5 - 1.1 * i, markersize=9 - 1.8 * i,
                    label=f"gain {gain:.2f}")
        if metric != "update:data":
            ax.set_yscale("log")
        else:
            ax.axhline(-3, color="black", linestyle="--", linewidth=1)
        if row == 0:
            ax.set_title(f"{metric}\n{SUBTITLE[metric]}", fontsize=10)
        ax.set_xlabel("layer nascosto")
        ax.grid(alpha=0.3)
        ax.set_xticks(list(depths))
        if col == 0:
            ax.set_ylabel("CON batchnorm" if batchnorm else "SENZA batchnorm",
                          fontsize=12, fontweight="bold")
            ax.legend(fontsize=7)

# stessa scala verticale fra le due righe, se no il confronto mente
for col in range(len(COLUMNS)):
    lo = min(axes[r][col].get_ylim()[0] for r in (0, 1))
    hi = max(axes[r][col].get_ylim()[1] for r in (0, 1))
    for r in (0, 1):
        axes[r][col].set_ylim(lo, hi)

plt.tight_layout()
plt.savefig(HERE / "10_out_batchnorm_robustness.png", bbox_inches="tight", dpi=100)
plt.close()
print(f"\n  grafico in {HERE / '10_out_batchnorm_robustness.png'}")
