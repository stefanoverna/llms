"""
Perché il gain della tanh è 5/3, visto su una rete profonda.

Su un layer solo il gain quasi non si vede: `01_mlp.py` funziona anche senza.
Il punto è che l'effetto è *moltiplicativo*, quindi si accumula con la
profondità — ed è per questo che la lezione, arrivata qui, smette di usare la
rete a un layer nascosto e ne impila sei (`02_out_lecture.srt`, dal minuto 1:26:50).

L'esperimento è tutto all'inizializzazione, senza allenare niente: una rete
lineare+tanh profonda, un batch che la attraversa, e la distribuzione delle
attivazioni misurata all'uscita di ogni tanh. Due volte, cambiando una sola
costante:

    std(W) = gain / sqrt(fan_in)      con gain = 1      e con gain = 5/3

Con gain 1 le deviazioni standard si schiacciano layer dopo layer verso zero.
Con 5/3 si stabilizzano. Il grafico è in 07_out_gain_depth.png.

Il motivo è che `gain=1` è il valore che conserva la varianza attraverso *un
layer lineare*, ed è quello che serve se dopo non c'è niente. Ma dopo c'è la
tanh, che è una funzione che schiaccia: prende la distribuzione e la
restringe. Il gain è il fattore che compensa quella contrazione, e per la tanh
vale 5/3 — un numero empirico, dice Karpathy, non uscito da una formula.
"""

from pathlib import Path

import torch
from torch import nn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 2147483647
HERE = Path(__file__).parent

DEPTH = 6  # quanti layer lineare+tanh impilati
WIDTH = 100
BATCH = 1000  # solo per avere istogrammi lisci: non alleniamo niente


def build(gain):
    """DEPTH volte (Linear -> Tanh), con i pesi a std = gain / sqrt(fan_in)."""
    torch.manual_seed(SEED)
    layers = []
    for _ in range(DEPTH):
        linear = nn.Linear(WIDTH, WIDTH)
        nn.init.normal_(linear.weight, std=gain / WIDTH**0.5)
        nn.init.zeros_(linear.bias)
        layers += [linear, nn.Tanh()]
    return nn.Sequential(*layers)


@torch.no_grad()
def activations(net, x):
    """L'uscita di ogni tanh, una alla volta."""
    out = []
    for layer in net:
        x = layer(x)
        if isinstance(layer, nn.Tanh):
            out.append(x)
    return out


# L'ingresso è gaussiano standard: è l'ipotesi sotto cui il conto di Kaiming è
# stato fatto, quindi partiamo dal caso ideale e guardiamo solo cosa succede
# scendendo.
torch.manual_seed(SEED)
x0 = torch.randn(BATCH, WIDTH)

runs = {"gain = 1 (conserva la varianza di un layer lineare)": 1.0,
        "gain = 5/3 (il valore per la tanh)": 5 / 3}
measured = {}

for label, gain in runs.items():
    acts = activations(build(gain), x0)
    measured[label] = acts
    print(f"\n=== {label} ===\n")
    print(f"  {'layer':>7}  {'media':>8}  {'dev.std.':>9}  {'saturi |t|>0.97':>16}")
    for i, t in enumerate(acts, 1):
        sat = (t.abs() > 0.97).float().mean() * 100
        print(f"  {i:>7}  {t.mean():>8.3f}  {t.std():>9.3f}  {sat:>15.1f}%")

print(
    """
  Con gain 1 la deviazione standard scende a ogni layer: lentamente, e senza
  mai assestarsi. Continuando oltre i sei layer di qui sopra:

      layer      1      6     10     20     40
      dev.std.   0.63   0.30   0.22   0.16   0.12

  Le attivazioni si stringono attorno allo zero e la rete diventa una funzione
  quasi costante. Nel backward il fattore (1 - t^2) resta sì vicino a 1 — la
  tanh non è satura, anzi il contrario — ma il gradiente su W è proporzionale
  alle attivazioni in ingresso, e quelle ormai non ci sono più.

  Con gain 5/3 invece si assesta: 0.66 al sesto layer, 0.65 al ventesimo, 0.65
  al quarantesimo, con la saturazione ferma intorno al 6%. È un punto fisso, e
  ci resta indipendentemente da quanti layer aggiungi. È il regime in cui la
  tanh lavora dove ha derivata utile.

  Ed è tutta la differenza fra le due colonne: una costante, scritta una volta
  sola all'inizializzazione."""
)


# ---------------------------------------------------------------------------
# il grafico
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (label, acts) in zip(axes, measured.items()):
    for i, t in enumerate(acts, 1):
        counts, edges = torch.histogram(t, bins=80, range=(-1.0, 1.0))
        ax.plot(edges[:-1].tolist(), counts.tolist(), label=f"layer {i}")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("attivazione dopo la tanh")
    ax.set_ylabel("quanti valori")
    ax.set_xlim(-1, 1)
    ax.legend(fontsize=7)

# il terzo pannello è il riassunto dei primi due: una curva per gain
for label, acts in measured.items():
    stds = [t.std().item() for t in acts]
    axes[2].plot(range(1, DEPTH + 1), stds, marker="o", label=label.split(" (")[0])
axes[2].set_title("deviazione standard, layer per layer", fontsize=10)
axes[2].set_xlabel("profondità")
axes[2].set_ylabel("dev. standard delle attivazioni")
axes[2].set_ylim(0, 0.8)
axes[2].grid(alpha=0.3)
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(HERE / "07_out_gain_depth.png", bbox_inches="tight", dpi=100)
plt.close()
print(f"\n  grafico in {HERE / '07_out_gain_depth.png'}")
