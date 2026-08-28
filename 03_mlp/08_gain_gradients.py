"""
Lo stesso esperimento di 07_gain_depth.py, ma guardando indietro: i gradienti.

`07_gain_depth.py` misura cosa arriva in fondo alla rete quando un batch la
attraversa in avanti. Ma una rete che non si allena non è rotta nel forward: è
rotta nel backward. Qui iniettiamo un gradiente in cima e misuriamo cosa arriva
a ogni profondità scendendo (`02_out_lecture.srt`, dal minuto 1:30).

Quello che si cerca è scritto bene nella lezione: "what we're looking for is
that all the different layers in this sandwich have roughly the same gradient,
things are not shrinking or exploding". Non conta il valore assoluto — conta
che sia lo stesso a tutte le profondità. Se i layer profondi ricevono un
gradiente cento volte più grande di quelli vicini all'ingresso, con un
learning rate solo per tutti, metà rete si allena e l'altra metà sta ferma.

La rete è identica a quella di 07_gain_depth.py, così i due grafici si leggono
uno accanto all'altro. La differenza è che invece di una loss vera iniettiamo
in cima un gradiente gaussiano di deviazione standard 1, che è il modo più
pulito di isolare la domanda: la propagazione all'indietro, da sola, questo
gradiente lo conserva, lo schiaccia o lo gonfia?

Il grafico è in 08_out_gain_gradients.png.
"""

from pathlib import Path

import torch
from torch import nn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 2147483647
HERE = Path(__file__).parent

DEPTH = 6
WIDTH = 100
BATCH = 1000

GAINS = {"0.5": 0.5, "1": 1.0, "5/3": 5 / 3, "3": 3.0}


def build(gain):
    """DEPTH volte (Linear -> Tanh), pesi a std = gain / sqrt(fan_in). Come in 07."""
    torch.manual_seed(SEED)
    layers = []
    for _ in range(DEPTH):
        linear = nn.Linear(WIDTH, WIDTH)
        nn.init.normal_(linear.weight, std=gain / WIDTH**0.5)
        nn.init.zeros_(linear.bias)
        layers += [linear, nn.Tanh()]
    return nn.Sequential(*layers)


def forward_backward(net, x, injected):
    """Un forward, poi un backward con un gradiente noto iniettato in cima.

    retain_grad() serve perché le uscite intermedie non sono foglie: di default
    PyTorch calcola il loro gradiente, lo usa per proseguire all'indietro e poi
    lo butta via. È la stessa riga che usa la lezione per poter fare questi
    istogrammi (`layer.out.retain_grad()`).
    """
    tanh_out, weights = [], []
    h = x
    for layer in net:
        h = layer(h)
        if isinstance(layer, nn.Tanh):
            h.retain_grad()
            tanh_out.append(h)
        else:
            weights.append(layer.weight)
    h.backward(injected)
    return tanh_out, weights


torch.manual_seed(SEED)
x0 = torch.randn(BATCH, WIDTH)
# il gradiente che entra dall'alto: gaussiano, deviazione standard 1. Lo stesso
# per tutti i gain, così l'unica variabile è la rete.
injected = torch.randn(BATCH, WIDTH)

measured = {}
summary = {}

for label, gain in GAINS.items():
    acts, weights = forward_backward(build(gain), x0, injected)
    measured[label] = [t.grad for t in acts]
    summary[label] = (acts[0].std().item(), acts[-1].std().item(),
                      acts[0].grad.std().item(), acts[-1].grad.std().item())
    print(f"\n=== gain = {label} ===\n")
    print(f"  {'layer':>7}  {'attivazione':>13}  {'gradiente':>13}  {'gradiente su W':>16}")
    print(f"  {'':>7}  {'(dev.std.)':>13}  {'(dev.std.)':>13}  {'(dev.std.)':>16}")
    for i, (t, w) in enumerate(zip(acts, weights), 1):
        print(f"  {i:>7}  {t.std():>13.3f}  {t.grad.std():>13.2e}  {w.grad.std():>16.2e}")
    print(f"\n  dal layer 1 al layer 6:  attivazione x{acts[-1].std() / acts[0].std():.2f}"
          f"   gradiente x{acts[-1].grad.std() / acts[0].grad.std():.2f}")


print("\n=== il riassunto ===\n")
print(f"  {'gain':>5}   {'attivazione, layer 1 -> 6':^30}   {'gradiente, layer 1 -> 6':^30}")
for label, (a1, a6, g1, g6) in summary.items():
    print(f"  {label:>5}   {a1:>10.2f} -> {a6:<8.2f} (x{a6 / a1:.2f})"
          f"    {g1:>10.2f} -> {g6:<8.2f} (x{g6 / g1:.2f})")

print(
    """
  Le due colonne si muovono in verso opposto lungo la profondità, ed è tutto
  qui. Con gain piccolo il layer 1 ha attivazioni larghe e un gradiente
  minuscolo, il layer 6 il contrario: i due estremi della rete vivono in regimi
  diversi, e con un learning rate solo per tutti non c'è modo di servirli
  entrambi. Con gain grande succede lo stesso a parti invertite.

  5/3 è l'unico valore che tiene le due colonne vicine a 1 nello stesso
  momento, ed è quello che si cerca: "all the different layers in this sandwich
  have roughly the same gradient, things are not shrinking or exploding".

  L'ultima colonna della tabella per gain, quella del gradiente su W, è la
  trappola. Il gradiente su un peso è il prodotto dell'attivazione che entra
  per il gradiente che esce, quindi quando la rete è mal calibrata i due errori
  si compensano a vicenda: con gain 1 quella colonna resta quasi piatta
  (10.5 -> 9.5) pur essendo la rete sbilanciata da tutte e due le parti.
  Guardare solo i gradienti dei pesi non basta a vedere il problema, e per
  questo la lezione guarda le attivazioni e i loro gradienti separatamente."""
)


# ---------------------------------------------------------------------------
# il grafico
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, label in zip(axes, ("1", "5/3")):
    grads = measured[label]
    edge = 4 * max(g.std().item() for g in grads)
    for i, g in enumerate(grads, 1):
        counts, edges = torch.histogram(g, bins=80, range=(-edge, edge))
        ax.plot(edges[:-1].tolist(), counts.tolist(), label=f"layer {i}")
    ax.set_title(f"gradienti, gain = {label}", fontsize=10)
    ax.set_xlabel("gradiente sull'uscita della tanh")
    ax.set_ylabel("quanti valori")
    ax.legend(fontsize=7)

for label, grads in measured.items():
    stds = [g.std().item() for g in grads]
    axes[2].plot(range(1, DEPTH + 1), stds, marker="o", label=f"gain = {label}")
axes[2].set_title("dev. standard del gradiente, layer per layer", fontsize=10)
axes[2].set_xlabel("profondità (il gradiente scende da destra a sinistra)")
axes[2].set_ylabel("dev. standard del gradiente")
axes[2].set_yscale("log")
axes[2].grid(alpha=0.3)
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(HERE / "08_out_gain_gradients.png", bbox_inches="tight", dpi=100)
plt.close()
print(f"\n  grafico in {HERE / '08_out_gain_gradients.png'}")
