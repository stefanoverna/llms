"""
L'ultimo dei quattro grafici diagnostici: quanto si muovono davvero i pesi.

`07_gain_depth.py` guarda le attivazioni, `08_gain_gradients.py` i gradienti.
Ma un gradiente grande non vuol dire un passo grande, e un gradiente piccolo non
vuol dire un passo piccolo: dipende da quanto sono grandi i pesi su cui va ad
applicarsi. Un update di 0.001 su un peso che vale 0.002 è enorme; lo stesso
update su un peso che vale 10 non si vede nemmeno.

Quindi il numero che conta è il rapporto fra i due (`02_out_lecture.srt`, dal
minuto 1:33):

    update:data = std(learning_rate * p.grad) / std(p)

cioè: di che frazione di sé stesso si sposta questo tensore a ogni passo. Lo
guardiamo in log10, e la regola pratica della lezione è che dovrebbe stare
intorno a **-3**, cioè un millesimo per passo:

    molto sotto -3   i pesi non si muovono: learning rate troppo basso
    intorno a -3     sano
    molto sopra -3   i pesi vengono ribaltati a ogni passo

È il grafico che rivela un learning rate sbagliato in trenta passi invece che
in trenta minuti, e l'unico dei quattro che si guarda *durante* il training e
non all'inizializzazione.

Il grafico è in 09_out_learning_speed.png: tre scenari, una curva per tensore.
"""

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
BATCH_SIZE = 32
DEPTH = 5  # quanti layer nascosti: qui la rete è profonda, come nella lezione
STEPS = 1000


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


def build(broken_init=False):
    """Embedding, poi DEPTH volte (Linear -> Tanh), poi i logits.

    Niente batchnorm: siamo nel punto della lezione che viene prima, quello in
    cui la scala giusta va ancora azzeccata a mano.
    """
    torch.manual_seed(SEED)
    layers = [nn.Embedding(VOCAB, EMB_DIM), nn.Flatten()]
    fan_in = BLOCK_SIZE * EMB_DIM
    for _ in range(DEPTH):
        layers += [nn.Linear(fan_in, HIDDEN), nn.Tanh()]
        fan_in = HIDDEN
    layers += [nn.Linear(HIDDEN, VOCAB)]
    net = nn.Sequential(*layers)

    for layer in net:
        if isinstance(layer, nn.Linear):
            if broken_init:
                # "let's say that we forgot to apply this fan-in normalization"
                nn.init.normal_(layer.weight, std=1.0)
            else:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="tanh")
            nn.init.zeros_(layer.bias)
    # l'ultimo layer rimpicciolito, perché i logits partano poco convinti: è la
    # sezione 2 di 02_optimizations.py, e qui diventa visibile nel grafico
    with torch.no_grad():
        net[-1].weight *= 0.1
    return net


def weight_tensors(net):
    """Le matrici di pesi, con un nome leggibile, in ordine dall'ingresso all'uscita.

    Guardiamo solo i tensori a due dimensioni: i bias hanno una dinamica loro e
    li salteremmo comunque nel grafico.

    Nota sulle shape: nn.Linear tiene `weight` come (out_features, in_features),
    cioè trasposta rispetto a come la si scriverebbe in `x @ W`. Per questo il
    primo layer nascosto, che va da 30 ingressi a 100 neuroni, ha shape
    (100, 30) e non (30, 100).
    """
    out, seen = [], 0
    for layer in net:
        if isinstance(layer, nn.Embedding):
            out.append(("embedding C", layer.weight))
        elif isinstance(layer, nn.Linear):
            seen += 1
            name = f"nascosto {seen}" if seen <= DEPTH else "logits (ultimo)"
            out.append((name, layer.weight))
    return [(f"{name} {tuple(w.shape)}", w) for name, w in out]


def train(net, lr, steps=STEPS):
    """Allena, e a ogni passo registra log10(update:data) per ogni matrice."""
    labels, params = zip(*weight_tensors(net))
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(SEED)
    history = [[] for _ in params]

    for _ in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
        loss = F.cross_entropy(net(Xtr[ix]), Ytr[ix])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # PRIMA dello step: l'update che stiamo per applicare è lr * p.grad, e
        # lo confrontiamo con i valori che il tensore ha adesso.
        with torch.no_grad():
            for series, p in zip(history, params):
                series.append(((lr * p.grad).std() / p.std()).log10().item())

        optimizer.step()

    return labels, params, history, loss.item()


SCENARIOS = [
    ("learning rate 0.1 (sano)", dict(lr=0.1)),
    ("learning rate 0.001 (troppo basso)", dict(lr=0.001)),
    ("init senza /sqrt(fan_in), lr 0.1", dict(lr=0.1, broken_init=True)),
]

results = []
stats = []
for title, cfg in SCENARIOS:
    net = build(broken_init=cfg.get("broken_init", False))
    labels, params, history, final_loss = train(net, lr=cfg["lr"])
    results.append((title, labels, history))

    tails = [sum(series[-100:]) / 100 for series in history]
    print(f"\n=== {title} ===\n")
    print(f"  {'tensore':>26}  {'log10(update:data) a fine training':>34}")
    for label, tail in zip(labels, tails):
        note = "   <-" if label is labels[-1] else ""
        print(f"  {label:>26}  {tail:>34.2f}{note}")
    hidden = tails[:-1]
    stats.append(dict(lo=min(hidden), hi=max(hidden), last=tails[-1],
                      last0=history[-1][0], spread=max(hidden) - min(hidden)))
    print(f"\n  dispersione fra i tensori nascosti: {max(hidden) - min(hidden):.2f} decadi")
    print(f"  loss all'ultimo batch: {final_loss:.4f}")

sane, slow, broken = stats

print(
    f"""
  Prima colonna. I sei tensori nascosti stanno fra {sane["lo"]:.1f} e {sane["hi"]:.1f}, cioè un filo
  sopra la riga di -3, e ci restano piatti per tutto il training: si spostano di
  qualche millesimo di sé stessi a ogni passo. È il commento che fa anche la
  lezione guardando il proprio grafico — "the learning rate here is a little bit
  on the higher side, we're somewhere around -2.5, it's okay".

  L'unico fuori dal gruppo è l'ultimo layer, e non è un difetto: è il tensore
  che abbiamo moltiplicato per 0.1 apposta perché i logits partissero piccoli.
  I suoi valori sono artificialmente bassi, quindi qualsiasi update è grande
  rispetto a loro. Nel grafico si vede partire da {sane["last0"]:.1f} e scendere fino a
  {sane["last"]:.1f} mentre quel tensore recupera la sua scala: è l'unica curva che si
  muove davvero.

  Seconda colonna. Tutto scivola di due decadi e mezza, a {slow["hi"]:.1f}/{slow["lo"]:.1f}. Nessun
  singolo numero è sbagliato — i gradienti sono gli stessi di prima, è solo il
  moltiplicatore a essere cambiato — ma i pesi si muovono di un centomillesimo
  per passo, e a quella velocità non si arriva da nessuna parte. È il tipo di
  problema che senza questo grafico si scopre dopo un'ora di attesa, ed è il
  motivo per cui il rapporto dice cose che i gradienti da soli non dicono: un
  grafico di gradienti qui sarebbe identico a quello della prima colonna.

  Nota l'ultimo layer anche qui, a {slow["last"]:.1f} mentre tutto il resto è sotto -4.7:
  stesso effetto di prima al contrario, i suoi pesi sono così piccoli che
  perfino un learning rate da 0.001 li muove parecchio.

  Terza colonna. Qui non c'è un valore da leggere, c'è la dispersione: i tensori
  nascosti si spargono su {broken["spread"]:.2f} decadi contro le {sane["spread"]:.2f} della prima colonna, e
  nel grafico le curve si aprono a ventaglio invece di sovrapporsi. Vuol dire
  che layer dello stesso identico modello stanno imparando a velocità che
  differiscono di un fattore {10 ** broken["spread"]:.0f}, e un learning rate solo non può servirli
  tutti: quello giusto per il primo layer è troppo per l'ultimo. È la diagnosi
  che la lezione riassume così: "this is how miscalibrations of your neural nets
  are going to manifest, and these kinds of plots are a good way of bringing
  them to your attention"."""
)


# ---------------------------------------------------------------------------
# il grafico
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

for ax, (title, labels, history) in zip(axes, results):
    for label, series in zip(labels, history):
        ax.plot(series, linewidth=0.8, label=label)
    ax.axhline(-3, color="black", linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("passo")
    ax.set_ylim(-6, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

axes[0].set_ylabel("log10( std(lr * grad) / std(peso) )")
axes[0].text(STEPS * 0.35, -3.12, "-3, il valore sano", va="top", fontsize=8)

plt.tight_layout()
plt.savefig(HERE / "09_out_learning_speed.png", bbox_inches="tight", dpi=100)
plt.close()
print(f"\n  grafico in {HERE / '09_out_learning_speed.png'}")
