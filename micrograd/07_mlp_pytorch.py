"""
Tappa 7: la stessa cosa della tappa 6, in PyTorch.

Il minimo indispensabile: solo tensori, requires_grad e backward(). Niente
nn.Linear, niente nn.Module, niente optim.SGD. Cosi' si vede esattamente cosa
sostituisce cosa:

  engine.py  ->  l'autograd dei tensori (che e' l'unica cosa che importiamo)
  nn.py      ->  tre righe di moltiplicazione fra matrici
  il resto   ->  identico, riga per riga

E si vede anche la differenza vera: qui i quattro esempi passano nella rete
tutti insieme, in una sola moltiplicazione, invece che uno alla volta con un
nodo scalare per ogni prodotto. E' solo efficienza, la matematica non cambia.

Nota: la rete parte dagli stessi 41 numeri della tappa 6, estratti con lo stesso
seed e nello stesso ordine, quindi la traiettoria della loss e' la stessa.
"""

import random

import torch

SEED = 1337

# gli stessi quattro esempi, ma impacchettati in due tensori invece che in
# quattro liste di float
X = torch.tensor(
    [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ],
    dtype=torch.double,
)
Y = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.double)


def make_mlp(nin, nouts):
    """Gli stessi parametri della MLP di nn.py, disposti in matrici.

    Un layer che in nn.py e' una lista di nout oggetti Neuron, ognuno con nin
    pesi scalari, qui e' una matrice W di forma (nin, nout) piu' un vettore b
    di nout elementi: la colonna j di W sono i pesi del neurone j.
    """
    random.seed(SEED)
    sz = [nin] + nouts
    layers = []
    for i in range(len(nouts)):
        # stesso ordine di estrazione di nn.py (per ogni neurone: prima i pesi,
        # poi il bias), cosi' i numeri di partenza sono gli stessi
        neurons = [
            [random.uniform(-1, 1) for _ in range(sz[i])] + [random.uniform(-1, 1)]
            for _ in range(sz[i + 1])
        ]
        W = torch.tensor(
            [[n[r] for n in neurons] for r in range(sz[i])], dtype=torch.double
        )
        b = torch.tensor([n[-1] for n in neurons], dtype=torch.double)
        # senza questo i gradienti non verrebbero tenuti: e' l'equivalente di
        # dire "questi sono i parametri", cioe' di parameters() in nn.py
        W.requires_grad = True
        b.requires_grad = True
        layers.append((W, b))
    return layers


def forward(layers, x):
    """Tutta nn.py in tre righe.

    '@' e' l'operatore di moltiplicazione fra matrici (in Python e' __matmul__,
    e ha senso solo per oggetti come i tensori: sui float normali non esiste).
    Non e' la moltiplicazione elemento per elemento, che invece e' '*': la
    riga i di 'x @ W' e' fatta dai prodotti scalari fra la riga i di x e ogni
    colonna di W.

    Qui e' esattamente il conto del neurone. x ha forma (esempi, input), W ha
    forma (input, neuroni), e la colonna j di W sono i pesi del neurone j:
    quindi la casella (i, j) del risultato e' la somma dei w*x del neurone j
    sull'esempio i, cioe' l'act di Neuron.__call__ in nn.py. La differenza e'
    che qui esce tutta insieme, per tutti gli esempi e tutti i neuroni, invece
    che un neurone e un esempio alla volta.

    Poi '+ b' somma il vettore dei bias (uno per neurone) a ogni riga, per
    broadcasting, e tanh schiaccia tutto elemento per elemento.
    """
    for W, b in layers:
        x = torch.tanh(x @ W + b)
    return x


# ---------------------------------------------------------------------------
# la rete e la loss
# ---------------------------------------------------------------------------

print("=== la stessa MLP della tappa 6, con i tensori ===\n")

layers = make_mlp(3, [4, 4, 1])
params = [p for W, b in layers for p in (W, b)]
print(f"  tensori: {[tuple(p.shape) for p in params]}")
print(f"  parametri totali: {sum(p.numel() for p in params)}")

ypred = forward(layers, X).squeeze(-1)
print(f"\n  predizioni iniziali: {[round(v, 4) for v in ypred.tolist()]}")
print(f"  target:              {Y.tolist()}")

loss = ((ypred - Y) ** 2).sum()
print(f"  loss: {loss.item():.4f}")


# ---------------------------------------------------------------------------
# il training loop
# ---------------------------------------------------------------------------

print("\n=== gradient descent ===\n")


def train(steps, lr, zero_grads, verbose=False):
    """Riga per riga lo stesso loop della tappa 6."""
    layers = make_mlp(3, [4, 4, 1])
    params = [p for W, b in layers for p in (W, b)]

    for k in range(steps):
        # forward
        ypred = forward(layers, X).squeeze(-1)
        loss = ((ypred - Y) ** 2).sum()

        # backward
        if zero_grads:
            # in PyTorch azzerare vuol dire buttare via il tensore dei
            # gradienti: al prossimo backward viene ricreato da zero
            for p in params:
                p.grad = None
        loss.backward()

        # aggiornamento: no_grad() dice a PyTorch di non registrare queste
        # operazioni nel grafo. Senza, l'aggiornamento dei pesi finirebbe a far
        # parte dell'espressione da derivare al passo dopo. In micrograd il
        # problema non si pone perche' scriviamo direttamente p.data.
        with torch.no_grad():
            for p in params:
                p -= lr * p.grad

        if verbose and (k % 10 == 0 or k == steps - 1):
            print(f"    passo {k:3d}   loss {loss.item():.6f}")

    return loss.item(), ypred.tolist()


print("  con lo zero dei gradienti (corretto):")
loss_ok, pred_ok = train(steps=100, lr=0.05, zero_grads=True, verbose=True)
print(f"  predizioni finali: {[round(v, 4) for v in pred_ok]}")
print(f"  target:            {Y.tolist()}")

# lo stesso bug della tappa 6: anche PyTorch accumula con +=, e non azzera da
# solo. E' il motivo per cui in ogni training loop scritto con PyTorch si trova
# una chiamata a optimizer.zero_grad().
loss_bug, pred_bug = train(steps=100, lr=0.05, zero_grads=False)
print(f"\n  senza azzerarli (buggato): loss {loss_bug:.6f} contro {loss_ok:.6f}")
print(f"  predizioni: {[round(v, 4) for v in pred_bug]}")
print("\n  gli stessi numeri della tappa 6, bug compreso: quello che cambia e'")
print("  solo come sono impacchettati i conti, non i conti")
