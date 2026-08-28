"""
Tappa 6: una rete neurale vera, e come si addestra.

Neuron, Layer e MLP stanno in nn.py e sono ~50 righe: il lavoro vero lo fa il
motore di autograd. Qui mettiamo insieme tutto: una MLP da 41 parametri, una
loss, e il ciclo forward / backward / aggiornamento che e' tutto quello che
serve per addestrare qualsiasi rete, da questa a GPT.

Corrisponde all'ultima parte della lezione, bug dello zero_grad compreso.
"""

import random

from nn import MLP

SEED = 1337

# quattro esempi, un classificatore binario giocattolo
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]


# ---------------------------------------------------------------------------
# la rete e la loss
# ---------------------------------------------------------------------------

print("=== una MLP: 3 input, due layer da 4, un'uscita ===\n")

random.seed(SEED)
n = MLP(3, [4, 4, 1])
print(f"  {n}")
print(f"  parametri totali: {len(n.parameters())}")

ypred = [n(x) for x in xs]
print(f"\n  predizioni iniziali: {[round(p.data, 4) for p in ypred]}")
print(f"  target:              {ys}")

# La loss e' l'unico numero che riassume quanto la rete sta andando male. Qui
# usiamo l'errore quadratico medio (MSE, Mean Squared Error): e' 0 solo se
# predizione e target coincidono, e il quadrato serve a rendere l'errore
# positivo comunque sia il segno.
loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
print(f"  loss: {loss.data:.4f}")

loss.backward()
w = n.layers[0].neurons[0].w[0]
print(f"\n  dopo backward, un peso a caso: data {w.data:+.4f}, grad {w.grad:+.4f}")
print("  il grafo di questa loss contiene i quattro forward pass, tutti insieme,")
print("  e la backward li attraversa tutti fino ai 41 parametri")


# ---------------------------------------------------------------------------
# il training loop
# ---------------------------------------------------------------------------

print("\n=== gradient descent ===\n")


def train(steps, lr, zero_grads, verbose=False):
    """Forward, backward, aggiornamento. Ripetuto. Questo e' tutto l'addestramento."""
    random.seed(SEED)  # stessa inizializzazione a ogni chiamata, per confrontare
    n = MLP(3, [4, 4, 1])

    for k in range(steps):
        # forward
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        # backward
        if zero_grads:
            n.zero_grad()
        loss.backward()

        # aggiornamento: il gradiente punta verso la loss che cresce, e noi la
        # vogliamo far scendere, quindi andiamo nel verso opposto (il meno).
        # lr troppo basso: lentissimo. Troppo alto: si scavalca il minimo e
        # l'ottimizzazione esplode.
        for p in n.parameters():
            p.data += -lr * p.grad

        if verbose and (k % 10 == 0 or k == steps - 1):
            print(f"    passo {k:3d}   loss {loss.data:.6f}")

    return n, loss.data, [p.data for p in ypred]


print("  con zero_grad (corretto):")
net, loss_ok, pred_ok = train(steps=100, lr=0.05, zero_grads=True, verbose=True)
print(f"  predizioni finali: {[round(p, 4) for p in pred_ok]}")
print(f"  target:            {ys}")

# Il bug numero tre della lista dei classici: dimenticare zero_grad. I gradienti
# si accumulano con += (tappa 3) e non vengono mai svuotati, quindi restano
# dentro quelli del passo precedente.
_, loss_bug, pred_bug = train(steps=100, lr=0.05, zero_grads=False)
print(f"\n  senza zero_grad (buggato): loss {loss_bug:.6f} contro {loss_ok:.6f}")
print(f"  predizioni: {[round(p, 4) for p in pred_bug]}")
print("  i gradienti non svuotati si sommano passo dopo passo, quindi il passo")
print("  effettivo cresce senza controllo: la rete satura e resta bloccata.")
print("  Su problemi facili a volte converge lo stesso, ed e' per questo che il")
print("  bug passa inosservato finche' il problema non diventa serio.")
