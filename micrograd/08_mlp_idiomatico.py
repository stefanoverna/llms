"""
Tappa 8: la stessa cosa della tappa 7, scritta come si scrive davvero.

La tappa 7 usava il minimo di PyTorch apposta, per far vedere che sotto c'e'
solo quello che abbiamo costruito noi. Qui invece usiamo le astrazioni vere, e
il codice si accorcia fino a sparire:

  make_mlp()          ->  nn.Linear, che tiene W e b e li inizializza
  forward()           ->  nn.Sequential, che li mette in fila
  la loss a mano      ->  nn.MSELoss (Mean Squared Error)
  p.grad = None       ->  optimizer.zero_grad()
  p -= lr * p.grad    ->  optimizer.step() (SGD, Stochastic Gradient Descent)

Il training loop diventa le cinque righe che si trovano identiche in qualsiasi
progetto PyTorch. Alla fine del file: le differenze fra questa versione e la
tappa 7 che non sono solo cosmetiche.
"""

import torch
from torch import nn

SEED = 1337
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# il modello
# ---------------------------------------------------------------------------

print("=== la stessa MLP, con nn.Sequential ===\n")

# Un nn.Linear(3, 4) e' esattamente la coppia (W, b) della tappa 7: tiene una
# matrice 4x3 e un vettore da 4, e li inizializza da solo. Le Tanh in mezzo sono
# le non linearita': senza, tre Linear di fila collasserebbero in una sola.
mlp = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),
    nn.Linear(4, 4),
    nn.Tanh(),
    nn.Linear(4, 1),
    nn.Tanh(),
)

print("  " + str(mlp).replace("\n", "\n  "))
print(f"\n  parametri totali: {sum(p.numel() for p in mlp.parameters())}")

# MSE = Mean Squared Error, errore quadratico medio: la stessa loss della
# tappa 6, la media (o la somma) dei quadrati degli scarti fra predizione e
# target. Qui reduction="sum" perche' la nostra sommava i quattro scarti; il
# default e' "mean", che divide per 4: stessa cosa a meno di un fattore
# costante, ma con lo stesso learning rate la rete imparerebbe quattro volte
# piu' piano. E' una delle differenze che si pagano senza accorgersene.
loss_fn = nn.MSELoss(reduction="sum")

# SGD = Stochastic Gradient Descent, discesa del gradiente stocastica: e'
# esattamente il "p -= lr * p.grad" della tappa 7, solo scritto da qualcun
# altro. "Stocastica" perche' di solito ogni passo si calcola su un campione
# casuale di esempi (un batch) invece che su tutti: qui gli esempi sono quattro
# e li usiamo sempre tutti, quindi di stocastico non c'e' niente, ma la classe
# di PyTorch si chiama cosi' lo stesso.
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.05)

# i quattro esempi, impacchettati in due tensori invece che in liste di float
X = torch.tensor(
    [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
)
# (4, 1) e non (4,): stessa forma dell'output del modello, altrimenti il
# broadcasting di MSELoss confronta ogni predizione con ogni target
Y = torch.tensor([[1.0], [-1.0], [-1.0], [1.0]])

print(f"\n  predizioni iniziali (garbage): {[round(v, 4) for v in mlp(X).flatten().tolist()]}")
print(f"  target: {Y.flatten().tolist()}")


# ---------------------------------------------------------------------------
# il training loop
# ---------------------------------------------------------------------------

print("\n=== gradient descent ===\n")

for k in range(100):
    ypred = mlp(X)
    loss = loss_fn(ypred, Y)

    # queste tre righe, in quest'ordine, sono il cuore di ogni progetto PyTorch:
    # svuota i gradienti, calcolali, applicali. Lo zero_grad e' lo stesso della
    # tappa 3 e della tappa 6: PyTorch accumula con += e non azzera da solo.
    optimizer.zero_grad()
    loss.backward()
    # step() modifica i pesi in place: nella tappa 7 quella riga andava
    # protetta con no_grad(), altrimenti l'aggiornamento stesso sarebbe entrato
    # nel grafo da derivare al giro dopo. Qui non serve perche' step() lo fa
    # gia' al posto nostro.
    optimizer.step()

    if k % 10 == 0 or k == 99:
        print(f"    passo {k:3d}   loss {loss.item():.6f}")

print(f"\n  predizioni finali: {[round(v, 4) for v in mlp(X).flatten().tolist()]}")
print(f"  target:            {Y.flatten().tolist()}")
