import torch
from torch import nn

SEED = 1337
torch.manual_seed(SEED)

X = torch.tensor(
    [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
)
Y = torch.tensor(
    [
        [1.0],
        [-1.0],
        [-1.0],
        [1.0],
    ]
)


# ---------------------------------------------------------------------------
# il modello
# ---------------------------------------------------------------------------

mlp = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),
    nn.Linear(4, 4),
    nn.Tanh(),
    nn.Linear(4, 1),
    nn.Tanh(),
)

print("  " + str(mlp).replace("\n", "\n  "))

params = list(mlp.parameters())
print(f"\n  tensori: {[tuple(p.shape) for p in params]}")
print(f"  parametri totali: {sum(p.numel() for p in params)}")

ypred = mlp(X)
print(f"\n  predizioni iniziali: {[round(v, 4) for v in ypred.flatten().tolist()]}")
print(f"  target:              {Y.flatten().tolist()}")

loss = ((ypred - Y) ** 2).sum()
print(f"  loss: {loss.item():.4f}")

print("\n=== gradient descent ===\n")

lr = 0.05
for k in range(100):
    ypred = mlp(X)
    loss = ((ypred - Y) ** 2).sum()

    for p in params:
        p.grad = None
    loss.backward()

    with torch.no_grad():
        for p in params:
            p -= lr * p.grad

    if k % 10 == 0 or k == 99:
        print(f"    passo {k:3d}   loss {loss.item():.6f}")

print(f"\n  predizioni finali: {[round(v, 4) for v in mlp(X).flatten().tolist()]}")
print(f"  target:            {Y.flatten().tolist()}")
print("\n  il loop non e' cambiato di una riga rispetto alla tappa 7: quello che")
print("  cambia e' solo chi tiene i pesi, e da quali numeri sono partiti")
