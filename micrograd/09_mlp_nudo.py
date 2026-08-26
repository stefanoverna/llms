"""Tappa 9: la tappa 8 senza commenti. Il codice, e basta."""

import torch
from torch import nn

torch.manual_seed(1337)

X = torch.tensor(
    [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
)
Y = torch.tensor([[1.0], [-1.0], [-1.0], [1.0]])

mlp = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),
    nn.Linear(4, 4),
    nn.Tanh(),
    nn.Linear(4, 1),
    nn.Tanh(),
)

loss_fn = nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.05)

for k in range(100):
    loss = loss_fn(mlp(X), Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if k % 10 == 0 or k == 99:
        print(f"passo {k:3d}   loss {loss.item():.6f}")

print(f"\npredizioni: {[round(v, 4) for v in mlp(X).flatten().tolist()]}")
print(f"target:     {Y.flatten().tolist()}")
