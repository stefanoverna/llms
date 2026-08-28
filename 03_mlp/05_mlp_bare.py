"""La stessa cosa di 04_mlp_idiomatic.py, senza commenti. Il codice, e basta."""

import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 2147483647
HERE = Path(__file__).parent
NAMES = HERE.parent / "02_bigram" / "names.txt"

BLOCK_SIZE = 3
EMB_DIM = 10
HIDDEN = 200
BATCH_SIZE = 32
FAN_IN = BLOCK_SIZE * EMB_DIM
EPOCHS = 36

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
Xte, Yte = build_dataset(words[n_dev:])

torch.manual_seed(SEED)

model = nn.Sequential(
    nn.Embedding(VOCAB, EMB_DIM),
    nn.Flatten(),
    nn.Linear(FAN_IN, HIDDEN, bias=False),
    nn.BatchNorm1d(HIDDEN, momentum=0.001),
    nn.Tanh(),
    nn.Linear(HIDDEN, VOCAB),
)

nn.init.kaiming_normal_(model[2].weight, nonlinearity="tanh")
nn.init.normal_(model[5].weight, std=0.01)
nn.init.zeros_(model[5].bias)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[EPOCHS // 2], gamma=0.1)
loader = DataLoader(
    TensorDataset(Xtr, Ytr),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    generator=torch.Generator().manual_seed(SEED),
)

model.train()
for epoch in range(EPOCHS):
    for xb, yb in loader:
        loss = loss_fn(model(xb), yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    scheduler.step()

    if epoch % 6 == 0 or epoch == EPOCHS - 1:
        print(f"epoca {epoch:2d}   loss {loss.item():.4f}   lr {optimizer.param_groups[0]['lr']:g}")

model.eval()


@torch.no_grad()
def evaluate(X, Y):
    return loss_fn(model(X), Y).item()


print(f"\ntrain: {evaluate(Xtr, Ytr):.4f}")
print(f"dev:   {evaluate(Xdev, Ydev):.4f}")
print(f"test:  {evaluate(Xte, Yte):.4f}")


@torch.no_grad()
def sample(num=10):
    g = torch.Generator().manual_seed(SEED)
    names = []
    for _ in range(num):
        out, context = [], [0] * BLOCK_SIZE
        while True:
            logits = model(torch.tensor([context]))
            ix = torch.multinomial(logits.softmax(1), num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            if ix == 0:
                break
            out.append(itos[ix])
        names.append("".join(out))
    return names


print("\n" + "  ".join(sample()))
