import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).parent.parent / "06_image_classification"))
import mnist

SEED = 1
HERE = Path(__file__).parent

HIDDEN = 200
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.1

torch.manual_seed(SEED)

(xtr, ytr), (xva, yva), (xte, yte) = mnist.load()

model = nn.Sequential(
    nn.Linear(784, HIDDEN),
    nn.Tanh(),
    nn.Linear(HIDDEN, 10),
)

n_params = sum(p.numel() for p in model.parameters())

print("=== la rete ===\n")
print(f"  {model}\n")
print(f"  {n_params:,} parametri")

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


@torch.no_grad()
def evaluate(model, x, y, chunk=5000):
    model.eval()
    total_loss = 0.0
    correct = 0
    for k in range(0, len(x), chunk):
        logits = model(x[k : k + chunk])
        total_loss += F.cross_entropy(logits, y[k : k + chunk], reduction="sum").item()
        correct += (logits.argmax(dim=1) == y[k : k + chunk]).sum().item()
    model.train()
    return total_loss / len(x), correct / len(x) * 100


loader = DataLoader(
    TensorDataset(xtr, ytr), batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
steps_per_epoch = len(loader)

print("\n=== il training ===\n")
print(f"  {EPOCHS} epoche, minibatch da {BATCH_SIZE}, SGD lr={LEARNING_RATE}")
print(f"  {steps_per_epoch} passi per epoca, {EPOCHS * steps_per_epoch:,} in tutto\n")
print(f"  {'epoca':>6}  {'loss train':>11}  {'loss val':>9}  {'acc train':>10}  {'acc val':>8}")

best_val_acc = 0.0
best_epoch = 0
best_state = None

for epoch in range(EPOCHS):
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb), yb).backward()
        optimizer.step()

    train_loss, train_acc = evaluate(model, xtr, ytr)
    val_loss, val_acc = evaluate(model, xva, yva)

    if val_acc > best_val_acc:
        best_val_acc, best_epoch = val_acc, epoch
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    marker = "  <- migliore finora" if epoch == best_epoch else ""
    print(f"  {epoch:>6}  {train_loss:>11.4f}  {val_loss:>9.4f}  {train_acc:>9.2f}%  {val_acc:>7.2f}%{marker}")

model.load_state_dict(best_state)

print(f"\n  {elapsed / 60:.1f} minuti.")
print(f"  ripresi i pesi dell'epoca {best_epoch}, validation {best_val_acc:.2f}%")

test_loss, test_acc = evaluate(model, xte, yte)
errors = int(round((100 - test_acc) * len(xte) / 100))

print("\n=== il test ===\n")
print(f"  loss {test_loss:.4f}, accuratezza {test_acc:.2f}%  ({errors} errori su {len(xte)})")
