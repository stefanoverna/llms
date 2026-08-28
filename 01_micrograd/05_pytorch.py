"""
Tappa 5: la controprova con PyTorch.

L'unico file che non insegna niente di micrograd: serve solo a verificare che
quello che abbiamo scritto dia gli stessi numeri di una libreria vera.

PyTorch fa esattamente questo, ma su tensori invece che su scalari: gli stessi
.data e .grad, lo stesso .backward(). La differenza e' solo efficienza (i
tensori permettono di fare tante operazioni in parallelo), la matematica e'
identica.
"""

import torch

print("=== lo stesso neurone della tappa 4, in PyTorch ===\n")

# .double() perche' Python usa float a 64 bit e i tensori di default stanno a
# 32: senza, i numeri non tornerebbero fino all'ultima cifra
x1 = torch.Tensor([2.0]).double()
x2 = torch.Tensor([0.0]).double()
w1 = torch.Tensor([-3.0]).double()
w2 = torch.Tensor([1.0]).double()
b = torch.Tensor([6.8813735870195432]).double()
# le foglie di default non tengono i gradienti (di solito sono i dati, che non
# si toccano): qui li vogliamo, quindi lo chiediamo esplicitamente
for t in (x1, x2, w1, w2, b):
    t.requires_grad = True

o = torch.tanh(x1 * w1 + x2 * w2 + b)
o.backward()

print(f"  forward:  {o.data.item():.4f}")
print(
    f"  backward: x1 {x1.grad.item():+.4f}, x2 {x2.grad.item():+.4f}, "
    f"w1 {w1.grad.item():+.4f}, w2 {w2.grad.item():+.4f}"
)
print("\n  gli stessi numeri della tappa 4: micrograd e' PyTorch ristretto al")
print("  caso in cui ogni tensore ha un elemento solo")
