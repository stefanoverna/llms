"""
Tappa 4: un neurone.

Da qui in poi il Value non cresce piu' a vista: e' quello finito in engine.py,
con tanh, exp, potenze, divisione e tutto il syntactic sugar.

Il neurone e' la prima espressione che somiglia a qualcosa: w . x + b passato
in una non linearita'. E la costruiamo due volte, per mostrare che il livello a
cui si spezzano le operazioni e' una scelta libera.

Corrisponde alla lezione da "let's back propagate through a neuron" fino alla
tanh riscritta con gli esponenziali.
"""

import math

from engine import Value, print_graph


def make_neuron():
    """Il neurone della lezione, con numeri scelti per far uscire tondi i grad."""
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    # i pesi sono le "forze sinaptiche" degli ingressi, il bias dice quanto il
    # neurone e' facile da eccitare a prescindere dall'input
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    b = Value(6.8813735870195432, label="b")
    x1w1 = x1 * w1
    x1w1.label = "x1*w1"
    x2w2 = x2 * w2
    x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1*w1 + x2*w2"
    n = x1w1x2w2 + b
    n.label = "n"
    return x1, x2, w1, w2, n


# ---------------------------------------------------------------------------
# con tanh come operazione atomica
# ---------------------------------------------------------------------------

print("=== un neurone: tanh(x1*w1 + x2*w2 + b) ===\n")

# tanh schiaccia qualsiasi input dentro (-1, 1): serve a rendere non lineare il
# neurone, altrimenti una pila di layer collasserebbe in una singola somma
# pesata. In engine.py e' implementata come operazione unica, perche' ne
# conosciamo la derivata: 1 - tanh(x)^2.
x1, x2, w1, w2, n = make_neuron()
o = n.tanh()
o.label = "o"
o.backward()

print_graph(o)
print("\n  i gradienti che contano sono quelli sui pesi:")
print(f"    w1.grad = {w1.grad:+.4f}   w2.grad = {w2.grad:+.4f}")
print("  w2 ha gradiente 0 perche' il suo input x2 e' 0: muoverlo non cambia nulla")

grad_with_tanh = [x1.grad, x2.grad, w1.grad, w2.grad]


# ---------------------------------------------------------------------------
# con tanh spezzata nei suoi pezzi
# ---------------------------------------------------------------------------

print("\n=== la stessa tanh, spezzata in exp / somma / divisione ===\n")

# tanh(n) = (e^2n - 1) / (e^2n + 1). Sopra l'abbiamo trattata come operazione
# atomica; qui la costruiamo con operazioni piu' piccole. Il grafo diventa piu'
# lungo ma forward e backward devono dare esattamente gli stessi numeri: il
# livello a cui si spezzano le operazioni e' una scelta di comodo, l'unica cosa
# che conta e' saper scrivere la derivata locale di quello che si implementa.

x1, x2, w1, w2, n = make_neuron()
ex = (2 * n).exp()
o = (ex - 1) / (ex + 1)
o.label = "o"
o.backward()

print(f"  forward:  {o.data:.4f}  (prima era lo stesso numero)")
print(f"  backward: x1 {x1.grad:+.4f}, x2 {x2.grad:+.4f}, w1 {w1.grad:+.4f}, w2 {w2.grad:+.4f}")
# non identici bit a bit (il percorso di calcolo e' diverso), ma uguali
same = all(
    math.isclose(new, old, abs_tol=1e-9)
    for new, old in zip([x1.grad, x2.grad, w1.grad, w2.grad], grad_with_tanh)
)
print(f"  identici a quelli di prima: {same}")

print("\n  qui '2 * n', '- 1' e '/' funzionano grazie a __rmul__, __sub__ e")
print("  __truediv__, che engine.py costruisce tutti sopra + * e **")
