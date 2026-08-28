"""
Tappa 1: che cos'e' una derivata.

Prima di scrivere qualsiasi cosa serve avere ben chiaro cosa misura una
derivata. Niente Value, niente grafi: solo numeri.

Corrisponde ai primi minuti della lezione.
"""


def numerical_derivative(fn, x, h=0.000001):
    """La definizione di derivata, senza fare il limite: (f(x+h) - f(x)) / h.

    E' la pendenza della funzione in x: di quanto risponde l'output se spingo
    l'input di un pelo. Con h troppo piccolo la precisione dei float si rompe.
    """
    return (fn(x + h) - fn(x)) / h


# ---------------------------------------------------------------------------
# una funzione di un solo scalare
# ---------------------------------------------------------------------------

print("=== una funzione di un solo scalare: f(x) = 3x^2 - 4x + 5 ===\n")


def f(x):
    return 3 * x**2 - 4 * x + 5


# In una rete neurale nessuno deriva a mano l'espressione: sarebbe lunga
# decine di migliaia di termini. Quindi non seguiamo la strada simbolica, ma
# quella numerica, che e' anche l'unica che dice davvero cosa sta succedendo.
# Analiticamente f'(x) = 6x - 4, quindi 14 in 3, -22 in -3, 0 in 2/3.
for x in (3.0, -3.0, 2 / 3):
    print(f"  f({x:+.4f}) = {f(x):8.4f}   f'({x:+.4f}) ~ {numerical_derivative(f, x):+.4f}")

print("\n  il segno dice se la funzione sale o scende, il modulo quanto forte")


# ---------------------------------------------------------------------------
# una funzione con piu' input
# ---------------------------------------------------------------------------

print("\n=== tre input, un output: d = a*b + c ===\n")

a, b, c = 2.0, -3.0, 10.0


def forward_d(a, b, c):
    return a * b + c


print(f"  d = {forward_d(a, b, c)}")
# derivata rispetto a un input solo: gli altri restano fissi
print(f"  dd/da ~ {numerical_derivative(lambda v: forward_d(v, b, c), a):+.4f}  (= b)")
print(f"  dd/db ~ {numerical_derivative(lambda v: forward_d(a, v, c), b):+.4f}  (= a)")
print(f"  dd/dc ~ {numerical_derivative(lambda v: forward_d(a, b, v), c):+.4f}  (= 1)")

print("\n  ogni input ha la sua sensibilita': e' questo l'insieme di numeri che")
print("  vogliamo saper calcolare per i pesi di una rete neurale.")
