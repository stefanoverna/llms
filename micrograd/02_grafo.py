"""
Tappa 2: il grafo delle espressioni, e la backpropagation fatta a mano.

Il Value di questo file sa costruire il grafo ma non sa derivare: i gradienti
li scriviamo noi, un nodo alla volta, verificandoli contro la stima numerica.
E' apposta: serve a vedere esattamente cosa dovra' fare backward() nella
tappa 3.

Corrisponde alla lezione da "let's build out this value object" fino al primo
passo di ottimizzazione.
"""

# print_graph e' solo un visualizzatore: guarda .data/.grad/.label/._op/._prev
# e va bene anche con il Value ridotto qui sotto. Non fa parte della lezione.
from engine import print_graph


class Value:
    """Prima versione: registra come e' stato calcolato, non sa ancora derivare."""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        # lo teniamo gia' qui, ma per ora lo riempiamo a mano
        self.grad = 0.0
        # il pezzo nuovo: i puntatori ai nodi da cui questo valore e' nato, e
        # l'operazione che li ha combinati. Senza questi il grafo non esiste.
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), "*")


def numerical_derivative(fn, x, h=0.000001):
    return (fn(x + h) - fn(x)) / h


# ---------------------------------------------------------------------------
# il grafo
# ---------------------------------------------------------------------------

print("=== il grafo delle espressioni: L = (a*b + c) * f ===\n")

# Le espressioni di una rete neurale sono enormi: nessuno le deriva a mano su un
# foglio. Serve una struttura dati che, mentre calcola, si ricordi come ci e'
# arrivata: quali nodi ha usato e con quale operazione.

a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a * b
e.label = "e"
d = e + c
d.label = "d"
ff = Value(-2.0, label="f")
L = d * ff
L.label = "L"

print(f"  forward pass: L = {L.data}\n")
print_graph(L)


# ---------------------------------------------------------------------------
# backpropagation a mano
# ---------------------------------------------------------------------------

print("\n=== backpropagation a mano, un nodo alla volta ===\n")


def forward_L(a, b, c, f):
    return (a * b + c) * f


# caso base: quanto cambia L se muovo L? Di altrettanto, quindi 1.
L.grad = 1.0

# L = d * f, e la derivata di un prodotto rispetto a un fattore e' l'altro
ff.grad = d.data  # dL/df = d
d.grad = ff.data  # dL/dd = f

# Qui c'e' il nocciolo di tutto. Il nodo '+' che ha prodotto d non sa niente del
# resto del grafo: sa solo che ha sommato c ed e, quindi la sua derivata locale
# e' 1 verso entrambi. La chain rule dice che per ottenere dL/dc basta
# moltiplicare la derivata locale per quella che arriva da sopra:
#   dL/dc = dL/dd * dd/dc = -2 * 1
#
# Il motivo per cui si moltiplica e' quello delle velocita': se un'auto va il
# doppio di una bici, e la bici va quattro volte un uomo a piedi, allora l'auto
# va 2 * 4 = 8 volte l'uomo a piedi. Ogni tratto della catena ha il suo rapporto
# di cambiamento, e per sapere come il primo influenza l'ultimo si moltiplicano
# i rapporti fra loro.
#
# Cioe': un '+' smista il gradiente ai figli senza cambiarlo.
c.grad = d.grad * 1.0
e.grad = d.grad * 1.0

# stessa cosa sul '*': la derivata locale rispetto ad a e' il valore di b
a.grad = e.grad * b.data
b.grad = e.grad * a.data

print("  gradienti calcolati a mano, verificati contro la stima numerica:")
for name, node, numeric in [
    ("a", a, numerical_derivative(lambda v: forward_L(v, b.data, c.data, ff.data), a.data)),
    ("b", b, numerical_derivative(lambda v: forward_L(a.data, v, c.data, ff.data), b.data)),
    ("c", c, numerical_derivative(lambda v: forward_L(a.data, b.data, v, ff.data), c.data)),
    ("f", ff, numerical_derivative(lambda v: forward_L(a.data, b.data, c.data, v), ff.data)),
]:
    print(f"    dL/d{name}: a mano {node.grad:+.4f}   numerico {numeric:+.4f}")

print("\n  il grafo, adesso con i gradienti dentro:\n")
print_graph(L)


# ---------------------------------------------------------------------------
# a cosa serve il gradiente
# ---------------------------------------------------------------------------

print("\n=== un passo nella direzione del gradiente ===\n")

# Il gradiente dice come muovere ogni input per far salire L. Se ci muoviamo in
# quella direzione, L deve salire: e' tutto quello che serve per ottimizzare.
# Nella tappa 6 faremo esattamente questo, ma col segno girato per far scendere
# una loss invece che salire una L.
step = 0.01
nudged = [n.data + step * n.grad for n in (a, b, c, ff)]
print(f"  L prima: {L.data:.4f}")
print(f"  L dopo un passo di {step}: {forward_L(*nudged):.4f}")

print("\n  fare tutto questo a mano e' insostenibile: nella tappa 3 lo automatizziamo")
