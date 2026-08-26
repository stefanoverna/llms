"""
Tappa 3: backward() automatica.

Stesso Value della tappa 2, con due aggiunte: ogni operazione si chiude dentro
una funzione che sa applicare la propria chain rule locale, e backward() le
chiama tutte nell'ordine giusto. Sa ancora fare solo '+' e '*': basta per il
grafo di prima, e tiene il file corto abbastanza da leggerlo tutto.

Qui salta fuori anche il bug piu' sottile di tutto micrograd.

Corrisponde alla lezione da "let's codify what we've seen" fino a "we have to
accumulate these gradients".
"""

from engine import print_graph


class Value:
    """Seconda versione: sa derivarsi da sola, ma conosce solo + e *."""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        # 0 = "questo nodo non influenza l'output": e' il valore neutro giusto
        # da cui partire, visto che i gradienti si accumulano in somma.
        self.grad = 0.0
        # per una foglia non c'e' niente da propagare: funzione vuota
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # la derivata locale del '+' e' 1 per entrambi gli addendi: il nodo
            # somma si limita a smistare il gradiente ai figli, invariato.
            # Sul += invece che = si torna sotto: e' il bug di questa tappa.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        # la funzione la salviamo, non la chiamiamo: verra' eseguita dopo,
        # quando out.grad sara' stato riempito da chi sta a valle
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b: ogni fattore riceve il valore dell'altro
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Riempie .grad di ogni nodo del grafo con la derivata di self."""
        # Non si puo' chiamare _backward su un nodo prima di aver finito tutti i
        # nodi che vengono dopo di lui, altrimenti il suo grad e' ancora
        # incompleto. L'ordinamento topologico garantisce proprio questo.
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                # un nodo entra in lista solo dopo tutti i suoi figli
                topo.append(v)

        build_topo(self)

        self.grad = 1.0  # caso base: d(self)/d(self) = 1
        for node in reversed(topo):
            node._backward()


def numerical_derivative(fn, x, h=0.000001):
    return (fn(x + h) - fn(x)) / h


# ---------------------------------------------------------------------------
# lo stesso grafo della tappa 2, senza scrivere un gradiente a mano
# ---------------------------------------------------------------------------

print("=== backward(): la stessa cosa della tappa 2, automatica ===\n")

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

L.backward()
print_graph(L)
print("\n  stessi numeri di prima, una riga di codice invece di sei")


# ---------------------------------------------------------------------------
# il bug dell'accumulo
# ---------------------------------------------------------------------------

print("\n=== perche' i gradienti si sommano (+=) e non si sovrascrivono (=) ===\n")

# Finche' ogni variabile compare una volta sola nell'espressione il bug non si
# vede, ed e' per questo che il grafo qui sopra non l'ha fatto emergere. Ma qui
# a compare due volte, e con '=' il secondo contributo sovrascriverebbe il primo
# lasciando 1 invece di 2.
a = Value(3.0, label="a")
b = a + a
b.label = "b"
b.backward()
print(f"  b = a + a  ->  db/da = {a.grad:.4f}  (giusto: 1 + 1 = 2, non 1)")

# caso meno ovvio: a e b sono usate da due rami diversi che si ricongiungono
a = Value(-2.0, label="a")
b = Value(3.0, label="b")
d = a * b
d.label = "d"
e = a + b
e.label = "e"
g = d * e
g.label = "g"
g.backward()
print(f"  g = (a*b) * (a+b)  ->  dg/da = {a.grad:+.4f}, dg/db = {b.grad:+.4f}")
print(
    f"    controprova numerica: {numerical_derivative(lambda v: (v * 3.0) * (v + 3.0), -2.0):+.4f}, "
    f"{numerical_derivative(lambda v: (-2.0 * v) * (-2.0 + v), 3.0):+.4f}"
)
print("\n  e' la regola della catena multivariata: quando una variabile influenza")
print("  l'output per piu' strade, i contributi delle singole strade si sommano.")
