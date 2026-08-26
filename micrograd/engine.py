"""
Il motore di autograd: la classe Value.

Ogni Value avvolge un singolo scalare e tiene traccia di quattro cose:
  - data:      il valore calcolato in avanti (forward pass)
  - grad:      la derivata dell'output finale rispetto a questo nodo
  - _prev/_op: da quali nodi e con quale operazione e' stato prodotto
  - _backward: la chiusura che applica la chain rule locale di quel nodo

Con queste quattro cose si costruisce un grafo delle espressioni e lo si
percorre all'indietro con backward().
"""

import math


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        # 0 = "questo nodo non influenza l'output": e' il valore neutro giusto
        # da cui partire, visto che i gradienti si accumulano in somma.
        self.grad = 0.0
        # per una foglia non c'e' niente da propagare: funzione vuota
        self._backward = lambda: None
        # nella lezione i figli arrivano come tupla ma vengono tenuti come set
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # -----------------------------------------------------------------------
    # operazioni
    # -----------------------------------------------------------------------

    def __add__(self, other):
        # cosi' `a + 1` funziona: se l'altro operando non e' un Value lo avvolgo
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # la derivata locale del '+' e' 1 per entrambi gli addendi: il nodo
            # somma si limita a smistare il gradiente ai figli, invariato.
            # ATTENZIONE al += invece di =: se una variabile compare piu' volte
            # nell'espressione (es. a + a) i contributi vanno sommati, non
            # sovrascritti. E' la regola della catena multivariata.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b: ogni fattore riceve il valore dell'altro
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        # solo esponenti costanti: con un esponente Value la derivata sarebbe
        # un'altra espressione (servirebbe anche il log)
        assert isinstance(other, (int, float)), "supportati solo int/float"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            # regola della potenza: d(x**n)/dx = n * x**(n-1)
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            # d(e**x)/dx = e**x, che e' esattamente out.data: gia' calcolato
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        # tanh non si costruisce con soli + e *: serve l'esponenziale. Qui pero'
        # la implementiamo come operazione unica invece di spezzarla, perche' il
        # livello di atomicita' delle operazioni e' una scelta libera: basta
        # saper scrivere la derivata locale.
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)**2, e tanh(x) e' gia' t
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        # la non linearita' che usa il micrograd vero: piu' spigolosa di tanh
        out = Value(self.data if self.data > 0 else 0.0, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    # -----------------------------------------------------------------------
    # syntactic sugar, tutto costruito sopra le operazioni di sopra
    # -----------------------------------------------------------------------

    def __neg__(self):  # -a
        return self * -1

    def __sub__(self, other):  # a - b
        return self + (-other)

    def __truediv__(self, other):  # a / b == a * b**-1
        return self * other**-1

    # le versioni "r" servono quando il Value sta a destra: Python prova prima
    # 2 * a, non sa cosa fare con un Value e ripiega su a.__rmul__(2)
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * self**-1

    # -----------------------------------------------------------------------
    # backpropagation
    # -----------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# visualizzazione del grafo
# ---------------------------------------------------------------------------

def print_graph(root, indent=0, _seen=None):
    """Stampa il grafo delle espressioni come albero indentato.

    Nella lezione questo si fa con graphviz; qui basta del testo per non tirarsi
    dietro una dipendenza di sistema. I nodi condivisi vengono ristampati, con
    un marcatore, perche' il grafo e' un DAG e non un albero.
    """
    if _seen is None:
        _seen = set()

    label = f"{root.label} " if root.label else ""
    op = f" [{root._op}]" if root._op else ""
    already_seen = " (gia' visto sopra)" if root in _seen else ""
    print(
        f"{'  ' * indent}{label}data {root.data:.4f} | "
        f"grad {root.grad:.4f}{op}{already_seen}"
    )

    if root in _seen:
        return
    _seen.add(root)

    # _prev e' un set, quindi l'ordine non e' quello di scrittura: ordiniamo
    # per avere un output stabile fra un'esecuzione e l'altra
    for child in sorted(root._prev, key=lambda v: (v.label, v._op, v.data)):
        print_graph(child, indent + 1, _seen)
