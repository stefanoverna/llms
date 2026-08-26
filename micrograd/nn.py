"""
La libreria di reti neurali costruita sopra il motore di autograd.

Un neurone e' w . x + b passato in una non linearita', un layer e' una lista di
neuroni valutati in parallelo, una MLP e' una sequenza di layer. Tutto qui: e'
il motore di autograd a fare il lavoro vero.

L'API (parameters, zero_grad, __call__) ricalca quella di torch.nn.
"""

import random

from engine import Value


class Module:
    def zero_grad(self):
        # va chiamata prima di ogni backward: i gradienti si accumulano con +=,
        # quindi senza reset quelli del passo precedente restano dentro
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin):
        # un peso per ogni input (la "forza sinaptica") piu' un bias, che regola
        # quanto il neurone e' facile da eccitare a prescindere dall'input
        self.w = [Value(random.uniform(-1, 1), label=f"w{i}") for i in range(nin)]
        self.b = Value(random.uniform(-1, 1), label="b")

    def __call__(self, x):
        # sum(..., self.b) parte dal bias invece che da 0: un nodo in meno
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout):
        # i neuroni di un layer non si parlano fra loro: sono indipendenti,
        # tutti collegati agli stessi input
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        # comodita': se il layer ha un solo neurone restituisce lo scalare
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nouts):
        # nouts descrive le dimensioni dei layer: MLP(3, [4, 4, 1]) sono 3
        # input, due layer da 4 neuroni e un'uscita singola
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(l) for l in self.layers)}]"
