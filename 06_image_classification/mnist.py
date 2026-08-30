"""
MNIST: 70.000 cifre scritte a mano, e il modo in cui il capitolo le divide.

`data/mnist.pkl.gz` è il file del repository di Nielsen, preso così com'è.
Dentro c'è un pickle di Python 2 con tre coppie `(immagini, etichette)`:

    50.000   train        le immagini su cui la rete impara
    10.000   validation   messe da parte, in questo capitolo non si usano
    10.000   test         il voto finale

La divisione ufficiale di MNIST è 60.000 + 10.000. Qui le 60.000 sono spezzate
in 50.000 + 10.000: la validation serve dal capitolo 3 in poi per scegliere gli
iperparametri — il learning rate e compagnia — senza sbirciare il test. Quando
il capitolo dice "i dati di training" intende le 50.000.

Le 10.000 di test vengono da 250 persone diverse da quelle del training: è
questo che rende il voto finale un voto, e non un'interrogazione su cose già
viste.

Ogni immagine è 28x28 in scala di grigi, appiattita in un vettore di 784
numeri fra 0.0 (bianco) e 1.0 (nero). Le due forme dell'etichetta:

  - per **allenare** serve il vettore a 10 componenti che la rete deve imitare:
    un 6 diventa (0,0,0,0,0,0,1,0,0,0). È `one_hot()`.
  - per **contare quante ne indovina** serve l'intero, da confrontare con
    l'indice del neurone di uscita più acceso.

Nielsen tiene i due formati in due strutture diverse (`training_data` con le
y-vettore, `test_data` con le y-intero). Qui `load()` restituisce sempre gli
interi e `one_hot()` si chiama dove serve: sono gli stessi dati.
"""

import gzip
import pickle
from pathlib import Path

import torch

# relativo a questo file, non alla directory da cui si lancia lo script
PATH = Path(__file__).parent / "data" / "mnist.pkl.gz"


def load():
    """Restituisce (train, validation, test), ognuno una coppia (x, y).

    x è (N, 784) float32 in [0, 1], y è (N,) int64 con la cifra giusta.
    """
    with gzip.open(PATH, "rb") as f:
        # encoding="latin1" perché il pickle è stato scritto da Python 2
        train, validation, test = pickle.load(f, encoding="latin1")

    def to_tensors(pair):
        x, y = pair
        return torch.from_numpy(x).float(), torch.from_numpy(y).long()

    return to_tensors(train), to_tensors(validation), to_tensors(test)


def one_hot(y):
    """Da (N,) di cifre a (N, 10) di vettori con un 1.0 nel posto giusto."""
    return torch.zeros(len(y), 10).scatter_(1, y[:, None], 1.0)


def draw(x, width=28):
    """Un'immagine (784,) come arte ASCII, per guardarla senza matplotlib."""
    levels = " .:-=+*#%@"
    rows = []
    for r in range(width):
        row = x[r * width : (r + 1) * width]
        rows.append("".join(levels[min(int(v * len(levels)), len(levels) - 1)] * 2 for v in row))
    return "\n".join(rows)


if __name__ == "__main__":
    (xtr, ytr), (xva, yva), (xte, yte) = load()
    print(f"train       {tuple(xtr.shape)}  etichette {tuple(ytr.shape)}")
    print(f"validation  {tuple(xva.shape)}  etichette {tuple(yva.shape)}")
    print(f"test        {tuple(xte.shape)}  etichette {tuple(yte.shape)}")
    print(f"\npixel: da {xtr.min():.1f} (bianco) a {xtr.max():.2f} (nero)")
    print(f"\nla prima immagine del training, etichetta {ytr[0].item()}:\n")
    print(draw(xtr[0]))
    print(f"\ncome la vuole il costo quadratico:\n  {one_hot(ytr[:1])[0]}")
