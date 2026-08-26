"""
makemore, parte 1: modello di linguaggio bigram a livello di carattere.

Ricostruzione dello script della lezione fino a 1h03m: si ferma subito prima
della riformulazione del bigram come rete neurale.

Contenuti:
  1. caricamento del dataset
  2. conteggio dei bigrammi (prima con un dict, poi con un tensore 27x27)
  3. visualizzazione della matrice dei conteggi
  4. normalizzazione in probabilita' (broadcasting + keepdim)
  5. campionamento di nomi nuovi con torch.multinomial
  6. valutazione del modello con la negative log likelihood
  7. model smoothing (+1)
"""

from pathlib import Path

import torch
import matplotlib

matplotlib.use("Agg")  # niente notebook: salviamo la figura su file
import matplotlib.pyplot as plt

SEED = 2147483647
HERE = Path(__file__).parent  # i file stanno accanto allo script, non nel cwd


# ---------------------------------------------------------------------------
# 1. il dataset
# ---------------------------------------------------------------------------

words = open(HERE / "names.txt", "r").read().splitlines()

print("primi 10 nomi:", words[:10])
print("numero di parole:", len(words))
print("parola piu' corta:", min(len(w) for w in words))
print("parola piu' lunga:", max(len(w) for w in words))


# ---------------------------------------------------------------------------
# 2a. conteggio dei bigrammi con un dizionario
# ---------------------------------------------------------------------------

b = {}
for w in words:
    chs = ["<S>"] + list(w) + ["<E>"]
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

# ordinati per conteggio decrescente: i bigrammi piu' probabili
print("\nbigrammi piu' frequenti:", sorted(b.items(), key=lambda kv: -kv[1])[:5])
print("bigrammi meno frequenti:", sorted(b.items(), key=lambda kv: kv[1])[:5])


# ---------------------------------------------------------------------------
# 2b. lookup table caratteri <-> interi, e conteggi in un tensore 27x27
# ---------------------------------------------------------------------------

# un solo token speciale '.' (invece di <S>/<E>), messo in posizione 0
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

N = torch.zeros((27, 27), dtype=torch.int32)

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1


# ---------------------------------------------------------------------------
# 3. visualizzazione della matrice dei conteggi
# ---------------------------------------------------------------------------

def plot_counts(path=HERE / "bigram_counts.png"):
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap="Blues")
    for i in range(27):
        for j in range(27):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", dpi=80)
    plt.close()
    print(f"\nmatrice dei conteggi salvata in {path}")


plot_counts()


# ---------------------------------------------------------------------------
# 4. dai conteggi alle probabilita'
# ---------------------------------------------------------------------------

# +1 = model smoothing: nessun bigramma ha probabilita' 0, quindi nessuna
# loss infinita. Mettere 0 al posto di 1 riproduce il modello non smoothed.
P = (N + 1).float()

# ATTENZIONE al broadcasting: keepdim=True tiene la shape (27, 1), cioe' un
# vettore colonna, che viene replicato orizzontalmente -> normalizza le RIGHE.
# Senza keepdim la shape diventa (27,), che broadcasting interpreta come riga
# (1, 27) replicata verticalmente -> normalizzerebbe le COLONNE. Bug silenzioso.
P /= P.sum(1, keepdim=True)

assert torch.allclose(P[0].sum(), torch.tensor(1.0))


# ---------------------------------------------------------------------------
# 5. campionamento dal modello
# ---------------------------------------------------------------------------

def sample(num=5, probs=None):
    g = torch.Generator().manual_seed(SEED)
    names = []
    for _ in range(num):
        out = []
        ix = 0  # si parte sempre dal token '.'
        while True:
            p = P[ix] if probs is None else probs
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            if ix == 0:
                break
            out.append(itos[ix])
        names.append("".join(out))
    return names


print("\nnomi campionati dal modello bigram:")
for name in sample(5):
    print(" ", name)

# per confronto: un modello completamente non allenato (distribuzione uniforme)
print("\nnomi campionati da un modello uniforme (non allenato):")
for name in sample(5, probs=torch.ones(27) / 27):
    print(" ", name)


# ---------------------------------------------------------------------------
# 6. valutazione: negative log likelihood media
# ---------------------------------------------------------------------------

def nll(dataset, verbose=False):
    """Average negative log likelihood: piu' e' bassa, meglio e'. Minimo 0."""
    log_likelihood = 0.0
    n = 0
    for w in dataset:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            logprob = torch.log(prob)
            log_likelihood += logprob
            n += 1
            if verbose:
                print(f"  {ch1}{ch2}: {prob:.4f} {logprob:.4f}")
    return -log_likelihood / n


print(f"\nloss sull'intero training set: {nll(words):.4f}")

# la likelihood si puo' valutare su qualsiasi parola
print("\nvalutazione di 'andrej':")
print(f"  loss: {nll(['andrej'], verbose=True):.4f}")

# senza smoothing questa sarebbe inf: il bigramma 'jq' non compare mai
print("\nvalutazione di 'andrejq':")
print(f"  loss: {nll(['andrejq']):.4f}")
