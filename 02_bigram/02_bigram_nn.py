"""
makemore, parte 1 (seguito): lo stesso bigram, ma come rete neurale.

`bigram.py` contava le coppie di lettere e le normalizzava. Qui non contiamo
niente: partiamo da una matrice di pesi a caso e lasciamo che sia la gradient
descent a trovare i numeri giusti. Il modello è lo stesso, il modo di
arrivarci no.

L'iter, per un carattere in ingresso (diciamo "n", indice 12):

  1. one-hot encoding    -> [0, 0, ..., 1, ..., 0], 27 numeri, un vettore riga
  2. un layer lineare    -> 27 output, senza bias e senza tanh, in (-inf, +inf)
  3. exp                 -> li porta in (0, +inf): sono i "counts" stimati
  4. normalizzazione     -> sommano a 1: sono le probabilità
  5. loss
  6. backward + step     -> i pesi si spostano, la loss scende

I passi 3 e 4 insieme sono la softmax. È un layer molto comune nelle NN e trasforma
un vettore di output (-inf, +inf) in un array di probabilità (0, 1) con somma 1.0.

Per il calcolo del loss useremo la "negative log likelihood media".

La *likelihood* è la probabilità che il modello assegna all'intero dataset:

likelihood = p(aa)^N[aa] * p(ab)^N[ab] * ... * p(zz)^N[zz]

Per ogni bigramma, elevo la probabilità stimata dalla NN al numero di volte
che quel bigramma compare nel dataset -> vogliamo probabilità alte sui
bigrammi frequenti e probabilità basse su quelli rari. Ma moltiplicare
migliaia di numeri < 1.0 tra di loro porta in underflow, quindi riformuliamo
il problema in modo equivalente:

log_likelihood = N[aa]*log p(aa) + N[ab]*log p(ab) + ... + N[zz]*log p(zz)

Ogni log p è ≤ 0 perché le probabilità stanno in (0,1] -> log_likelihood va da (-inf, 0]

Ottimo! È quasi un loss, basta invertirlo! Così loss = 0 -> obiettivo, loss
alto -> ancora da minimizzare.

negative_log_likelihood = -( ... )

Ancora meglio se dividiamo per il numero di esempi: il valore non dipende più da quanto
è grande il dataset, e vengono fuori numeri leggibili come 2.4 invece di somme da
centinaia di migliaia.

negative_log_likelihood_media = - (...) / sum(N)

Nel codice, però, né questa formula né la matrice N (frequenza per bigramma) compaiono mai.
N si può scrivere solo perché qui i tipi di evento sono appena 27x27 = 729, cioè
ci stanno in una tabella. Il calcolo effettivo invece non conta/raggruppa niente: scorre
gli esempi del dataset uno per uno, e per ognuno prende la probabilità che il
modello ha dato alla lettera che è davvero arrivata.

  negative_log_likelihood_media = - ( log p(y_0|x_0) + ... + log p(y_n-1|x_n-1) ) / n

che in PyTorch è l'indicizzazione che si vede in giro dappertutto:

  loss = -probs.log()[range(n), ys].mean()

`probs` ha una riga per *esempio* (n x 27), `ys` dice quale colonna è quella giusta,
e [range(n), ys] pesca una casella per riga: riga 0 colonna ys[0], riga 1 colonna
ys[1], e così via. n numeri, di cui si fa la media.

Le due forme danno lo stesso identico numero. Perché in pratica si usa sempre la seconda
forma?

  - non richiede di enumerare i tipi di evento. Qui sono 27x27 = 729, ma con contesto
    9 caratteri diventa 27^9: tabelle impossibili.
  - i minibatch diventano gratis: un minibatch è semplicemente un sottoinsieme di
    righe, e la formula non cambia di una virgola. Con N andrebbero ricontati i
    bigrammi del batch ogni volta.

Questa è la loss: il numero che stampa il training loop, e l'unico che serve
per allenare. Qualche valore di riferimento, per sapere cosa è buono:

  loss 0        il modello perfetto: probabilità 1.0 su ogni bigramma visto.
                È il limite teorico, ma qui è irraggiungibile: p=1 su una
                lettera vuol dire p=0 su tutte le altre della riga, cioè dopo
                ogni lettera ne può venire una sola. Nei nomi non è così.
  loss 3.2958   -log(1/27), quello che tira a caso.
  loss +inf     aver dato probabilità zero a qualcosa che poi è successo:
                ed è il motivo per cui serve lo smoothing.

In micrograd la loss era nn.MSELoss, la media dei quadrati degli scarti. Lì il
modello doveva indovinare un numero (regressione) e la cosa sensata da misurare
era di quanto avesse sbagliato; Qui deve scegliere fra 27 possibilità (classificazione),
e l'unica cosa che e' interessante valorizzare è quanta probabilità ha dato a quella giusta.

I passi 3, 4 e 5 insieme sono la "cross entropy". Che in PyTorch idiomatico è una riga sola:

  loss a mano (exp, /sum, log, mean)   ->  F.cross_entropy
  one_hot(x) @ W                       ->  nn.Embedding
  W.grad = None                        ->  optimizer.zero_grad()
  W -= lr * W.grad                     ->  optimizer.step()
  0.01 * (W**2).mean()                 ->  weight_decay dell'optimizer

In fondo al file: cosa cambia davvero rispetto alla versione a conteggi.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

SEED = 2147483647
HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# 1. dataset: non più una matrice di conteggi, ma coppie (input, target)
# ---------------------------------------------------------------------------

words = open(HERE / "names.txt", "r").read().splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
VOCAB = len(itos)  # 27

# Per la rete ogni bigramma è un esempio di training a sé: xs[i] è il
# carattere che ho, ys[i] quello che dovrebbe venire dopo. La stessa
# informazione della matrice N, srotolata in una lista lunga.
xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)

print(f"esempi di training: {xs.nelement()}")
print(f"primi 5: {[(itos[a.item()], itos[b.item()]) for a, b in zip(xs[:5], ys[:5])]}")


# ---------------------------------------------------------------------------
# 2. il modello: un layer, 27 -> 27
# ---------------------------------------------------------------------------

print("\n=== il modello ===\n")

torch.manual_seed(SEED)

# Il layer descritto dall'iter: 27 ingressi, 27 neuroni, niente bias, niente
# non linearità. Il one-hot va in input, i logits escono.
linear = nn.Linear(VOCAB, VOCAB, bias=False)

# Ma one_hot(12) @ W non è una vera moltiplicazione: 26 termini su 27 sono
# moltiplicati per zero. Quello che resta è *la riga 12 di W*, copiata. Cioè
# il layer lineare su un one-hot è una lookup table, e PyTorch ha un layer
# apposta che salta la matmul e va a prendere la riga: nn.Embedding.
#
# nn.Linear tiene W trasposta (out, in), quindi copiamo di conseguenza e
# verifichiamo che le due strade diano gli stessi numeri.
model = nn.Embedding(VOCAB, VOCAB)

with torch.no_grad():
    model.weight.copy_(linear.weight.T)

probe = xs[:8]
assert torch.allclose(linear(F.one_hot(probe, VOCAB).float()), model(probe), atol=1e-6)
print("  one_hot(x) @ W  ==  Embedding(x)  ->  stessi logits, senza la matmul")

print(f"\n  parametri: {sum(p.numel() for p in model.parameters())} (la matrice 27x27, e basta)")

# I 27 output di una riga sono log-counts: exp li porta in (0, +inf) e la
# normalizzazione in probabilità. Ma exp -> normalizza -> log -> media è
# esattamente la definizione della cross entropy, e F.cross_entropy la calcola
# in un colpo solo, prendendo i logits grezzi. Non passargli mai le
# probabilità già fatte: le rifarebbe, ed è proprio quello che vogliamo
# evitare (vedi "cosa cambia" in fondo).
loss_fn = F.cross_entropy

# lr enorme perché i gradienti qui sono minuscoli: la loss è una media su
# 228k esempi, e ogni riga di W viene toccata solo dagli esempi che partono da
# quel carattere. weight_decay è il model smoothing di bigram.py: spinge i
# pesi verso lo zero, cioè i counts verso l'uniforme. Nella lezione lo
# smoothing è un termine 0.01*(W**2).mean() aggiunto alla loss; il suo
# gradiente è 0.02/27**2 * W, che è quello che weight_decay somma al grad.
optimizer = torch.optim.SGD(model.parameters(), lr=50.0, weight_decay=0.01 * 2 / VOCAB**2)

print(f"  loss iniziale (pesi a caso): {loss_fn(model(xs), ys).item():.4f}")
print(f"  loss di un modello uniforme: {torch.tensor(1 / VOCAB).log().neg().item():.4f}")


# ---------------------------------------------------------------------------
# 3. il training loop: le solite cinque righe
# ---------------------------------------------------------------------------

print("\n=== gradient descent ===\n")

for k in range(1000):
    # forward su tutti i 228k esempi insieme: il dataset è minuscolo, non
    # servono minibatch. logits ha shape (228146, 27).
    loss = loss_fn(model(xs), ys)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if k % 100 == 0 or k == 999:
        print(f"    passo {k:3d}   loss {loss.item():.4f}")


# ---------------------------------------------------------------------------
# 4. campionamento: identico a prima, cambia solo da dove arrivano le probabilità
# ---------------------------------------------------------------------------

print("\n=== nomi campionati dalla rete ===\n")


@torch.no_grad()
def sample(num=5):
    g = torch.Generator().manual_seed(SEED)
    names = []
    for _ in range(num):
        out = []
        ix = 0
        while True:
            # softmax = exp + normalizzazione, i passi 3 e 4 dell'iter. Qui li
            # scriviamo perché ci servono le probabilità vere da campionare;
            # nel training invece restano dentro cross_entropy.
            probs = F.softmax(model(torch.tensor([ix])), dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            if ix == 0:
                break
            out.append(itos[ix])
        names.append("".join(out))
    return names


for name in sample(5):
    print("  ", name)

# gli stessi nomi di bigram.py, con lo stesso seed: è lo stesso modello


# ---------------------------------------------------------------------------
# 5. e infatti la rete ha imparato la matrice dei conteggi
# ---------------------------------------------------------------------------

print("\n=== rete vs conteggi ===\n")

N = torch.zeros((VOCAB, VOCAB), dtype=torch.int32)
for a, b in zip(xs, ys):
    N[a, b] += 1
P = (N + 1).float()
P /= P.sum(1, keepdim=True)

with torch.no_grad():
    Q = F.softmax(model.weight, dim=1)

nll_counts = -P[xs, ys].log().mean()
print(f"  loss del modello a conteggi: {nll_counts.item():.4f}")
print(f"  loss della rete:             {loss.item():.4f}")
print(f"  scarto massimo fra le due tabelle di probabilità: {(P - Q).abs().max().item():.4f}")

print(
    """
  Le due loss non coincidono al quarto decimale, e non è training incompleto:
  è il weight_decay che tira i pesi verso lo zero un po' più forte di quanto
  faccia il +1 sui conteggi. Togliendolo la rete arriva a 2.4554, contro il
  2.4540 dei conteggi non smoothed.

  A parte quello, la rete non ha trovato "un altro" modello: ha ritrovato
  quello a conteggi. Con un solo carattere di contesto la soluzione ottima è
  una sola, ed è la frequenza osservata, e la rete non poteva che finire lì.

  Quello che è cambiato è dove stanno i parametri. Nel modello a conteggi i
  parametri *sono* le probabilità: una per casella, tenute esplicitamente in
  una tabella. Nella rete la tabella non c'è più: ci sono dei pesi, e le
  probabilità vengono calcolate al volo quando servono. Il punto della
  riformulazione non è fare meglio qui, è che tenere la tabella non scala:
  per due caratteri di contesto diventa 27x27x27, per tre 27^4, e presto non
  ci sono abbastanza nomi al mondo per riempirne le caselle. I pesi invece
  cambiano solo di shape.
"""
)


# ---------------------------------------------------------------------------
# cosa cambia rispetto alla versione a mano (non è solo cosmetica)
# ---------------------------------------------------------------------------
#
# 1. F.cross_entropy non fa davvero exp -> /sum -> log. Con logits grandi exp
#    va a inf e la loss diventa nan; cross_entropy sottrae prima il massimo
#    (la softmax non cambia se trasli tutti i logits della stessa costante) e
#    poi usa log-sum-exp. È l'unico motivo per cui non le si passano mai le
#    probabilità già calcolate.
#
# 2. cross_entropy non alloca il tensore intermedio dei counts e ha un
#    backward scritto a mano: il gradiente rispetto ai logits è
#    (probs - onehot(target)), una riga, invece del grafo di tre operazioni.
#
# 3. weight_decay non è esattamente un termine in più nella loss: SGD lo
#    somma direttamente al gradiente. Coincide con la regolarizzazione L2 solo
#    per SGD puro, e infatti con Adam serve AdamW proprio per questo.
#
# 4. nn.Embedding non è solo comodo: il suo backward è sparso (scatter-add
#    sulle righe usate) invece di una matmul 228146x27 di quasi tutti zeri.
#    Su vocabolari da decine di migliaia di token è la differenza fra
#    fattibile e non fattibile.
