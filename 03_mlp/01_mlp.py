"""
makemore, parte 2: lo stesso problema, ma con una MLP (Bengio et al. 2003).

Il bigram funzionava, ma guardava un carattere solo. Allungare il contesto
nella versione a conteggi non si può: la tabella ha una riga per ogni contesto
possibile, quindi 27 righe con un carattere, 27^2 = 729 con due, 27^3 = 19683
con tre. Le righe esplodono esponenzialmente mentre gli esempi restano quelli
che sono, e i conteggi per riga si assottigliano finché non c'è più niente da
contare: 228k bigrammi spalmati su 19683 righe fanno una decina di esempi per
riga, e le probabilità stimate diventano rumore.

La strada è quella del paper di Bengio et al. 2003, che è il riferimento della
lezione:

  https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

Nel paper il modello è a livello di *parola*, con un vocabolario di 17k parole:

  - durante l'inferenza si prendono le ultime 3 parole (block size, o context
    length) e si fa il lookup nella matrice di embedding: ogni parola delle 17k
    diventa un vettore di 30 elementi;
  - si ottiene un vettore da 90 numeri, [...emb1, ...emb2, ...emb3], e questo è
    l'input di un layer MLP (con bias e tanh). Quanti neuroni abbia quel layer
    lo possiamo decidere indipendentemente, ma i suoi neuroni avranno sempre 90
    input;
  - il layer successivo ha 17k neuroni, uno per parola, ciascuno connesso a
    tutti i neuroni del layer precedente: sputa fuori i logits;
  - una softmax sopra ai logits dà le probabilità della prossima parola;
  - nel training si usa la negative log likelihood come funzione di loss,
    confrontando l'output con i risultati effettivi (le "labels");
  - la gradient descent gira su *tutti* i parametri: la matrice di embedding
    più i pesi di tutti i neuroni dei layer.

La cosa interessante è perché questo generalizza dove la tabella no. Se
"a dog was running in a ___" non è mai comparso nel training set, la tabella
non ha niente da dirti. La rete invece può aver visto "the dog was running in
a ___", e se ha imparato a mettere gli embedding di "a" e "the" vicini nello
spazio, la conoscenza si trasferisce. Stessa cosa per "cats"/"dogs", o per
"walking"/"running": è l'embedding a fare da ponte fra contesti diversi ma
simili. Le probabilità non stanno più in una casella, sono calcolate.

Nel nostro caso non parole ma caratteri, e contesto di 3 caratteri:

  X è la matrice di input,  N x 3 caratteri (N = numero di esempi "c1 c2 c3"
                            che ci dà il dataset)
  Y è il vettore di label,  N

I nostri embedding partono a 2 dimensioni: i 27 caratteri stanno in un piano,
quindi la matrice C è 27x2, inizializzata a caso. Il vantaggio di 2 è che si
può disegnare (lo facciamo, in 01_out_embeddings.png); alla fine passiamo a 10.

Cosa c'è di nuovo rispetto a bigram_nn.py, oltre all'architettura:

  1. il minibatch: 228k esempi a ogni passo sono troppi. Meglio un gradiente
     approssimato e mille passi che il gradiente esatto e dieci passi.
  2. come si sceglie il learning rate, invece di indovinarlo.
  3. il learning rate decay: alla fine si rallenta.
  4. lo split train / dev / test, la differenza fra underfitting e overfitting,
     e come si capisce quando ci si deve fermare.

Alla fine del file: cosa ci portiamo dietro da questa tappa.
"""

import random
from pathlib import Path

import torch
from torch.nn import functional as F

import matplotlib

matplotlib.use("Agg")  # niente notebook: salviamo le figure su file
import matplotlib.pyplot as plt

SEED = 2147483647
HERE = Path(__file__).parent
NAMES = HERE.parent / "02_bigram" / "names.txt"  # lo stesso dataset della parte 1

BLOCK_SIZE = 3  # quanti caratteri di contesto per predire il prossimo
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# 1. il dataset: da parole a esempi (contesto -> carattere)
# ---------------------------------------------------------------------------

words = open(NAMES, "r").read().splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
VOCAB = len(itos)  # 27


def build_dataset(word_list, verbose=False):
    """Da una lista di parole a (X, Y): X è (N, BLOCK_SIZE), Y è (N,)."""
    X, Y = [], []
    for word in word_list:
        # si parte da un contesto tutto di '.', e la finestra scorre: a ogni
        # carattere si butta via il più vecchio e si accoda quello appena letto
        context = [0] * BLOCK_SIZE
        for ch in word + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            if verbose:
                print(f"    {''.join(itos[i] for i in context)} -> {ch}")
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


print("=== il dataset ===\n")
print(f"  parole: {len(words)}")
print(f"\n  gli esempi che si ricavano da '{words[0]}':")
build_dataset(words[:1], verbose=True)

# Lo split. Il training set serve a trovare i *parametri*; il dev set a
# scegliere gli *iperparametri* (quanti neuroni, quante dimensioni di
# embedding, quanto contesto, quanta regolarizzazione), cioè tutto quello che
# non impara la gradient descent ma decidiamo noi; il test set serve a dare un
# numero finale, e va guardato pochissime volte. Ogni volta che ci guardi e ne
# impari qualcosa, un pezzetto di test set diventa training set.
# 42 e non SEED: è il seed che usa la lezione per mescolare le parole, e
# tenerlo uguale vuol dire ritrovarsi con gli stessi identici split.
random.seed(42)
random.shuffle(words)
n_train = int(0.8 * len(words))
n_dev = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n_train])
Xdev, Ydev = build_dataset(words[n_train:n_dev])
Xte, Yte = build_dataset(words[n_dev:])

print(f"\n  train: {tuple(Xtr.shape)}  dev: {tuple(Xdev.shape)}  test: {tuple(Xte.shape)}")


# ---------------------------------------------------------------------------
# 2. l'embedding: una lookup table, o un layer lineare, fa lo stesso
# ---------------------------------------------------------------------------

print("\n=== l'embedding ===\n")

g = torch.Generator().manual_seed(SEED)
C_demo = torch.randn((VOCAB, 2), generator=g)

# Dato un carattere, per ottenere il suo embedding posso fare C[ix]. Ma anche:
#
#   one_hot = F.one_hot(torch.tensor(ix), num_classes=27)   # [0, ..., 1, ..., 0]
#   embedding = one_hot.float() @ C
#
# perché per come funziona la moltiplicazione di matrici, di fatto considero
# solo la riga ix di C, moltiplicando per 1 i suoi elementi e per 0 tutto il
# resto. Il secondo modo è quello che rende l'embedding *un primo layer della
# rete*: neuroni senza non linearità e senza bias, la cui matrice dei pesi è C.
# È la stessa identità di bigram_nn.py, vista da qui.
ix = 5
one_hot = F.one_hot(torch.tensor(ix), num_classes=VOCAB).float()
assert torch.allclose(one_hot @ C_demo, C_demo[ix])
print(f"  one_hot({ix}) @ C  ==  C[{ix}]  ->  {C_demo[ix].tolist()}")

# In pratica si indicizza, che è molto più veloce. E l'indicizzazione di
# PyTorch è abbastanza flessibile da prendere in un colpo solo tutti e tre i
# caratteri di tutti gli esempi: C[X] con X di shape (N, 3) dà (N, 3, 2).
print(f"  C[X] con X {tuple(Xtr.shape)}  ->  {tuple(C_demo[Xtr].shape)}")


# ---------------------------------------------------------------------------
# 3. la rete: due layer, scritti a mano
# ---------------------------------------------------------------------------


def make_params(emb_dim, hidden):
    """C + i due layer. emb_dim e hidden sono i due iperparametri principali."""
    g = torch.Generator().manual_seed(SEED)
    # il primo layer ha BLOCK_SIZE * emb_dim input (le coordinate dei 3
    # caratteri precedenti, messe in fila) e `hidden` neuroni
    params = [
        torch.randn((VOCAB, emb_dim), generator=g),  # C:  lookup carattere -> embedding
        torch.randn((BLOCK_SIZE * emb_dim, hidden), generator=g),  # W1
        torch.randn(hidden, generator=g),  # b1
        torch.randn((hidden, VOCAB), generator=g),  # W2: un neurone per carattere
        torch.randn(VOCAB, generator=g),  # b2
    ]
    for p in params:
        p.requires_grad = True
    return params


def forward(params, X):
    """Da (N, BLOCK_SIZE) indici a (N, 27) logits."""
    C, W1, b1, W2, b2 = params

    emb = C[X]  # (N, 3, emb_dim): gli embedding dei 3 caratteri di ogni esempio

    # Ora servono in una forma più comoda: (N, 3*emb_dim), cioè i tre vettori
    # concatenati in una riga sola. view() fa esattamente questo e non costa
    # niente: la memoria di un tensore è sempre un vettore piatto, e view
    # cambia solo gli attributi (shape, stride) con cui viene interpretata.
    # Nessun byte viene copiato o spostato. torch.cat darebbe lo stesso
    # risultato ma allocando un tensore nuovo.
    #
    # emb.shape[0] invece di scrivere N, e -1 invece di 3*emb_dim: il -1 lo
    # deduce PyTorch dal fatto che il numero totale di elementi non cambia.
    # Hardcodare 32 e 6 funziona finché non cambi un iperparametro.
    flat = emb.view(emb.shape[0], -1)

    # (N, 3*emb) @ (3*emb, hidden) -> (N, hidden), e b1 è (hidden,): il
    # broadcasting allinea a destra, tratta b1 come (1, hidden) e lo somma a
    # tutte le righe. Che è quello che vogliamo, ma vale la pena controllarlo:
    # se le shape si allineassero male il codice girerebbe lo stesso.
    h = torch.tanh(flat @ W1 + b1)  # (N, hidden), valori in (-1, 1)

    return h @ W2 + b2  # (N, 27), i logits in (-inf, +inf)


print("\n=== la rete, passo per passo ===\n")

demo_params = make_params(emb_dim=2, hidden=100)
batch = Xtr[:BATCH_SIZE]
targets = Ytr[:BATCH_SIZE]

logits = forward(demo_params, batch)
print(f"  logits: {tuple(logits.shape)}  (una riga per esempio, 27 numeri per riga)")

# Da qui in poi è identico al bigram: exp per tornare a dei "counts", poi
# normalizzazione per avere probabilità.
counts = logits.exp()  # (N, 27), valori in (0, +inf)
probs = counts / counts.sum(1, keepdims=True)  # (N, 27), ogni riga somma a 1

# Per vedere quanto bene predice: per ogni esempio, dalla matrice delle
# probabilità prendo solo la probabilità del carattere che *so* essere quello
# giusto. Idealmente dovrebbero essere tutti 1.
p_correct = probs[torch.arange(len(targets)), targets]
# a pesi random sono numeri assurdi: i logits partono grandi, exp li fa esplodere
# e quasi tutta la probabilità finisce su una casella a caso. Da qui si parte.
print(f"  probabilità date alla risposta giusta: {[f'{v:.1e}' for v in p_correct[:5].tolist()]} ...")

loss_manual = -p_correct.log().mean()

# ...che è esattamente la cross entropy, e F.cross_entropy la calcola in un
# colpo solo dai logits grezzi, senza costruire counts e probs e senza andare
# in overflow quando i logits sono grandi.
loss = F.cross_entropy(logits, targets)
assert torch.allclose(loss_manual, loss)
print(f"  loss a mano: {loss_manual.item():.4f}   F.cross_entropy: {loss.item():.4f}")


# ---------------------------------------------------------------------------
# 4. come si sceglie il learning rate
# ---------------------------------------------------------------------------

print("\n=== ricerca del learning rate ===\n")


def train(params, steps, lr_of_step, log_every=None):
    """Il training loop: minibatch, forward, backward, passo. Ritorna le loss."""
    g = torch.Generator().manual_seed(SEED)
    losses = []
    for i in range(steps):
        # il minibatch: BATCH_SIZE indici a caso nel training set. Il gradiente
        # che ne esce non è quello vero, è una stima rumorosa, ma la direzione è
        # abbastanza buona ed è centinaia di volte più economica. Meglio molti
        # passi approssimati che pochi passi esatti.
        # Quel "meglio molti passi approssimati che pochi passi esatti": su 1.6M di esempi
        # processati per tutti, quindi stesso lavoro totale, ma cambiando solo come sono
        # distribuiti fra passi grandi e piccoli:
        #
        #     batch      passi     dev     sec
        #         4     400000  2.2928      93
        #         8     200000  2.2277      50     <- il migliore
        #        32      50000  2.2474      15
        #       512       3125  2.7856       3
        #      2048        781  4.5324       2
        #
        # Sotto una certa taglia il gradiente è così rumoroso che i passi si sprecano.

        ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)

        loss = F.cross_entropy(forward(params, Xtr[ix]), Ytr[ix])

        for p in params:
            p.grad = None
        loss.backward()

        lr = lr_of_step(i)
        with torch.no_grad():
            for p in params:
                p -= lr * p.grad

        losses.append(loss.item())
        if log_every and (i % log_every == 0 or i == steps - 1):
            print(f"    passo {i:6d}   lr {lr:.3f}   loss (minibatch) {loss.item():.4f}")
    return losses



# Il metodo: si parte da un lr palesemente troppo basso (la loss non si muove)
# e si sale fino a dove esplode, poi si cerca il punto buono in mezzo. Invece
# di provare a mano, facciamo un'unica corsa in cui il lr cresce a ogni passo
# da 10^-3 a 10^0, e guardiamo dove la loss smette di scendere. Si spazia sugli
# *esponenti*, non sui valori: fra 0.001 e 1 la scala interessante è quella.
lr_exponents = torch.linspace(-3, 0, 1000)
lr_probe = make_params(emb_dim=2, hidden=100)
lr_losses = train(lr_probe, 1000, lambda i: (10 ** lr_exponents[i]).item())

best_exponent = lr_exponents[torch.tensor(lr_losses).argmin()].item()
print(f"  minimo intorno all'esponente {best_exponent:.2f}, cioè lr ~ {10**best_exponent:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(lr_exponents.tolist(), lr_losses)
plt.xlabel("esponente del learning rate (lr = 10^x)")
plt.ylabel("log10(loss)")
plt.savefig(HERE / "01_out_lr_search.png", bbox_inches="tight", dpi=100)
plt.close()
print(f"  grafico in {HERE / '01_out_lr_search.png'}")

# # La valle è larga e il grafico rumoroso: il minimo puntuale cade dove capita
# # (qui intorno a 0.16), ma quello che si legge davvero è che sotto 10^-2 non
# # succede niente e sopra 10^-0.5 comincia a saltare. Non è una misura di
# # precisione, è un modo per trovare l'ordine di grandezza: da qui usiamo 0.1.


# ---------------------------------------------------------------------------
# 5. primo giro: embedding a 2 dimensioni
# ---------------------------------------------------------------------------

print("\n=== training, embedding 2D, 100 neuroni ===\n")


@torch.no_grad()
def evaluate(params, X, Y):
    """La loss su un dataset intero, senza costruire il grafo per il backward."""
    return F.cross_entropy(forward(params, X), Y).item()


@torch.no_grad()
def sample(params, num=10):
    """Genera nomi partendo da un contesto di soli '.', un carattere alla volta."""
    g = torch.Generator().manual_seed(SEED)
    names = []
    for _ in range(num):
        out = []
        context = [0] * BLOCK_SIZE
        while True:
            # softmax perché qui servono le probabilità vere da campionare; nel
            # training invece restano dentro cross_entropy
            probs = F.softmax(forward(params, torch.tensor([context])), dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]  # la finestra scorre, come nel dataset
            if ix == 0:
                break
            out.append(itos[ix])
        names.append("".join(out))
    return names


params_small = make_params(emb_dim=2, hidden=100)
print(f"  parametri: {sum(p.nelement() for p in params_small)}")

# 50k passi, con il learning rate decay in mezzo: si allena un po' a 0.1, poi
# si scende a 0.01 per gli ultimi ritocchi. Anche questo è a occhio.
train(params_small, 50_000, lambda i: 0.1 if i < 25_000 else 0.01, log_every=10_000)

print(f"\n  loss train: {evaluate(params_small, Xtr, Ytr):.4f}")
print(f"  loss dev:   {evaluate(params_small, Xdev, Ydev):.4f}")
print("  (il bigram si fermava a 2.45)")

print("\n  nomi campionati:")
for name in sample(params_small):
    print("   ", name)


# ---------------------------------------------------------------------------
# 6. cosa ha imparato: gli embedding, disegnati
# ---------------------------------------------------------------------------

C = params_small[0].detach()

plt.figure(figsize=(8, 8))
plt.scatter(C[:, 0], C[:, 1], s=200, c="white", edgecolors="black")
for i in range(VOCAB):
    plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center", color="black")
plt.grid(True, linestyle=":")
plt.savefig(HERE / "01_out_embeddings.png", bbox_inches="tight", dpi=100)
plt.close()

print(f"\n  embedding salvati in {HERE / '01_out_embeddings.png'}")
print(
    """
  Nessuno ha detto alla rete che a, e, i, o sono vocali, eppure finiscono
  vicine e nettamente staccate dal grumo di consonanti: la rete le tratta come
  intercambiabili, perché nei nomi si comportano in modo simile. Il '.', che
  non è una lettera, se ne sta per conto suo in un angolo. Qualche carattere
  viene spinto lontano da tutto (qui u, l, g, p): con due sole dimensioni lo
  spazio è stretto e la disposizione precisa cambia col seed, quindi la cosa da
  guardare è che ci *sia* struttura, non dove finisca ogni singola lettera. È
  l'effetto descritto nel paper di Bengio, su 27 caratteri invece che su 17k
  parole.
"""
)


# ---------------------------------------------------------------------------
# 7. underfitting, overfitting, e quando ci si ferma
# ---------------------------------------------------------------------------
#
# Da qui in avanti i numeri da guardare sono due, e nessuno dei due dice niente
# da solo: la loss sul training set, cioè sui dati su cui la rete ha fatto i
# passi di gradient descent, e la loss sul dev set, che sono dati che non ha mai
# visto. Il senso di ciascuno sta nel confronto con l'altro.
#
# UNDERFITTING — le due loss restano appaiate.
#
# Il modello non ha abbastanza capacità nemmeno per memorizzare quello che vede,
# figurarsi per trovarci una struttura.
# È il caso della rete qui sopra, 2.3380 contro 2.3393, che distano lo 0.001.
# La cura è più capacità: più parametri, o gli stessi parametri messi dove
# servono davvero.
#
# Due loss appaiate però, da sole, non bastano a diagnosticarlo: sarebbero
# appaiate anche se il modello fosse arrivato al suo ottimo. Per distinguere i
# due casi non c'è una misura, c'è un esperimento:
#
#     ingrandisci il modello e guarda se la dev loss scende
#
# Se scende, eri in underfitting; se non si muove, eri già all'ottimo di quella
# architettura. È una diagnosi che si fa all'indietro. Qui la prova è proprio il
# passo che stiamo per fare: da 2D/100 neuroni a 10D/200 la dev va da 2.3393 a
# 2.1731, quindi sì, eravamo in underfitting.
#
# OVERFITTING — la loss di training precipita, quella di dev risale.
#
# Il modello ha abbastanza capacità da imparare i nomi specifici invece di come
# sono fatti i nomi. Continua a migliorare sui dati che ha visto e intanto
# peggiora su tutto il resto. Il caso limite è la training loss che va a zero:
# il dataset imparato a memoria, e allora campionando escono i nomi che c'erano
# già. Un modello in overfitting non è solo inutile, è peggio che tirare a caso,
# perché è *sicuro* di risposte sbagliate. La cura è meno capacità, o più dati,
# o regolarizzazione.
#
# Due cose che confondono:
#   - il gap fra le due loss non è il criterio. +0.05 non è "meglio" di +0.30 in
#     assoluto: se allargando la rete la dev scende e il gap si allarga, hai
#     fatto un passo avanti. Quello che conta è la dev; il gap è solo il sintomo
#     che dice in che direzione muoversi quando la dev è ferma — gap piccolo,
#     serve più capacità; gap grande, ne serve meno.
#   - una training loss che scende verso lo zero non è un traguardo, è un
#     allarme.
#
# Qualche riferimento, buono per sapere se si sta avanzando, non per sapere se
# si è all'ottimo:
#
#     3.2958   tirare a caso, -log(1/27)
#     2.4540   il bigram della parte 1, un carattere di contesto
#     2.17     questa MLP, tre caratteri
#
# Il nostro caso: siamo in underfitting? Ipotesi: 27 caratteri schiacciati in un piano
# non hanno spazio per rappresentare distinzioni utili. Quindi 10 dimensioni di
# embedding, e 200 neuroni.

print("=== training, embedding 10D, 200 neuroni ===\n")

params_big = make_params(emb_dim=10, hidden=200)
print(f"  parametri: {sum(p.nelement() for p in params_big)}")

loss_curve = train(params_big, 200_000, lambda i: 0.1 if i < 100_000 else 0.01, log_every=20_000)

# Assi logaritmici su entrambi i lati. Sulle y perché altrimenti il crollo
# iniziale (da 28 a 3 in un paio di centinaia di passi) schiaccia tutto il resto
# contro il fondo: è la classica forma a mazza da hockey, e il log la raddrizza.
# Sulle x perché in scala lineare i primi mille passi, dove succede quasi tutto,
# occupano mezzo pixel su 200000. Il passo 0 in scala log non è disegnabile,
# quindi si parte dall'1.
#
# Che poi quella mazza da hockey sia colpa nostra, e non una legge di natura, si
# scopre nella sezione 9.
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(loss_curve) + 1), loss_curve, linewidth=0.5)
plt.xscale("log")
plt.yscale("log")
# il momento in cui il learning rate passa da 0.1 a 0.01: senza la riga non si
# distinguerebbe, perché la banda si stringe troppo gradualmente
plt.axvline(100_000, color="black", linestyle="--", linewidth=0.8)
plt.text(95_000, 25, "lr 0.1 -> 0.01 ", fontsize=8, va="top", ha="right")
plt.xlabel("passo")
plt.ylabel("loss sul minibatch")
# in scala log matplotlib etichetta anche le tacche intermedie ("6 x 10^0"):
# le spegniamo e teniamo solo dei numeri leggibili
plt.yticks([2, 3, 5, 10, 20, 30], ["2", "3", "5", "10", "20", "30"])
plt.gca().yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
plt.grid(True, which="both", linestyle=":", linewidth=0.5)
plt.savefig(HERE / "01_out_training_loss.png", bbox_inches="tight", dpi=100)
plt.close()
print(f"\n  curva di training in {HERE / '01_out_training_loss.png'}")
print("  (è spessa perché ogni punto è la loss su 32 esempi diversi, non su tutto)")

loss_tr = evaluate(params_big, Xtr, Ytr)
loss_dev = evaluate(params_big, Xdev, Ydev)
print(f"\n  loss train: {loss_tr:.4f}")
print(f"  loss dev:   {loss_dev:.4f}")

# Il test set si guarda una volta sola, alla fine, quando tutti gli
# iperparametri sono già stati scelti sul dev.
print(f"  loss test:  {evaluate(params_big, Xte, Yte):.4f}   <- guardata una volta sola")

print(
    f"""
  Le due loss adesso si separano di {loss_dev - loss_tr:.2f}, contro lo 0.00 della rete piccola:
  il modello ha abbastanza capacità da memorizzare qualcosa del training set.
  Non è un problema — la dev è scesa da 2.34 a {loss_dev:.2f}, e finché scende quel
  gap è un prezzo che conviene pagare. Il segnale per fermarsi sarebbe la dev
  che risale mentre la train continua a scendere: lì il modello non starebbe
  più imparando i nomi, starebbe imparando *questi* nomi.
"""
)


# ---------------------------------------------------------------------------
# 8. campionamento
# ---------------------------------------------------------------------------

# Stessa funzione della sezione 5, stesso seed: le due liste sono confrontabili
# a occhio, e la differenza fra 2.34 e 2.17 di loss si vede.
print("=== nomi campionati ===\n")

for name in sample(params_big, 20):
    print("  ", name)


# ---------------------------------------------------------------------------
# 9. quello che questo file sbaglia: l'inizializzazione
# ---------------------------------------------------------------------------

print("\n=== l'inizializzazione ===\n")

params_fresh = make_params(emb_dim=10, hidden=200)
with torch.no_grad():
    logits0 = forward(params_fresh, Xtr)

print(f"  loss della rete appena inizializzata: {evaluate(params_fresh, Xtr, Ytr):.2f}")
print(f"  loss di chi non sa niente, -log(1/27): {-torch.tensor(1 / VOCAB).log().item():.2f}")
print(f"  logits al passo 0: da {logits0.min():.0f} a {logits0.max():.0f}, invece che intorno a 0")

print(
    """
  Inizializzando i parametri con randn partiamo malissimo, e la catena è questa:
  pesi a caso -> al primo giro i logits che escono dai vari passaggi hanno valori
  molto grandi in valore assoluto, qui da -51 a +55 -> e un logit grande, passato
  per la softmax, diventa una probabilità vicina a 1, con tutte le altre
  schiacciate a 0.

  Cioè i parametri di partenza non ci mettono in una situazione neutra — "non so
  niente, tutto è possibile con la stessa probabilità", 1/27 su ogni lettera,
  loss 3.30 — ma in una situazione di estrema sicurezza, e del tutto infondata:
  il modello è convintissimo di lettere prese a caso. Da lì il 26.

  Il conto lo paghiamo in passi: i primi migliaia non servono a imparare i nomi,
  servono a spegnere le assurdità che ci abbiamo messo dentro noi.

  E si vede nel grafico. Il crollo verticale nelle prime centinaia di passi, la
  mazza da hockey che in 01_out_training_loss.png abbiamo raddrizzato con l'asse
  logaritmico, non è apprendimento: è la rete che disfa la propria
  inizializzazione. Partendo da 3.32 invece che da 26 quel tratto verticale non
  esiste proprio, e la mazza da hockey sparisce. Al passo 3000 la rete con
  l'init sistemato è a 2.11, questa qui è ancora a 2.96.
"""
)

# Sistemarlo costa due righe, zero parametri e zero secondi:
#
#     W2 *= 0.01                                  # logits iniziali quasi a zero
#     b2 *= 0.0
#     W1 *= (5/3) / sqrt(BLOCK_SIZE * EMB_DIM)    # kaiming, il 5/3 è per la tanh
#     b1 *= 0.01
#
# e la dev, a parità di tutto il resto, migliora di più di quanto abbia reso
# qualsiasi altra manopola provata finora:
#
#     seed          init di qui   init sistemato
#     2147483647         2.1698           2.1070
#     1337               2.1561           2.1079
#     20250826           2.1644           2.1033
#
# C'è un secondo effetto, meno ovvio e più insidioso: finché l'init è rotto, gli
# esperimenti sugli iperparametri mentono. Allungando il contesto a 5 caratteri
# la dev *peggiora*, 2.2049 contro 2.1698 — che sarebbe una conclusione assurda,
# perché più contesto non può contenere meno informazione. E infatti è falsa:
# con l'init sistemato lo stesso blocco 5 fa 2.1228. 
#
# Qui non lo sistemiamo, perché è il contenuto della lezione 3 della serie,
# "Activations & Gradients, BatchNorm", che parte esattamente da questa
# osservazione: https://www.youtube.com/watch?v=P6sfmUTpUmc


# ---------------------------------------------------------------------------
# cosa ci portiamo dietro
# ---------------------------------------------------------------------------
#
# 1. La tabella dei conteggi è sparita e non tornerà più. Le probabilità non
#    stanno da nessuna parte: vengono calcolate. Il costo di allungare il
#    contesto ora è lineare (BLOCK_SIZE * emb_dim input in più al primo layer)
#    invece che esponenziale.
#
# 2. Gli embedding sono parametri come gli altri, imparati dalla stessa
#    gradient descent, e la struttura che ci finisce dentro (le vocali vicine)
#    è un effetto collaterale del predire bene, non un obiettivo.
#
# 3. Il minibatch introduce rumore nel gradiente e lo si accetta volentieri:
#    la curva di training è spessa, ma si fanno mille volte più passi — fino a
#    un limite, misurato nel commento sotto train().
#
# 4. Ci sono due categorie di numeri da scegliere. I parametri li trova la
#    gradient descent; gli iperparametri (emb_dim, hidden, BLOCK_SIZE,
#    BATCH_SIZE, il learning rate e il suo decay, quanti passi) li scegliamo
#    noi, guardando il dev set. Il fatto che qui siano scelti a occhio è
#    esattamente quello che in un lavoro serio diventa una griglia di
#    esperimenti.
#
# 5. Le manopole rimaste, se si vuole scendere sotto il 2.17: più neuroni, più
#    dimensioni di embedding, più contesto (BLOCK_SIZE), batch più grandi,
#    schedule del learning rate meno improvvisati, più passi. Ma la prima da
#    girare non è nessuna di queste: è l'inizializzazione (sezione 9), che rende
#    da sola più di tutte le altre e senza la quale le altre si misurano male.
