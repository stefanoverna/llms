"""
makemore, parte 1 (coda): lo stesso identico bigram, servito in una forma diversa.

Questo file non insegna un modello nuovo. Il modello è quello di `02_bigram_nn.py`,
riga per riga: una `nn.Embedding(27, 27)`, `F.cross_entropy`, SGD. Quello che
cambia è la *forma dei tensori* che gli passiamo, e cambia in modo dichiaratamente
inutile: alla fine la rete ritrova la stessa tabella di probabilità e sputa fuori
gli stessi nomi sgangherati di prima.

Perché farlo allora? Perché da qui in avanti tutti gli esempi che incontrerai
hanno gli input in questa forma, e conviene vederla nascere su un modello che
già conosci a memoria piuttosto che scoprirla dentro un modello nuovo.

In `02_bigram_nn.py` il dataset era una lista piatta di coppie:

    xs = [ ., m, a, r, i, ., l, u, c, a, ... ]     shape (228146,)
    ys = [ m, a, r, i, o, l, u, c, a, .,  ... ]    shape (228146,)

Qui gli stessi identici numeri vengono impacchettati in una griglia:

    xb = [[ ., m, a, r, i ],      yb = [[ m, a, r, i, o ],
          [ ., l, u, c, a ],            [ l, u, c, a, . ],
          [ ., s, t, e, f ]]            [ s, t, e, f, a ]]

    shape (3, 5)                        shape (3, 5)

Le tre lettere che vedrai dappertutto, e da dove vengono i nomi:

  B  "batch"     quante sequenze indipendenti processi insieme. Qui 3. Le righe
                 non comunicano mai fra loro, in nessuna operazione: sono
                 semplicemente esempi diversi messi nella stessa chiamata per
                 tenere occupata la macchina. È il minibatch della gradient
                 descent, che in `02_bigram_nn.py` non avevamo perché il dataset
                 ci stava tutto in memoria.

  T  "time"      la posizione dentro la sequenza. Qui 5. I caratteri di una
                 riga sono *consecutivi nel testo*. Il nome viene dalle RNN,
                 che il testo lo leggevano davvero un carattere alla volta in
                 un loop: ogni carattere era un istante, e l'algoritmo si
                 chiamava "backpropagation through time". Oggi i T caratteri
                 li processiamo tutti insieme, ma il nome dell'asse è rimasto.

  C  "channels"  quanti numeri descrivono una posizione. Qui 27: dietro ogni
                 casella della griglia non c'è un numero, c'è il vettore dei
                 27 logits. Il nome viene dalle reti convoluzionali, dove ogni
                 pixel porta con sé i canali R, G, B. Stessa idea: l'asse delle
                 feature, quello che dice *cosa so* di questo punto.
                 Attenzione alla coincidenza: qui C == 27 == VOCAB solo perché
                 la tabella è 27x27 e i suoi output sono già i logits. In
                 generale C è "quanto è largo il vettore", non "quante lettere".

Il fatto centrale, che questo file verifica con degli assert invece di
raccontarlo: per il bigram **B e T sono la stessa cosa**. I logits in posizione
(b, t) dipendono solo da `xb[b, t]`, non da chi ha intorno. Quindi una griglia
3x5 sono 15 esempi indipendenti, esattamente come i 15 della lista piatta, e
`emb(xb)` produce numero per numero lo stesso risultato di `emb(x_flat)`.

L'asse T qui non porta nessuna informazione che il modello usi. Porta però
informazione che nei dati adesso c'è: **quali caselle sono vicine nel testo**.
La lista piatta la buttava via, la griglia la conserva. Il bigram continuerà a
ignorarla; i modelli successivi non faranno altro che leggerla.

Cosa cambia in pratica, e sono tre cose sole:

  1. il dataset non è più una lista di coppie, è un corpus continuo da cui si
     ritagliano finestre. Meno codice, e smette di dipendere dal fatto che i
     nostri esempi siano parole corte.
  2. `nn.Embedding` non se ne accorge: stessa operazione, stessi numeri.
  3. `F.cross_entropy` invece sì, e vuole un reshape. È l'unico vero attrito
     della nuova forma, ed è meglio incontrarlo qui che altrove.

In fondo al file: cosa abbiamo guadagnato e cosa no.
"""

from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

SEED = 2147483647
HERE = Path(__file__).parent

B = 32  # batch: sequenze indipendenti per passo
T = 8  # time: caratteri consecutivi per sequenza
STEPS = 5000


# ---------------------------------------------------------------------------
# 1. il dataset: da lista di coppie a corpus continuo
# ---------------------------------------------------------------------------

words = open(HERE / "names.txt", "r").read().splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
VOCAB = len(itos)  # 27, cioè C

decode = lambda t: "".join(itos[i.item()] for i in t.reshape(-1))

# Il vecchio dataset: una coppia (input, target) per bigramma, appiattita.
xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs, ys = torch.tensor(xs), torch.tensor(ys)

# Il nuovo: un unico testo lungo, i nomi separati dal punto. Il punto che
# chiude un nome è lo stesso che apre il successivo, quindi i bigrammi che si
# leggono scorrendo il corpus sono esattamente gli stessi di prima -- il
# dataset non è cambiato, è cambiato solo come lo teniamo.
text = "." + ".".join(words) + "."
data = torch.tensor([stoi[c] for c in text])

print("=== il dataset ===\n")
print(f"  corpus: {len(text)} caratteri, {text[:40]!r} ...")
print(f"  esempi ricavabili: {len(data) - 1}   (coppie di 02_bigram_nn.py: {len(xs)})")

assert Counter(zip(xs.tolist(), ys.tolist())) == Counter(
    zip(data[:-1].tolist(), data[1:].tolist())
)
print("  stesso identico multiset di bigrammi: il modello vedrà gli stessi dati")


# ---------------------------------------------------------------------------
# 2. get_batch: ritagliare la griglia (B, T)
# ---------------------------------------------------------------------------

print("\n=== la griglia (B, T) ===\n")


def get_batch(batch_size, block_size, generator=None):
    # Una riga nasce da UNA finestra di block_size+1 caratteri, tagliata due
    # volte: i primi block_size sono l'input, gli ultimi block_size il target.
    # Ecco perché yb è xb shiftato di uno, e perché l'ultima colonna di yb
    # contiene un carattere che in xb non compare.
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    xb = torch.stack([data[i : i + block_size] for i in ix])
    yb = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return xb, yb


g = torch.Generator().manual_seed(1337)
xb, yb = get_batch(4, T, g)

print(f"  xb {tuple(xb.shape)}          yb {tuple(yb.shape)}")
for rx, ry in zip(xb, yb):
    print(f"    {decode(rx)!r}  ->  {decode(ry)!r}")

print("\n  la prima riga, esempio per esempio (T esempi da una sola finestra):")
for t in range(T):
    print(f"    dopo {itos[xb[0, t].item()]!r} deve venire {itos[yb[0, t].item()]!r}")

print(
    "\n  gli offset sono casuali, quindi una riga attraversa più nomi. I bigrammi di\n"
    "  confine -- ultima lettera seguita da '.', '.' seguito dall'iniziale -- sono\n"
    "  quelli che insegnano al modello dove le parole finiscono e ricominciano."
)


# ---------------------------------------------------------------------------
# 3. il forward: nn.Embedding non si accorge di niente
# ---------------------------------------------------------------------------

print("\n=== il forward ===\n")

torch.manual_seed(SEED)
model = nn.Embedding(VOCAB, VOCAB)

# nn.Embedding.forward, tutto qui:
#
#     flat = idx.reshape(-1)              -1 appiattisci tutte le dimensioni (4, 8) → (32,), per riga. quindi con l'asse piu' interno, il secondo, che scorre piu' velocemente
#     rows = W[flat]                    una copia di riga per ogni intero -> (32, 27)
#     return rows.reshape(*idx.shape, C)   (32, 27) → riporto in "griglia" (4, 8, 27)
#
# Il passo di mezzo è l'unico che fa lavoro (e non è nemmeno un calcolo: sono
# copie, zero moltiplicazioni). Gli altri due sono contabilità sulle shape.
# Quindi il forward su (B, T) *è* il forward su (B*T,) più un reshape.
with torch.no_grad():
    logits_grid = model(xb)
    logits_flat = model(xb.reshape(-1))

print(f"  model(xb) con xb {tuple(xb.shape)}  ->  {tuple(logits_grid.shape)}   (B, T, C)")
print(f"  model(xb.reshape(-1))       ->  {tuple(logits_flat.shape)}      (B*T, C)")

assert torch.equal(logits_grid, logits_flat.view(*xb.shape, VOCAB))
print("  gli stessi numeri, negli stessi posti: cambia solo dove PyTorch mette le parentesi")

print(
    f"\n  logits[0, 0] ha {logits_grid[0, 0].numel()} numeri: i punteggi che il modello dà\n"
    f"  a ciascuna lettera come successore di {itos[xb[0, 0].item()]!r}. Immagina la griglia\n"
    f"  {xb.shape[0]}x{xb.shape[1]} con una colonnina di 27 valori dietro ogni casella: è il cubo (B, T, C)."
)


# ---------------------------------------------------------------------------
# 4. la loss: qui invece qualcosa cambia
# ---------------------------------------------------------------------------

print("\n=== la loss ===\n")

# Prima le parole, che nella documentazione di PyTorch sono date per scontate.
#
#   classe   una delle risposte possibili. Qui le classi sono le 27 lettere:
#            il modello, per ogni esempio, deve sceglierne una.
#   C        QUANTE classi ci sono. Qui 27. È lo stesso C di "channels" della
#            sezione 3 solo per la coincidenza del bigram (la tabella è 27x27,
#            e i suoi 27 output sono già i punteggi delle 27 classi).
#   logits   i punteggi grezzi, uno per classe: C numeri per ogni esempio.
#   target   QUALE classe è quella giusta. Un intero solo, non un vettore:
#            un numero da 0 a 26. Questa è la "label".
#
# Quindi, per il calcolo del loss, per ogni esempio: C numeri in ingresso (i logits),
# 1 intero come risposta corretta (l'indice del target). Ma queste forme non piacciono
# a cross_entropy.
#
# F.cross_entropy accetta due forme, e due soltanto.
#
# FORMA 1 -- la lista piatta:  logits (N, C)  +  target (N,)
#
#   N esempi, una riga di C punteggi ciascuno. Con N=3 esempi e C=4 classi:
#
#       logits (3, 4)                       target (3,)
#         [ 2.1,  0.3, -1.0,  0.5 ]            1     <- la giusta era la classe 1
#         [ 0.0,  1.2,  3.4,  0.1 ]            2     <- la classe 2
#         [ 1.1,  1.0,  0.9,  5.0 ]            3     <- la classe 3
#           ^0    ^1    ^2    ^3
#           le 4 classi, in colonna
#
#   cross_entropy guarda riga per riga: quanta probabilità ha dato il modello
#   alla colonna indicata dal target?
#
# FORMA 2 -- la stessa cosa ripetuta su una griglia:
#            logits (N, C, d1...dk)  +  target (N, d1...dk)
#
#   Serve quando ogni esempio non è una classificazione sola ma tante, disposte
#   in una griglia. Il caso per cui è nata: segmentare un'immagine, cioè dare
#   una classe a ogni pixel. Con 1 immagine 2x2 e 4 classi:
#
#       logits (1, 4, 2, 2)                 target (1, 2, 2)
#         classe 0:  [ 2.1  0.3 ]              [ 1  2 ]
#                    [ 0.9  1.4 ]              [ 3  0 ]
#         classe 1:  [ 5.0  1.1 ]
#                    [ 0.2  0.7 ]            un intero per pixel: quale
#         classe 2:  [ 0.1  4.2 ]            classe è quella giusta lì
#                    [ 1.0  0.3 ]
#         classe 3:  [ 0.4  0.8 ]
#                    [ 6.1  0.1 ]
#
#   Come si legge: i punteggi di UN pixel non stanno affiancati, stanno
#   impilati -- bisogna guardare la stessa casella in tutte e 4 le matrici.
#   Per il pixel in alto a sinistra:
#
#         classe 0 -> 2.1        il punteggio più alto è 5.0, quindi il
#         classe 1 -> 5.0  <--   modello dice "classe 1"
#         classe 2 -> 0.1        e target[0,0] vale 1: ci ha preso.
#         classe 3 -> 0.4
#
#   Stessa lettura per gli altri tre pixel, e qui il modello li azzecca tutti
#   (in alto a destra vince 4.2, classe 2; in basso a sinistra 6.1, classe 3;
#   in basso a destra 1.4, classe 0 -- e il target dice 2, 3, 0).
#
#   Nota dov'è finito C: NON in fondo, ma in *seconda posizione*, subito dopo
#   il batch. È il layout (N, C, H, W) che PyTorch usa per le immagini, e
#   cross_entropy lo ha ereditato. Le dimensioni dopo C (qui H e W) sono
#   semplicemente "quante volte ripeti la classificazione dentro un esempio".
#
# E QUI CASCA IL NOSTRO CASO. Noi abbiamo:
#
#       logits (B, T, C) = (4, 8, 27)       target (B, T) = (4, 8)
#
#   con C in ULTIMA posizione, non in seconda. cross_entropy prova a leggerlo
#   come FORMA 2 e conta le posizioni:
#
#       (  4  ,  8  , 27 )
#          ^N    ^C    ^d1      "4 esempi, 8 classi, ripetuto 27 volte"
#
#   cioè scambia T per il numero di classi. Con quella lettura si aspetta un
#   target (N, d1) = (4, 27), noi gliene diamo uno (4, 8), e si lamenta.
#   Il messaggio d'errore qui sotto è esattamente questo.
try:
    F.cross_entropy(logits_grid, yb)
except RuntimeError as e:
    print(f"  F.cross_entropy(logits (B,T,C), targets (B,T))  ->  RuntimeError: {e}")

# Due rimedi, uno per forma accettata.
#
#   A) piegarsi alla FORMA 1: srotolare la griglia in una lista di B*T esempi.
#
#        (4, 8, 27) --view(-1, 27)--> (32, 27)      logits
#        (4, 8)     --view(-1)------> (32,)         target
#
#      È lecito perché per il bigram le 32 caselle sono 32 esempi indipendenti
#      (lo dimostra l'assert più sotto), ed è la forma che si usa sempre.
#
#   B) piegarsi alla FORMA 2: spostare C in seconda posizione, lasciando T
#      nel ruolo di d1 ("8 classificazioni per esempio").
#
#        (4, 8, 27) --transpose(1,2)--> (4, 27, 8)  logits
#        (4, 8)      invariato                      target
#
#      Il prima/dopo, in piccolo: B=2 sequenze, T=3 posizioni, C=4 classi.
#
#      "Sequenza" = una riga del batch, cioè una delle B finestre di testo
#      contiguo che get_batch ha ritagliato. Nel file vero è una riga come
#      '.gracie.'; qui, per far stare il disegno nella pagina, l'alfabeto ha
#      4 lettere -- '.', 'a', 'b', 'c', cioè le classi 0, 1, 2, 3 -- e ogni
#      sequenza è lunga 3:
#
#        sequenza 0:  xb = '.ab'   ->  yb = 'abc'
#        sequenza 1:  xb = 'ca.'   ->  yb = 'a.b'
#
#      PRIMA -- logits (2, 3, 4): una riga per posizione, le 4 classi in
#      colonna. È la disposizione naturale, la stessa della FORMA 1.
#
#        sequenza 0  '.ab'                   sequenza 1  'ca.'
#          t=0 '.' [ 2.1,  0.3, -1.0,  0.5 ]   t=0 'c' [ 0.7,  2.9,  0.4,  1.3 ]
#          t=1 'a' [ 0.0,  1.2,  3.4,  0.1 ]   t=1 'a' [ 0.0,  1.2,  3.4,  0.1 ]
#          t=2 'b' [ 1.1,  1.0,  0.9,  5.0 ]   t=2 '.' [ 2.1,  0.3, -1.0,  0.5 ]
#                    ^'.'  ^'a'  ^'b'  ^'c'
#                    le 4 classi, in colonna
#
#      Come si legge una riga -- prendiamo t=1 della sequenza 0, cioè
#      [ 0.0, 1.2, 3.4, 0.1 ]:
#
#        il carattere in ingresso è 'a' (la posizione 1 di '.ab'), e quei
#        quattro numeri sono i punteggi che il modello dà alle quattro
#        continuazioni possibili:
#
#              '.' -> 0.0        il più alto è 3.4, quindi il modello
#              'a' -> 1.2        scommette su 'b'. Il target in quella
#              'b' -> 3.4  <--   posizione vale 2, cioè 'b': ci ha preso.
#              'c' -> 0.1
#
#      DOPO -- logits.transpose(1, 2), shape (2, 4, 3): ogni sequenza è stata
#      ribaltata. Adesso una riga è una CLASSE e una colonna è una posizione.
#
#        sequenza 0                          sequenza 1
#          classe '.'  [ 2.1,  0.0,  1.1 ]     classe '.'  [ 0.7,  0.0,  2.1 ]
#          classe 'a'  [ 0.3,  1.2,  1.0 ]     classe 'a'  [ 2.9,  1.2,  0.3 ]
#          classe 'b'  [-1.0,  3.4,  0.9 ]     classe 'b'  [ 0.4,  3.4, -1.0 ]
#          classe 'c'  [ 0.5,  0.1,  5.0 ]     classe 'c'  [ 1.3,  0.1,  0.5 ]
#                        ^t=0  ^t=1  ^t=2
#
#      È esattamente la disposizione dell'immagine segmentata qui sopra, con
#      le classi impilate: T ha preso il posto che lì avevano H e W.
#
#      Il target non si tocca: resta (2, 3), un intero per posizione.
#
#        [ 1,  2,  3 ]     sequenza 0: 'abc', cioè le classi 1, 2, 3
#        [ 1,  0,  2 ]     sequenza 1: 'a.b'
#
#      Da qui cross_entropy legge (N=2, C=4, d1=3) e i conti tornano.
#
loss_view = F.cross_entropy(logits_grid.view(-1, VOCAB), yb.view(-1))
loss_transpose = F.cross_entropy(logits_grid.transpose(1, 2), yb)

def show(label, logits, target, value, note=""):
    pair = f"logits {tuple(logits.shape)} + target {tuple(target.shape)}"
    print(f"  {label} {pair:<38} ->  {value:.6f}{note}")


print()
show("A)", logits_grid.view(-1, VOCAB), yb.view(-1), loss_view.item(), "   <- la forma che si usa")
show("B)", logits_grid.transpose(1, 2), yb, loss_transpose.item())
assert torch.allclose(loss_view, loss_transpose)

# E la prova che il flatten è lecito: la loss è identica a quella calcolata
# sugli stessi esempi presi come lista piatta. Per il bigram, B e T sono
# entrambi "batch".
with torch.no_grad():
    loss_flat = F.cross_entropy(model(xb.reshape(-1)), yb.reshape(-1))
assert torch.allclose(loss_view, loss_flat)
print(
    f"  {'C) srotolando PRIMA del forward, non dopo':<41} ->  {loss_flat.item():.6f}"
    "   <- niente griglia, mai"
)

print(
    f"\n  ⚠ il pericolo: qui T={T} e C={VOCAB} sono diversi, quindi l'errore salta fuori.\n"
    "    Se fossero uguali, cross_entropy leggerebbe T come numero di classi senza\n"
    "    lamentarsi e calcolerebbe in silenzio una loss senza senso. Il reshape non\n"
    "    è una formalità per far contento PyTorch: è quello che dice chi sono le\n"
    "    classi e chi sono gli esempi."
)


# ---------------------------------------------------------------------------
# 5. il training: adesso a minibatch, ed è l'unico vero guadagno
# ---------------------------------------------------------------------------

print("\n=== gradient descent a minibatch ===\n")

# In 02_bigram_nn.py ogni passo faceva il forward su tutti i 228146 esempi. Qui
# ogni passo ne vede B*T = 256, ritagliati a caso. Il codice della loss non
# cambia di una virgola: un minibatch è semplicemente una griglia più piccola.
optimizer = torch.optim.SGD(
    model.parameters(), lr=10.0, weight_decay=0.01 * 2 / VOCAB**2
)

g = torch.Generator().manual_seed(1337)
for k in range(STEPS):
    # lr più basso di quello di 02_bigram_nn.py (50) e dimezzato a metà corsa: il
    # minibatch introduce rumore nel gradiente, e con passi troppo lunghi i
    # pesi rimbalzano intorno all'ottimo senza mai arrivarci. La misura è nella
    # tabella qui sotto.
    if k == STEPS // 2:
        for group in optimizer.param_groups:
            group["lr"] = 1.0

    xb, yb = get_batch(B, T, g)
    loss = F.cross_entropy(model(xb).view(-1, VOCAB), yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if k % 1000 == 0 or k == STEPS - 1:
        print(f"    passo {k:5d}   loss del batch {loss.item():.4f}")


@torch.no_grad()
def full_loss(m):
    """La loss vera, su tutti i 228146 esempi. Quella del batch è troppo rumorosa."""
    return F.cross_entropy(m(data[:-1]), data[1:]).item()


print(f"\n  loss sul corpus intero: {full_loss(model):.4f}")
print(
    f"  esempi visti: {STEPS} passi x {B * T} = {STEPS * B * T:,} visite\n"
    f"  (02_bigram_nn.py: 1000 passi x {len(xs):,} = {1000 * len(xs):,}, cioè {1000 * len(xs) // (STEPS * B * T)}x tanto)"
)


# ---------------------------------------------------------------------------
# 6. il prezzo del minibatch: il rumore
# ---------------------------------------------------------------------------

print("\n=== quanto costa guardare 256 esempi invece di 228146 ===\n")

# Il pavimento teorico: la tabella dei conteggi è la soluzione ottima esatta
# per un bigram, e la sua loss è il minimo che qualunque training possa fare.
N = torch.zeros((VOCAB, VOCAB), dtype=torch.int32)
for a, b in zip(data[:-1], data[1:]):
    N[a, b] += 1
P = (N + 1).float()
P /= P.sum(1, keepdim=True)
floor = -P[data[:-1], data[1:]].log().mean().item()


def train(lr, decay):
    torch.manual_seed(SEED)
    m = nn.Embedding(VOCAB, VOCAB)
    opt = torch.optim.SGD(m.parameters(), lr=lr, weight_decay=0.01 * 2 / VOCAB**2)
    gen = torch.Generator().manual_seed(1337)
    for k in range(STEPS):
        if decay and k == STEPS // 2:
            for group in opt.param_groups:
                group["lr"] = lr / 10
        bx, by = get_batch(B, T, gen)
        loss = F.cross_entropy(m(bx).view(-1, VOCAB), by.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return full_loss(m)


print(f"  {'lr = 50 (quello di 02_bigram_nn.py)':<40} {train(50.0, False):.4f}")
print(f"  {'lr = 10':<40} {train(10.0, False):.4f}")
print(f"  {'lr = 10, /10 a metà corsa':<40} {full_loss(model):.4f}")
print(f"  {'tabella dei conteggi (ottimo esatto)':<40} {floor:.4f}")

print(
    "\n  Con il dataset intero il gradiente è esatto e lr=50 va benissimo. Con 256\n"
    "  esempi il gradiente è una stima rumorosa: lr=50 fa rimbalzare i pesi\n"
    "  intorno all'ottimo senza mai posarsi. Abbassare il passo verso la fine è il\n"
    "  rimedio standard, ed è la prima volta che ci serve. Non c'entra l'asse T:\n"
    "  è il conto del minibatch, che la forma (B, T) rende semplicemente comodo."
)


# ---------------------------------------------------------------------------
# 7. il campionamento: qui l'asse T sparisce del tutto
# ---------------------------------------------------------------------------

print("\n=== nomi campionati ===\n")


@torch.no_grad()
def generate(idx, max_new_tokens, generator=None):
    """idx è (B, T). Restituisce (B, T + max_new_tokens)."""
    # Due tensori con due ruoli distinti, che è facile confondere:
    #
    #   chunks   l'output che stiamo costruendo, una colonna (B, 1) alla volta.
    #            Alla fine viene concatenato: è il testo che vogliamo leggere.
    #   cur      l'input del modello: (B,), un intero per sequenza. Al bigram
    #            basta l'ultimo carattere per predire il prossimo, quindi non
    #            ha senso ripassargli tutto il prefisso -- sarebbe la stessa
    #            risposta, ricalcolata su una sequenza sempre più lunga.
    #
    # Dal lato del modello, quindi, qui l'asse T non c'è proprio: generare con
    # un bigram è B catene indipendenti di lookup, una lettera alla volta. La
    # griglia resta solo nell'output, dove serve a impilare i pezzi.
    chunks = [idx]
    cur = idx[:, -1]  # (B,): di tutto quello che ci danno, l'ultimo carattere
    for _ in range(max_new_tokens):
        logits = model(cur)  # (B, C)
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1, generator=generator)  # (B, 1)
        chunks.append(nxt)
        cur = nxt[:, 0]  # (B,): multinomial dà una colonna, al modello serve un vettore
    return torch.cat(chunks, dim=1)  # (B, T + max_new_tokens)


g = torch.Generator().manual_seed(SEED)
start = torch.zeros((1, 1), dtype=torch.long)  # una sequenza sola, che parte dal punto
out = generate(start, 200, g)

# L'output è un flusso continuo, esattamente come il corpus della sezione 1: i
# nomi si ricavano tagliando sui punti. L'ultimo pezzo è troncato a metà, via.
stream = decode(out)
names = [n for n in stream.split(".") if n][:-1]

print(f"  partiti da {tuple(start.shape)}, arrivati a {tuple(out.shape)}\n")
print(f"    {stream[:64]!r} ...\n")
print(f"  {len(names)} nomi, tagliando il flusso sui punti:\n")
for n in names[:8]:
    print(f"    {n}")

print(
    "\n  Notare cosa NON c'è. Al modello arriva un (B,), non un (B, T): un intero per\n"
    "  sequenza, l'ultimo carattere estratto. Per il bigram è tutto quello che serve,\n"
    "  e la griglia sopravvive solo nell'output, dove impila i pezzi generati.\n"
    "\n"
    "  Qui B vale 1 perché un flusso basta: 200 caratteri contengono già decine di\n"
    "  nomi. Con start (5, 1) si genererebbero 5 flussi in parallelo, uno per riga,\n"
    "  senza cambiare una virgola di generate -- ma non c'è motivo di farlo.\n"
    "\n"
    "  È qui che si vedrà la differenza, quando arriverà un modello che il contesto lo\n"
    "  usa davvero. Lì `cur` non potrà più essere il solo ultimo carattere: i logits\n"
    "  dipenderanno da tutto il prefisso, quindi `cur` tornerà a essere un (B, T) che\n"
    "  cresce, `model(cur)` darà un (B, T, C), e servirà una riga in più --\n"
    "  `logits[:, -1, :]` -- per tenere solo l'ultima posizione, l'unica che continua\n"
    "  la sequenza. Quella riga, che si vede in tutti i generate in giro, esiste per\n"
    "  quel caso: qui sarebbe stata un no-op travestito da codice necessario."
)


# ---------------------------------------------------------------------------
# 8. e infatti è sempre lo stesso modello
# ---------------------------------------------------------------------------

print("\n=== rete vs conteggi ===\n")

with torch.no_grad():
    Q = F.softmax(model.weight, dim=1)

print(f"  scarto massimo fra le due tabelle di probabilità: {(P - Q).abs().max().item():.4f}")

# Per riga usiamo la distanza in variazione totale invece dello scarto massimo:
# "quanta probabilità, in totale, è finita nella casella sbagliata". Lo scarto
# massimo guarda una casella sola ed è troppo rumoroso per dire qualcosa.
tv = 0.5 * (P - Q).abs().sum(dim=1)
counts = N.sum(dim=1)
order = tv.argsort(descending=True)

print(f"\n  {'riga':<6} {'occorrenze':>11} {'probabilità sbagliata':>23}")
for i in list(order[:4]) + list(order[-3:]):
    print(f"    {itos[i.item()]!r:<4} {counts[i].item():>11} {tv[i].item():>22.1%}")

correlation = torch.corrcoef(torch.stack([counts.float().log(), tv]))[0, 1]
print(f"\n  correlazione fra log(occorrenze) e errore della riga: {correlation.item():+.2f}")

print(
    """
  Le righe imprecise sono quelle delle lettere rare, e la correlation lo dice
  senza ambiguità. La riga W[q] viene toccata solo dagli esempi che cominciano
  per 'q', 272 su 228146: nella stragrande maggioranza dei minibatch quella riga
  non riceve *nessun* gradiente, e resta ferma. Le righe di 'a' o 'e' vengono
  aggiornate a ogni singolo passo.

  È lo stesso squilibrio per cui in 02_bigram_nn.py serviva un lr enorme, qui reso
  più visibile dal minibatch: con il full batch ogni riga riceveva comunque il
  suo contributo a ogni passo, solo piccolo.

  A parte quello: stesso modello, stessa tabella, stessi nomi sgangherati. La
  griglia (B, T) non ha reso il bigram né migliore né peggiore, ed era il punto.
"""
)


# ---------------------------------------------------------------------------
# cosa abbiamo guadagnato, e cosa no
# ---------------------------------------------------------------------------
#
# NON abbiamo guadagnato:
#
#   - accuratezza. Il bigram guarda un carattere e ne predice uno. Gli 8
#     caratteri di contesto che ora stanno in ogni riga li ignora tutti tranne
#     quello sotto la casella: logits[b, t] dipende solo da xb[b, t]. La loss
#     finale è quella di sempre, ~2.46, il minimo raggiungibile con un bigram.
#   - una nuova capacità del modello. `emb(xb)` e `emb(xb.reshape(-1))` sono
#     bit per bit lo stesso risultato: lo verifica un assert qui sopra.
#
# Abbiamo guadagnato:
#
#   1. il minibatch, gratis. In 02_bigram_nn.py il full batch era possibile solo
#      perché il dataset era minuscolo; qui bastano 256 esempi per passo per
#      arrivare alla stessa loss con 178 volte meno lavoro. E il codice della
#      loss non è cambiato di una riga, perché un minibatch è solo una griglia
#      più piccola.
#   2. un dataset che non dipende più dalla forma dei dati. `get_batch`
#      ritaglia finestre da un testo qualsiasi: nomi, Shakespeare, codice. La
#      lista di coppie costruita a mano con `zip(chs, chs[1:])` funzionava
#      perché i nostri esempi erano parole corte e ben separate.
#   3. l'informazione di contesto *presente nei dati*. Nella lista piatta,
#      "quale casella viene prima di quale" era perduto. Nella griglia c'è, e
#      c'è nell'unica forma che serve: dentro una riga, e nell'order giusto.
#      Il bigram continua a ignorarla. Ma la stampa della sezione 2 -- dove si
#      vede che una sola finestra da T+1 caratteri contiene T esempi annidati,
#      di contesto lungo 1, 2, ... T -- è esattamente il materiale che un
#      modello capace di guardare indietro userebbe. Noi lo stiamo già
#      caricando in memoria e buttando via.
#
# Il prezzo, tutto sommato:
#
#   - `.view(-1, C)` prima di ogni cross_entropy, per sempre;
#   - in generazione, due tensori da tenere distinti invece di uno: l'output
#     che cresce, (B, T), e l'input del modello, che qui resta un (B,);
#   - un learning rate da ritarare, perché il gradiente adesso è rumoroso.
