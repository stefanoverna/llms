"""
GPT: i token passati influenzano il presente, e la media è il primo tentativo.

Questo file copre la lezione — `01_out_lecture.srt` — dal minuto 42:29 al
1:05:00, e non allena niente: sono tensori giocattolo attorno a un'idea sola.

L'idea è che i token di una sequenza si devono influenzare a vicenda, e in una
direzione precisa: il passato sul presente. Fino alla lezione scorsa ogni
posizione se ne stava per conto suo, e il contesto era una finestra di
lunghezza fissa appiccicata in un vettore. Qui invece la posizione t deve poter
guardare tutte le posizioni da 0 a t — mai il futuro, perché il futuro è
esattamente quello che stiamo cercando di indovinare.

Detta così però è un programma, non un calcolo: "influenzarsi" non vuol ancora
dire niente di preciso. Quindi si sperimenta, partendo dalla forma più stupida
che funzioni: la media di tutto quello che c'è stato fino a lì. È
un'interazione debolissima — l'ordine dei token precedenti va perso, resta solo
il mucchio, ed è per questo che si chiama *bag of words*, un sacchetto di
parole senza posizioni — ma è già passato che entra nel presente. E alla
self-attention basterà cambiare i pesi di quella media per avere il resto.

Poi c'è la parte tecnica, che è quella che occupa il segmento: quella media si
scrive come un prodotto fra matrici, e allora si calcola per tutte le posizioni
in una volta sola invece che una alla volta. Lo stesso identico calcolo, tre
volte:

    1. due for annidati                     (si legge, è lento)
    2. wei @ x, con wei triangolare         (il trucco)
    3. softmax su una matrice mascherata    (la forma che userà la self-attention)

Poi le sezioni 6, 7 e 8 aprono quel triangolo: perché i pesi non devono essere
tutti uguali, da dove arriva davvero il tensore che stiamo aggregando, e la
prima testa di self-attention.

Il seed è 1337, quello della lezione. Nelle sezioni 1-5 i numeri stampati sono
gli stessi del video; dalla 7 in poi no, perché lì x smette di essere
torch.randn e diventa un embedding vero.
"""

import time

import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(1337)

B, T, C = 4, 8, 2  # sequenze, posizioni, canali


# ---------------------------------------------------------------------------
# 0. il problema
# ---------------------------------------------------------------------------

x = torch.randn(B, T, C)

print("=== 0. il punto di partenza ===\n")
print(f"  x: {tuple(x.shape)}   (B={B} sequenze, T={T} posizioni, C={C} canali)")
print(
    """
  Otto posizioni che non si parlano: ognuna porta i suoi due numeri e basta.
  Quello che vogliamo è che la posizione t risenta di tutte le posizioni da 0
  a t compresa — e di nessuna oltre, perché oltre c'è quello che dobbiamo
  indovinare.

  "Risentire di" non è ancora un calcolo. Il primo che proviamo, e il più
  stupido possibile, è la media."""
)


# ---------------------------------------------------------------------------
# 1. versione 1: due for annidati
# ---------------------------------------------------------------------------
#
# La traduzione letterale della frase "media di tutte le posizioni fino a
# questa". Serve a fissare cosa vogliamo calcolare: le due versioni dopo
# dovranno dare esattamente questi numeri.

xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]  # (t+1, C): il passato, questa posizione inclusa
        xbow[b, t] = xprev.mean(0)  # media sul tempo -> (C,)

print("\n\n=== 1. i due for ===\n")
print("  la prima sequenza, prima e dopo:\n")
print(f"    {'t':>3}  {'x[0, t]':>34}  {'xbow[0, t]':>34}")
for t in range(T):
    a = "[" + ", ".join(f"{v:7.4f}" for v in x[0, t].tolist()) + "]"
    c = "[" + ", ".join(f"{v:7.4f}" for v in xbow[0, t].tolist()) + "]"
    print(f"    {t:>3}  {a:>34}  {c:>34}")
print(
    """
  La riga t=0 è identica: la media di un elemento solo è l'elemento. La riga
  t=1 è la media delle prime due righe di x, la t=2 delle prime tre, e così
  via fino all'ultima, che è la media di tutte e otto.

  Funziona ed è illeggibilmente lento: un ciclo Python per ogni sequenza e per
  ogni posizione, e dentro una slice e una media. La sezione 5 misura quanto."""
)


# ---------------------------------------------------------------------------
# 2. l'esempio giocattolo: come una matrice diventa "fai le medie"
# ---------------------------------------------------------------------------
#
# Prima di vettorizzare, il trucco in miniatura: 3x3 per 3x2, con numeri
# abbastanza piccoli da poterli seguire a mente.

print("\n\n=== 2. il trucco, in miniatura ===\n")

torch.manual_seed(42)
b_mat = torch.randint(0, 10, (3, 2)).float()


def show(a, b, c, titolo, nota):
    print(f"  {titolo}\n")
    print(f"    {'a (3x3)':<26}   {'b (3x2)':<14}   {'c = a @ b':<16}")
    for i in range(3):
        ra = "[" + " ".join(f"{v:5.2f}" for v in a[i].tolist()) + "]"
        rb = "[" + " ".join(f"{v:4.0f}" for v in b[i].tolist()) + "]"
        rc = "[" + " ".join(f"{v:5.2f}" for v in c[i].tolist()) + "]"
        print(f"    {ra:<26}   {rb:<14}   {rc:<16}")
    print(f"\n    {nota}\n")


# tutti uno: ogni riga di c è la somma di tutte le righe di b
a1 = torch.ones(3, 3)
show(a1, b_mat, a1 @ b_mat, "a) tutti uno",
     "ogni riga di c è la somma di tutte le righe di b, tre volte uguale")

# triangolare inferiore: la riga i somma solo le prime i+1 righe di b
a2 = torch.tril(torch.ones(3, 3))
show(a2, b_mat, a2 @ b_mat, "b) torch.tril: sopra la diagonale, zeri",
     "gli zeri spengono le righe future: c[i] somma solo le righe 0..i di b")

# righe normalizzate: la somma diventa media
a3 = a2 / a2.sum(1, keepdim=True)
show(a3, b_mat, a3 @ b_mat, "c) righe che sommano a 1",
     "stessa cosa, ma ogni riga è una media invece che una somma")

print(
    """  È tutto qui. Una moltiplicazione fra matrici è, riga per riga, una somma
  pesata delle righe della seconda matrice — e i pesi li decidiamo noi nella
  prima. Zero vuol dire "questa riga non la voglio", e normalizzare le righe
  perché sommino a 1 trasforma le somme in medie.

  Quindi "media di tutto il passato" non è un ciclo: è una matrice."""
)


# ---------------------------------------------------------------------------
# 3. versione 2: wei @ x
# ---------------------------------------------------------------------------

wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x

print("\n\n=== 3. wei @ x ===\n")
print(f"  wei ({T}x{T}), quanto pesa ogni posizione (colonna) per ogni riga:\n")
for i in range(T):
    print("    " + " ".join(f"{v:5.2f}" for v in wei[i].tolist()))
print(
    f"""
  Riga t: 1/(t+1) sulle prime t+1 colonne, zero sulle altre. Il triangolo
  vuoto in alto a destra è il vincolo causale, scritto in numeri.

  Le forme: wei è ({T}, {T}), x è ({B}, {T}, {C}). PyTorch non trova la stessa
  forma, quindi tratta wei come se fosse ({B}, {T}, {T}) — la stessa matrice
  ripetuta per ogni sequenza — e fa {B} prodotti ({T}x{T}) @ ({T}x{C}) in
  parallelo. Il risultato è {tuple(xbow2.shape)}.

  identico alla versione con i for: {torch.allclose(xbow, xbow2)}"""
)


# ---------------------------------------------------------------------------
# 4. versione 3: la stessa cosa con la softmax
# ---------------------------------------------------------------------------
#
# Un terzo modo di ottenere la stessa wei, che sembra un giro inutile e non lo
# è: è la forma in cui i pesi smetteranno di essere costanti.

tril = torch.tril(torch.ones(T, T))
wei3 = torch.zeros((T, T))  # le "affinità": per ora tutte uguali
wei3 = wei3.masked_fill(tril == 0, float("-inf"))  # il futuro: spento
wei3 = F.softmax(wei3, dim=-1)  # normalizza ogni riga
xbow3 = wei3 @ x

print("\n\n=== 4. la stessa wei, per un'altra strada ===\n")
print("  1. si parte da zeri, cioè da 'tutti i token si interessano uguale'")
print("  2. masked_fill mette -inf dove tril è zero, cioè sul futuro")
print("  3. la softmax esponenzia e normalizza: exp(-inf) = 0, e il resto\n"
      "     si divide equamente\n")
print("  le prime tre righe, passo per passo:\n")
zero = torch.zeros((T, T))
masked = zero.masked_fill(tril == 0, float("-inf"))
for i in range(3):
    z = " ".join(f"{v:5.0f}" for v in zero[i].tolist())
    m = " ".join(f"{v:5.0f}" for v in masked[i].tolist())
    s = " ".join(f"{v:5.2f}" for v in wei3[i].tolist())
    print(f"    zeri     {z}")
    print(f"    masked   {m}")
    print(f"    softmax  {s}\n")

print(f"  identico alle altre due versioni: "
      f"{torch.allclose(xbow, xbow3) and torch.allclose(wei, wei3)}")
print(
    """
  Il giro in più serve a questo: nella versione 2 i pesi *sono* il triangolo,
  e il triangolo è una costante. Qui i pesi partono da una matrice di numeri
  — che adesso è tutta zeri, ma non deve esserlo — e il triangolo interviene
  solo a spegnere il futuro.

  Quei numeri sono le affinità: quanto il token t trova interessante il token
  t'. Nella self-attention non saranno più zeri, saranno calcolati dai token
  stessi, e ogni posizione si costruirà la sua media pesata invece di
  prendersi quella uniforme. Il resto della riga — maschera, softmax,
  moltiplicazione — resta identico a questo.

  E si capisce perché la maschera è -inf e non 0: -inf è lo zero *prima* della
  softmax. Mettere 0 direbbe "peso medio", non "peso nullo"."""
)


# ---------------------------------------------------------------------------
# 5. quanto costa il ciclo
# ---------------------------------------------------------------------------
#
# La lezione dice che i due for sono inefficienti e passa oltre. Con (4, 8, 2)
# non si vede: sono 32 medie. Con le dimensioni di un GPT giocattolo sì.

print("\n\n=== 5. i due for contro la moltiplicazione ===\n")

Bb, Tt, Cc = 64, 256, 64
xb = torch.randn(Bb, Tt, Cc)

t0 = time.time()
loop = torch.zeros((Bb, Tt, Cc))
for b in range(Bb):
    for t in range(Tt):
        loop[b, t] = xb[b, : t + 1].mean(0)
t_loop = time.time() - t0

t0 = time.time()
w = torch.tril(torch.ones(Tt, Tt))
w = w / w.sum(1, keepdim=True)
matmul = w @ xb
t_matmul = time.time() - t0

print(f"  su (B, T, C) = ({Bb}, {Tt}, {Cc}):\n")
print(f"    due for annidati       {t_loop * 1000:8.1f} ms   ({Bb * Tt} medie, una per volta)")
print(f"    wei @ x                {t_matmul * 1000:8.1f} ms")
print(f"    rapporto               {t_loop / t_matmul:8.0f}x")
print(f"\n  stessi numeri: {torch.allclose(loop, matmul, atol=1e-6)}")
print(
    """
  E il divario cresce con T, perché il ciclo fa T passi mentre la
  moltiplicazione resta una chiamata sola — dentro cui BLAS lavora su tutte le
  posizioni insieme. Su GPU la differenza è ancora più larga.

  Non è un dettaglio da ottimizzatori: è il motivo per cui questa architettura
  è allenabile. La stessa somma pesata scritta come ciclo non ci porterebbe da
  nessuna parte."""
)
