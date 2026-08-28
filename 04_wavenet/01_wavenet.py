"""
makemore, parte 5: da una MLP piatta a una WaveNet.

La rete delle lezioni precedenti prende tre caratteri, li appiccica in un
vettore solo e ci passa sopra un layer nascosto. Funziona, ma ha un limite che
si vede appena si prova a darle più contesto: qualunque sia il numero di
caratteri in ingresso, vengono schiacciati tutti insieme al primo passo. Otto
caratteri che diventano 200 neuroni in una moltiplicazione sola sono
informazione buttata via troppo in fretta.

L'alternativa è fondere il contesto *piano*, un livello alla volta: prima le
coppie di caratteri, poi le coppie di coppie, poi le coppie di quelle. Otto
caratteri diventano quattro bigrammi, poi due quadrigrammi, poi un vettore
solo, e ogni fusione ha in mezzo un layer con i suoi pesi. È la struttura ad
albero di [WaveNet](https://arxiv.org/abs/1609.03499) (DeepMind, 2016), che
nel paper predice campioni audio invece di caratteri ma è lo stesso identico
problema: un modello autoregressivo che indovina il prossimo elemento di una
sequenza.

Nel paper quella struttura si chiama "stack of dilated causal convolutions", e
il nome fa più paura dell'idea. Le convoluzioni sono un dettaglio di
efficienza, non di modello: servono a far scorrere la stessa rete su tutte le
posizioni della sequenza in una volta sola. L'albero, che è il modello, si
costruisce con le `view` — e la sezione 8 spiega cosa cambierebbe a farlo con
le convoluzioni davvero.

Due cose che questo file dà per fatte, e che nella lezione invece si fanno:

  - i layer sono quelli di `torch.nn`, non riscritti a mano. Il giro di
    ricostruirli l'abbiamo fatto in `03_mlp/06_modules_from_scratch.py`, dove
    le nostre versioni sono state confrontate numero per numero con quelle
    vere. Di tutto quello che serve qui, PyTorch non ha una cosa sola — il
    flatten che raggruppa — e la sezione 1 è lunga due classi in croce.

  - la MLP piatta non c'è: è la rete della lezione 3 con una costante cambiata.
    Serve solo come termine di paragone, e quel numero — misurato a parte con
    lo stesso `train()` di qui — è nella tabella sotto.

Due training da 200k passi, in tutto una decina di minuti:

    2.1095   MLP piatta, 3 caratteri di contesto    (03_mlp/03_batchnorm.py)
    2.0263   MLP piatta, 8 caratteri                (misurata a parte, stesso train())
    2.0149   WaveNet a 3 livelli, stessi parametri  (sezione 5)
    1.9908   ...e con la rete più grande            (sezione 6)

Le due righe di mezzo sono quelle da leggere insieme: a parità di parametri
l'albero da solo non regala quasi niente. Quasi tutto il guadagno rispetto
alla lezione 3 viene dal contesto più lungo. Quello che l'albero dà è una
manopola in più — la profondità — e la si vede nell'ultima riga.

Le forme dei tensori, tutte insieme e con lo schema dell'albero, stanno in
`01_out_shapes.md`.
"""

import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 2147483647
HERE = Path(__file__).parent
NAMES = HERE.parent / "02_bigram" / "names.txt"

BLOCK_SIZE = 8  # era 3: è l'unica riga che cambia nel dataset
BATCH_SIZE = 32
STEPS = 200_000
LR = 0.1


# ---------------------------------------------------------------------------
# 0. il dataset: lo stesso di sempre, con la finestra più larga
# ---------------------------------------------------------------------------
#
# Non c'è niente da riscrivere. La finestra scorrevole che costruiva esempi da
# tre caratteri ne costruisce da otto cambiando una costante, e il numero di
# esempi resta identico (182625): sono sempre le stesse posizioni dentro le
# stesse parole, viste con più passato alle spalle.

words = open(NAMES, "r").read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
VOCAB = len(itos)

random.seed(42)
random.shuffle(words)
n_train, n_dev = int(0.8 * len(words)), int(0.9 * len(words))


def build_dataset(word_list):
    X, Y = [], []
    for word in word_list:
        context = [0] * BLOCK_SIZE
        for ch in word + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


Xtr, Ytr = build_dataset(words[:n_train])
Xdev, Ydev = build_dataset(words[n_train:n_dev])
Xte, Yte = build_dataset(words[n_dev:])

print("=== dataset ===\n")
print(f"  contesto: {BLOCK_SIZE} caratteri")
print(f"  train {tuple(Xtr.shape)}   dev {tuple(Xdev.shape)}   test {tuple(Xte.shape)}\n")
for i in range(5):
    print("   ", "".join(itos[ix.item()] for ix in Xtr[i]), "-->", itos[Ytr[i].item()])


# ---------------------------------------------------------------------------
# 1. i due pezzi che torch.nn non ha
# ---------------------------------------------------------------------------
#
# Tutto il resto viene da `torch.nn`: `Embedding`, `Linear`, `Tanh`,
# `BatchNorm1d`, `Sequential`, e più sotto `optim.SGD` e `MultiStepLR`.
#
# Restano due cose da mettere noi, e sono due cose diverse: la prima è un
# layer che in PyTorch non esiste, la seconda è solo una questione di dove
# stanno gli assi.


class FlattenConsecutive(nn.Module):
    """Fonde `n` elementi consecutivi dell'asse centrale in quello dei canali.

        (B, T, C)  ->  (B, T // n, C * n)

    Con n uguale a T è `nn.Flatten`: un gruppo solo, cioè tutto in una riga.
    Con n = 2 è un livello dell'albero di WaveNet.

    In PyTorch non c'è: `nn.Flatten` prende un intervallo di assi e li
    appiattisce tutti, non sa raggruppare a n a n. Il nome quindi cambia,
    perché l'interfaccia è un'altra.
    """

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            # Rimasto un gruppo solo: l'asse delle posizioni non serve più.
            # `squeeze(1)` e non `squeeze()`: vogliamo un errore se a essere
            # uno è una dimensione che non ci aspettavamo.
            x = x.squeeze(1)
        return x

    def extra_repr(self):
        return f"n={self.n}"


class BatchNorm1dNLC(nn.BatchNorm1d):
    """`nn.BatchNorm1d`, con i canali sull'ultimo asse invece che in mezzo.

    Non è una reimplementazione: è `nn.BatchNorm1d`, chiamata con gli assi
    messi come li vuole lei. Serve perché le due convenzioni non coincidono.

    I nostri tensori sono (N, T, C) — esempi, posizioni, canali — perché è la
    forma che vuole `nn.Linear`, che moltiplica sempre sull'ultimo asse. È
    anche la convenzione dei Transformer.

    `nn.BatchNorm1d`, sugli input a tre assi, vuole invece (N, C, L): i canali
    in mezzo, che è la convenzione delle convoluzioni. Passarle il nostro
    tensore così com'è non darebbe nessun errore e sarebbe sbagliato: si
    metterebbe a tenere una statistica per ogni posizione, invece che una per
    canale.

    La strada ovvia sarebbe trasporre, `x.transpose(1, 2)`. Funziona, ma esce
    un tensore non contiguo, e la `view` del FlattenConsecutive successivo si
    rifiuta di lavorarci: bisognerebbe aggiungere un `.contiguous()`, che è
    una copia dell'intero tensore delle attivazioni a ogni layer.

    Meglio l'altra strada: le T posizioni sono gruppi di caratteri che passano
    tutti negli stessi pesi, cioè un asse di batch esattamente come N. Allora
    li fondiamo, (N, T, C) -> (N*T, C), e usiamo il ramo a due assi, dove i
    canali sono già in fondo. Le statistiche sono le stesse che uscirebbero da
    (N, C, L) — lì riduce su (0, 2), qui su tutte le N*T righe — ma `flatten` e
    `view_as` su un tensore contiguo non copiano niente.
    """

    def forward(self, x):
        if x.ndim == 2:
            return super().forward(x)
        return super().forward(x.flatten(0, 1)).view_as(x)


# ---------------------------------------------------------------------------
# 2. la moltiplicazione fra matrici accetta più assi di quanti ne servano
# ---------------------------------------------------------------------------
#
# Il primo dei due ingredienti, ed è una proprietà di `@` che si usa di
# continuo senza saperla: la moltiplicazione lavora sull'*ultimo* asse, e
# tutti quelli davanti se li porta dietro come assi di batch.

print("\n\n=== 2. `@` e gli assi in più ===\n")

W = torch.randn(20, 200)
for shape in [(4, 20), (4, 5, 20), (4, 5, 6, 20)]:
    print(f"  {str(shape):<16} @ (20, 200)  ->  {tuple((torch.randn(shape) @ W).shape)}")

print(
    """
  Quindi un Linear non ha bisogno di sapere quante posizioni ci sono davanti:
  ne processa in parallelo quante gliene arrivano. Ed è per questo che non
  scriveremo nessun ciclo per applicare lo stesso layer ai quattro bigrammi di
  ogni esempio — basta che i bigrammi stiano in un asse a sinistra, e la
  moltiplicazione li tratta come batch.

  Ed è anche il motivo per cui i nostri tensori tengono i canali per ultimi:
  è la forma che serve a nn.Linear. L'unico layer che la vuole diversa è la
  batchnorm, ed è tutto il contenuto di BatchNorm1dNLC."""
)


# ---------------------------------------------------------------------------
# 3. FlattenConsecutive: raggruppare a due a due è una `view`
# ---------------------------------------------------------------------------
#
# Il secondo ingrediente: mettere i bigrammi in quell'asse. Detto
# esplicitamente sarebbe "prendi le posizioni pari, prendi le dispari, e
# concatenale sui canali". Solo che la memoria di un tensore è già ordinata
# così, quindi non serve copiare niente: basta rileggere gli stessi numeri con
# un'altra forma.

print("\n\n=== 3. la view che raggruppa ===\n")

e = torch.randn(4, 8, 10)  # (batch, posizioni, canali)
explicit = torch.cat([e[:, ::2, :], e[:, 1::2, :]], dim=2)
viewed = e.view(4, 4, 20)

print(f"  concatenazione esplicita di pari e dispari: {tuple(explicit.shape)}")
print(f"  e.view(4, 4, 20):                           {tuple(viewed.shape)}")
print(f"  identici numero per numero: {torch.equal(explicit, viewed)}")
print(
    """
  Una `view` non copia e non calcola: cambia solo come si leggono i byte che
  ci sono già. Quindi il primo livello dell'albero — "fondi i caratteri a due
  a due" — costa zero, e tutto il costo sta nel Linear che viene dopo.

  È anche il motivo per cui questa architettura si scrive con le view: la
  concatenazione fra elementi vicini, in un tensore contiguo, è gratis."""
)


# ---------------------------------------------------------------------------
# 4. la rete, e le sue shape
# ---------------------------------------------------------------------------
#
# Tre livelli: 8 caratteri -> 4 bigrammi -> 2 quadrigrammi -> 1 vettore. Ogni
# livello è la stessa sequenza di quattro layer, e l'unica cosa che cambia fra
# uno e l'altro è quanti gruppi restano.


def wavenet_model(emb_dim=10, hidden=68):
    torch.manual_seed(SEED)
    model = nn.Sequential(
        nn.Embedding(VOCAB, emb_dim),
        FlattenConsecutive(2), nn.Linear(emb_dim * 2, hidden, bias=False), BatchNorm1dNLC(hidden), nn.Tanh(),
        FlattenConsecutive(2), nn.Linear(hidden * 2, hidden, bias=False), BatchNorm1dNLC(hidden), nn.Tanh(),
        FlattenConsecutive(2), nn.Linear(hidden * 2, hidden, bias=False), BatchNorm1dNLC(hidden), nn.Tanh(),
        nn.Linear(hidden, VOCAB),
    )

    # L'unica inizializzazione che tocchiamo: l'ultimo layer, più piccolo.
    # Senza, la loss al passo 0 vale 26 invece di -log(1/27) = 3.30, e la rete
    # spende i primi mille passi solo a spegnere i logits sbagliati (è la
    # sezione 2 di 03_mlp/02_optimizations.py).
    #
    # Su tutti gli altri layer lasciamo il default di nn.Linear, che per la
    # tanh sarebbe troppo stretto (il conto sta in 03_mlp/04_mlp_idiomatic.py)
    # — ma qui dopo ognuno c'è una batchnorm, che rinormalizza le
    # pre-attivazioni comunque siano uscite. Con una normalizzazione in mezzo
    # la scala dei pesi smette di essere una cosa da azzeccare, ed è tutto il
    # senso della seconda metà della lezione 3.
    nn.init.normal_(model[-1].weight, std=0.01)
    nn.init.zeros_(model[-1].bias)
    return model


print("\n\n=== 4. le shape, layer per layer ===\n")


@torch.no_grad()
def show_shapes(model, x):
    """Su un modello appena costruito: le shape non dipendono
    dall'allenamento, e così non tocchiamo i buffer della batchnorm di una
    rete che stiamo per allenare."""
    print(f"  {'input':<22} {tuple(x.shape)}")
    for layer in model:
        x = layer(x)
        print(f"  {type(layer).__name__:<22} {tuple(x.shape)}")


show_shapes(wavenet_model(), Xtr[:4])
print(
    """
  Le righe da guardare sono le tre FlattenConsecutive: l'asse di mezzo si
  dimezza (4 -> 2 -> sparisce) e quello dei canali raddoppia. È l'albero, e
  non c'è nessun ciclo da nessuna parte — i quattro bigrammi sono un asse, e
  il Linear che segue li macina in parallelo.

  All'ultima l'asse dei gruppi arriva a uno e viene tolto: da lì in poi il
  tensore ha due assi e la rete torna a essere una MLP normale."""
)


# ---------------------------------------------------------------------------
# il training
# ---------------------------------------------------------------------------


def train(model, steps=STEPS, label=""):
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[steps * 3 // 4], gamma=0.1
    )

    g = torch.Generator().manual_seed(SEED)
    losses = []
    model.train()
    t0 = time.time()

    for _ in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
        loss = F.cross_entropy(model(Xtr[ix]), Ytr[ix])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {label}: {n_params} parametri, {steps} passi in {time.time() - t0:.0f}s"
          f"   -->  train {evaluate(model, Xtr, Ytr):.4f}   dev {evaluate(model, Xdev, Ydev):.4f}")
    return losses


@torch.no_grad()
def evaluate(model, X, Y, chunk=8192):
    """La loss media su tutto uno split, a blocchi.

    A blocchi non per eleganza: 182625 esempi che attraversano tutti insieme
    tre livelli da 128 canali sono qualche gigabyte di attivazioni. Il
    risultato è identico perché la media di medie pesate per la dimensione dei
    blocchi è la media.

    E `eval()` prima di misurare: senza, la batchnorm normalizzerebbe con le
    statistiche del blocco che le passiamo, e il numero che esce è un altro.
    """
    model.eval()
    total = 0.0
    for i in range(0, len(X), chunk):
        xb, yb = X[i : i + chunk], Y[i : i + chunk]
        total += F.cross_entropy(model(xb), yb).item() * len(xb)
    model.train()
    return total / len(X)


# ---------------------------------------------------------------------------
# 5. a parità di parametri
# ---------------------------------------------------------------------------
#
# 68 neuroni per livello non è un numero scelto per la rete: è scelto perché i
# parametri vengano 22397, contro i 22097 della MLP piatta con lo stesso
# contesto. Non stiamo confrontando una rete grande con una piccola, stiamo
# confrontando due modi di spendere gli stessi parametri.

print("\n\n=== 5. la WaveNet, stessi parametri (qualche minuto) ===\n")

wavenet = wavenet_model()
wavenet_losses = train(wavenet, label="wavenet 10/68")
print(
    """
  Contro il 2.0263 della MLP piatta con lo stesso contesto, allenata con lo
  stesso codice: un centesimo di loss. A parità di parametri l'albero da solo
  non regala praticamente niente — e la lezione lo dice: non abbiamo torturato
  l'architettura, i canali per livello sono la prima cosa provata.

  Quello che l'albero regala non è la loss, è che adesso esiste una manopola
  per la profondità. Con la rete piatta l'unico modo di crescere era allargare
  l'unico layer nascosto, e schiacciare comunque tutto al primo passo."""
)


# ---------------------------------------------------------------------------
# 6. la stessa rete, più grande
# ---------------------------------------------------------------------------
#
# Farla crescere è una questione di due numeri: embedding da 24 invece di 10,
# 128 canali invece di 68. Nessuna riga di struttura cambia, e i parametri
# passano da 22k a 76k.

print("\n\n=== 6. la stessa rete, 76k parametri (qualche minuto) ===\n")

big = wavenet_model(emb_dim=24, hidden=128)
big_losses = train(big, label="wavenet 24/128")
print(
    """
  Sotto il 2.0. Ma qui bisogna essere onesti su cosa è appena successo: non
  abbiamo un'infrastruttura per gli esperimenti, guardiamo solo la loss di
  training mentre scorre, e i learning rate sono quelli di tre lezioni fa
  scelti su una rete diversa. Stiamo tirando a indovinare con metodo, che è
  meglio di niente ma non è misurare."""
)


# ---------------------------------------------------------------------------
# 7. il grafico delle loss, e perché va mediato
# ---------------------------------------------------------------------------
#
# La loss di un batch da 32 esempi è rumorosissima: 32 esempi sono pochi, e
# capitare in un batch fortunato o sfortunato sposta il numero più di quanto
# lo sposti l'allenamento. Il grafico grezzo è una banda spessa in cui non si
# legge niente.
#
# Il trucco è di nuovo una view: 200000 numeri visti come (200, 1000), poi la
# media su ogni riga. Duecento punti, ognuno la media di mille passi.

print("\n\n=== 7. il grafico ===\n")

WINDOW = 1000


def smooth(losses):
    return torch.tensor(losses).view(-1, WINDOW).mean(1)


fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

averaged = smooth(wavenet_losses)
axes[0].plot(wavenet_losses, linewidth=0.3, alpha=0.5, label="loss dei singoli batch")
axes[0].plot(torch.arange(len(averaged)) * WINDOW + WINDOW // 2, averaged,
             linewidth=2, label=f"media ogni {WINDOW} passi")
axes[0].set_title("perché la media: la stessa curva, due volte", fontsize=10)
axes[0].set_xlabel("passo")
axes[0].set_ylabel("loss di training")
axes[0].set_ylim(1.5, 3.5)
axes[0].legend(fontsize=8)

for losses, label in [
    (wavenet_losses, "wavenet 10/68, 22k parametri"),
    (big_losses, "wavenet 24/128, 76k parametri"),
]:
    axes[1].plot(smooth(losses), linewidth=1.5, label=label)
axes[1].set_title("i due training", fontsize=10)
axes[1].set_xlabel(f"passo / {WINDOW}")
axes[1].set_ylabel(f"loss di training (media ogni {WINDOW})")
axes[1].set_ylim(1.7, 2.6)
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(HERE / "01_out_loss_curves.png", bbox_inches="tight", dpi=100)
plt.close()
print(f"  grafico in {HERE / '01_out_loss_curves.png'}")
print("\n  Nel pannello di destra si vede anche il gradino del learning rate a\n"
      "  tre quarti del training: da lì in poi la curva non scende, si posa.")


# ---------------------------------------------------------------------------
# 8. e le convoluzioni?
# ---------------------------------------------------------------------------
#
# Nel paper questa architettura è fatta di "dilated causal convolutions", e
# qui non ne abbiamo scritta una. Non è una semplificazione: il modello è lo
# stesso, la convoluzione è solo il modo di calcolarlo in fretta.
#
# Il punto è che finora abbiamo predetto *una* posizione alla volta. Una
# parola di sette lettere sono otto esempi indipendenti, e li mandiamo dentro
# la rete come otto righe di un batch. La convoluzione invece fa scorrere la
# rete lungo la sequenza e calcola tutte le posizioni in una passata, con il
# ciclo dentro un kernel CUDA invece che in Python — e soprattutto riusando i
# nodi intermedi, che fra una finestra e la successiva si ripetono.

print("\n\n=== 8. cosa farebbe una convoluzione ===\n")

# Ogni parola occupa len(word)+1 righe consecutive di Xtr (una per lettera,
# più quella che predice il punto finale): scorriamo fino alla prima da sette.
offset = 0
for word in words[:n_train]:
    if len(word) == 7:
        break
    offset += len(word) + 1
rows = range(offset, offset + len(word) + 1)

big.eval()
with torch.no_grad():
    one_by_one = torch.cat([big(Xtr[r : r + 1]) for r in rows])  # otto chiamate
    all_at_once = big(Xtr[offset : offset + len(rows)])  # una sola
big.train()

print(f"  la parola: {word!r}, cioè {len(rows)} esempi")
print(f"  otto forward separati contro uno solo, scarto massimo sui logits: "
      f"{(one_by_one - all_at_once).abs().max():.2e}")

# Quanti nodi calcola davvero il primo livello dell'albero, sulle otto
# finestre, e quanti sarebbero se li riusassimo: una finestra lunga 8 contiene
# 4 bigrammi, e finestre consecutive li condividono a due a due.
computed, distinct = 0, set()
for k in range(len(rows)):  # la finestra k copre le posizioni assolute k-8 .. k-1
    for j in range(BLOCK_SIZE // 2):
        computed += 1
        distinct.add(k - BLOCK_SIZE + 2 * j)  # dove comincia questo bigramma

print(f"\n  primo livello: {computed} bigrammi calcolati, {len(distinct)} diversi"
      f"  ({computed / len(distinct):.1f}x di lavoro ripetuto)")
print(
    """
  Ed è tutto qui il guadagno delle convoluzioni: lo stesso nodo dell'albero è
  figlio destro di una finestra e figlio sinistro di quella dopo, e noi lo
  ricalcoliamo ogni volta. Un layer convoluzionale è un filtro lineare — la
  stessa matrice di pesi dei nostri Linear — fatto scorrere sulla sequenza, e
  la sovrapposizione la paga una volta sola.

  Il modello non cambia. Cambia quanto costa allenarlo, che è poi la ragione
  per cui il paper è scritto in quel modo."""
)


# ---------------------------------------------------------------------------
# 9. i nomi
# ---------------------------------------------------------------------------

print("\n=== 9. nomi campionati dalla rete grande ===\n")

big.eval()


@torch.no_grad()
def sample(model, num=20):
    g = torch.Generator().manual_seed(SEED)
    names = []
    for _ in range(num):
        out, context = [], [0] * BLOCK_SIZE
        while True:
            logits = model(torch.tensor([context]))
            ix = torch.multinomial(logits.softmax(1), num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            if ix == 0:
                break
            out.append(itos[ix])
        names.append("".join(out))
    return names


generated = sample(big)
known = set(words)
for i in range(0, len(generated), 5):
    print("  " + "  ".join(f"{n}{'*' if n in known else ''}" for n in generated[i:i + 5]))
print(f"\n  (* = nome che esiste davvero nel dataset: {sum(n in known for n in generated)} su {len(generated)})")


# ---------------------------------------------------------------------------
# cosa resta fuori
# ---------------------------------------------------------------------------
#
# Del paper WaveNet abbiamo preso la struttura ad albero e basta. Dentro ogni
# livello loro non hanno un Linear e una tanh, ma un blocco con una gated
# linear unit (due rami, uno che calcola e uno che decide quanto farlo
# passare), connessioni residue e skip connection. Sono le cose che servono a
# far funzionare la stessa idea con molti più livelli.
#
# E resta fuori la cosa che la lezione stessa indica come il buco più grosso:
# non c'è nessuna infrastruttura sperimentale. Due run scelti a mano, una
# curva guardata a occhio, nessuna ricerca di iperparametri, nessun confronto
# fra train e dev durante il training. Con esperimenti che durano minuti si
# tira a indovinare; con esperimenti che durano ore non si può più.
