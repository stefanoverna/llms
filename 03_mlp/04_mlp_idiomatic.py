"""
La stessa rete delle tre tappe precedenti, scritta come si scrive davvero.

`01_mlp.py`, `02_optimizations.py` e `03_batchnorm.py` tengono tutto a mano
apposta: un dizionario di tensori al posto del modello, un `forward` scritto a
mano, i batch pescati con `torch.randint`, il learning rate cambiato con un
`if`, media e deviazione standard della batchnorm calcolate riga per riga. Era
il punto: far vedere che sotto non c'è niente di magico.

Qui la stessa identica rete usa le astrazioni vere di PyTorch. Non impara
niente di diverso e non arriva a una loss diversa — cambia solo che ogni pezzo
che avevamo costruito a mano ha un nome e sta in una classe:

  dict di tensori             ->  nn.Sequential, che tiene i parametri e li nomina
  C[X]                        ->  nn.Embedding
  emb.view(N, -1)             ->  nn.Flatten
  W1 e basta, senza b1        ->  nn.Linear(..., bias=False)
  W1 * (5/3)/sqrt(fan_in)     ->  nn.init.kaiming_normal_(..., nonlinearity="tanh")
  bngain, bnbias, running_*   ->  nn.BatchNorm1d
  stats="batch" / "running"   ->  model.train() / model.eval()
  torch.tanh                  ->  nn.Tanh
  F.cross_entropy             ->  nn.CrossEntropyLoss
  torch.randint per i batch   ->  DataLoader
  p.grad = None               ->  optimizer.zero_grad(set_to_none=True)
  p -= lr * p.grad            ->  optimizer.step()
  0.1 e poi 0.01              ->  lr_scheduler.MultiStepLR

Quello che PyTorch *non* fa al posto tuo, e che resta scritto a mano anche qui,
è l'inizializzazione: il default di `nn.Linear` non è quello che serve per la
tanh, e l'ultimo layer va rimpicciolito a parte. Sul secondo punto il default se
la cava molto meglio di quanto ci si aspetti, e nel file c'è il perché con i
numeri. In fondo: le differenze che non sono solo cosmetiche.

`05_mlp_bare.py` è questo stesso file senza i commenti.
"""

import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 2147483647
HERE = Path(__file__).parent
NAMES = HERE.parent / "02_bigram" / "names.txt"

BLOCK_SIZE = 3
EMB_DIM = 10
HIDDEN = 200
BATCH_SIZE = 32
FAN_IN = BLOCK_SIZE * EMB_DIM

EPOCHS = 36  # ~205k passi da 32 esempi, cioè i 200k delle tappe precedenti


# ---------------------------------------------------------------------------
# 0. il dataset: identico alle tappe precedenti
# ---------------------------------------------------------------------------

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

torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# 1. il modello
# ---------------------------------------------------------------------------

print("=== il modello ===\n")

# Tutto il forward delle tappe precedenti, in sei righe. nn.Sequential chiama i
# moduli in fila e tiene i loro parametri: al posto del dizionario `net` e
# della funzione `forward()` che lo leggeva.
model = nn.Sequential(
    # C[X]: prende (N, 3) di indici e restituisce (N, 3, 10). È letteralmente
    # una tabella di lookup, come la nostra C, con in più un backward sparso
    # (vedi la nota 4 in fondo).
    nn.Embedding(VOCAB, EMB_DIM),
    # emb.view(emb.shape[0], -1): (N, 3, 10) -> (N, 30). Di default parte
    # dall'asse 1, cioè non tocca mai l'asse degli esempi.
    nn.Flatten(),
    # bias=False perché subito dopo c'è una batchnorm, che sottrae la media del
    # batch e quindi cancella il bias: 200 parametri con gradiente esattamente
    # zero. È la sezione 8 di 03_batchnorm.py, srotolata in 03_out_batchnorm_bias.md.
    nn.Linear(FAN_IN, HIDDEN, bias=False),
    # bngain, bnbias, running_mean, running_std, il momentum e l'epsilon: tutto
    # dentro qui. eps=1e-5 è già il default; il momentum no (vedi sotto). I due
    # buffer si aggiornano da soli e fuori dal grafo: qui non c'è nessun
    # torch.no_grad() da scrivere, mentre in 03_batchnorm.py serviva (nota 3).
    nn.BatchNorm1d(HIDDEN, momentum=0.001),
    nn.Tanh(),
    nn.Linear(HIDDEN, VOCAB),
)

# Il momentum della media mobile: PyTorch usa 0.1 di default, che è tarato su
# batch grandi. Con batch da 32 la media di un batch balla parecchio intorno a
# quella vera, e con 0.1 la stima oscilla senza assestarsi mai. È lo stesso
# 0.001 di 03_batchnorm.py.
#
# Una differenza di formula, piccola ma vera: la nostra divideva per
# (std + eps), PyTorch per sqrt(var + eps). Cambia nella terza cifra.

first_linear = model[2]
last_linear = model[5]

# --- l'inizializzazione, che PyTorch NON indovina al posto tuo ---
#
# Il default di nn.Linear è kaiming_uniform_(a=sqrt(5)), una scelta storica che
# non corrisponde a nessuna non-linearità in particolare: dà una deviazione
# standard di 1/sqrt(3*fan_in), circa tre volte più piccola di quella che
# 02_optimizations.py aveva ricavato per la tanh, che è (5/3)/sqrt(fan_in).
default_std = first_linear.weight.std().item()

# kaiming_normal_ con nonlinearity="tanh" chiede a PyTorch il gain giusto (5/3)
# e divide per sqrt(fan_in): è esattamente `W1 * (5/3) / FAN_IN**0.5`.
nn.init.kaiming_normal_(first_linear.weight, nonlinearity="tanh")

print(f"  std di W1 con il default di nn.Linear: {default_std:.4f}")
print(f"  std di W1 dopo kaiming_normal_(tanh):  {first_linear.weight.std().item():.4f}")
print(f"  (il conto a mano di 02_optimizations.py: {(5 / 3) / FAN_IN**0.5:.4f})\n")

# L'altra riga che PyTorch non scrive al posto tuo: i logits devono partire
# piccoli, altrimenti i primi mille passi se ne vanno solo a schiacciare
# l'ultimo layer. È la sezione 2 di 02_optimizations.py.
#
# Qui il default non è affatto un disastro. La loss 26 delle tappe precedenti 
# era colpa del nostro `torch.randn` puro, ma nn.Linear inizializza i pesi uniformi in 
#
# ±1/sqrt(fan_in)
#
# e il bias nello stesso intervallo. Quel 1/sqrt(fan_in) è proprio quello che tiene buoni i logits:
#
#     torch.randn puro, come nelle tappe 01-03   std(logits) 10.20   loss 20.93
#     default di nn.Linear                       std(logits)  0.39   loss  3.32
#     std=0.01 e bias a zero, questa riga        std(logits)  0.10   loss  3.30
#                                                       (l'ideale è log(27) = 3.2958)
#
# Quindi queste due righe non salvano il training, lo rifiniscono: portano la
# loss iniziale da 3.32 a 3.30 e allineano questo file alle tappe precedenti. Se
# le togli non succede niente di grave — ed è una cosa che è meglio sapere che
# credere il contrario.
nn.init.normal_(last_linear.weight, std=0.01)
nn.init.zeros_(last_linear.bias)

# La controprova, che costa un forward: la loss al passo 0 con i pesi appena
# inizializzati. Se non è vicina a log(27) vuol dire che la rete ha già delle
# opinioni prima di aver visto un dato, e le prossime centinaia di passi le
# servono solo per disfarle.
with torch.no_grad():
    loss_zero = nn.functional.cross_entropy(model(Xtr[:BATCH_SIZE]), Ytr[:BATCH_SIZE]).item()
model[3].reset_running_stats()  # quel forward ha mosso la media mobile di 1/1000: la rimettiamo a nuovo
print(f"  loss al passo 0: {loss_zero:.4f}   (-log(1/27) = {math.log(VOCAB):.4f})\n")

print("  " + str(model).replace("\n", "\n  "))
# 12097 parametri e 401 buffer: running_mean, running_var e num_batches_tracked,
# un contatore che serve solo se si usa momentum=None (media cumulativa esatta
# invece che mobile). Sono i due tensori che tenevamo di lato con il prefisso
# "running_", più uno che non ci era servito.
print(f"\n  parametri: {sum(p.numel() for p in model.parameters())}", end="")
print(f"  (+ {sum(b.numel() for b in model.buffers())} buffer, che non si allenano)")


# ---------------------------------------------------------------------------
# 2. loss, optimizer, scheduler, dati
# ---------------------------------------------------------------------------
#
# Quattro oggetti che nella versione a mano erano quattro righe sparse dentro
# il loop. Messi in fila, ognuno ha un compito solo e non invade quello degli
# altri — ed è questa divisione dei ruoli, più che il codice risparmiato, il
# motivo per cui il training loop poi diventa uguale in tutti i progetti:
#
#   loader      decide QUALI esempi il modello vede a questo passo
#   model       trasforma quel batch in logits
#   loss_fn     trasforma logits + risposte giuste in un numero solo
#   .backward() riempie p.grad per ogni parametro
#   optimizer   legge p.grad e scrive p          <- l'unico che tocca i pesi
#   scheduler   cambia il learning rate che l'optimizer userà al passo dopo

# La versione classe di F.cross_entropy. Identica: si usa l'una o l'altra per
# gusto. Vuole i logits, mai le probabilità (nota 1 in fondo).
#
# Il suo compito è ridurre i (32, 27) logits e le 32 risposte giuste a un
# singolo scalare, perché è da uno scalare che backward() sa partire: la
# derivata di un vettore rispetto ai pesi sarebbe una matrice, quella di un
# numero è un gradiente per parametro. È il collo di bottiglia da cui passa
# tutto il training.
loss_fn = nn.CrossEntropyLoss()

# `p -= lr * p.grad` per ogni parametro, scritto da qualcun altro. Il primo
# argomento è l'elenco dei parametri, che il modello sa già quali sono: al
# posto del nostro trainable_params(), che doveva filtrare a mano i buffer.
#
# Il suo compito è possedere la regola di aggiornamento e i pesi su cui
# applicarla: tiene un riferimento ai tensori (non una copia), e a ogni step()
# legge il `.grad` che backward() ci ha appena scritto dentro e sposta il peso.
# È l'unico oggetto in tutto il file che modifica i parametri — il modello li
# usa e basta, la loss non li vede nemmeno. Cambiare da SGD ad Adam è cambiare
# questa riga e nient'altro, ed è il motivo per cui esiste come oggetto
# separato invece che come due righe dentro il loop.
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# `0.1 if i < steps // 2 else 0.01`, ma dichiarato invece che nascosto dentro
# il loop: a metà training moltiplica il learning rate per 0.1.
#
# Il suo compito è cambiare un numero nel tempo, e quel numero è dentro
# l'optimizer: non tocca né i pesi né i gradienti, riscrive soltanto
# optimizer.param_groups[0]["lr"]. Sta un piano sopra l'optimizer, che a sua
# volta sta un piano sopra i pesi. È per questo che va costruito passandogli
# l'optimizer, e che il suo step() è una cosa diversa da quello dell'optimizer:
# uno scandisce le epoche, l'altro i batch.
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[EPOCHS // 2], gamma=0.1)

# TensorDataset è il compito più piccolo dei quattro: dice cos'è "un esempio" e
# come si prende l'i-esimo. È solo una coppia di tensori affiancati con un
# __len__ e un __getitem__ che restituisce (Xtr[i], Ytr[i]) — non sa niente di
# batch, di ordine, di training. Su dati veri è qui che si mette la lettura da
# disco o l'augmentation, e tutto il resto del file non se ne accorge.
#
# DataLoader ci sta sopra e decide l'unica cosa che restava da decidere: in che
# ordine, quanti alla volta, e come impilarli in un tensore. Al posto di
# `ix = torch.randint(0, Xtr.shape[0], (32,))`. Non è la stessa cosa: randint
# pesca con reimmissione, il DataLoader mescola una volta e poi scorre, quindi
# ogni esempio compare esattamente una volta per epoca. Con 180k esempi la
# differenza non si vede, ma è quella che rende "epoca" una parola con un
# significato — e che rende `for xb, yb in loader` un loop che finisce.
loader = DataLoader(
    TensorDataset(Xtr, Ytr),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,  # l'ultimo batch spaiato è più corto, e la batchnorm preferisce di no
    generator=torch.Generator().manual_seed(SEED),
)


# ---------------------------------------------------------------------------
# 3. il training loop
# ---------------------------------------------------------------------------

print(f"\n=== training: {EPOCHS} epoche da {len(loader)} passi (un paio di minuti) ===\n")

# model.train() non calcola niente e non tocca nessun tensore: gira un flag,
# ricorsivamente, su tutti i sottomoduli.
#
#     def train(self, mode=True):          # nn.Module, in pseudo-codice
#         self.training = mode
#         for child in self.children():
#             child.train(mode)
#         return self
#
# Poi ogni modulo decide da solo cosa farsene di quel flag. nn.Embedding,
# nn.Linear e nn.Tanh lo ignorano — calcolano la stessa cosa in ogni caso.
# nn.BatchNorm1d invece:
#
#     def forward(self, x):                # BatchNorm1d, in pseudo-codice
#         if self.training:
#             mean, var = x.mean(0), x.var(0)        # usa le statistiche di QUESTO batch
#             self.running_mean = (1 - m) * self.running_mean + m * mean
#             self.running_var  = (1 - m) * self.running_var  + m * var
#         else:
#             mean, var = self.running_mean, self.running_var   # usa i buffer, fissi
#         return gain * (x - mean) / sqrt(var + eps) + bias
#

model.train()

for epoch in range(EPOCHS):
    for xb, yb in loader:
        loss = loss_fn(model(xb), yb)

        # Le tre righe che si trovano identiche in qualsiasi progetto PyTorch.
        # set_to_none=True è il nostro `p.grad = None`: liberare il tensore
        # invece di riempirlo di zeri. Dalla 2.0 è il default, lo scriviamo per
        # dire che è una scelta.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # una volta per epoca, non una volta per passo
    scheduler.step()

    if epoch % 6 == 0 or epoch == EPOCHS - 1:
        lr = optimizer.param_groups[0]["lr"]
        print(f"    epoca {epoch:2d}   loss (ultimo batch) {loss.item():.4f}   lr {lr:g}")


# ---------------------------------------------------------------------------
# 4. valutazione
# ---------------------------------------------------------------------------

print("\n=== valutazione ===\n")

# model.eval() è model.train(False): stesso flag.
#
# Tre conseguenze:
#
# 1. la rete torna a essere una funzione di UN esempio. In train() la
#    previsione su un esempio dipendeva dagli altri 31 capitati nel batch (è
#    03_out_bn_jitter.png); qui no. 
#
# 2. durante il training il layer di normalizzazione effettivamente normalizza
#    le attivazioni di quel batch. Ma durante .eval() invece utilizza le mean/std 
#    accumulate durante il training (buffer). Sono costanti ormai, non viene fatta nessuna 
#    analisi di distribuzione sugli esempi/singolo esempio. Quindi e' solo una 
#    moltiplicazione/somma fissa che ci portiamo dietro.
#
# E non è la stessa cosa del torch.no_grad() qui sotto, anche se si scrivono
# quasi sempre insieme. Sono due interruttori indipendenti:
#
#     model.eval()     ->  QUALI statistiche usa la batchnorm (e spegne il dropout)
#     torch.no_grad()  ->  SE costruire il grafo per il backward
#
# Dentro un no_grad con il modello ancora in train(), la batchnorm continua a
# usare le statistiche del batch e continua ad aggiornare la media mobile: il
# no_grad risparmia memoria, non cambia cosa calcola la rete.
model.eval()


@torch.no_grad()
def evaluate(X, Y):
    return loss_fn(model(X), Y).item()


print(f"  train: {evaluate(Xtr, Ytr):.4f}")
print(f"  dev:   {evaluate(Xdev, Ydev):.4f}")
print(f"  test:  {evaluate(Xte, Yte):.4f}")
print("\n  (03_batchnorm.py, la stessa rete a mano: dev 2.1095)")


# ---------------------------------------------------------------------------
# 5. campionamento
# ---------------------------------------------------------------------------

print("\n=== nomi campionati ===\n")


@torch.no_grad()
def sample(num=10):
    g = torch.Generator().manual_seed(SEED)
    names = []
    for _ in range(num):
        out, context = [], [0] * BLOCK_SIZE
        while True:
            # un esempio solo: (1, 3). In model.train() questo esploderebbe —
            # la deviazione standard di un batch da uno non esiste — ed è il
            # motivo per cui i buffer della batchnorm devono esistere.
            logits = model(torch.tensor([context]))
            ix = torch.multinomial(logits.softmax(1), num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            if ix == 0:
                break
            out.append(itos[ix])
        names.append("".join(out))
    return names


print("  " + "  ".join(sample()))


# ---------------------------------------------------------------------------
# cosa cambia rispetto alla versione a mano (non è solo cosmetica)
# ---------------------------------------------------------------------------
#
# 1. nn.CrossEntropyLoss non fa davvero exp -> /sum -> log. Con logits grandi
#    exp va a inf e la loss diventa nan; internamente sottrae il massimo (la
#    softmax non cambia se trasli tutti i logits della stessa costante) e usa
#    log-sum-exp. È l'unico motivo per cui non le si passano mai le
#    probabilità già calcolate.
#
# 2. model.train() / model.eval() non riguardano solo la batchnorm: sono un
#    flag che attraversa tutto il modello, e ogni modulo decide cosa farne. Il
#    dropout, per dirne un altro, si spegne in eval. La nostra versione a mano
#    passava stats="batch"/"running" come argomento a forward(), che è la
#    stessa cosa scritta peggio: bisogna ricordarsi di propagarla a mano
#    attraverso ogni funzione della catena.
#
# 3. I buffer non sono parametri, ma vanno salvati lo stesso. model.state_dict()
#    contiene sia parameters che buffers, model.parameters() solo i primi — ed
#    è per questo che l'optimizer non tocca la media mobile della batchnorm.
#    Nella versione a mano quella distinzione era il prefisso "running_" nelle
#    chiavi del dizionario, e una funzione che filtrava.
#
#    E l'aggiornamento della media mobile non entra nel grafo, senza che tu
#    debba fare niente: i buffer nascono con requires_grad=False, e
#    nn.BatchNorm1d li aggiorna in place dentro il kernel, sotto lo strato di
#    autograd. Dopo un forward in training, running_mean ha grad_fn None ed è
#    ancora una foglia.
#
#    In 03_batchnorm.py quella stessa riga andava protetta a mano:
#
#        with torch.no_grad():
#            net["running_mean"] = (1 - MOMENTUM) * net["running_mean"] + MOMENTUM * mean
#
#    perché lì il buffer veniva ricombinato con `mean`, che nel grafo c'è
#    eccome — arriva da hpreact. Senza il no_grad ogni passo si sarebbe portato
#    dietro il grafo del passo precedente, che è una perdita di memoria che
#    cresce fino al crash.
#
#    PyTorch è anche più severo di così: quei due argomenti sono dichiarati non
#    differenziabili, e se provi a metterci requires_grad=True si rifiuta di
#    procedere invece di ignorarti in silenzio.
#
#        RuntimeError: The function 'native_batch_norm' is not differentiable
#        with respect to argument 'running_mean'.
#
# 4. nn.Embedding non è solo comodo: il suo backward è sparso (scatter-add
#    sulle righe usate) invece di una matmul contro una matrice one-hot di
#    quasi tutti zeri. Con 27 caratteri non si nota; su vocabolari da decine
#    di migliaia di token è la differenza fra fattibile e non fattibile.
#
# 5. Il DataLoader qui sembra un giro lungo per fare una cosa che randint
#    faceva in una riga, e su un dataset che sta tutto in RAM lo è. Serve
#    quando i dati non ci stanno: num_workers carica i batch in processi
#    separati mentre la GPU lavora, pin_memory prepara i trasferimenti, e il
#    Dataset può leggere da disco o applicare augmentation. L'interfaccia è la
#    stessa, il che è tutto il punto.
#
# 6. Restano scritte a mano, qui come in qualsiasi progetto vero, le due righe
#    di inizializzazione. PyTorch ha un default ragionevole ma generico, e la
#    scala giusta dipende dalla non-linearità che viene dopo e da come deve
#    partire l'ultimo layer: sono decisioni del modello, non della libreria.
#    L'unica alternativa a saperle è la batchnorm, che è esattamente il motivo
#    per cui esiste.
