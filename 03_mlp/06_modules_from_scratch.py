"""
Il giro di ritorno: ricostruiamo le interfacce di PyTorch, invece di usarle.

`04_mlp_idiomatic.py` va da sinistra a destra — il codice sparso delle tappe
01-03 diventa `nn.Sequential`, `nn.BatchNorm1d`, `torch.optim.SGD`. Questo file
va da destra a sinistra: prende quelle stesse interfacce e le riscrive con il
codice sparso che avevamo già, dentro classi che espongono i metodi giusti.

È il bonus con cui si chiude la lezione 3 (`02_out_lecture.srt`, dal minuto 1:18):
"I would like us to start by torchifying our code a little bit so it looks much
more like what you would encounter in PyTorch". Il senso non è riscrivere
PyTorch — è che dopo aver visto entrambe le direzioni non resta niente di
opaco: `nn.Linear` è quattro righe, `nn.BatchNorm1d` è dodici, `optim.SGD` è
sei, e sono le stesse quattro/dodici/sei che avevamo già scritto sparse.

Quello che ricostruiamo, e che sta tutto nella prima metà del file:

    nn.Module          il pezzo che tiene insieme gli altri: parameters(),
                       train()/eval() ricorsivi, la chiamata come funzione
    nn.Linear          x @ W (+ b)
    nn.BatchNorm1d     due parametri, due buffer, due rami
    nn.Tanh            una riga
    nn.Embedding       una tabella e un'indicizzazione
    nn.Flatten         una view
    nn.Sequential      un for
    F.cross_entropy    log-sum-exp, che è l'unico pezzo con un'idea dentro
    optim.SGD          p -= lr * p.grad, e il possesso dei parametri
    MultiStepLR        un numero che cambia dentro l'optimizer
    DataLoader         una permutazione tagliata a fette

Poi la controprova, che è il motivo per cui vale la pena scriverlo: costruiamo
la stessa rete due volte, una con i moduli di PyTorch e una con i nostri, ci
copiamo dentro gli stessi pesi, e verifichiamo che i due output coincidano
numero per numero. Se coincidono, le nostre dodici righe *sono* nn.BatchNorm1d.

Infine alleniamo la nostra e controlliamo di arrivare allo stesso dev loss.
"""

import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

SEED = 2147483647
HERE = Path(__file__).parent
NAMES = HERE.parent / "02_bigram" / "names.txt"

BLOCK_SIZE = 3
EMB_DIM = 10
HIDDEN = 200
BATCH_SIZE = 32
FAN_IN = BLOCK_SIZE * EMB_DIM

EPOCHS = 36


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


# ---------------------------------------------------------------------------
# 1. la classe base: nn.Module
# ---------------------------------------------------------------------------


class Module:
    """Quello che nn.Module fa e che nessun layer singolo potrebbe fare.

    Preso da solo, un layer sa calcolare la sua roba. Quello che serve in più è
    tutto di coordinamento, e sono le tre cose qui sotto:

      parameters()   raccogliere i tensori da allenare risalendo l'albero dei
                     moduli, così che l'optimizer ne riceva uno solo
      train()/eval() propagare un flag verso il basso, così che si imposti da
                     un punto solo invece che layer per layer
      __call__       chiamare il modulo come una funzione

    Nel resto della lezione questa classe non c'è: Karpathy scrive parameters()
    su ogni layer e mette `.training` a mano dove serve. La scriviamo perché è
    esattamente il punto in cui nn.Module smette di essere magia.
    """

    training = True

    def __call__(self, x):
        raise NotImplementedError

    def parameters(self):
        """I tensori che l'optimizer deve aggiornare. NON i buffer."""
        return []

    def children(self):
        """I sottomoduli, per la ricorsione di train()."""
        return []

    def train(self, mode=True):
        # Il vero nn.Module fa esattamente questo, riga per riga.
        self.training = mode
        for child in self.children():
            child.train(mode)
        return self

    def eval(self):
        return self.train(False)


# ---------------------------------------------------------------------------
# 2. i layer
# ---------------------------------------------------------------------------


class Linear(Module):
    """nn.Linear.

    Due differenze di interfaccia con quello vero, tutte e due visibili:

    1. la nostra `weight` ha forma (fan_in, fan_out) e si usa come `x @ W`.
       PyTorch la tiene trasposta, (fan_out, fan_in), e calcola `x @ W.T`. È
       una convenzione ereditata dalle BLAS, e si paga solo quando si copiano
       i pesi da una parte all'altra — cosa che facciamo nella sezione 4.

    2. l'inizializzazione di default. Qui `randn / sqrt(fan_in)`, che è la
       formula di 02_optimizations.py senza gain. PyTorch usa un'uniforme in
       ±1/sqrt(fan_in), che ha deviazione standard 1/sqrt(3*fan_in), tre volte
       più stretta (il perché sta in 04_mlp_idiomatic.py).
    """

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out = self.out + self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d(Module):
    """nn.BatchNorm1d, cioè le sezioni 2-7 di 03_batchnorm.py dentro una classe.

    Qui si vede a occhio la distinzione che nella versione a mano era il
    prefisso "running_" nelle chiavi del dizionario:

        gamma, beta                  parametri  -> li restituisce parameters()
        running_mean, running_var    buffer     -> non li restituisce nessuno

    I buffer non ricevono gradiente e l'optimizer non li vede mai, ma sono
    indispensabili all'inferenza. È il motivo per cui in PyTorch state_dict()
    contiene entrambi mentre parameters() contiene solo i primi.
    """

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        # parametri: partono neutri, gain 1 e traslazione 0
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffer: la stima che useremo all'inferenza
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            mean = x.mean(0)
            # Dettaglio che PyTorch documenta e quasi nessuno legge: per
            # normalizzare usa la varianza *biased* (divide per N), per
            # aggiornare il buffer usa quella *unbiased* (divide per N-1). Con
            # N=32 sono lo stesso numero a meno del 3%, ma se vogliamo che la
            # controprova della sezione 4 torni al settimo decimale va scritto
            # come lo scrivono loro.
            var = x.var(0, unbiased=False)
        else:
            mean, var = self.running_mean, self.running_var

        # niente keepdim: mean e var hanno forma (dim,) e il broadcasting le
        # allinea a destra contro (N, dim) da solo. È anche la forma con cui
        # PyTorch tiene i buffer.
        self.out = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

        if self.training:
            # Il no_grad è obbligatorio, non un'ottimizzazione: senza, il nuovo
            # running_mean porterebbe un grad_fn e ogni passo si trascinerebbe
            # dietro il grafo del passo precedente. nn.BatchNorm1d non ha
            # bisogno di scriverlo perché aggiorna i buffer dentro il kernel,
            # sotto lo strato di autograd.
            with torch.no_grad():
                var_unbiased = x.var(0, unbiased=True)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_unbiased

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh(Module):
    """nn.Tanh. Nessun parametro, nessuno stato, ignora il flag training."""

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out


class Embedding(Module):
    """nn.Embedding: una tabella e un'indicizzazione, cioè il nostro C[X].

    Unica differenza vera con quello di PyTorch: il backward. Il nostro passa
    dall'autograd generico dell'indicizzazione, il loro è uno scatter-add
    sparso sulle sole righe usate. Con 27 caratteri non si nota, su un
    vocabolario da 50k token è la differenza fra fattibile e non fattibile.
    """

    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, ix):
        self.out = self.weight[ix]
        return self.out

    def parameters(self):
        return [self.weight]


class Flatten(Module):
    """nn.Flatten: la view che concatena i BLOCK_SIZE embedding, (N,3,10)->(N,30).

    Parte dall'asse 1, cioè non tocca mai l'asse degli esempi.
    """

    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out


class Sequential(Module):
    """nn.Sequential: un for, più la raccolta ricorsiva di parametri e figli.

    Le due righe di parameters() e children() sono quello che fa funzionare
    tutto il resto: chiamare parameters() sul contenitore raggiunge ogni
    tensore di ogni layer, e chiamare eval() sul contenitore raggiunge ogni
    flag. È l'unica cosa che questa classe aggiunge al `for`.
    """

    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def children(self):
        return self.layers

    def __getitem__(self, i):
        return self.layers[i]


# ---------------------------------------------------------------------------
# 3. la loss, l'optimizer, lo scheduler, i dati
# ---------------------------------------------------------------------------


def cross_entropy(logits, targets):
    """F.cross_entropy.

    Le tre righe sembrano una riscrittura letterale di exp / normalizza / log,
    e non lo sono: la prima è quella che rende la funzione utilizzabile. Con
    logits grandi `exp` va a `inf` e la loss diventa `nan`; sottraendo prima il
    massimo di ogni riga i numeri restano piccoli, e la softmax non cambia
    perché traslare tutti i logits della stessa costante non la altera.

    È l'unico motivo per cui a cross_entropy non si passano mai le probabilità
    già calcolate: dentro non fa quello che sembra.
    """
    logits = logits - logits.max(1, keepdim=True).values
    logprobs = logits - logits.exp().sum(1, keepdim=True).log()
    return -logprobs[torch.arange(len(targets)), targets].mean()


class SGD:
    """torch.optim.SGD.

    Il metodo interessante è __init__: tiene un *riferimento* ai tensori, non
    una copia. È per questo che step() può modificarli e il modello se ne
    accorge, pur non essendoci nessun collegamento fra i due oggetti.

    param_groups è una lista invece che un dizionario solo perché PyTorch
    permette learning rate diversi per gruppi diversi di parametri; qui il
    gruppo è uno, ma teniamo la struttura perché è quella che lo scheduler
    andrà a modificare.
    """

    def __init__(self, params, lr):
        self.param_groups = [{"params": list(params), "lr": lr}]

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for p in group["params"]:
                if set_to_none:
                    p.grad = None  # libera il tensore
                elif p.grad is not None:
                    p.grad.zero_()  # lo tiene e lo riempie di zeri

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p -= group["lr"] * p.grad


class MultiStepLR:
    """torch.optim.lr_scheduler.MultiStepLR.

    Non tocca né i pesi né i gradienti: riscrive un numero dentro l'optimizer.
    Sta un piano sopra l'optimizer, che sta un piano sopra i pesi — e si vede
    dal fatto che l'unica cosa che riceve nel costruttore è l'optimizer.
    """

    def __init__(self, optimizer, milestones, gamma=0.1):
        self.optimizer = optimizer
        self.milestones = set(milestones)
        self.gamma = gamma
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        if self.last_epoch in self.milestones:
            for group in self.optimizer.param_groups:
                group["lr"] *= self.gamma


class DataLoader:
    """TensorDataset + DataLoader, che qui collassano in una cosa sola.

    PyTorch li tiene separati perché hanno due compiti diversi: il Dataset dice
    cos'è un esempio e come si prende l'i-esimo (ed è lì che su dati veri si
    mette la lettura da disco o l'augmentation), il DataLoader decide ordine,
    dimensione del batch e come impilare. Con due tensori già in RAM la
    separazione non compra niente, quindi la saltiamo.

    Il cuore è randperm: una permutazione tagliata a fette. Ed è la differenza
    con `torch.randint` delle tappe 01-03, che pescava con reimmissione — è
    questa riga che rende "epoca" una parola con un significato.
    """

    def __init__(self, X, Y, batch_size, shuffle=True, drop_last=True, generator=None):
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

    def __iter__(self):
        n = len(self.X)
        order = torch.randperm(n, generator=self.generator) if self.shuffle else torch.arange(n)
        stop = n - self.batch_size + 1 if self.drop_last else n
        for start in range(0, stop, self.batch_size):
            ix = order[start : start + self.batch_size]
            yield self.X[ix], self.Y[ix]

    def __len__(self):
        div, rem = divmod(len(self.X), self.batch_size)
        return div if self.drop_last or rem == 0 else div + 1


# ---------------------------------------------------------------------------
# 4. la controprova: le nostre classi contro quelle vere
# ---------------------------------------------------------------------------

print("=== controprova: i nostri moduli contro nn.* ===\n")

torch.manual_seed(SEED)

# La rete di riferimento è esattamente quella di 04_mlp_idiomatic.py.
ref = nn.Sequential(
    nn.Embedding(VOCAB, EMB_DIM),
    nn.Flatten(),
    nn.Linear(FAN_IN, HIDDEN, bias=False),
    nn.BatchNorm1d(HIDDEN, momentum=0.001),
    nn.Tanh(),
    nn.Linear(HIDDEN, VOCAB),
)
nn.init.kaiming_normal_(ref[2].weight, nonlinearity="tanh")
nn.init.normal_(ref[5].weight, std=0.01)
nn.init.zeros_(ref[5].bias)

# La nostra, con la stessa struttura.
model = Sequential(
    Embedding(VOCAB, EMB_DIM),
    Flatten(),
    Linear(FAN_IN, HIDDEN, bias=False),
    BatchNorm1d(HIDDEN, momentum=0.001),
    Tanh(),
    Linear(HIDDEN, VOCAB),
)

# Ci copiamo dentro gli stessi identici numeri. Le due .T sono la differenza di
# convenzione di Linear, i due nomi diversi (weight/bias contro gamma/beta)
# sono l'unico posto in cui PyTorch non segue la notazione del paper.
with torch.no_grad():
    model[0].weight.copy_(ref[0].weight)
    model[2].weight.copy_(ref[2].weight.T)
    model[3].gamma.copy_(ref[3].weight)
    model[3].beta.copy_(ref[3].bias)
    model[5].weight.copy_(ref[5].weight.T)
    model[5].bias.copy_(ref[5].bias)

batch = Xtr[:BATCH_SIZE]

ref.train(), model.train()
with torch.no_grad():
    diff_train = (ref(batch) - model(batch)).abs().max().item()
    diff_buffer = (ref[3].running_mean - model[3].running_mean).abs().max().item()

ref.eval(), model.eval()
with torch.no_grad():
    diff_eval = (ref(batch) - model(batch)).abs().max().item()
    diff_one = (ref(batch[:1]) - model(batch[:1])).abs().max().item()

logits = ref(batch)
diff_loss = abs(cross_entropy(logits, Ytr[:BATCH_SIZE]).item()
                - F.cross_entropy(logits, Ytr[:BATCH_SIZE]).item())

print(f"  scarto massimo sui logits, in train():        {diff_train:.2e}")
print(f"  scarto massimo su running_mean dopo il passo: {diff_buffer:.2e}")
print(f"  scarto massimo sui logits, in eval():         {diff_eval:.2e}")
print(f"  idem su un esempio solo (batch da 1):         {diff_one:.2e}")
print(f"  scarto sulla loss, nostra contro F.cross_entropy: {diff_loss:.2e}")
print(f"\n  parametri: {sum(p.numel() for p in model.parameters())}"
      f" (nn.*: {sum(p.numel() for p in ref.parameters())})")
print(
    """
  Zero a meno dell'errore in virgola mobile, in tutte le righe. Le nostre
  dodici righe di BatchNorm1d non assomigliano a nn.BatchNorm1d: sono
  nn.BatchNorm1d. E il conto dei parametri torna perché in entrambi i casi i
  buffer restano fuori."""
)

# La controprova ha fatto due forward in training, quindi ha mosso i buffer di
# due millesimi. Li rimettiamo a nuovo prima di allenare davvero — è il nostro
# reset_running_stats().
model[3].running_mean.zero_()
model[3].running_var.fill_(1.0)


# ---------------------------------------------------------------------------
# 5. training, con i nostri oggetti e nient'altro
# ---------------------------------------------------------------------------

# nn.Parameter fa due cose che qui facciamo a mano: accende requires_grad, e
# fa sapere a nn.Module che quel tensore è un parametro. La seconda è il vero
# motivo per cui esiste — è come parameters() sa cosa raccogliere senza che
# tu glielo dica. Noi lo diciamo, scrivendo parameters() su ogni classe.
for p in model.parameters():
    p.requires_grad_(True)

optimizer = SGD(model.parameters(), lr=0.1)
scheduler = MultiStepLR(optimizer, milestones=[EPOCHS // 2], gamma=0.1)
loader = DataLoader(
    Xtr, Ytr, batch_size=BATCH_SIZE, drop_last=True,
    generator=torch.Generator().manual_seed(SEED),
)

print(f"\n=== training: {EPOCHS} epoche da {len(loader)} passi (un paio di minuti) ===\n")

# Da qui in giù il codice è identico a quello di 04_mlp_idiomatic.py, carattere
# per carattere. È l'unico modo di dimostrare che le interfacce sono le stesse:
# non che il risultato coincide, ma che il codice che le usa non cambia.
model.train()

for epoch in range(EPOCHS):
    for xb, yb in loader:
        loss = cross_entropy(model(xb), yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    scheduler.step()

    if epoch % 6 == 0 or epoch == EPOCHS - 1:
        lr = optimizer.param_groups[0]["lr"]
        print(f"    epoca {epoch:2d}   loss (ultimo batch) {loss.item():.4f}   lr {lr:g}")


# ---------------------------------------------------------------------------
# 6. valutazione e campionamento
# ---------------------------------------------------------------------------

print("\n=== valutazione ===\n")

model.eval()


@torch.no_grad()
def evaluate(X, Y):
    return cross_entropy(model(X), Y).item()


print(f"  train: {evaluate(Xtr, Ytr):.4f}")
print(f"  dev:   {evaluate(Xdev, Ydev):.4f}")
print(f"  test:  {evaluate(Xte, Yte):.4f}")
print("\n  (04_mlp_idiomatic.py, con i moduli veri: dev 2.1112)")


@torch.no_grad()
def sample(num=10):
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


print("\n=== nomi campionati ===\n")
print("  " + "  ".join(sample()))


# ---------------------------------------------------------------------------
# cosa manca, rispetto a nn.Module vero
# ---------------------------------------------------------------------------
#
# Le nostre classi calcolano la stessa cosa, ma sono un giocattolo. Quello che
# nn.Module ha in più non riguarda il calcolo, riguarda tutto il resto:
#
# 1. La registrazione automatica. nn.Module intercetta __setattr__: quando
#    assegni un nn.Parameter o un nn.Module a un attributo, lui lo annota in un
#    registro interno. È per questo che il vero parameters() funziona senza che
#    tu scriva niente, mentre noi abbiamo dovuto scriverlo su ogni classe — e
#    perché nn.Parameter esiste come tipo separato invece di essere un tensore
#    con requires_grad=True.
#
# 2. state_dict() e load_state_dict(). Salvare e ricaricare un modello, con i
#    nomi giusti ("layers.3.running_mean"), parametri e buffer insieme. È la
#    ragione pratica per cui la distinzione parametro/buffer deve stare nel
#    framework e non nella tua testa: le due categorie si allenano
#    diversamente ma si salvano insieme.
#
# 3. .to(device) e .to(dtype). Spostare ricorsivamente tutto su GPU o in
#    float16. Con i nostri tensori sparsi bisognerebbe farlo a mano, uno per
#    uno, ricordandosi anche dei buffer.
#
# 4. Gli hook. forward_hook, backward_hook, e tutta la strumentazione che
#    permette di guardare dentro una rete senza modificarla. Il nostro
#    equivalente è `self.out`, che è comodo per fare istogrammi ma è anche una
#    perdita di memoria: tiene vivo l'output di ogni layer fino al forward
#    dopo. Il vero nn.Module non ha un .out, ed è un bene.
#
# 5. __repr__, il conteggio dei sottomoduli, la gestione dei nomi annidati, e
#    le duecento righe di casi limite che rendono una cosa che funziona una
#    cosa su cui puoi costruire.
#
# Ma il calcolo — la parte che si può sbagliare in silenzio, quella che decide
# se la rete impara o no — è quello che c'è qui sopra, e ci sta in mezza
# pagina.
