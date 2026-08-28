"""
makemore, parte 3 (seconda metà): la batch normalization.

`02_optimizations.py` si chiude con una formula: `std = gain / sqrt(fan_in)`. Con
quella le pre-attivazioni escono più o meno gaussiane, la tanh non satura, e la
loss scende meglio. Ma quella formula l'abbiamo ricavata per *un* layer lineare
con in ingresso una gaussiana. Cambia la funzione di attivazione, aggiungi una
connessione residua, metti cinquanta layer di tipi diversi uno sopra l'altro, e
non c'è più nessun conto da fare a mano: azzeccare la scala di ogni matrice di
pesi perché le attivazioni restino ragionevoli ovunque diventa impraticabile.

Nel 2015 un gruppo di Google propone una scorciatoia che sembra barare: se
vogliamo che le pre-attivazioni siano gaussiane, invece di sperarci,
*normalizziamole*. Sottraiamo la media e dividiamo per la deviazione standard,
lì dentro, a ogni passo. Si può fare, perché media e deviazione standard sono
formule differenziabili come tutto il resto, quindi la backpropagation ci passa
attraverso senza accorgersene.

Il nome viene da come si calcolano quella media e quella deviazione standard:
sul batch. Ed è anche da lì che vengono tutti i guai, che sono la parte
interessante di questo file.

La rete è sempre quella: stesso dataset, stessi iperparametri, stesso seed. Il
punto di partenza è l'ultima riga di `02_optimizations.py`, l'inizializzazione di
Kaiming, che sul dev set fa 2.1070.
"""

import random
from pathlib import Path

import torch
from torch.nn import functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 2147483647
HERE = Path(__file__).parent
NAMES = HERE.parent / "02_bigram" / "names.txt"

BLOCK_SIZE = 3
EMB_DIM = 10
HIDDEN = 200
BATCH_SIZE = 32
FAN_IN = BLOCK_SIZE * EMB_DIM

EPS = 1e-5  # sezione 7
MOMENTUM = 0.001  # sezione 6


# ---------------------------------------------------------------------------
# 0. dataset e rete: gli stessi di 02_optimizations.py
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


def make_net(batchnorm=False, with_b1=True):
    """I parametri, stavolta in un dizionario invece che in una lista.

    Serve perché da qui in poi l'insieme dei parametri cambia: con la batchnorm
    arrivano bngain e bnbias, e b1 se ne va (sezione 8). I randn vengono estratti
    sempre nello stesso ordine, così le reti che confrontiamo partono davvero
    dagli stessi numeri.

    L'inizializzazione è quella con cui si chiude 02_optimizations.py: Kaiming su
    W1, W2 rimpicciolita e b2 azzerato perché i logits partano piccoli.
    """
    g = torch.Generator().manual_seed(SEED)
    C = torch.randn((VOCAB, EMB_DIM), generator=g)
    W1 = torch.randn((FAN_IN, HIDDEN), generator=g) * (5 / 3) / FAN_IN**0.5
    b1 = torch.randn(HIDDEN, generator=g) * 0.01
    W2 = torch.randn((HIDDEN, VOCAB), generator=g) * 0.01
    b2 = torch.randn(VOCAB, generator=g) * 0.0

    net = {"C": C, "W1": W1, "W2": W2, "b2": b2}
    if with_b1:
        net["b1"] = b1
    if batchnorm:
        # il gain a 1 e il bias a 0: all'inizio la batchnorm non sposta niente,
        # lascia esattamente la gaussiana che ha appena costruito
        net["bngain"] = torch.ones((1, HIDDEN))
        net["bnbias"] = torch.zeros((1, HIDDEN))
    for p in net.values():
        p.requires_grad = True

    if batchnorm:
        # questi due invece NON sono parametri: sono buffer. Non ricevono
        # gradiente e non vengono aggiornati dalla discesa del gradiente, li
        # teniamo aggiornati a mano di lato (sezione 6)
        net["running_mean"] = torch.zeros((1, HIDDEN))
        net["running_std"] = torch.ones((1, HIDDEN))
    return net


def trainable_params(net):
    return [v for k, v in net.items() if not k.startswith("running_")]


def linear_output(net, X):
    """Il primo layer e basta: embedding, concatenazione, matrice, bias."""
    emb = net["C"][X]
    hpreact = emb.view(emb.shape[0], -1) @ net["W1"]
    if "b1" in net:
        hpreact = hpreact + net["b1"]
    return hpreact


def forward(net, X, stats="batch", update_running=False):
    """Come sempre, con la batchnorm infilata fra il layer lineare e la tanh.

    `stats` dice da dove prendere media e deviazione standard: dal batch che
    stiamo processando (durante il training) o dai buffer (all'inferenza).
    `update_running` è quello che tiene aggiornati i buffer, e lo usa solo train().
    """
    hpreact = linear_output(net, X)

    if "bngain" in net:
        if stats == "batch":
            mean = hpreact.mean(0, keepdim=True)
            std = hpreact.std(0, keepdim=True)
            if update_running:
                with torch.no_grad():
                    net["running_mean"] = (1 - MOMENTUM) * net["running_mean"] + MOMENTUM * mean
                    net["running_std"] = (1 - MOMENTUM) * net["running_std"] + MOMENTUM * std
        else:
            mean, std = net["running_mean"], net["running_std"]
        hpreact = net["bngain"] * (hpreact - mean) / (std + EPS) + net["bnbias"]

    h = torch.tanh(hpreact)
    return h @ net["W2"] + net["b2"], hpreact, h


@torch.no_grad()
def evaluate(net, X, Y, stats="running"):
    return F.cross_entropy(forward(net, X, stats)[0], Y).item()


def train(net, steps=200_000):
    params = trainable_params(net)
    g = torch.Generator().manual_seed(SEED)
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
        loss = F.cross_entropy(forward(net, Xtr[ix], "batch", update_running=True)[0], Ytr[ix])
        for p in params:
            p.grad = None
        loss.backward()
        with torch.no_grad():
            for p in params:
                p -= (0.1 if i < steps // 2 else 0.01) * p.grad


# ---------------------------------------------------------------------------
# 1. il punto di partenza: quanto sono gaussiane le pre-attivazioni?
# ---------------------------------------------------------------------------

print("=== 1. le pre-attivazioni con l'inizializzazione di Kaiming ===\n")

net = make_net()
batch = Xtr[:BATCH_SIZE]
with torch.no_grad():
    hpreact = linear_output(net, batch)

neuron_means = hpreact.mean(0)  # una media per neurone, sui 32 esempi del batch
neuron_stds = hpreact.std(0)

print(f"  hpreact ha forma {tuple(hpreact.shape)}: {BATCH_SIZE} esempi x {HIDDEN} neuroni\n")
print(f"  tutto insieme:        media {hpreact.mean():.2f}, dev.standard {hpreact.std():.2f}")
print(f"  neurone per neurone:  medie da {neuron_means.min():.2f} a {neuron_means.max():.2f},"
      f" dev.standard da {neuron_stds.min():.2f} a {neuron_stds.max():.2f}")
print(f"  già saturi (|tanh| > 0.99): {torch.tanh(hpreact).abs().gt(0.99).float().mean() * 100:.1f}%")

print(
    """
  Kaiming ha fatto il suo lavoro nel senso che conta: siamo a 1.5, non a 15.
  Ma il conto prometteva 1.00, e la differenza non è arrotondamento. Neurone
  per neurone si va da 0.79 a 2.56: c'è chi sta già sul punto di saturare, e
  infatti un'attivazione su tredici è già fuori.

  Il motivo è che la formula `gain / sqrt(fan_in)` viene da un conto sulla
  varianza che assume un ingresso gaussiano con componenti indipendenti. Il
  nostro ingresso non è né l'una né l'altra cosa: sono righe pescate da una
  tabella di embedding, ripetute su tre posizioni, con la stessa lettera che
  ricompare. La formula resta una buona stima — ed è tantissimo, rispetto a
  indovinare — ma è una stima, tarata su un caso ideale, per questo layer in
  questa rete.

  E qui c'è un layer solo e il conto lo sappiamo fare. L'idea del paper del
  2015 è di smettere di sperare: se vogliamo che questi numeri siano gaussiani,
  li rendiamo gaussiani.
"""
)


# ---------------------------------------------------------------------------
# 2. normalizzare a mano
# ---------------------------------------------------------------------------

print("=== 2. la normalizzazione ===\n")

# Le due righe che fanno tutto. mean(0) e std(0) vogliono dire "lungo la
# dimensione degli esempi": per ogni neurone, come si è comportato sui 32
# esempi del batch. keepdim=True tiene la forma (1, 200) invece di (200,), così
# il broadcasting contro (32, 200) è ovvio da leggere.

mean = hpreact.mean(0, keepdim=True)
std = hpreact.std(0, keepdim=True)
hpreact_norm = (hpreact - mean) / std

print(f"  hpreact {tuple(hpreact.shape)}  -  mean {tuple(mean.shape)}  /  std {tuple(std.shape)}\n")
print(f"  prima:  media per neurone da {neuron_means.min():>6.2f} a {neuron_means.max():>5.2f},"
      f" dev.std. da {neuron_stds.min():.2f} a {neuron_stds.max():.2f}")
print(f"  dopo:   media per neurone da {hpreact_norm.mean(0).min():>6.2f} a {hpreact_norm.mean(0).max():>5.2f},"
      f" dev.std. da {hpreact_norm.std(0).min():.2f} a {hpreact_norm.std(0).max():.2f}")

print(
    """
  Adesso ogni neurone ha esattamente media zero e deviazione standard uno su
  questo batch. Non "circa": esattamente, per costruzione.

  La cosa che sembra barare e non lo è: questa operazione sta dentro al grafo
  come tutte le altre. Sottrazione, divisione, media, radice quadrata — sono
  tutte funzioni derivabili, e micrograd ci ha insegnato che se sai derivare i
  pezzi sai derivare la composizione. Quindi la backpropagation ci passa
  attraverso e i pesi a monte continuano a ricevere il loro gradiente.
"""
)

# Controprova, perché è il punto su cui si può restare scettici: mettiamo la
# batchnorm nel forward, chiamiamo backward, e guardiamo se W1 riceve gradiente.
probe = make_net(batchnorm=True)
F.cross_entropy(forward(probe, batch)[0], Ytr[:BATCH_SIZE]).backward()
print(f"  gradiente che arriva a W1 attraverso la batchnorm: norma {probe['W1'].grad.norm():.4f}")
print(f"  gradiente su bngain: norma {probe['bngain'].grad.norm():.4f}\n")


# ---------------------------------------------------------------------------
# 3. gain e bias: non vogliamo *costringerla* a essere gaussiana
# ---------------------------------------------------------------------------

print("=== 3. gain e bias ===\n")

print(
    """  Normalizzare a ogni passo è giusto all'inizio, ma sarebbe una gabbia se
  restasse per sempre. La rete deve poter decidere che un certo neurone va
  tenuto più schiacciato, o più largo, o spostato di lato — magari proprio
  saturo, se saturare è quello che serve per quel neurone. Quello che vogliamo
  è che la distribuzione parta gaussiana, non che ci resti.

  Quindi dopo aver normalizzato rimettiamo due manopole:

      hpreact = bngain * (hpreact - mean) / std + bnbias

  bngain parte a 1 e bnbias a 0, cioè all'inizio non fanno niente e la
  distribuzione è esattamente quella normalizzata. Ma sono parametri come
  tutti gli altri: ricevono gradiente, e la backpropagation decide dove
  spostarli. Sono 400 numeri in più su 11897, e sono il prezzo per non dover
  più indovinare nessuna scala.
"""
)


# ---------------------------------------------------------------------------
# 4. alleniamola
# ---------------------------------------------------------------------------

print("=== 4. il training ===\n")
print("  (200k passi, un paio di minuti)\n")

net_bn = make_net(batchnorm=True)
train(net_bn)

print(f"  train: {evaluate(net_bn, Xtr, Ytr):.4f}")
print(f"  dev:   {evaluate(net_bn, Xdev, Ydev):.4f}")

# I numeri che escono, accanto all'ultima riga di 02_optimizations.py, che è la
# stessa identica rete senza la batchnorm:
#
#     inizializzazione di Kaiming, senza batchnorm      dev  2.1070
#     inizializzazione di Kaiming, con batchnorm        dev  2.1095
#
# Cioè: non cambia niente, anzi perde due millesimi, che è rumore. Ed è
# esattamente quello che dice la lezione, che si ferma sul 2.10 senza esserne
# sorpresa e senza aspettarsi di meglio. Qui c'è un layer nascosto solo, e per
# un layer solo la scala giusta dei pesi la sappiamo calcolare: la batchnorm
# non ha niente da sistemare. Il suo valore non è battere Kaiming su una rete
# con un layer, è rendere inutile Kaiming su una rete con cinquanta.


# ---------------------------------------------------------------------------
# 5. il prezzo: gli esempi del batch non sono più indipendenti
# ---------------------------------------------------------------------------

print("\n=== 5. quello che abbiamo rotto ===\n")

print(
    """  Fino a ieri il batch era solo un fatto di efficienza. Trentadue esempi
  passavano dentro la rete affiancati, ma ognuno per conto suo: i logits del
  primo non sapevano niente del secondo. La rete era una funzione di un
  esempio.

  Da adesso non più. Media e deviazione standard sono calcolate sul batch,
  quindi le pre-attivazioni di un esempio dipendono da chi altro è capitato
  nel batch con lui — e chi capita nel batch è deciso a caso. Lo stesso
  esempio, valutato due volte, dà due risposte diverse.
"""
)

# Misuriamolo: prendiamo un esempio del dev set e infiliamolo in 500 batch
# diversi, ognuno riempito con 31 esempi presi a caso dal training set.
dev_idx = next(i for i in range(len(Xdev)) if (Xdev[i] > 0).all())  # un contesto senza punti
example = Xdev[dev_idx : dev_idx + 1]
target = Ydev[dev_idx].item()
probs = []
g = torch.Generator().manual_seed(SEED)
with torch.no_grad():
    for _ in range(500):
        ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE - 1,), generator=g)
        logits, _, _ = forward(net_bn, torch.cat([example, Xtr[ix]]), "batch")
        probs.append(F.softmax(logits[0], 0)[target].item())

probs = torch.tensor(probs)
p_inference = F.softmax(forward(net_bn, example, "running")[0][0], 0)[target].item()

context = "".join(itos[i] for i in Xdev[dev_idx].tolist())
print(f"  esempio: contesto '{context}' -> lettera giusta '{itos[target]}'\n")
print(f"  probabilità che la rete le assegna, su 500 batch diversi:")
print(f"    da {probs.min():.4f} a {probs.max():.4f}, media {probs.mean():.4f},"
      f" dev.std. {probs.std():.4f}")
print(f"  la stessa cosa in inferenza, con le statistiche fisse: {p_inference:.4f}")

plt.figure(figsize=(9, 4))
plt.hist(probs.tolist(), bins=50)
plt.axvline(p_inference, color="red", label="in inferenza (statistiche fisse)")
plt.title(f"stesso esempio, 500 batch diversi: p('{itos[target]}' | '{context}')")
plt.xlabel("probabilità assegnata alla lettera giusta")
plt.ylabel("quanti batch")
plt.legend()
plt.tight_layout()
plt.savefig(HERE / "03_out_bn_jitter.png", bbox_inches="tight", dpi=100)
plt.close()
print(f"\n  il grafico è in {HERE / '03_out_bn_jitter.png'}")

print(
    """
  Senza batchnorm quell'istogramma sarebbe un bastoncino solo: una previsione,
  sempre la stessa. Con la batchnorm è una distribuzione.

  Sembra un bug, ed è un bug — ma per un effetto collaterale fa bene. Ogni
  volta che la rete vede quell'esempio lo vede leggermente diverso, spostato a
  caso da chi gli sta intorno, e quindi non riesce a impararlo a memoria: è
  rumore aggiunto ai dati, cioè un regolarizzatore, cugino della data
  augmentation. È anche il motivo per cui la batchnorm è così difficile da
  togliere: nessuno vuole esempi accoppiati fra loro, ma quando la si toglie
  spesso peggiora, e ci vuole un po' per capire che manca la regolarizzazione,
  non la normalizzazione.

  Le alternative che sono nate per questo — layer norm, instance norm, group
  norm — normalizzano lungo altre dimensioni e non accoppiano gli esempi. La
  layer normalization è quella che si trova nei transformer.
"""
)


# ---------------------------------------------------------------------------
# 6. e all'inferenza? un esempio solo non ha una media di batch
# ---------------------------------------------------------------------------

print("=== 6. l'inferenza, e i due modi di stimare le statistiche ===\n")

print(
    """  C'è un problema pratico prima ancora che concettuale. In produzione arriva
  un esempio alla volta: qual è la media di un batch da uno? Zero, e la
  deviazione standard non esiste. La rete così non si può usare.

  La proposta del paper è di fissare quei numeri una volta sola, alla fine del
  training, stimandoli su tutto il training set.
"""
)

with torch.no_grad():
    hpreact_train = linear_output(net_bn, Xtr)
    calibrated_mean = hpreact_train.mean(0, keepdim=True)
    calibrated_std = hpreact_train.std(0, keepdim=True)

print(f"  {'':<12} {'primi tre neuroni: media':>34} {'dev.standard':>28}")
print(f"  {'calibrate':<12} {str([round(v, 3) for v in calibrated_mean[0, :3].tolist()]):>34}"
      f" {str([round(v, 3) for v in calibrated_std[0, :3].tolist()]):>28}")
print(f"  {'running':<12} {str([round(v, 3) for v in net_bn['running_mean'][0, :3].tolist()]):>34}"
      f" {str([round(v, 3) for v in net_bn['running_std'][0, :3].tolist()]):>28}")

print(
    """
  Le due righe sono le due strade. La prima è la calibrazione esplicita: un
  secondo giro su tutto il training set, alla fine, per stimare media e
  deviazione standard di ogni neurone. Funziona, ma è una fase in più che
  bisogna ricordarsi di fare, e nessuno se la ricorda.

  La seconda è quella che il paper propone subito dopo, e che è finita in
  tutte le implementazioni: teniamo una stima aggiornata *durante* il training,
  di lato. A ogni passo

      running = (1 - momentum) * running + momentum * batch_stat

  con momentum piccolo (qui 0.001). È una media mobile: ogni batch sposta la
  stima di un millesimo, e dopo 200k passi la stima è quella giusta, senza
  nessuna fase due. Quei due tensori non sono parametri — non ricevono
  gradiente, non compaiono in loss.backward() — sono buffer, e in PyTorch si
  chiamano proprio così.

  Nota il momentum piccolo. Con batch da 32 la media di un batch balla parecchio
  intorno a quella vera, quindi la stima va aggiornata piano. Il default di
  PyTorch è 0.1, che va bene per batch grandi ma con batch piccoli fa oscillare
  la stima senza farla mai assestare.
"""
)

# E le tre valutazioni possibili sul dev set. La prima è sbagliata, e vale la
# pena vederla stampata:
print(f"  dev con le statistiche del dev set stesso: {evaluate(net_bn, Xdev, Ydev, 'batch'):.4f}  <- sbagliata")
print(f"  dev con le statistiche calibrate a mano:   ", end="")
saved_mean, saved_std = net_bn["running_mean"], net_bn["running_std"]
net_bn["running_mean"], net_bn["running_std"] = calibrated_mean, calibrated_std
print(f"{evaluate(net_bn, Xdev, Ydev):.4f}")
net_bn["running_mean"], net_bn["running_std"] = saved_mean, saved_std
print(f"  dev con le statistiche running:            {evaluate(net_bn, Xdev, Ydev):.4f}")

print(
    """
  La prima riga è sbagliata anche se il numero è quasi identico: normalizzare
  il dev set con le statistiche del dev set vuol dire far entrare nella
  previsione di un esempio l'informazione di tutti gli altri esempi di
  valutazione. È una fuga di informazione piccola qui, dove i due insiemi si
  assomigliano molto, ma è della stessa famiglia di errori che rende un
  risultato non riproducibile in produzione. In PyTorch è esattamente ciò che
  distingue model.train() da model.eval().
"""
)


# ---------------------------------------------------------------------------
# 7. l'epsilon
# ---------------------------------------------------------------------------

print("=== 7. l'epsilon ===\n")

# La formula vera non è (x - mean) / std ma (x - mean) / sqrt(var + eps), con
# eps piccolo (il default di PyTorch è 1e-5). Serve solo a non dividere per
# zero: se un neurone in questo batch ha sparato lo stesso identico valore su
# tutti e 32 gli esempi, la sua deviazione standard è zero. Con dati veri non
# capita quasi mai, ma "quasi mai" moltiplicato per milioni di passi capita.
#
# Noi lo sommiamo alla deviazione standard invece che alla varianza sotto la
# radice: è quello che fa la lezione, cambia nella terza cifra e non cambia
# niente di sostanziale.

print(f"  usiamo eps = {EPS}, sommato alla deviazione standard\n")


# ---------------------------------------------------------------------------
# 8. b1 non serve più
# ---------------------------------------------------------------------------

print("=== 8. il bias inutile ===\n")

# Dettaglio sottile ma vero. hpreact = X @ W1 + b1, e subito dopo la batchnorm
# sottrae la media del batch. Ma b1 è lo stesso su tutti gli esempi, quindi
# finisce dentro la media, e sottraendo la media viene sottratto via. b1 non ha
# nessun effetto sull'uscita — e quindi il suo gradiente è zero.

net_bn["b1"].grad = None
F.cross_entropy(forward(net_bn, Xtr[:BATCH_SIZE], "batch")[0], Ytr[:BATCH_SIZE]).backward()
print(f"  gradiente su b1, il più grande dei 200: {net_bn['b1'].grad.abs().max():.2e}")
print(f"  (per confronto, quello su bnbias:        {net_bn['bnbias'].grad.abs().max():.2e})")

# E la controprova: la stessa rete senza b1 calcola la stessa identica cosa.
net_without_b1 = make_net(batchnorm=True, with_b1=False)
net_with_b1 = make_net(batchnorm=True, with_b1=True)
with torch.no_grad():
    diff = (forward(net_without_b1, batch)[0] - forward(net_with_b1, batch)[0]).abs().max()
print(f"\n  logits con b1 vs senza b1, differenza massima: {diff:.2e}")

print(
    """
  Zero a meno dell'errore in virgola mobile, in entrambi i casi. Quel bias è
  200 parametri che non imparano niente e non fanno niente: la traslazione la
  fa bnbias, che è il bias della batchnorm. Per questo, nelle reti vere, ogni
  layer che sta subito prima di una normalizzazione viene creato con
  bias=False. Se apri il codice di una ResNet in PyTorch lo vedi scritto: nn.
  Conv2d(..., bias=False), e subito sotto la BatchNorm2d.

  Non è un bug, se lo lasci: è solo spreco. Ma è il tipo di dettaglio che si
  capisce solo avendo guardato dentro.
"""
)


# ---------------------------------------------------------------------------
# 9. com'è fatta davvero, in PyTorch
# ---------------------------------------------------------------------------

print("=== 9. torch.nn.BatchNorm1d ===\n")

print(
    """  Tutto quello che abbiamo scritto a mano sta in una riga:

      torch.nn.BatchNorm1d(200)

  e i suoi argomenti sono, uno per uno, le cose di questo file:

      num_features=200      quanti neuroni: serve a dimensionare gain, bias e
                            i due buffer
      eps=1e-5              la sezione 7
      momentum=0.1          la sezione 6. Il default è per batch grandi; noi
                            usiamo 0.001
      affine=True           se avere gain e bias imparabili. Praticamente
                            sempre True
      track_running_stats   se tenere i buffer aggiornati durante il training.
                            False vuol dire che li calibrerai a mano dopo

  Dentro ha quindi due parametri (gain e bias, allenati) e due buffer (media e
  deviazione standard running, non allenati). È la distinzione che in PyTorch
  separa .parameters() da .buffers(): entrambi vengono salvati nel checkpoint,
  ma solo i primi finiscono nell'optimizer.

  Il motivo per cui la si incontra ovunque è il motivo per cui esiste: nelle
  reti profonde il motivo ricorrente è

      layer che moltiplica  ->  normalizzazione  ->  non-linearità

  ripetuto decine di volte — nelle ResNet è conv, batchnorm, relu, e da capo.
  Così non c'è nessuna scala da calcolare a mano da nessuna parte: le
  attivazioni vengono rimesse in riga a ogni tappa, per costruzione.
"""
)


# ---------------------------------------------------------------------------
# cosa ci portiamo dietro
# ---------------------------------------------------------------------------
#
# 1. L'idea è banale e per questo è grande: se vuoi che una quantità interna
#    sia gaussiana, normalizzala. Si può fare perché media e deviazione
#    standard sono differenziabili, quindi la normalizzazione è un layer come
#    gli altri e la backpropagation ci passa attraverso.
#
# 2. Normalizzare da solo sarebbe una gabbia. Servono gain e bias, inizializzati
#    a 1 e 0, perché la distribuzione *parta* gaussiana ma la rete possa
#    spostarla dove vuole.
#
# 3. Su una rete con un layer nascosto non guadagna niente, anzi perde due
#    millesimi (2.1095 contro 2.1070): con un layer la scala giusta si calcola
#    a mano. Il guadagno è che su cinquanta layer non c'è più niente da
#    calcolare.
#
# 4. Il prezzo è che gli esempi di un batch smettono di essere indipendenti.
#    La stessa previsione cambia a seconda di chi capita nel batch. Come
#    regolarizzatore aiuta, come sorgente di bug è famigerata, ed è il motivo
#    per cui esistono layer norm e compagnia.
#
# 5. Dall'accoppiamento discende tutto il resto dell'apparato: le statistiche
#    running, i buffer che non sono parametri, la distinzione fra modalità
#    training e modalità inferenza. Non sono dettagli implementativi, sono
#    conseguenze.
#
# 6. E un bias prima di una normalizzazione non serve: viene sottratto via.
