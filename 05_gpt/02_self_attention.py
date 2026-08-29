import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(1337)

def lezione1():
  B,T,C = 1,3,4 # batch, time, embeddings

  # i miei input embeddings
  x = torch.rand(B, T, C)

  # wei e' una matrice costruita appositamente per ottenere nuovi
  # embeddings, che sono per ogni T-th embedding la media di se stesso 
  # con i suoi precedenti (per t=2, media del token t=0, t=1 e t=2)

  tril = torch.tril(torch.ones(T, T))
  wei = torch.zeros((T,T))                           # inizializzo tutto a zero
  wei = wei.masked_fill(tril == 0, float('-inf'))    # forzo il triangolo superiore destro a zero
  wei = F.softmax(wei, dim=-1)                       # softmax converte i -inf in 0, e gli zero in 1.0, poi normalizza per fare somma per riga = 1.0

  # wei -> tensor([[1.0000, 0.0000, 0.0000],
  #                [0.5000, 0.5000, 0.0000],
  #                [0.3333, 0.3333, 0.3333]])

  out = wei @ x

  print(out)

def lezione2():
  # nel mondo reale vogliamo una media PESATA tra i token (ie. quello che viene
  # subito PRIMA pesa di piu' dei precedenti), e per fare cio' la parte da
  # modificare e' `wei = torch.zeros((T,T))` -> vogliamo dare pesi differenti,
  # non tutti 0 (che diventeranno === pari peso)!

  # ed e' proprio quello che vuole fare il self-attention. ma, dato un token
  # d'ingresso m per cui vogliamo calcolare la sua versione "mixata" con gli
  # altri token precedenti, che pesi mettiamo a ogni token precedente p? 
  # 
  # * ipotizziamo "the cat ate the mat" come i nostri token d'ingresso (5)
  # * ipotizziamo che la nostra self-attention head sia specializzata in
  #   "accoppiare i nomi con i loro articoli"
  #
  # per lavorare a questo compito la testa di self-attention ha bisogno di
  # lavorare in uno spazio dimensionale nuovo, differente da quello degli
  # embedding di partenza. e' uno spazio che serve a lui per questo specifico
  # scopo. ipotizziamo che questo spazio abbia 4 dimensioni, che possiamo
  # interpretare come
  # 
  # [articolo, nome, verbo, posizione]
  # 
  # il primo step e' quello di "proiettare" i nostri token su questo nuovo
  # spazio dimensionale, ed e' molto facile: basta una singola matmul
  # 
  # punto_nel_nuovo_spazio = embedding_token * M
  # 
  # ora però, siamo interessati a DUE diverse proiezioni di ogni token su questo
  # nuovo spazio!
  #
  # nel primo, siamo interessati alla domanda "ciò che sono":
  #
  # K = X * Wk
  #
  # +---------+-----+------+-------+-----+
  # | token   | ART | NOME | VERBO | POS |
  # +---------+-----+------+-------+-----+
  # | the (1) |  1  |  0   |   0   |  1  |
  # | cat (2) |  0  |  1   |   0   |  2  |
  # | ate (3) |  0  |  0   |   1   |  3  |
  # | the (4) |  1  |  0   |   0   |  4  |
  # | mat (5) |  0  |  1   |   0   |  5  |
  # +---------+-----+------+-------+-----+
  #
  # La matrice di proiezione che ci fa arrivare a questi risultati la chiamiamo
  # Wk (n_embed, head_dim) -> n_head = 4 dimensioni in questo caso
  #
  # nel secondo, siamo interessati alla domanda "ciò che cerco":
  #
  # Q = X * Wq
  #
  # +---------+-----+------+-------+-----+
  # | token   | ART | NOME | VERBO | POS |
  # +---------+-----+------+-------+-----+
  # | the (1) |  0  |  0   |   0   |  0  |
  # | cat (2) |  3  |  0   |   0   | 0.2 | <- "voglio fortemente un articolo, e a parità di tutto, 
  # | ate (3) |  0  |  0   |   0   |  0  |     preferisco chi ha indice alto, cioè chi mi sta vicino"
  # | the (4) |  0  |  0   |   0   |  0  |
  # | mat (5) |  3  |  0   |   0   | 0.2 |
  # +---------+-----+------+-------+-----+
  #
  # Ora arriva la genialata: devo trovare una operazione che mi faccia da
  # "matcher" tra domanda e offerta, in modo che il peso sia alto dove c'e' un
  # match. E l'operazione e':
  #
  # Omega = Q · K^T
  #
  # Ovvero sia il dot-product delle righe di Q con quelle di T. Praticamente
  # associo numeri grandi di uno con quelli grandi dell'altro tramite
  # moltiplicazione.
  #
  # +---------+--------+--------+--------+--------+--------+
  # |         | the(1) | cat(2) | ate(3) | the(4) | mat(5) |
  # +---------+--------+--------+--------+--------+--------+
  # | the (1) |   0    |   0    |   0    |   0    |   0    |
  # | cat (2) |  3.2   |  0.4   |  0.6   |  3.8   |  1.0   |
  # | ate (3) |   0    |   0    |   0    |   0    |   0    |
  # | the (4) |   0    |   0    |   0    |   0    |   0    |
  # | mat (5) |  3.2   |  0.4   |  0.6   |  3.8   |  1.0   |
  # +---------+--------+--------+--------+--------+--------+
  #
  # Eccoli qua i nostri punteggi! Da qui possiamo, come abbiamo fatto
  # nell'esempio precedente:
  # 1. mascherare il triangolo superiore a -inf
  # 2. applicare softmax
  # 
  # Ed otteniamo i nostri pesi. In realta', per tenere la softmax lontana dalla
  # saturazione, c'e uno step 1b intermedio ma e' una technicality:
  # 1. mascherare il triangolo superiore a -inf
  # 2. divido tutti gli elementi per sqrt(head_dim)
  # 3. applicare softmax
  #
  # +---------+--------+--------+--------+--------+--------+
  # |         | the(1) | cat(2) | ate(3) | the(4) | mat(5) |
  # +---------+--------+--------+--------+--------+--------+
  # | the (1) |  1.00  |   --   |   --   |   --   |   --   |
  # | cat (2) |  0.80  |  0.20  |   --   |   --   |   --   | <- "cat" ha un peso molto maggiore su "the"
  # | ate (3) |  0.33  |  0.33  |  0.33  |   --   |   --   |    che addirittura se stesso!
  # | the (4) |  0.25  |  0.25  |  0.25  |  0.25  |   --   |
  # | mat (5) |  0.33  |  0.02  |  0.02  |  0.59  |  0.04  | <- "mat" ha un peso piu' grande sul "the" subito
  # +---------+--------+--------+--------+--------+--------+    prima di lui che su quello a inizio frase
  #
  # Ma a questo punto c'e' un altro colpo di scena. I pesi li abbiamo, ma NON li
  # usiamo più per pesare i token d'ingresso, come abbiamo fatto sopra!
  # Effettivamente, non funzionerebbe molto. Se guardiamo quale sarebbe il
  # risultato per "cat":
  #
  # 1. "cat" sarebbe praticamente "the", sporcato da un po' di "cat" ->
  #    perderemmo informazione fondamentale per altri contesti che non siano
  #    quello di questa testa
  # 2. nello spazio dimensionale degli embedding, non e' che fare una media tra
  #    due punti (che equivale a spostare il punto a metà tra i due) ha alcun
  #    significato semantico
  #
  # Ed e' qui che entra in gioco una TERZA proiezione!
  #
  # Il punto e' che Wk e Wq rispondono a domande che servono al MATCHING ("cosa
  # sono" / "cosa cerco"). Fatto, missione compiuta.
  #
  # Ora dobbiamo pensare a una nuova missione: ciò che un token deve CONSEGNARE
  # a chi lo ha guardato. E' un compito diverso, e le dimensioni [articolo,
  # nome, verbo, posizione] non hanno più alcun senso in questo contesto. Quindi
  # ipotizziamo di avere un nuovo spazio, per esempio a 3 dimensioni:
  #
  # [determinatezza, animato, superficie]
  #
  # Immaginiamo di nuovo di avere una matrice di proiezione Wv in grado di
  # portarci in questo spazio:
  #
  # V = X * Wv
  #
  # +---------+------+------+------+
  # | token   | DET  | ANIM | SUP  |
  # +---------+------+------+------+
  # | the (1) |  1   |  0   |  0   |  <- offro DETERMINATEZZA a chiunque mi guardi
  # | cat (2) |  0   |  1   |  0   |
  # | ate (3) |  0   |  0   |  0   |  <- un verbo non ha niente da offrire a QUESTA testa
  # | the (4) |  1   |  0   |  0   |
  # | mat (5) |  0   |  0   |  1   |
  # +---------+------+------+------+
  #
  # Il "the", nello spazio K diceva "sono un articolo in posizione 1", perche'
  # serviva a farsi TROVARE. Nello spazio V dice tutt'altro, perche' ora deve
  # farsi USARE. A "mat" non serve sapere che il suo articolo stava in posizione
  # 4, gli serve la determinatezza.
  #
  # E finalmente, possiamo unire i pezzi: i nostri pesi li usiamo su V
  #
  # C = A · V
  #
  # +---------+------+------+------+
  # |         | DET  | ANIM | SUP  |
  # +---------+------+------+------+
  # | the (1) | 1.00 | 0.00 | 0.00 |
  # | cat (2) | 0.80 | 0.20 | 0.00 |  <- "+determinatezza". "cat" NON e' stato sostituito da "the", abbiamo aggiunto determinatezza a "gatto", è "IL GATTO", non "UN GATTO"
  # | ate (3) | 0.33 | 0.33 | 0.00 |  <- pesi uniformi -> poltiglia innocua
  # | the (4) | 0.50 | 0.25 | 0.00 |
  # | mat (5) | 0.73 | 0.08 | 0.10 |  <- "+determinatezza", raccolta da entrambi i "the"
  # +---------+------+------+------+
  #
  # I due problemi che avevamo elencato sono risolti entrambi:
  # 1. non perdiamo piu' informazione, perche' C non e' la nuova versione del
  #    token: e' un DELTA che verra' sommato al flusso residuo (x = x +
  #    attention(x), lo vedremo). "cat" resta cat, e guadagna determinatezza.
  # 2. la media ora ha senso semantico, perche' la facciamo in uno spazio
  #    costruito apposta perche' ce l'abbia.
  #
  # Va da se' che per sommare questi "delta" ai token, servirà una ulteriore
  # proiezione che riporti questi delta nello spazio dimensionale degli
  # embedding.
  
  T = 5

  # facciamo finta che le tre proiezioni siano gia' avvenute, e che siano venute cosi' pulite.
  # nella realta' queste tre matrici escono da X @ Wq, X @ Wk, X @ Wv, e sono illeggibili.
  Q = torch.tensor([[0., 0., 0., 0.0],
                    [3., 0., 0., 0.2],
                    [0., 0., 0., 0.0],
                    [0., 0., 0., 0.0],
                    [3., 0., 0., 0.2]])

  K = torch.tensor([[1., 0., 0., 1.],
                    [0., 1., 0., 2.],
                    [0., 0., 1., 3.],
                    [1., 0., 0., 4.],
                    [0., 1., 0., 5.]])

  V = torch.tensor([[1., 0., 0.],      # nota: 3 dimensioni, non 4. spazio diverso.
                    [0., 1., 0.],
                    [0., 0., 0.],
                    [1., 0., 0.],
                    [0., 0., 1.]])

  head_dim = K.shape[-1]

  omega = Q @ K.T                                   # (T,T) punteggi grezzi
  tril = torch.tril(torch.ones(T, T))

  A = omega.masked_fill(tril == 0, float('-inf')) # maschera causale
  A = A / head_dim**0.5                         # scaling (technicality: tiene la softmax lontana dalla saturazione)
  A = F.softmax(A, dim=-1)                      # -> A, ogni riga somma a 1

  C = A @ V

  print(A)
  print(C)

# ok, ora implementamolo bene

class Head(nn.Module):
  """una singola testa di self-attention"""

  def __init__(self, n_embed, head_dim, block_size):
    super().__init__()

    # nn.Linear(a, b) calcola x @ W.T con W di shape (b, a): e' la nostra X @ Wk, solo memorizzata trasposta
    self.key   = nn.Linear(n_embed, head_dim, bias=False)
    self.query = nn.Linear(n_embed, head_dim, bias=False)
    self.value = nn.Linear(n_embed, head_dim, bias=False)

    # tril non e' un parametro addestrabile: register_buffer lo attacca al modulo senza che l'ottimizzatore lo tocchi
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B, T, n_embed = x.shape

    K = self.key(x)      # (B, T, head_dim)
    Q = self.query(x)    # (B, T, head_dim)
    V = self.value(x)    # (B, T, head_dim)

    head_dim = K.shape[-1]

    # -2,-1 e non 0,1: la prima dimensione e' il batch, trasponiamo solo le ultime due
    A = Q @ K.transpose(-2, -1) * head_dim**-0.5    # (B, T, head_dim) @ (B, head_dim, T) => (B, T, T)

    # :T, :T perche' la sequenza in ingresso puo' essere piu' corta di block_size
    A = A.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    A = F.softmax(A, dim=-1)

    C = A @ V   # (B, T, head_dim)

    return C

def lezione3():
  # facciamolo davvero: le tre proiezioni sono nn.Linear addestrabili, e lavoriamo in batch.

  # 4 samples, sequenze da 8 caratteri, embedding da 32
  B, T, n_embed = 4, 8, 32
  # lo spazio dimensionale di K,Q e, incidentalmente, anche V (anche se sono spazi diversi)
  head_dim = 16

  # simuliamo input con numeri casuali
  x = torch.randn(B, T, n_embed)

  head = Head(n_embed, head_dim, block_size=T)
  C = head(x)

  print(C.shape)

lezione3()
