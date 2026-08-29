# primo training vero con la nostra testa di self-attention: stesso modello
# della lezione sui bigrammi (una lookup table token -> logits del prossimo
# token), ma con due aggiunte:
#
# 1. gli embedding non sono piu' direttamente i logits: c'e' un vero spazio
#    latente (n_embed) e un lm_head finale che lo proietta sul vocabolario
# 2. in mezzo, la Head: ogni posizione si mixa con quelle precedenti
#
# dataset: tinyshakespeare, a livello di carattere
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Nel file prima abbiamo dato un senso intuitivo importante alle head di
# self-attention, che possiamo ora sfruttare in questo esempio.

# Cosa c'è di importante in questa versione (vedi il grafico in
# 05_gpt/03_out_architecture.md)

# Abbiamo sia embedding di carattere, che embedding di posizione nella sequenza.
# Sommandoli, per la proprietà distributiva della moltiplicazione, abbiamo che
# tutto quello che avviene dopo di lineare lavora sia sulla componente di carattere
# che di posizione, sommando gli effetti. Quindi key, query, value... tengono
# tutte conto della posizione. Quando si arriva a Q @ K^T perdiamo la linearità,
# ma e' negli step prima di quello che era importante avere il contributo di 
# entrambi.

# Le 4 teste si occupano ciascuna di generare 8 dimensioni di un embedding
# nuovo, e ciascuna testa partendo da valori casuali differenti, dovrebbe 
# porre l'attenzione a parti diverse di semantica/struttura.

# lm_head (l'ultimo) ha il compito di convertire da embeddings a logits.

# Il feed forward ha il compito "ipotetico" (e' sempre wishful thinking con le
# reti neurali) di non sovraccaricare lm_head con anche il compito di ragionare
# sul risultato degli embedding post self-attention. Ed è un ragionamento per-token
# non c'è "comunicazione"/passaggio di info tra differenti token.

# Il ReLu in fondo feed forward e' fondamentale per aggiungere non linearità e
# forzare due compiti differenti per le parti lineari di feedfoward e lm_head,
# altrimenti, sarebbero una moltiplicazione di matrice e poi un'altra, e quindi
# sarebbero praticamente una sola matrice di moltiplicazione, quindi inutile.

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# ---- hyperparameters --------------------------------------------------------

batch_size = 32      # quante sequenze indipendenti processiamo in parallelo
block_size = 8       # la lunghezza massima del contesto (T)
max_iters = 5000
eval_interval = 500
eval_iters = 200     # su quanti batch mediamo la loss quando la stimiamo
learning_rate = 1e-3
n_embed = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

# ---- dataset ----------------------------------------------------------------

text = (Path(__file__).parent / 'input.txt').read_text(encoding='utf-8')

# il nostro vocabolario sono i singoli caratteri che compaiono nel testo
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]           # stringa -> lista di interi
decode = lambda l: ''.join(itos[i] for i in l)    # lista di interi -> stringa

data = torch.tensor(encode(text), dtype=torch.long)

# primo 90% train, ultimo 10% validation: ci serve per accorgerci se stiamo
# semplicemente memorizzando Shakespeare invece di imparare qualcosa
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  data = train_data if split == 'train' else val_data

  # batch_size punti di partenza casuali nel testo
  ix = torch.randint(len(data) - block_size, (batch_size,))

  x = torch.stack([data[i:i + block_size] for i in ix])
  y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])   # y e' x shiftato di uno: il target di ogni posizione e' il carattere dopo

  return x.to(device), y.to(device)

@torch.no_grad()   # niente grafo dei gradienti: qui stiamo solo misurando
def estimate_loss(model):
  out = {}

  model.eval()     # per ora non cambia nulla (niente dropout/batchnorm), ma prendiamo l'abitudine

  for split in ['train', 'val']:
    # la loss di un singolo batch e' rumorosissima: ne mediamo eval_iters
    losses = torch.zeros(eval_iters)

    for k in range(eval_iters):
      X, Y = get_batch(split)
      _, loss = model(X, Y)
      losses[k] = loss.item()

    out[split] = losses.mean().item()

  model.train()

  return out

# ---- modello ----------------------------------------------------------------

class Head(nn.Module):
  """una singola testa di self-attention"""

  def __init__(self, n_embed, head_dim, block_size):
    super().__init__()

    # nn.Linear(a, b) calcola x @ W.T con W di shape (b, a): e' la nostra X @ Wk, solo memorizzata trasposta
    self.key   = nn.Linear(n_embed, head_dim, bias=False)
    self.query = nn.Linear(n_embed, head_dim, bias=False)
    self.value = nn.Linear(n_embed, head_dim, bias=False)

    # tril non e' un parametro addestrabile: register_buffer lo attacca al modulo
    # (segue il .to(device), finisce nello state_dict) senza che l'ottimizzatore lo tocchi
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

class MultiHeadAttention(nn.Module):
  """piu' teste di self-attention in parallelo"""

  def __init__(self, num_heads, head_dim):
    super().__init__()

    # una testa sola sa fare una cosa sola. nell'esempio della lezione2 la
    # nostra testa immaginaria era specializzata in "accoppiare i nomi con i
    # loro articoli": con piu' teste ognuna si sceglie il suo mestiere (chi
    # guarda la punteggiatura, chi il soggetto lontano, chi il carattere
    # precedente) e i risultati si sommano nel flusso residuo
    #
    # ModuleList e non una lista Python: cosi' le teste sono davvero
    # sotto-moduli, e parameters() / .to(device) / state_dict le vedono
    self.heads = nn.ModuleList([Head(n_embed, head_dim, block_size) for _ in range(num_heads)])

  def forward(self, x):
    # ogni testa produce (B, T, head_dim): le concateniamo lungo l'ultima
    # dimensione => (B, T, num_heads * head_dim)
    return torch.cat([h(x) for h in self.heads], dim=-1)

class FeedForward(nn.Module):
  """un piccolo MLP applicato a ogni token per conto suo"""

  def __init__(self, n_embed):
    super().__init__()

    # nn.Linear lavora sull'ultima dimensione: applicato a (B, T, n_embed)
    # tratta ogni token indipendentemente dagli altri. qui non si comunica
    self.net = nn.Sequential(
      nn.Linear(n_embed, n_embed),
      nn.ReLU(),
    )

  def forward(self, x):
    return self.net(x)

class AttentionLanguageModel(nn.Module):
  def __init__(self, num_heads):
    super().__init__()

    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

    # novita': ora la posizione conta. nel bag of words la media era cieca
    # all'ordine, e anche la self-attention di per se' non sa "dove" sono i
    # token: glielo diciamo noi sommando all'embedding del token un embedding
    # (addestrabile) della sua posizione 0..block_size-1
    self.position_embedding_table = nn.Embedding(block_size, n_embed)

    # spartiamo lo spazio fra le teste: num_heads * head_dim == n_embed, cosi'
    # l'uscita ha la stessa shape a prescindere da quante teste usiamo, e il
    # numero di parametri resta lo stesso (4 teste da 8 invece di 1 da 32)
    self.sa_heads = MultiHeadAttention(num_heads, n_embed // num_heads)

    # la self-attention e' COMUNICAZIONE: i token si guardano e raccolgono dati.
    # ma finora da li' andavamo dritti ai logits, troppo in fretta: i token non
    # avevano un attimo per RAGIONARE su quello che avevano appena raccolto.
    # ecco il feed-forward, che e' computazione pura, ognuno per conto suo
    self.ffwd = FeedForward(n_embed)

    # dallo spazio latente ai logits, uno per carattere del vocabolario
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    tok_emb = self.token_embedding_table(idx)                                  # (B, T, n_embed)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))    # (T, n_embed), broadcastato su tutti i batch

    x = tok_emb + pos_emb    # (B, T, n_embed)
    x = self.sa_heads(x)     # (B, T, n_embed) -> ogni posizione ha guardato le precedenti
    x = self.ffwd(x)         # (B, T, n_embed) -> ogni posizione ragiona su quello che ha raccolto

    logits = self.lm_head(x) # (B, T, vocab_size)

    if targets is None:
      return logits, None

    # cross_entropy vuole (N, C) e (N): appiattiamo batch e tempo in un'unica
    # dimensione, tanto ogni posizione e' una predizione indipendente
    B, T, vocab = logits.shape
    loss = F.cross_entropy(logits.view(B * T, vocab), targets.view(B * T))

    return logits, loss

  @torch.no_grad()
  def generate(self, idx, max_new_tokens):
    # idx e' (B, T): il contesto corrente. a ogni giro ne appendiamo un carattere
    for _ in range(max_new_tokens):
      # la position_embedding_table ha solo block_size righe: teniamo gli ultimi block_size token
      idx_cond = idx[:, -block_size:]

      logits, _ = self(idx_cond)

      # ci interessa solo la predizione dell'ultima posizione
      logits = logits[:, -1, :]                            # (B, vocab_size)
      probs = F.softmax(logits, dim=-1)                    # (B, vocab_size)

      # campioniamo, non prendiamo l'argmax: vogliamo varieta'
      idx_next = torch.multinomial(probs, num_samples=1)   # (B, 1)
      idx = torch.cat((idx, idx_next), dim=1)              # (B, T+1)

    return idx

# ---- training ---------------------------------------------------------------

def train(num_heads):
  # stesso seed per ogni versione: stessa inizializzazione e stessi batch, cosi'
  # l'unica differenza fra i due run e' il numero di teste
  torch.manual_seed(1337)

  model = AttentionLanguageModel(num_heads).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

  for iter in range(max_iters):
    if iter % eval_interval == 0:
      losses = estimate_loss(model)
      print(f"  step {iter:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

  losses = estimate_loss(model)
  print(f"  step {max_iters:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  return model, losses

# ---- confronto: 1 testa da 32 vs 4 teste da 8 -------------------------------

results = {}

for num_heads in [1, 4]:
  print(f"\n=== {num_heads} test{'a' if num_heads == 1 else 'e'} da {n_embed // num_heads} dimensioni ===\n")

  model, losses = train(num_heads)
  results[num_heads] = losses

  # partiamo da un singolo token 0, che nel nostro vocabolario e' il newline
  context = torch.zeros((1, 1), dtype=torch.long, device=device)

  print()
  print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

print("\n=== riepilogo ===\n")

for num_heads, losses in results.items():
  print(f"  {num_heads} test{'a' if num_heads == 1 else 'e'}: train {losses['train']:.4f}, val {losses['val']:.4f}")
