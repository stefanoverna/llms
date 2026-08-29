# "intersperse communication with computation"
#
# nel file 03 il flusso era: embedding -> self-attention -> feed forward -> logits.
# una sola volta: i token si parlano una volta sola, poi ragionano una volta sola,
# poi si decide il carattere. e' poco: il Transformer del paper impila lo stesso
# schema N volte (parla, ragiona, parla, ragiona, ...), e quel gruppetto e' il
# Block.
#
# ma impilare non basta: appena la rete diventa profonda smette di ottimizzare
# bene. qui trainiamo tre versioni per vederlo con i numeri:
#
#   1. 3 blocchi "ingenui"          -> peggio del modello a 1 blocco del file 03
#   2. + residui e proiezioni       -> molto meglio
#   3. + feed forward largo (x4)    -> meglio ancora
#
# dataset: tinyshakespeare, a livello di carattere

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# ---- hyperparameters --------------------------------------------------------

batch_size = 32      # quante sequenze indipendenti processiamo in parallelo
block_size = 8       # la lunghezza massima del contesto (T)
max_iters = 5000
eval_interval = 1000
eval_iters = 200     # su quanti batch mediamo la loss quando la stimiamo
learning_rate = 1e-3
n_embed = 32
n_head = 4           # 32 / 4 = 8 dimensioni per testa
n_layer = 3          # quanti Block impilati
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

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  data = train_data if split == 'train' else val_data

  ix = torch.randint(len(data) - block_size, (batch_size,))

  x = torch.stack([data[i:i + block_size] for i in ix])
  y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

  return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
  out = {}

  model.eval()

  for split in ['train', 'val']:
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

    self.key   = nn.Linear(n_embed, head_dim, bias=False)
    self.query = nn.Linear(n_embed, head_dim, bias=False)
    self.value = nn.Linear(n_embed, head_dim, bias=False)

    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B, T, n_embed = x.shape

    K = self.key(x)      # (B, T, head_dim)
    Q = self.query(x)    # (B, T, head_dim)
    V = self.value(x)    # (B, T, head_dim)

    head_dim = K.shape[-1]

    A = Q @ K.transpose(-2, -1) * head_dim**-0.5    # (B, T, T)

    A = A.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    A = F.softmax(A, dim=-1)

    C = A @ V   # (B, T, head_dim)

    return C

class MultiHeadAttention(nn.Module):
  """piu' teste di self-attention in parallelo"""

  def __init__(self, num_heads, head_dim, proj):
    super().__init__()

    self.heads = nn.ModuleList([Head(n_embed, head_dim, block_size) for _ in range(num_heads)])

    # NOVITA': la proiezione. la concatenazione delle teste e' un accrocchio -
    # pezzi di vettore prodotti da teste che non si sono mai parlate, appiccicati
    # uno dopo l'altro. la proj e' una nn.Linear che li rimescola e li riporta
    # "nella lingua" del flusso residuo, prima di sommarli dentro
    self.proj = nn.Linear(num_heads * head_dim, n_embed) if proj else None

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, num_heads * head_dim)

    if self.proj is not None:
      out = self.proj(out)

    return out

class FeedForward(nn.Module):
  """un piccolo MLP applicato a ogni token per conto suo"""

  def __init__(self, n_embed, wide, proj):
    super().__init__()

    # NOVITA': nel paper l'input/output del feed forward e' 512 ma lo strato
    # interno e' 2048, cioe' x4. e' li' che sta la "computazione" vera: si
    # espande in uno spazio piu' grande, si applica la non linearita', e si
    # torna giu'. il ritorno giu' e' anche qui la proiezione nel flusso residuo
    hidden = 4 * n_embed if wide else n_embed

    layers = [nn.Linear(n_embed, hidden), nn.ReLU()]

    if proj:
      layers.append(nn.Linear(hidden, n_embed))

    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  """un blocco di Transformer: comunicazione seguita da computazione"""

  def __init__(self, n_embed, n_head, residual, wide):
    super().__init__()

    head_dim = n_embed // n_head

    self.sa = MultiHeadAttention(n_head, head_dim, proj=residual)
    self.ffwd = FeedForward(n_embed, wide=wide, proj=residual)

    self.residual = residual

  def forward(self, x):
    if not self.residual:
      # la versione ingenua: ogni stadio SOSTITUISCE x
      x = self.sa(x)
      x = self.ffwd(x)
      return x

    # NOVITA': i residui (skip connections, dal paper ResNet del 2015).
    #
    # x resta il "flusso residuo": non lo sostituiamo piu', ci scostiamo di
    # lato, calcoliamo qualcosa, e torniamo dentro SOMMANDO. il motivo e' il
    # backward: la somma distribuisce il gradiente identico a tutti e due i
    # rami, quindi c'e' un'"autostrada" che porta il gradiente dalla loss fino
    # agli embedding senza attraversare nessuna moltiplicazione.
    #
    # e all'inizio i rami laterali partono con pesi piccoli e contano quasi
    # nulla: la rete profonda nasce quasi come una rete che non fa niente
    # (quindi facile da ottimizzare) e i blocchi "si accendono" col training

    # l'idea quindi e' quella di "forkare", fare della computazione all'interno
    # di un blocco, e tornare con un contributo
    x = x + self.sa(x)
    x = x + self.ffwd(x)

    return x

class AttentionLanguageModel(nn.Module):
  def __init__(self, residual, wide):
    super().__init__()

    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)

    # NOVITA': non piu' una self-attention + un feed forward, ma n_layer
    # blocchi in fila. nn.Sequential li chiama uno dopo l'altro passandosi x
    self.blocks = nn.Sequential(*[
      Block(n_embed, n_head, residual=residual, wide=wide) for _ in range(n_layer)
    ])

    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    tok_emb = self.token_embedding_table(idx)                                  # (B, T, n_embed)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))    # (T, n_embed)

    x = tok_emb + pos_emb    # (B, T, n_embed)
    x = self.blocks(x)       # (B, T, n_embed) -> parla, ragiona, parla, ragiona, ...

    logits = self.lm_head(x) # (B, T, vocab_size)

    if targets is None:
      return logits, None

    B, T, vocab = logits.shape
    loss = F.cross_entropy(logits.view(B * T, vocab), targets.view(B * T))

    return logits, loss

  @torch.no_grad()
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]

      logits, _ = self(idx_cond)

      logits = logits[:, -1, :]                            # (B, vocab_size)
      probs = F.softmax(logits, dim=-1)                    # (B, vocab_size)

      idx_next = torch.multinomial(probs, num_samples=1)   # (B, 1)
      idx = torch.cat((idx, idx_next), dim=1)              # (B, T+1)

    return idx

# ---- training ---------------------------------------------------------------

def train(residual, wide):
  # stesso seed per ogni versione: stessa inizializzazione e stessi batch
  torch.manual_seed(1337)

  model = AttentionLanguageModel(residual, wide).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

  n_params = sum(p.numel() for p in model.parameters())
  print(f"  {n_params} parametri")

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

# ---- confronto: 3 blocchi ingenui / con residui / con feed forward largo -----

versions = [
  ('3 blocchi, niente residui',          dict(residual=False, wide=False)),
  ('3 blocchi + residui e proiezioni',   dict(residual=True,  wide=False)),
  ('3 blocchi + residui + ffwd x4',      dict(residual=True,  wide=True)),
]

results = {}

for name, kwargs in versions:
  print(f"\n=== {name} ===\n")

  model, losses = train(**kwargs)
  results[name] = losses

  context = torch.zeros((1, 1), dtype=torch.long, device=device)

  print()
  print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))

print("\n=== riepilogo ===\n")

print(f"  {'1 blocco (file 03)':38s}  val 2.2495")

for name, losses in results.items():
  print(f"  {name:38s}  val {losses['val']:.4f}  (train {losses['train']:.4f})")
