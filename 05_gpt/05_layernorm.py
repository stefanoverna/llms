# LayerNorm, Dropout, e la scala
#
# nel file 04 abbiamo reso ottimizzabile una rete profonda con i residui. qui
# aggiungiamo le ultime due rifiniture e poi alziamo tutti i numeri:
#
#   1. LayerNorm  - normalizza le feature di ogni token, stabilizza il training
#   2. Dropout    - regolarizzazione, serve solo quando il modello e' grosso
#   3. scale-up   - 384 dimensioni, 6 teste, 6 layer, 256 caratteri di contesto
#
# --- a cosa serve la LayerNorm -----------------------------------------------
#
# il problema e' lo stesso di makemore parte 3, quello che li' risolvevamo con
# la batch normalization: tenere le attivazioni in un intervallo ragionevole man
# mano che la rete diventa profonda. l'inizializzazione di Kaiming
# (gain / sqrt(fan_in)) sistema le cose una volta sola, all'inizio, e un layer
# alla volta: dopo qualche migliaio di passi i pesi si sono spostati e quel
# conto non vale piu'.
#
# con i residui del file 04 il problema si aggrava, perche' ogni blocco SOMMA il
# suo contributo dentro x invece di sostituirlo: la scala del flusso residuo
# cresce di blocco in blocco, e dopo 6 blocchi non e' piu' una scala che
# abbiamo scelto noi.
#
# e qui la scala conta in modo particolare, per via della softmax dentro ogni
# testa: A = Q @ K^T * head_dim**-0.5. se x arriva grosso, le affinita' sono
# grosse, la softmax si satura e diventa quasi one-hot (ogni token guarda un
# token solo), e i gradienti che la attraversano vanno a zero. e' esattamente la
# tanh in saturazione di makemore, con un altro nome.
#
# la LayerNorm risolve come la BatchNorm: invece di sperare che le feature
# escano ben scalate, le normalizza a mano prima di ogni sotto-livello. media e
# varianza sono operazioni differenziabili come le altre, quindi il backward ci
# passa attraverso senza accorgersene.
#
# la differenza fra le due sta in QUALI numeri si mediano:
#
#   BatchNorm: per ogni feature, media su tutti gli esempi del batch
#              (una colonna)  ->  gli esempi si "parlano"
#   LayerNorm: per ogni esempio, media su tutte le sue feature
#              (una riga)     ->  ogni token per conto suo
#
# ed e' questa scelta a far sparire tutti i guai che la BatchNorm ci aveva
# procurato: niente accoppiamento fra esempi finiti per caso nello stesso
# batch, niente medie mobili da accumulare, nessuna differenza di comportamento
# fra training e inference. cosa che qui non e' un dettaglio: con la BatchNorm
# la predizione per un token dipenderebbe dalle altre frasi del batch, il che
# non ha alcun senso, e in generazione il batch e' comunque di 1.
#
# il guadagno che misuriamo a questa scala e' piccolo (2.0705 -> 2.0563): tre
# blocchi da 32 dimensioni sono pochi, non c'e' granche' da stabilizzare. e'
# nella configurazione 'big', a 6 blocchi da 384, che diventa quello che
# permette alla rete di essere addestrabile.
#
# dopo di che il modello e' un GPT: un Transformer decoder-only completo,
# identico nell'architettura a quello del paper (a meno del ramo encoder e
# della cross-attention, che non ci servono perche' non stiamo traducendo).
#
# due configurazioni:
#
#   python 05_layernorm.py          -> 'small', confrontabile col file 04, ~2 min
#   python 05_layernorm.py big      -> lo scale-up vero, lungo (ore su CPU)

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# ---- hyperparameters --------------------------------------------------------

mode = sys.argv[1] if len(sys.argv) > 1 else 'small'

if mode == 'small':
  # stessi numeri del file 04, cosi' l'effetto della LayerNorm si misura da solo
  batch_size = 32
  block_size = 8
  n_embed = 32
  n_head = 4
  n_layer = 3
  dropout = 0.0        # a questa scala il modello e' in underfitting: regolarizzare fa solo danni
  learning_rate = 1e-3
  max_iters = 5000
  eval_interval = 1000
else:
  # la configurazione di Karpathy. 384/6 = 64 dimensioni per testa, che e' la
  # taglia standard. learning rate piu' basso perche' la rete e' molto piu' grande
  batch_size = 64
  block_size = 256     # 256 caratteri di contesto invece di 8
  n_embed = 384
  n_head = 6
  n_layer = 6
  dropout = 0.2        # ora i parametri sono ~10M su ~1M di caratteri: qui l'overfitting c'e' davvero
  learning_rate = 3e-4
  max_iters = 5000
  eval_interval = 500

eval_iters = 200

# mps e' la GPU integrata dei Mac Apple Silicon. per il modello 'small' non
# conviene (i tensori sono minuscoli, il costo di spostarli mangia il guadagno)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if mode != 'small' and torch.backends.mps.is_available():
  device = 'mps'

torch.manual_seed(1337)

# ---- dataset ----------------------------------------------------------------

text = (Path(__file__).parent / 'input.txt').read_text(encoding='utf-8')

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

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

  # ora model.eval() conta davvero: spegne il Dropout. in valutazione vogliamo
  # la rete intera, non una delle sue sottoreti campionate a caso
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

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, n_embed = x.shape

    K = self.key(x)      # (B, T, head_dim)
    Q = self.query(x)    # (B, T, head_dim)
    V = self.value(x)    # (B, T, head_dim)

    head_dim = K.shape[-1]

    A = Q @ K.transpose(-2, -1) * head_dim**-0.5    # (B, T, T)

    A = A.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    A = F.softmax(A, dim=-1)

    # dropout sulle affinita': a ogni forward alcune coppie di token vengono
    # scollegate a caso, cosi' la testa non puo' dipendere sempre dalle stesse
    A = self.dropout(A)

    C = A @ V   # (B, T, head_dim)

    return C

class MultiHeadAttention(nn.Module):
  """piu' teste di self-attention in parallelo"""

  def __init__(self, num_heads, head_dim):
    super().__init__()

    self.heads = nn.ModuleList([Head(n_embed, head_dim, block_size) for _ in range(num_heads)])

    self.proj = nn.Linear(num_heads * head_dim, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, num_heads * head_dim)

    # il dropout va sempre proprio prima del rientro nel flusso residuo
    out = self.dropout(self.proj(out))

    return out

class FeedForward(nn.Module):
  """un piccolo MLP applicato a ogni token per conto suo"""

  def __init__(self, n_embed):
    super().__init__()

    self.net = nn.Sequential(
      nn.Linear(n_embed, 4 * n_embed),
      nn.ReLU(),
      nn.Linear(4 * n_embed, n_embed),   # la proiezione di ritorno nel flusso residuo
      nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  """un blocco di Transformer: comunicazione seguita da computazione"""

  def __init__(self, n_embed, n_head, layer_norm):
    super().__init__()

    head_dim = n_embed // n_head

    self.sa = MultiHeadAttention(n_head, head_dim)
    self.ffwd = FeedForward(n_embed)

    # NOVITA': la LayerNorm (il perche' e' spiegato in testa al file). per ogni
    # singolo token le sue n_embed feature vengono portate a media 0 e varianza
    # 1, e poi riscalate da gamma e beta addestrabili - che sono li' apposta per
    # permettere all'ottimizzatore di disfare la normalizzazione, se scopre che
    # gli conviene.
    #
    # nn.Identity e' un modulo che restituisce l'input tale e quale: ci serve
    # solo per poter spegnere la LayerNorm nel confronto senza toccare forward()
    self.ln1 = nn.LayerNorm(n_embed) if layer_norm else nn.Identity()
    self.ln2 = nn.LayerNorm(n_embed) if layer_norm else nn.Identity()

  def forward(self, x):
    # ATTENZIONE, qui deviamo dal paper: nella figura originale e' "Add & Norm",
    # cioe' la norma DOPO il sotto-modulo. oggi si usa quasi sempre la variante
    # pre-norm, con la norma PRIMA: cosi' il flusso residuo resta un'autostrada
    # pulita di sole somme, senza normalizzazioni in mezzo
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))

    return x

class GPTLanguageModel(nn.Module):
  def __init__(self, layer_norm=True):
    super().__init__()

    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)

    self.blocks = nn.Sequential(*[
      Block(n_embed, n_head, layer_norm=layer_norm) for _ in range(n_layer)
    ])

    # una LayerNorm finale, dopo tutti i blocchi e prima della testa di
    # decodifica: e' l'ultimo pezzo mancante rispetto al Transformer del paper
    self.ln_f = nn.LayerNorm(n_embed) if layer_norm else nn.Identity()

    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    tok_emb = self.token_embedding_table(idx)                                  # (B, T, n_embed)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))    # (T, n_embed)

    x = tok_emb + pos_emb    # (B, T, n_embed)
    x = self.blocks(x)       # (B, T, n_embed)
    x = self.ln_f(x)         # (B, T, n_embed)

    logits = self.lm_head(x) # (B, T, vocab_size)

    if targets is None:
      return logits, None

    B, T, vocab = logits.shape
    loss = F.cross_entropy(logits.view(B * T, vocab), targets.view(B * T))

    return logits, loss

  @torch.no_grad()
  def generate(self, idx, max_new_tokens):
    self.eval()

    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]

      logits, _ = self(idx_cond)

      logits = logits[:, -1, :]                            # (B, vocab_size)
      probs = F.softmax(logits, dim=-1)                    # (B, vocab_size)

      idx_next = torch.multinomial(probs, num_samples=1)   # (B, 1)
      idx = torch.cat((idx, idx_next), dim=1)              # (B, T+1)

    self.train()

    return idx

# ---- training ---------------------------------------------------------------

def train(layer_norm=True):
  torch.manual_seed(1337)

  model = GPTLanguageModel(layer_norm).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

  n_params = sum(p.numel() for p in model.parameters())
  print(f"  {n_params / 1e6:.2f}M parametri, su {device}")

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

if mode == 'small':
  # confronto: gli stessi 3 blocchi del file 04, senza e con LayerNorm
  results = {}

  for name, layer_norm in [('senza LayerNorm', False), ('con LayerNorm', True)]:
    print(f"\n=== {name} ===\n")

    model, losses = train(layer_norm)
    results[name] = losses

    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    print()
    print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))

  print("\n=== riepilogo ===\n")

  for name, losses in results.items():
    print(f"  {name:20s}  val {losses['val']:.4f}  (train {losses['train']:.4f})")

else:
  print("\n=== GPT completo ===\n")

  model, _ = train()

  context = torch.zeros((1, 1), dtype=torch.long, device=device)

  # 10.000 caratteri su file: a questa scala vale la pena leggerne un bel pezzo
  out = decode(model.generate(context, max_new_tokens=10000)[0].tolist())

  (Path(__file__).parent / '05_out_sample.txt').write_text(out, encoding='utf-8')

  print()
  print(out[:1000])
  print("\n(campione completo in 05_out_sample.txt)")
