"""GPT decoder-only su corpus di caratteri. Versione operativa del 05.

  python 06_gpt.py train                          config 'small', corpus italiano
  python 06_gpt.py train --config big
  python 06_gpt.py train --config big --corpus input.txt
  python 06_gpt.py train --config big --resume    riprende un run interrotto
  python 06_gpt.py generate --tokens 2000
  python 06_gpt.py generate --checkpoint 06_out_model_big_step3500.pt

Il training salva i pesi con la val loss piu' bassa in 06_out_model_<config>.pt
(quelli che carica generate), l'ultimo stato completo di ottimizzatore in
06_out_model_<config>_last.pt (per --resume), e i grafici di diagnostica in
06_out_training_<config>.png.
"""

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
EVAL_ITERS = 200

CONFIGS = {
  'small': dict(batch_size=32, block_size=8,   n_embed=32,  n_head=4, n_layer=3,
                dropout=0.0, learning_rate=1e-3, max_iters=5000, eval_interval=500),
  'big':   dict(batch_size=64, block_size=256, n_embed=384, n_head=6, n_layer=6,
                dropout=0.2, learning_rate=3e-4, max_iters=5000, eval_interval=500),
}

def pick_device():
  if torch.cuda.is_available():
    return 'cuda'
  if torch.backends.mps.is_available():
    return 'mps'
  return 'cpu'

# ---- dati -------------------------------------------------------------------

class CharDataset:
  def __init__(self, path, block_size, batch_size, device, chars=None):
    text = path.read_text(encoding='utf-8')

    self.chars = chars if chars is not None else sorted(set(text))
    self.stoi = {ch: i for i, ch in enumerate(self.chars)}
    self.itos = {i: ch for i, ch in enumerate(self.chars)}

    self.block_size = block_size
    self.batch_size = batch_size
    self.device = device

    data = torch.tensor(self.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    self.splits = {'train': data[:n], 'val': data[n:]}

  @property
  def vocab_size(self):
    return len(self.chars)

  def encode(self, s):
    return [self.stoi[c] for c in s]

  def decode(self, ids):
    return ''.join(self.itos[i] for i in ids)

  def batch(self, split):
    data = self.splits[split]
    ix = torch.randint(len(data) - self.block_size, (self.batch_size,))

    x = torch.stack([data[i:i + self.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])

    return x.to(self.device), y.to(self.device)

# ---- modello ----------------------------------------------------------------

_RECORD = False   # vedi recording(): abilita le statistiche di diagnostica

@contextmanager
def recording(model):
  global _RECORD

  was_training = model.training
  model.eval()
  _RECORD = True

  try:
    yield
  finally:
    _RECORD = False
    model.train(was_training)

class Head(nn.Module):
  def __init__(self, n_embed, head_dim, block_size, dropout):
    super().__init__()

    self.key   = nn.Linear(n_embed, head_dim, bias=False)
    self.query = nn.Linear(n_embed, head_dim, bias=False)
    self.value = nn.Linear(n_embed, head_dim, bias=False)

    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)
    self.entropy = float('nan')

  def forward(self, x):
    B, T, _ = x.shape

    K, Q, V = self.key(x), self.query(x), self.value(x)

    A = Q @ K.transpose(-2, -1) * K.shape[-1]**-0.5
    A = A.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    A = F.softmax(A, dim=-1)

    if _RECORD:
      # entropia di ogni riga, normalizzata sul massimo possibile a quella
      # posizione: 1 = attenzione uniforme, 0 = softmax satura su un token solo
      P = A.clamp_min(1e-9)
      H = -(P * P.log2()).sum(-1)
      H_max = torch.arange(1, T + 1, device=A.device).float().log2()
      self.entropy = (H[:, 1:] / H_max[1:]).mean().item()

    return self.dropout(A) @ V

class MultiHeadAttention(nn.Module):
  def __init__(self, n_embed, n_head, block_size, dropout):
    super().__init__()

    head_dim = n_embed // n_head

    self.heads = nn.ModuleList([
      Head(n_embed, head_dim, block_size, dropout) for _ in range(n_head)
    ])
    self.proj = nn.Linear(n_head * head_dim, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    return self.dropout(self.proj(out))

class FeedForward(nn.Module):
  def __init__(self, n_embed, dropout):
    super().__init__()

    self.net = nn.Sequential(
      nn.Linear(n_embed, 4 * n_embed),
      nn.ReLU(),
      nn.Linear(4 * n_embed, n_embed),
      nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_embed, n_head, block_size, dropout):
    super().__init__()

    self.sa = MultiHeadAttention(n_embed, n_head, block_size, dropout)
    self.ffwd = FeedForward(n_embed, dropout)

    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

    self.rms = float('nan')

  def forward(self, x):
    if _RECORD:
      self.rms = x.pow(2).mean(-1).sqrt().mean().item()

    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))

    return x

class GPT(nn.Module):
  def __init__(self, vocab_size, block_size, n_embed, n_head, n_layer, dropout):
    super().__init__()

    self.block_size = block_size

    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)

    self.blocks = nn.Sequential(*[
      Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer)
    ])
    self.ln_f = nn.LayerNorm(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))

    x = self.ln_f(self.blocks(tok_emb + pos_emb))
    logits = self.lm_head(x)

    if targets is None:
      return logits, None

    B, T, vocab = logits.shape
    loss = F.cross_entropy(logits.view(B * T, vocab), targets.view(B * T))

    return logits, loss

  def generate(self, idx, max_new_tokens):
    """generatore: restituisce un indice alla volta, appena campionato.

    Un generatore e non una lista perche' il testo si scrive da solo mentre
    esce, invece di comparire tutto insieme alla fine.
    """
    was_training = self.training
    self.eval()

    try:
      with torch.no_grad():
        for _ in range(max_new_tokens):
          # il contesto e' lungo al massimo block_size: la tabella delle
          # posizioni non ha righe oltre
          logits, _ = self(idx[:, -self.block_size:])

          probs = F.softmax(logits[:, -1, :], dim=-1)
          nxt = torch.multinomial(probs, num_samples=1)

          idx = torch.cat((idx, nxt), dim=1)

          yield nxt.item()
    finally:
      self.train(was_training)

# ---- diagnostica ------------------------------------------------------------

def family(name):
  if 'token_embedding' in name:                                   return 'token emb'
  if 'position_embedding' in name:                                return 'pos emb'
  if any(k in name for k in ('.key.', '.query.', '.value.')):     return 'K/Q/V'
  if 'sa.proj' in name:                                           return 'proj (attn)'
  if 'ffwd' in name:                                              return 'feed forward'
  if 'lm_head' in name:                                           return 'lm_head'
  return 'altro'

def trainable_matrices(model):
  return {n: p for n, p in model.named_parameters() if p.ndim >= 2}

@torch.no_grad()
def update_data_ratio(model, before):
  """log10( std(update) / std(pesi) ), per famiglia di tensori.

  Riferimento empirico: -3. Misuriamo l'update davvero applicato e non
  lr * grad, che con AdamW non coincidono.
  """
  groups = {}

  for name, p in trainable_matrices(model).items():
    r = ((p - before[name]).std() / before[name].std()).log10().item()
    groups.setdefault(family(name), []).append(r)

  return {k: sum(v) / len(v) for k, v in groups.items()}

@torch.no_grad()
def estimate_loss(model, data, eval_iters):
  out = {}

  was_training = model.training
  model.eval()

  for split in ('train', 'val'):
    losses = torch.zeros(eval_iters)

    for k in range(eval_iters):
      X, Y = data.batch(split)
      _, loss = model(X, Y)
      losses[k] = loss.item()

    out[split] = losses.mean().item()

  model.train(was_training)

  return out

@torch.no_grad()
def attention_stats(model, data):
  X, _ = data.batch('val')

  with recording(model):
    model(X)

  return {
    'entropy': [[h.entropy for h in b.sa.heads] for b in model.blocks],
    'rms': [b.rms for b in model.blocks],
  }

def plot_history(history, path, title):
  iters = [h['iter'] for h in history]

  fig, axes = plt.subplots(2, 2, figsize=(13, 8))
  fig.suptitle(title, fontsize=11)

  ax = axes[0][0]
  ax.plot(iters, [h['train'] for h in history], label='train')
  ax.plot(iters, [h['val'] for h in history], label='val')
  ax.set_title('loss', fontsize=10)
  ax.legend()

  ax = axes[0][1]
  for fam in sorted({k for h in history for k in h['ud']}):
    ax.plot(iters, [h['ud'].get(fam, float('nan')) for h in history], label=fam)
  ax.axhline(-3, color='k', ls='--', lw=1)
  ax.set_title(r'update:data   log10( std(update) / std(W) )', fontsize=10)
  ax.legend(fontsize=8)

  ax = axes[1][0]
  n_layer = len(history[0]['entropy'])
  n_head = len(history[0]['entropy'][0])
  colors = plt.cm.viridis([i / max(n_layer - 1, 1) for i in range(n_layer)])
  for b in range(n_layer):
    for h in range(n_head):
      ax.plot(iters, [x['entropy'][b][h] for x in history],
              color=colors[b], lw=1.2,
              label=f'blocco {b}' if h == 0 else None)
  ax.set_ylim(0, 1.05)
  ax.set_title(f'entropia dell\'attenzione, una curva per testa '
               f'({n_layer} blocchi x {n_head} teste)\n'
               f'1 = guarda tutti, 0 = softmax satura', fontsize=10)
  ax.legend(fontsize=8)

  ax = axes[1][1]
  for i in range(n_layer):
    ax.plot(iters, [h['rms'][i] for h in history], label=f'blocco {i}')
  ax.set_title('scala del flusso residuo in ingresso a ogni blocco (RMS)', fontsize=10)
  ax.legend(fontsize=8)

  for row in axes:
    for ax in row:
      ax.set_xlabel('iterazione')
      ax.grid(alpha=0.3)

  plt.tight_layout()
  plt.savefig(path, bbox_inches='tight', dpi=100)
  plt.close()

# ---- checkpoint -------------------------------------------------------------

def checkpoint_path(config):
  """i pesi con la val loss piu' bassa: e' questo che carica generate"""
  return HERE / f'06_out_model_{config}.pt'

def last_path(config):
  """l'ultimo stato, ottimizzatore compreso: serve solo a --resume"""
  return HERE / f'06_out_model_{config}_last.pt'

def save_checkpoint(model, cfg, data, corpus, path, optimizer=None, it=None, best_val=None):
  payload = {
    'config': cfg,
    'chars': data.chars,
    'corpus': corpus,
    'state_dict': model.state_dict(),
  }

  if optimizer is not None:
    # le medie mobili di AdamW: senza queste, riprendere un training significa
    # ricominciare con un ottimizzatore che non sa niente della storia passata
    payload |= {'optimizer': optimizer.state_dict(), 'iter': it, 'best_val': best_val}

  torch.save(payload, path)

def build(cfg, vocab_size, device):
  return GPT(
    vocab_size=vocab_size,
    block_size=cfg['block_size'],
    n_embed=cfg['n_embed'],
    n_head=cfg['n_head'],
    n_layer=cfg['n_layer'],
    dropout=cfg['dropout'],
  ).to(device)

# ---- comandi ----------------------------------------------------------------

def cmd_train(args):
  # nel training il seed fisso serve: due run vanno confrontati a parita' di
  # inizializzazione e di batch
  torch.manual_seed(args.seed if args.seed is not None else 1337)

  cfg = dict(CONFIGS[args.config])
  if args.max_iters:
    cfg['max_iters'] = args.max_iters
    cfg['eval_interval'] = max(args.max_iters // 10, 1)

  device = args.device or pick_device()

  resumed = (torch.load(last_path(args.config), map_location=device, weights_only=False)
             if args.resume else None)

  if resumed:
    # riprendiamo esattamente la situazione salvata, corpus e vocabolario compresi
    cfg = resumed['config']
    if args.max_iters:
      cfg['max_iters'] = args.max_iters

  corpus = resumed['corpus'] if resumed else args.corpus
  chars = resumed['chars'] if resumed else None

  data = CharDataset(HERE / corpus, cfg['block_size'], cfg['batch_size'], device, chars=chars)

  model = build(cfg, data.vocab_size, device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])

  start, best_val = 0, float('inf')

  if resumed:
    model.load_state_dict(resumed['state_dict'])
    optimizer.load_state_dict(resumed['optimizer'])
    start = resumed['iter'] + 1
    best_val = resumed['best_val']
    print(f"ripreso da {last_path(args.config).name}: step {start}, "
          f"migliore val finora {best_val:.4f}")

  n_params = sum(p.numel() for p in model.parameters())
  print(f"{corpus}: {len(data.splits['train']) + len(data.splits['val'])} caratteri, "
        f"vocabolario di {data.vocab_size}")
  print(f"config '{args.config}': {n_params / 1e6:.2f}M parametri su {device}\n")

  history = []
  t0 = time.time()

  for it in range(start, cfg['max_iters']):
    xb, yb = data.batch('train')

    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    checkpoint = it % cfg['eval_interval'] == 0 or it == cfg['max_iters'] - 1
    before = ({n: p.detach().clone() for n, p in trainable_matrices(model).items()}
              if checkpoint else None)

    optimizer.step()

    if not checkpoint:
      continue

    losses = estimate_loss(model, data, args.eval_iters)
    stats = attention_stats(model, data)
    ud = update_data_ratio(model, before)

    history.append(dict(iter=it, ud=ud, **losses, **stats))

    flat = [e for block in stats['entropy'] for e in block]
    elapsed = time.time() - t0
    eta = elapsed / (it + 1) * (cfg['max_iters'] - it - 1)

    print(f"  step {it:5d}: train {losses['train']:.4f}  val {losses['val']:.4f}"
          f"  | attn {sum(flat) / len(flat):.2f}"
          f"  rms {stats['rms'][-1]:6.2f}"
          f"  | {elapsed / 60:5.1f}m, eta {eta / 60:5.1f}m")
    print("               " + "  ".join(f"{k} {v:+.1f}" for k, v in sorted(ud.items())))

    # il checkpoint "buono" e' quello con la val piu' bassa, non l'ultimo: se la
    # val risale, continuare ad addestrare peggiora il modello su testo nuovo
    if losses['val'] < best_val:
      best_val = losses['val']
      save_checkpoint(model, cfg, data, corpus, checkpoint_path(args.config))
      print(f"               nuovo minimo di val, salvato in {checkpoint_path(args.config).name}")

    save_checkpoint(model, cfg, data, corpus, last_path(args.config),
                    optimizer=optimizer, it=it, best_val=best_val)

  plot = HERE / f'06_out_training_{args.config}.png'
  plot_history(history, plot, f"{args.config}, {args.corpus}")

  print(f"\nmigliore val {best_val:.4f}, pesi in {checkpoint_path(args.config).name}"
        f" (ultimo stato in {last_path(args.config).name}), grafici in {plot.name}")

  sample(model, data, args.tokens, args.config)

def cmd_generate(args):
  # in generazione un seed fisso darebbe sempre lo stesso testo: senza --seed
  # ne peschiamo uno a caso, e lo stampiamo per poterlo rigiocare
  seed = args.seed if args.seed is not None else torch.seed()
  torch.manual_seed(seed)

  # --checkpoint punta a un file preciso; senza, si usa quello della config
  path = Path(args.checkpoint) if args.checkpoint else checkpoint_path(args.config)
  saved = torch.load(path, map_location='cpu', weights_only=False)

  device = args.device or pick_device()
  cfg = saved['config']

  # il vocabolario e' quello del corpus su cui il modello e' stato addestrato:
  # ricostruirlo da un corpus diverso cambierebbe il significato degli indici
  data = CharDataset(HERE / saved['corpus'], cfg['block_size'], cfg['batch_size'],
                     device, chars=saved['chars'])

  model = build(cfg, data.vocab_size, device)
  model.load_state_dict(saved['state_dict'])

  label = path.stem.replace('06_out_model_', '')

  print(f"{path.name}: corpus {saved['corpus']}, "
        f"{cfg['n_layer']} blocchi x {cfg['n_head']} teste da {cfg['n_embed'] // cfg['n_head']}, "
        f"su {device}")
  print(f"seed {seed}\n")

  sample(model, data, args.tokens, label)

def sample(model, data, tokens, label):
  context = torch.zeros((1, 1), dtype=torch.long, device=next(model.parameters()).device)

  out = []

  for idx in model.generate(context, tokens):
    ch = data.itos[idx]
    out.append(ch)
    print(ch, end='', flush=True)

  print()

  path = HERE / f'06_out_sample_{label}.txt'
  path.write_text(''.join(out), encoding='utf-8')

  print(f"\n[{tokens} caratteri in {path.name}]")

def main():
  # un run da ore va seguito mentre gira: senza questo, stdout rediretto su
  # file resta bufferizzato e non si vede niente fino alla fine
  sys.stdout.reconfigure(line_buffering=True)

  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('command', choices=('train', 'generate'))
  parser.add_argument('--config', choices=tuple(CONFIGS), default='small')
  parser.add_argument('--corpus', default='input_it.txt')
  parser.add_argument('--tokens', type=int, default=1000)
  parser.add_argument('--eval-iters', type=int, default=EVAL_ITERS,
                      help='batch su cui mediare la loss a ogni valutazione')
  parser.add_argument('--max-iters', type=int, default=None,
                      help='sovrascrive max_iters della config (utile per provare)')
  parser.add_argument('--checkpoint', default=None,
                      help='file di pesi da cui generare (default: quello di --config)')
  parser.add_argument('--resume', action='store_true',
                      help='riprende dal checkpoint _last, ottimizzatore compreso')
  parser.add_argument('--device', default=None)
  parser.add_argument('--seed', type=int, default=None,
                      help='default: 1337 per train, casuale per generate')

  args = parser.parse_args()

  try:
    (cmd_train if args.command == 'train' else cmd_generate)(args)
  except BrokenPipeError:
    # generate scrive in streaming: se chi legge chiude prima (`... | head`)
    # non e' un errore, e non deve stampare un traceback
    sys.stdout = None

if __name__ == '__main__':
  main()
