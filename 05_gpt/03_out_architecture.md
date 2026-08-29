# Schema a blocchi — `03_training.py`, versione a 4 teste

Iperparametri di riferimento: `B = 32` sequenze, `T = 8` posizioni,
`n_embed = 32`, `num_heads = 4`, `head_dim = n_embed // num_heads = 8`,
`vocab_size = 65`.

## Il forward completo

```mermaid
flowchart TD
  idx["idx — (B, T) interi<br/>indici dei caratteri"]

  tok["token_embedding_table<br/>nn.Embedding(65, 32)"]
  pos["position_embedding_table<br/>nn.Embedding(8, 32)"]
  arange["torch.arange(T) — (T)<br/>0, 1, 2 ... T-1"]

  sum(["+ somma<br/>(B, T, 32)"])

  h0["Head 0<br/>(B, T, 8)"]
  h1["Head 1<br/>(B, T, 8)"]
  h2["Head 2<br/>(B, T, 8)"]
  h3["Head 3<br/>(B, T, 8)"]

  cat(["torch.cat dim=-1<br/>(B, T, 32)"])

  ffwd["FeedForward<br/>Linear(32, 32) + ReLU<br/>(B, T, 32)"]
  lm["lm_head<br/>Linear(32, 65)"]
  logits["logits — (B, T, 65)"]
  loss["F.cross_entropy<br/>vs targets (B, T)"]

  idx --> tok
  arange --> pos
  tok -->|"(B, T, 32)"| sum
  pos -->|"(T, 32) broadcast"| sum

  sum --> h0 & h1 & h2 & h3
  h0 & h1 & h2 & h3 --> cat

  cat --> ffwd --> lm --> logits --> loss

  subgraph EMB ["embedding: chi sono + dove sono"]
    tok
    pos
    arange
    sum
  end

  subgraph MHA ["MultiHeadAttention — COMUNICAZIONE"]
    h0
    h1
    h2
    h3
    cat
  end

  subgraph FF ["FeedForward — COMPUTAZIONE, per-token"]
    ffwd
  end
```

Le quattro teste lavorano **in parallelo** sullo stesso input e ognuna scrive in
una fetta disgiunta dell'uscita: le dimensioni 0-7 vengono solo dalla testa 0,
le 8-15 solo dalla testa 1, e così via. Sono spazi `V` diversi e indipendenti,
non le dimensioni dell'embedding di partenza.

## Dentro una singola `Head`

```mermaid
flowchart TD
  xin["x — (B, T, 32)"]

  k["key<br/>Linear(32, 8, bias=False)"]
  q["query<br/>Linear(32, 8, bias=False)"]
  v["value<br/>Linear(32, 8, bias=False)"]

  K["K — (B, T, 8)<br/>cio' che sono"]
  Q["Q — (B, T, 8)<br/>cio' che cerco"]
  V["V — (B, T, 8)<br/>cio' che offro"]

  dot(["Q @ K.transpose(-2,-1)<br/>(B, T, T)"])
  scale(["* head_dim ** -0.5<br/>scaling"])
  mask(["masked_fill(tril == 0, -inf)<br/>maschera causale"])
  soft(["F.softmax(dim=-1)<br/>A — (B, T, T)"])
  agg(["A @ V<br/>(B, T, 8)"])
  out["C — (B, T, 8)"]

  xin --> k --> K
  xin --> q --> Q
  xin --> v --> V

  Q --> dot
  K --> dot
  dot --> scale --> mask --> soft --> agg
  V --> agg
  agg --> out

  tril["tril — buffer (8, 8)<br/>non addestrabile"] -.-> mask
```

`A` è la matrice dei pesi: la riga `t` dice quanto la posizione `t` pesa ognuna
delle posizioni `0..t`. Il triangolo superiore è a `-inf` **prima** della
softmax, cioè peso zero **dopo**: nessuno guarda il futuro.

## Parametri (versione a 4 teste)

| blocco | parametri |
| --- | ---: |
| `token_embedding_table` (65 × 32) | 2 080 |
| `position_embedding_table` (8 × 32) | 256 |
| `sa_heads` — 4 teste × 3 proiezioni × (32 × 8) | 3 072 |
| `ffwd` — Linear(32, 32) + bias | 1 056 |
| `lm_head` — Linear(32, 65) + bias | 2 145 |
| **totale** | **8 609** |

La versione a 1 testa ha esattamente gli stessi 8 609 parametri: cambia solo
come sono spartiti, 1 × (32 → 32) invece di 4 × (32 → 8). Val loss 2.3509
contro 2.2495.

## Cosa NON c'è ancora

```mermaid
flowchart LR
  a["oggi<br/>x = sa_heads(x)<br/>x = ffwd(x)"] --> b["poi<br/>x = x + sa_heads(x)<br/>x = x + ffwd(x)"]
```

- **il flusso residuo**: qui ogni stadio *sostituisce* `x` invece di aggiungergli
  un delta. Dopo l'attention, `tok_emb + pos_emb` non esiste più.
- **la proiezione finale** dentro `MultiHeadAttention` (`proj`), che riporterebbe
  la concatenazione nello spazio degli embedding prima di sommarla al residuo.
- **il fattore 4** nel feed-forward: nel paper è `Linear(32, 128) + ReLU + Linear(128, 32)`.
- **il `Block`** ripetuto in profondità, e i **LayerNorm**.
