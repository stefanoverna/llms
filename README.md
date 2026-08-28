# llms

Ricostruzioni delle lezioni della serie [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
di Andrej Karpathy. Ogni esempio sta in una cartella a se', con dentro gli
script, i dati e la trascrizione della lezione da cui e' ricavato (i file
`.srt`).

| cartella | lezione | argomento |
| --- | --- | --- |
| [`01_micrograd/`](01_micrograd) | [building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) | backpropagation da zero, motore di autograd scalare, MLP |
| [`02_bigram/`](02_bigram) | [building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) | modello di linguaggio bigram a livello di carattere |
| [`03_mlp/`](03_mlp) | [building makemore part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I) | MLP alla Bengio 2003, embedding imparati, minibatch, split train/dev/test |
| [`03_mlp/`](03_mlp) | [building makemore part 3: activations & gradients, batchnorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) | inizializzazione, tanh satura, Kaiming, batch normalization |

## Come si lancia

Serve solo [uv](https://docs.astral.sh/uv/), che si occupa di venv e dipendenze:

```sh
uv run 01_micrograd/01_derivate.py   # micrograd e' diviso in nove tappe, in ordine
uv run 02_bigram/01_bigram.py
uv run 02_bigram/02_bigram_nn.py         # lo stesso bigram come rete neurale
uv run 02_bigram/03_bigram_batched.py    # ...e con gli input in forma (B, T)
uv run 03_mlp/01_mlp.py              # lezione 2: la MLP
uv run 03_mlp/02_optimizations.py    # lezione 3, prima meta': l'inizializzazione
uv run 03_mlp/03_batchnorm.py        # lezione 3, seconda meta': la batch normalization
uv run 03_mlp/04_mlp_idiomatic.py    # la stessa rete con nn.Module, optim, DataLoader
uv run 03_mlp/06_modules_from_scratch.py   # e quelle interfacce ricostruite da zero
uv run 03_mlp/07_gain_depth.py       # perche' il gain della tanh e' 5/3
uv run 03_mlp/08_gain_gradients.py   # ...e cosa succede ai gradienti
uv run 03_mlp/09_learning_speed.py     # quanto si spostano i pesi: update:data
uv run 03_mlp/10_batchnorm_robustness.py   # cosa resta da calibrare con la batchnorm
```

Gli script si possono lanciare da qualsiasi directory: i percorsi dei file sono
relativi allo script, non alla directory corrente.
