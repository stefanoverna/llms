# llms

Ricostruzioni commentate di lezioni e capitoli sulle reti neurali. Il grosso e'
la serie [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
di Andrej Karpathy; l'ultima cartella prende invece il problema e i dati dal
capitolo 1 di [Neural Networks and Deep
Learning](http://neuralnetworksanddeeplearning.com/chap1.html) di Michael
Nielsen, e li rifa' con quello che le lezioni precedenti hanno stabilito. Ogni
esempio sta in una cartella a se', con dentro gli
script, i dati e il materiale di partenza (per le lezioni video, la trascrizione
nei file `.srt`).

| cartella | lezione | argomento |
| --- | --- | --- |
| [`01_micrograd/`](01_micrograd) | [building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) | backpropagation da zero, motore di autograd scalare, MLP |
| [`02_bigram/`](02_bigram) | [building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) | modello di linguaggio bigram a livello di carattere |
| [`03_mlp/`](03_mlp) | [building makemore part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I) | MLP alla Bengio 2003, embedding imparati, minibatch, split train/dev/test |
| [`03_mlp/`](03_mlp) | [building makemore part 3: activations & gradients, batchnorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) | inizializzazione, tanh satura, Kaiming, batch normalization |
| [`04_wavenet/`](04_wavenet) | [building makemore part 5: building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) | contesto piu' lungo, fusione gerarchica del contesto, batchnorm su tensori a 3 assi |
| [`05_gpt/`](05_gpt) | [let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) | il Transformer decoder, un pezzo per volta |
| [`06_image_classification/`](06_image_classification) | Nielsen, [capitolo 1](http://neuralnetworksanddeeplearning.com/chap1.html) | classificare immagini: MNIST con una MLP, e cosa cambia a scriverla bene |

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
uv run 04_wavenet/01_wavenet.py      # lezione 5: la MLP che diventa un albero
uv run 05_gpt/01_bag_of_words.py     # lezione 6: dalla media del passato...
uv run 05_gpt/02_self_attention.py   # ...alla prima testa di self-attention
uv run 05_gpt/03_training.py         # il primo training vero: 1 testa contro 4
uv run 05_gpt/04_blocks.py           # i Block impilati, e i residui che li rendono allenabili
uv run 05_gpt/05_layernorm.py        # LayerNorm e dropout: da qui in poi e' un GPT
uv run 05_gpt/06_gpt.py train        # la versione operativa: checkpoint, resume, diagnostica
uv run 06_image_classification/01_network.py    # MNIST: la MLP del capitolo 1, scritta bene
```

Gli script si possono lanciare da qualsiasi directory: i percorsi dei file sono
relativi allo script, non alla directory corrente.
