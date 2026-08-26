# micrograd

Backpropagation da zero, in Python puro: nessun tensore, nessuna libreria, solo
scalari e la regola della catena.

Ricostruzione della lezione [The spelled-out intro to neural networks and
backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
(2h25m, per intero). La trascrizione e' in `lezione.srt`.

[`appunti.md`](appunti.md) riassume a parole il quadro d'insieme: com'e' fatto
un MLP, cos'e' un gradiente e com'e' fatto un loop di training.

## Le tappe

Vanno lette e lanciate in ordine: ognuna aggiunge un pezzo alla precedente.

| | | |
| --- | --- | --- |
| `01_derivate.py` | che cos'e' una derivata, stimata numericamente | nessun `Value` |
| `02_grafo.py` | il grafo delle espressioni, backpropagation a mano con verifica numerica, un passo lungo il gradiente | `Value` #1, inline |
| `03_backward.py` | `backward()`: chain rule locale, ordinamento topologico, e perche' i gradienti si accumulano | `Value` #2, inline |
| `04_neurone.py` | un neurone con `tanh`, poi la stessa `tanh` spezzata in `exp`/somma/divisione: stessi gradienti | `engine.py` |
| `05_pytorch.py` | la controprova: gli stessi numeri, con PyTorch | solo `torch` |
| `06_mlp.py` | una MLP da 41 parametri, la loss, il training loop, e il bug dello `zero_grad` | `engine.py` + `nn.py` |
| `07_mlp_pytorch.py` | la stessa MLP e lo stesso loop, riscritti col minimo di PyTorch: stessi numeri | solo `torch` |
| `08_mlp_idiomatico.py` | ancora la stessa rete, ma con `nn.Sequential`, `nn.MSELoss` e `optim.SGD`: PyTorch come si scrive davvero | solo `torch` |
| `09_mlp_nudo.py` | la tappa 8 senza una riga di commento: 30 righe in tutto | solo `torch` |

```sh
uv run micrograd/03_backward.py
```

La tappa 7 riparte dagli stessi 41 numeri della 6, estratti con lo stesso seed
e nello stesso ordine: la traiettoria della loss coincide cifra per cifra. Usa
solo tensori, `requires_grad` e `backward()` — niente `nn.Linear`, niente
`nn.Module`, niente `optim.SGD` — cosi' si vede cosa sostituisce cosa:
`engine.py` diventa l'autograd dei tensori, `nn.py` tre righe di
moltiplicazione fra matrici, il resto resta identico.

La tappa 8 rifa' la 7 con le astrazioni vere, e chiude il giro: il training loop
si riduce alle cinque righe che si trovano identiche in qualsiasi progetto
PyTorch. La traiettoria della loss pero' non coincide piu' con quella delle
tappe 6 e 7, e il file finisce spiegando perche' — inizializzazione di
`nn.Linear`, `float32`, `reduction` di `MSELoss`: le scelte di default che la
libreria fa al posto tuo.

Le tappe 2 e 3 hanno la **loro** versione di `Value` scritta dentro al file, non
importata. E' voluto: il `Value` della tappa 2 sa costruire il grafo ma non sa
derivare, ed e' proprio quello che rende evidente perche' serve `_backward`; il
`Value` della tappa 3 sa derivare ma conosce solo `+` e `*`, ed e' quello che
rende evidente perche' serve `tanh`. Dalla tappa 4 in poi l'engine e' finito e
si importa, come fa la lezione quando alla fine apre i due file veri.

## La libreria

- **`engine.py`** — il motore di autograd: la classe `Value` completa. Avvolge
  uno scalare, registra da quali nodi e con quale operazione e' stato prodotto,
  e sa applicare la propria derivata locale. `backward()` ordina il grafo
  topologicamente e propaga i gradienti all'indietro.
- **`nn.py`** — `Neuron`, `Layer`, `MLP`, con l'API di `torch.nn`
  (`parameters()`, `zero_grad()`). Sono ~50 righe: il lavoro vero lo fa il
  motore.

Il grafo delle espressioni viene stampato come albero indentato (`print_graph`,
in `engine.py`) invece che con graphviz, per non tirarsi dietro una dipendenza
di sistema.
