# llms

Ricostruzioni delle lezioni della serie [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
di Andrej Karpathy. Ogni esempio sta in una cartella a se', con dentro gli
script, i dati e la trascrizione della lezione da cui e' ricavato
(`lezione.srt`).

| cartella | lezione | argomento |
| --- | --- | --- |
| [`micrograd/`](micrograd) | [building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) | backpropagation da zero, motore di autograd scalare, MLP |
| [`bigram/`](bigram) | [building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) | modello di linguaggio bigram a livello di carattere |

## Come si lancia

Serve solo [uv](https://docs.astral.sh/uv/), che si occupa di venv e dipendenze:

```sh
uv run micrograd/01_derivate.py   # micrograd e' diviso in sei tappe, in ordine
uv run bigram/bigram.py
```

Gli script si possono lanciare da qualsiasi directory: i percorsi dei file sono
relativi allo script, non alla directory corrente.
