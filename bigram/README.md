# bigram

Modello di linguaggio bigram a livello di carattere: conta le coppie di lettere
che compaiono nel dataset, le normalizza in probabilita' e campiona nomi nuovi.

Ricostruzione della lezione [The spelled-out intro to language modeling:
building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) fino a 1h03m: si
ferma subito prima della riformulazione del bigram come rete neurale. La
trascrizione e' in `lezione.srt`.

```sh
uv run bigram/bigram.py
```

## Cosa mostra lo script

1. caricamento del dataset
2. conteggio dei bigrammi (prima con un dict, poi con un tensore 27x27)
3. visualizzazione della matrice dei conteggi (salvata in `bigram_counts.png`)
4. normalizzazione in probabilita' (broadcasting + `keepdim`)
5. campionamento di nomi nuovi con `torch.multinomial`
6. valutazione del modello con la negative log likelihood
7. model smoothing (+1)

Il dataset `names.txt` viene da https://github.com/karpathy/makemore.
