# bigram

Modello di linguaggio bigram a livello di carattere: conta le coppie di lettere
che compaiono nel dataset, le normalizza in probabilita' e campiona nomi nuovi.

Ricostruzione della lezione [The spelled-out intro to language modeling:
building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) fino a 1h03m: si
ferma subito prima della riformulazione del bigram come rete neurale. La
trascrizione e' in `lezione.srt`.

```sh
uv run 02_bigram/01_bigram.py
```

## Cosa mostra lo script

1. caricamento del dataset
2. conteggio dei bigrammi (prima con un dict, poi con un tensore 27x27)
3. visualizzazione della matrice dei conteggi (salvata in `02_out_bigram_counts.png`)
4. normalizzazione in probabilita' (broadcasting + `keepdim`)
5. campionamento di nomi nuovi con `torch.multinomial`
6. valutazione del modello con la negative log likelihood
7. model smoothing (+1)

Il dataset `names.txt` viene da https://github.com/karpathy/makemore.

## 02_bigram_nn.py

Lo stesso modello riformulato come rete neurale (la lezione da 1h03m in poi),
scritto in PyTorch idiomatico: `nn.Embedding` al posto di `one_hot @ W`,
`F.cross_entropy` al posto di exp / normalizza / log / media, `weight_decay` al
posto del model smoothing, `optimizer.step()` al posto dell'aggiornamento a
mano.

```sh
uv run 02_bigram/02_bigram_nn.py
```

Non conta niente: parte da pesi a caso e li fa scendere con la gradient
descent. Alla fine `softmax(W)` e' la stessa tabella di probabilita' di
`01_bigram.py`, e i nomi campionati sono gli stessi. Il punto non e' fare meglio,
e' che questa strada scala a contesti piu' lunghi mentre la matrice dei
conteggi no.

## 03_bigram_batched.py

Lo stesso identico modello, con gli input in forma `(B, T)` invece che in una
lista piatta di coppie. Non c'e' nessuna lezione nuova dietro: e' una tappa di
preparazione, perche' da qui in avanti tutti gli esempi hanno i tensori in
questa forma e conviene vederla nascere su un modello gia' noto.

```sh
uv run 02_bigram/03_bigram_batched.py
```

Cosa mostra:

1. il dataset come corpus continuo invece che come lista di coppie (con un
   assert che verifica che i bigrammi siano esattamente gli stessi)
2. `get_batch`, che ritaglia finestre da `block_size + 1` caratteri
3. che `nn.Embedding` non si accorge della forma: `emb(xb)` e
   `emb(xb.reshape(-1))` danno numero per numero lo stesso risultato
4. che `F.cross_entropy` invece se ne accorge, e vuole `.view(-1, C)`
5. il training a minibatch, che con 178 volte meno lavoro arriva alla stessa
   loss, e il learning rate che va ritarato perche' il gradiente ora e' rumoroso
6. la generazione in forma `(B, T)`, con `logits[:, -1, :]`

Le tre lettere: B "batch" (sequenze indipendenti), T "time" (posizione dentro
la sequenza, nome ereditato dalle RNN), C "channels" (quanti numeri descrivono
una posizione, nome ereditato dalle convoluzionali). Per il bigram B e T sono
indistinguibili -- sono entrambi "batch" -- e il file lo verifica con degli
assert invece di raccontarlo.
