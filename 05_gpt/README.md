# gpt

[Let's build GPT: from scratch, in code, spelled
out](https://www.youtube.com/watch?v=kCc8FmEb1nY), la lezione più lunga della
serie: quasi due ore che arrivano a un Transformer decoder allenato su Shakespeare.
La trascrizione intera è in `01_out_lecture.srt`, e vale per tutti gli script
della cartella.

## 01_bag_of_words.py

Il pezzo dal minuto **42:29 all'1:05:00**. Non allena niente — sono tensori
giocattolo attorno a un'idea sola.

```sh
uv run 05_gpt/01_bag_of_words.py
```

Un secondo scarso.

**L'idea**: i token di una sequenza si devono influenzare a vicenda, e in una
direzione precisa, il passato sul presente. Il punto di partenza è un tensore
`(B, T, C)` — `B` sequenze, `T` posizioni ciascuna, `C` numeri per posizione —
in cui le `T` posizioni non si parlano affatto. Vogliamo che la posizione `t`
risenta di tutte quelle da `0` a `t`, e di nessuna oltre, perché oltre c'è
quello che stiamo cercando di indovinare.

"Risentire di" non è ancora un calcolo, quindi si sperimenta, partendo dalla
forma più stupida che funzioni: **la media** di tutto quello che precede. È
un'interazione debolissima — l'ordine dei token precedenti va perso, resta solo
il mucchio, ed è per questo che si chiama *bag of words* — ma è già passato che
entra nel presente, e alla self-attention basterà cambiare i pesi di quella
media per avere il resto.

**La parte tecnica**, che è quella che occupa il segmento: quella media si
scrive come un prodotto fra matrici, e allora si calcola per tutte le posizioni
in una volta sola.

Lo stesso identico calcolo, tre volte:

| | come | a cosa serve |
| --- | --- | --- |
| 1 | due `for` annidati, `x[b, :t+1].mean(0)` | si legge, e fissa il numero da ottenere |
| 2 | `wei @ x`, con `wei` triangolare inferiore | il trucco |
| 3 | `softmax(zeros.masked_fill(tril == 0, -inf))` | la forma che userà la self-attention |

Lo script verifica con `torch.allclose` che le tre versioni diano gli stessi
numeri, e in mezzo c'è l'esempio in miniatura della lezione — `3x3 @ 3x2`, con
numeri abbastanza piccoli da seguirli a mente — che mostra il passaggio in tre
tappe:

    tutti uno            ->  ogni riga è la somma di tutte le righe
    torch.tril           ->  gli zeri spengono le righe future
    righe che fanno 1    ->  le somme diventano medie

Cioè: una moltiplicazione fra matrici è, riga per riga, una somma pesata delle
righe della seconda matrice, e i pesi li decidiamo noi nella prima. "Media di
tutto il passato" non è un ciclo, è una matrice.

### Perché la terza versione

Nella versione 2 i pesi *sono* il triangolo, e il triangolo è una costante.
Nella 3 i pesi partono da una matrice separata — per ora tutta zeri — e il
triangolo interviene solo a spegnere il futuro.

Quei numeri sono le affinità: quanto la posizione `t` trova interessante la
posizione `t'`. Nella self-attention non saranno più zeri, li calcoleranno i
token stessi, e ogni posizione si costruirà la sua media pesata invece di
prendersi quella uniforme. Maschera, softmax e moltiplicazione restano
identiche a queste.

Si vede anche perché la maschera è `-inf` e non `0`: `-inf` è lo zero *prima*
della softmax. Mettere `0` direbbe "peso medio", non "peso nullo".

### Il buco lasciato aperto (sezione 6, 1:02:00-1:05:00)

L'ultima sezione dello script è solo commenti: la lezione qui dice cosa vuole
prima di scrivere come si fa, e il codice arriva nel file successivo.

Il difetto della media uniforme è che è una scelta fatta a priori: i pesi sono
una costante che non dipende né dai token né dall'addestramento. Ma se sono
una vocale mi interessano soprattutto le consonanti che ho alle spalle, non il
mucchio indistinto — i pesi devono essere *data dependent*.

Il meccanismo: ogni posizione emette due vettori, una **query** ("che cosa sto
cercando") e una **key** ("che cosa contengo"), e l'affinità fra `t` e `t'` è
il prodotto scalare fra la query di `t` e la key di `t'`. Due proiezioni
diverse dello stesso vettore, non il vettore stesso: altrimenti l'affinità
sarebbe decisa dall'embedding, non ci sarebbe niente da imparare, e sarebbe
simmetrica — mentre "quanto `t` si interessa a `t'`" e il contrario sono due
domande diverse.

Di tutto il file cambia una riga:

    wei = torch.zeros(T, T)   ->   wei = q @ k.transpose(-2, -1)

Maschera, softmax e moltiplicazione restano dove sono. La causalità in
particolare non c'entra con le query e le key, che si calcolano su tutte le
posizioni in parallelo e senza guardarsi: a spegnere il futuro continua a
pensarci `masked_fill`.

### Che cos'è x (sezioni 7 e 8, 58:21-1:05:00)

Nelle prime sei sezioni `x` è `torch.randn`: serviva solo qualcosa da mediare.
Ma `key` e `query` sono funzioni di `x[t]` **e di nient'altro**, quindi quello
che c'è dentro `x` decide cosa la self-attention può guardare — e allora vale
la pena costruirlo davvero. La catena è questa, e i due estremi non cambieranno
più fino alla fine della lezione:

    idx  →  token embedding  →  + position embedding  →  x  →  … →  lm_head  →  logits

| layer | a cosa serve |
| --- | --- |
| `nn.Embedding(vocab_size, n_embd)` | un id è un nome, non una quantità: `'c' = 2` non vuol dire "sta fra `b` e `d`". Una riga imparata per simbolo, in uno spazio dove la distanza significa qualcosa — cosa, lo decide la loss. Non sa niente di posizione né di contesto: stesso carattere, stessa riga, sempre |
| `nn.Embedding(block_size, n_embd)` | una riga imparata per casella, **sommata** alla prima, perché `x` dica anche *dove* sono e non solo *cosa* sono |
| `key`, `query` — `nn.Linear(n_embd, head_size, bias=False)` | due proiezioni diverse dello stesso `x`: "cosa contengo" e "cosa sto cercando". Applicate a ogni posizione in parallelo e in isolamento: qui non comunica ancora nessuno |
| `q @ k.transpose(-2, -1)` | il momento in cui le posizioni si incontrano: riga `t`, colonna `t'` è `q[t] · k[t']`, la *mia* query contro le key *degli altri* |
| `masked_fill` + `softmax` | quelli sopra sono punteggi, non pesi: numeri qualsiasi, righe che non sommano a niente. La maschera spegne il futuro, la softmax fa di ogni riga una distribuzione |
| `wei @ x` | la somma pesata, identica alla sezione 3 |
| `nn.Linear(n_embd, vocab_size)` | il *language modeling head*: da "il vettore che descrive questa posizione" a un punteggio per carattere |

Tre conseguenze che lo script verifica o mostra:

- **`wei` non è un parametro.** I pesi di una testa sono solo `key` e `query`
  (e la `value` che arriva subito dopo). La matrice `T × T` si ricalcola ogni
  volta dai dati, ed è `(B, T, T)`: `B` matrici diverse, una per sequenza.
  `tril` invece era una sola, costante, uguale per tutti.
- **L'affinità è asimmetrica.** Due matrici separate invece del prodotto
  `x[t] · x[t']`, che sarebbe simmetrico e deciso dall'embedding, senza niente
  da imparare. "Quanto `t` si interessa a `t'`" e il contrario sono due
  domande diverse.
- **L'ordine non lo introduce l'attenzione.** La sezione 8b rimescola i primi
  sette caratteri di ogni sequenza e guarda l'ultima posizione: con `x` fatto
  del solo token embedding l'output è **identico**, perché una somma non si
  accorge dell'ordine. Con la posizione dentro `x`, cambia. L'attenzione
  aggrega e basta; l'ordine entra da `x`, ed è per questo che la tabella delle
  posizioni esiste.

Quello che manca alla testa della sezione 8 per essere quella vera: la `value`
(si aggrega `x`, non `value(x)`), il fattore `1/sqrt(head_size)` sui punteggi,
più teste in parallelo, e il resto del blocco. Niente di tutto questo tocca lo
scheletro — punteggi, maschera, softmax, moltiplicazione.

### Quanto costa il ciclo

La lezione dice che i due `for` sono inefficienti e passa oltre. Con
`(4, 8, 2)` non si vede — sono 32 medie — quindi lo script rimisura su
`(64, 256, 64)`, che sono le dimensioni di un GPT giocattolo:

| | tempo |
| --- | --- |
| due `for` annidati | ~113 ms |
| `wei @ x` | ~1.5 ms |

Circa **75x**, e il divario cresce con `T`: il ciclo fa `T` passi, la
moltiplicazione resta una chiamata sola. Non è un dettaglio da ottimizzatori —
è il motivo per cui l'architettura è allenabile.


## 02_self_attention.py

Il pezzo dall'**1:05:00 all'1:11:00**: la self-attention vera, prima a parole e
poi in codice.

```sh
uv run 05_gpt/02_self_attention.py
```

Istantaneo: stampa una shape e basta. Il valore del file sta nei commenti.

Tre tappe, tre funzioni.

**`lezione1()`** rifà in tre righe la media pesata del file 01, per avere sotto
gli occhi il punto di partenza: `wei` costruita a mano, uguale per tutti.

**`lezione2()`** è il cuore, ed è tutta commenti. Prende cinque token —
*the cat ate the mat* — e ipotizza una testa specializzata in **accoppiare i
nomi con i loro articoli**. Da lì costruisce a mano, con numeri inventati ma
plausibili, le due proiezioni:

| | domanda | esempio |
| --- | --- | --- |
| `K = X · Wk` | "che cosa sono" | `cat` → `[ART 0, NOME 1, VERBO 0, POS 2]` |
| `Q = X · Wq` | "che cosa cerco" | `cat` → `[ART 3, NOME 0, VERBO 0, POS 0.2]`, cioè "voglio fortemente un articolo, e a parità di tutto preferisco chi mi sta vicino" |

E poi mostra perché il prodotto scalare è l'operazione giusta per far
incontrare domanda e offerta: `Q · K^T` associa i numeri grandi dell'una con
quelli grandi dell'altra, quindi il punteggio è alto esattamente dove c'è un
match. Le tabelle nel file portano il calcolo fino in fondo, e si vede la riga
di `cat` illuminarsi sulle due occorrenze di `the`.

Vale la pena leggerlo perché è la parte che il codice *non* dice: nel codice
`Wk` e `Wq` sono due `nn.Linear` inizializzate a caso, e cosa ci finisca dentro
dopo l'addestramento non lo sa nessuno. Lo spazio a quattro dimensioni
`[articolo, nome, verbo, posizione]` è un'illustrazione, non quello che la rete
impara davvero — ma è il modo per capire *che tipo* di cosa sta imparando.

**`lezione3()`** scrive la `Head` come `nn.Module`, che da qui in poi resta
uguale in tutti i file successivi:

- `key`, `query`, `value` sono tre `nn.Linear(n_embed, head_dim, bias=False)`.
  `nn.Linear(a, b)` calcola `x @ W.T` con `W` di shape `(b, a)`: è la nostra
  `X · Wk`, solo memorizzata trasposta;
- `tril` sta in un `register_buffer`, non in un parametro: segue il
  `.to(device)` e finisce nello `state_dict`, ma l'ottimizzatore non lo tocca;
- `A = Q @ K.transpose(-2, -1) * head_dim**-0.5` — si trasponogono le *ultime
  due* dimensioni, non le prime due, perché la prima è il batch;
- la maschera usa `self.tril[:T, :T]`, perché la sequenza in ingresso può
  essere più corta di `block_size`.

Il fattore `head_dim**-0.5` non è cosmetico: senza, i punteggi crescono con la
dimensione della testa, la softmax satura verso una one-hot, e i gradienti che
la attraversano vanno a zero. È lo stesso problema della `tanh` in saturazione
di `03_mlp`, e tornerà — con la LayerNorm — nel file 05.

## 03_training.py

Dall'**1:11:00 all'1:26:00**: il primo training vero.

```sh
uv run 05_gpt/03_training.py
```

Circa un minuto: due modelli da 5000 iterazioni ciascuno, su CPU.

Rispetto al bigram della lezione 2 cambiano tre cose. Gli embedding non sono
più direttamente i logits: c'è uno spazio latente da `n_embed = 32` e un
`lm_head` che lo proietta sul vocabolario. C'è una tabella di embedding **per
la posizione**, sommata a quella del token. E in mezzo c'è la self-attention.

Il file introduce due pezzi nuovi:

**`MultiHeadAttention`** — più teste in parallelo, e i loro output concatenati.
`num_heads * head_dim == n_embed`, quindi quattro teste da 8 al posto di una da
32 hanno lo *stesso numero di parametri*: non stiamo aggiungendo capacità,
stiamo distribuendola. Una testa sola sa fare una cosa sola; con quattro,
ognuna si sceglie il suo mestiere. È `nn.ModuleList` e non una lista Python,
altrimenti `parameters()`, `.to(device)` e `state_dict` non vedrebbero le teste.

**`FeedForward`** — la self-attention è **comunicazione**: i token si guardano
e raccolgono dati. Ma da lì si andava dritti ai logits, senza che nessuno
avesse un attimo per *ragionare* su quello che aveva appena raccolto. Il feed
forward è **computazione** pura, per-token: `nn.Linear` lavora sull'ultima
dimensione, quindi applicato a `(B, T, n_embed)` tratta ogni posizione per
conto suo. Qui non si comunica.

Lo script allena la versione a 1 testa e quella a 4, con lo stesso seed
reinizializzato dentro `train()` — così le due corse vedono la stessa
inizializzazione e gli stessi batch, e i due testi generati sono confrontabili
riga per riga.

| | val loss |
| --- | --- |
| 1 testa da 32 | 2.3509 |
| 4 teste da 8 | **2.2495** |

Entrambi i modelli hanno 8609 parametri.

## 04_blocks.py

Dall'**1:26:00 all'1:32:00**: *intersperse communication with computation*.

```sh
uv run 05_gpt/04_blocks.py
```

Circa quattro minuti: tre modelli da 5000 iterazioni.

Nel file 03 il flusso era: embedding → self-attention → feed forward → logits.
Una volta sola. Il Transformer del paper impila lo stesso schema `N` volte, e
quel gruppetto — parla, poi ragiona — è il **`Block`**.

Ma impilare non basta, ed è il motivo per cui questo file allena *tre* modelli
invece di uno:

| | parametri | val loss |
| --- | --- | --- |
| 1 blocco (file 03) | 8 609 | 2.2495 |
| 3 blocchi, niente residui | 16 865 | **2.3400** |
| 3 blocchi + residui e proiezioni | 23 201 | 2.1341 |
| 3 blocchi + residui + feed forward ×4 | 41 921 | **2.0705** |

Tre blocchi ingenui, con il doppio dei parametri, fanno **peggio** di un blocco
solo. Non è capacità che manca: è che una rete profonda senza una corsia per il
gradiente non si ottimizza.

I due rimedi, entrambi dal paper:

- **i residui** (skip connections, da ResNet 2015). `x = x + sa(x)` invece di
  `x = sa(x)`: il flusso non viene mai sostituito, ci si scosta di lato, si
  calcola qualcosa, e si rientra sommando. Il motivo è il backward: la somma
  distribuisce il gradiente identico a entrambi i rami, quindi c'è
  un'autostrada che porta il gradiente dalla loss fino agli embedding senza
  attraversare nessuna moltiplicazione. E all'inizio i rami laterali contano
  quasi nulla — la rete profonda nasce quasi come una rete che non fa niente,
  quindi facile da ottimizzare, e i blocchi "si accendono" col training;
- **le proiezioni**. La concatenazione delle teste è un accrocchio: pezzi di
  vettore prodotti da teste che non si sono mai parlate, appiccicati uno dopo
  l'altro. La `proj` li rimescola e li riporta nella lingua del flusso residuo,
  prima di sommarli dentro;
- **il feed forward largo**: `32 → 128 → 32`, con il ×4 del paper
  (`512 → 2048`). È lì che sta la computazione vera — ci si espande, si applica
  la non linearità, e si torna giù.

Lo schema è in `04_out_architecture.png`.

## 05_layernorm.py

Dall'**1:32:00 all'1:41:00**: le ultime due rifiniture, e la scala.

```sh
uv run 05_gpt/05_layernorm.py          # 'small': confrontabile col file 04, ~3 minuti
uv run 05_gpt/05_layernorm.py big      # lo scale-up: 10.8M parametri, ore su CPU
```

Con questo file il modello è un **GPT**: un Transformer decoder-only completo,
identico nell'architettura a quello del paper a meno del ramo encoder e della
cross-attention, che non servono perché non stiamo traducendo.

**La LayerNorm.** La testata del file spiega per esteso *quale problema
risolve*, che è lo stesso di `03_mlp` parte 3: tenere le attivazioni in un
intervallo ragionevole man mano che la rete si fa profonda. Kaiming
(`gain / sqrt(fan_in)`) sistema le cose una volta sola, all'inizio, e un layer
alla volta. Coi residui il problema si aggrava, perché ogni blocco *somma* il
suo contributo dentro `x`: la scala cresce di blocco in blocco. E se `x` arriva
grosso alla self-attention, le affinità sono grosse, la softmax satura e i
gradienti spariscono — la `tanh` in saturazione di makemore, con un altro nome.

La differenza con la BatchNorm sta in *quali numeri si mediano*:

| | media su | conseguenza |
| --- | --- | --- |
| BatchNorm | per ogni feature, tutti gli esempi del batch (una colonna) | gli esempi si "parlano" |
| LayerNorm | per ogni esempio, tutte le sue feature (una riga) | ogni token per conto suo |

Ed è questa scelta a far sparire i guai della BatchNorm: niente accoppiamento
fra esempi finiti per caso nello stesso batch, niente medie mobili da
accumulare, nessuna differenza fra training e inference. Cosa che qui non è un
dettaglio: con la BatchNorm la predizione per un token dipenderebbe dalle altre
frasi del batch, e in generazione il batch è comunque di 1.

**Attenzione: qui si devia dal paper.** Nella figura originale è *Add & Norm*,
la norma **dopo** il sotto-livello; oggi si usa quasi sempre la variante
**pre-norm**, con la norma **prima**, così il flusso residuo resta una corsia
pulita di sole somme. Il diagramma `05_out_architecture.png` lo mostra: i
LayerNorm gialli stanno dentro la deviazione laterale, non sul flusso.

**Il dropout** sta in tre punti: sulle affinità dentro ogni `Head` subito dopo
la softmax, e subito prima di ogni rientro nel flusso residuo (dopo la `proj`
della multi-head e dopo il secondo `Linear` del feed forward). In `small` è a
zero: a quella taglia il modello è in underfitting, regolarizzarlo farebbe solo
danni.

| | val loss |
| --- | --- |
| senza LayerNorm | 2.0705 |
| con LayerNorm | **2.0563** |

Guadagno piccolo, e il motivo è nel numero stesso: tre blocchi da 32 dimensioni
sono pochi, non c'è granché da stabilizzare. Il senso della LayerNorm si vede
nella configurazione `big`, a sei blocchi da 384.

## 06_gpt.py

Non è una tappa della lezione: è il `05` riscritto in forma operativa, per
quando il modello va allenato sul serio invece che letto.

```sh
uv run 05_gpt/06_gpt.py train                        # 'small', corpus italiano
uv run 05_gpt/06_gpt.py train --config big
uv run 05_gpt/06_gpt.py train --config big --resume  # riprende un run interrotto
uv run 05_gpt/06_gpt.py generate --tokens 2000       # dai pesi salvati, senza ri-addestrare
```

Cosa cambia rispetto al 05:

- **niente variabili globali**: ogni classe riceve i suoi parametri, ed è
  precisamente quello che serve per ricostruire un modello a partire da un
  checkpoint;
- **`CharDataset`** al posto delle lambda `encode`/`decode`, con il vocabolario
  iniettabile dall'esterno;
- **checkpoint**: `06_out_model_<config>.pt` tiene i pesi con la val loss più
  bassa — non gli ultimi, che se la val risale sono i peggiori che abbiamo
  avuto in mano. `06_out_model_<config>_last.pt` tiene l'ultimo stato completo
  di ottimizzatore, per `--resume`. Dentro ci vanno anche config, corpus e
  vocabolario: senza quei tre, i pesi da soli non bastano;
- **diagnostica**, nello spirito dei controlli di `03_mlp`, salvata in
  `06_out_training_<config>.png`.

I quattro pannelli:

| pannello | cosa dice |
| --- | --- |
| loss train/val | se il divario si allarga, il modello sta memorizzando invece di generalizzare |
| `update:data` | `log10( std(update) / std(pesi) )` per famiglia di tensori, con il riferimento a −3 del video `09_learning_speed.py`. **Nota**: lì l'update era `lr * grad` perché l'ottimizzatore era SGD; qui è AdamW, che rinormalizza il gradiente, quindi si misura la differenza vera dei pesi prima e dopo lo step |
| entropia dell'attenzione | per ogni riga di `A`, l'entropia normalizzata sul massimo possibile a quella posizione: **1** = la testa media su tutto il contesto, **0** = softmax satura su un token solo. Una curva per **testa**, colore per blocco — mediando sulle teste si perderebbe proprio la specializzazione |
| RMS del flusso residuo | la scala in ingresso a ogni blocco: il numero che cresce con la profondità, e il motivo per cui la LayerNorm serve |

Il corpus di default è `input_it.txt` (Pirandello, 979k caratteri, vocabolario
di 75); con `--corpus input.txt` si torna a tinyshakespeare (1.1M caratteri,
vocabolario di 65).

Un risultato che vale la pena guardare, dal run `small` in italiano: nel
**blocco 0** le quattro teste si separano nettamente — una scende a **0.22** di
entropia, molto selettiva, un'altra resta a **0.84**, quasi una media uniforme.
Nel **blocco 2** stanno tutte fra 0.55 e 0.62. Le medie dei due blocchi sono
quasi identiche: senza le curve per testa, le due situazioni sarebbero
indistinguibili. E partono tutte da ~0.98, cioè da un'attenzione uniforme: la
specializzazione non è progettata, emerge dal training a partire dalla sola
inizializzazione casuale.

## I diagrammi

`03_out_architecture`, `04_out_architecture` e `05_out_architecture` sono lo
stesso modello a tre stadi di completezza, disegnati in TikZ nello stile della
Figura 1 di *Attention Is All You Need* — di cui questi sono in pratica la metà
destra, il decoder. Ogni `.tex` è un documento `standalone` autosufficiente, e
la nota tratteggiata in fondo elenca cosa manca ancora rispetto al paper.

Per rigenerare i PDF serve un TeX. Il più leggero è
[tectonic](https://tectonic-typesetting.github.io/), un singolo eseguibile che
si scarica da sé solo i pacchetti che servono:

```sh
brew install tectonic
tectonic 05_gpt/05_out_architecture.tex
```
