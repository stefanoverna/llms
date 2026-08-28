# mlp

Modello di linguaggio a livello di carattere con una MLP: contesto di 3
caratteri, embedding imparati, un layer nascosto con tanh, 27 logits in uscita.

Dieci script e due lezioni. `01_mlp.py` ricostruisce [Building makemore Part 2:
MLP](https://www.youtube.com/watch?v=TCH_1BHY58I), che segue l'architettura di
[Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf);
`02_optimizations.py` e `03_batchnorm.py` si dividono [Building makemore Part 3:
Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc),
una metà per uno. Gli script dal `04_` in poi non stanno in nessuna lezione:
riscrivono la stessa rete con le astrazioni di PyTorch, e poi le rifanno a mano.

Ogni file che non è uno script si chiama `NN_out_...`, dove `NN` è il numero
dello script a cui appartiene: grafici, trascrizioni e appunti finiscono così
subito sotto il `.py` che li produce o che li riguarda. (Perché funzioni, il
nome dello script deve venire prima di `out` in ordine alfabetico — con `ls` in
locale en_US la punteggiatura viene ignorata, quindi è la prima lettera a
decidere.) Le trascrizioni sono in
`01_out_lecture.srt` (parte 2) e `02_out_lecture.srt` (parte 3, che vale sia per
`02_` che per `03_`).

```sh
uv run 03_mlp/01_mlp.py
```

Ci mette circa un minuto e mezzo: sono due training, 50k e 200k passi.

## Cosa mostra lo script

1. costruzione del dataset (finestra scorrevole di `BLOCK_SIZE` caratteri) e
   split 80/10/10 in train / dev / test
2. l'embedding come lookup table, ovvero come layer lineare su un one-hot
3. la rete scritta a mano (`C`, `W1`, `b1`, `W2`, `b2`), con `view` per
   concatenare gli embedding e la loss calcolata prima a mano e poi con
   `F.cross_entropy`
4. come si sceglie il learning rate, invece di indovinarlo (`01_out_lr_search.png`)
5. training con minibatch e learning rate decay (`01_out_training_loss.png`)
6. visualizzazione degli embedding a 2 dimensioni (`01_out_embeddings.png`): le vocali
   finiscono vicine da sole
7. underfitting, overfitting e quando ci si ferma: train e dev loss uguali
   vogliono dire rete troppo piccola, non overfitting. Da embedding 2D / 100
   neuroni a 10D / 200 neuroni
8. campionamento di nomi da entrambe le reti, con lo stesso seed
9. l'inizializzazione: la rete parte da loss 26 invece che da 3.30, e questo
   costa piu' di qualsiasi altra manopola. E' l'aggancio alla lezione 3

Il bigram si fermava a 2.45. Qui si arriva a circa 2.19 sul dev set, con 11897
parametri, e i nomi campionati iniziano a sembrare nomi.

## 02_optimizations.py

La sezione 9 di `01_mlp.py` si limita a constatare che la rete parte da loss 26
invece che da 3.30. Qui il problema si sistema, seguendo la lezione 3.

```sh
uv run 03_mlp/02_optimizations.py
```

Ci mettera' qualche minuto: allena quattro volte la stessa rete di `01_mlp.py`,
200k passi ciascuna, cambiando solo come partono i pesi.

1. perche' la loss al passo 0 dovrebbe essere `-log(1/27)` e non 26, e cosa
   vuol dire avere logits che vanno da -21 a +26
2. la cura: `W2 * 0.01` e `b2 * 0`, e perche' 0.01 invece di 0
3. il secondo problema, che la loss non rivela: la tanh e' satura al 61%, e il
   fattore `(1 - t^2)` del backward azzera i gradienti (`02_out_init_activations.png`,
   `02_out_init_saturation.png`)
4. la cura: `W1 * 0.2`
5. da dove viene quel numero: `gain / sqrt(fan_in)`, l'inizializzazione di
   Kaiming, e perche' per la tanh il gain e' 5/3

Risultato, che ricalca la lezione: dev da **2.1731** a **2.1296** sistemando i
logits, a **2.1043** sistemando la tanh. Zero parametri in piu', zero tempo in
piu'.

## 03_batchnorm.py

`02_optimizations.py` finisce con una formula per la scala dei pesi, che pero'
vale per un layer lineare con in ingresso una gaussiana. Su una rete profonda
non c'e' piu' niente da calcolare a mano, e la seconda meta' della lezione 3
propone la scorciatoia: se vuoi che le pre-attivazioni siano gaussiane,
normalizzale e basta.

```sh
uv run 03_mlp/03_batchnorm.py
```

Un paio di minuti: un solo training da 200k passi, la stessa rete di
`02_optimizations.py` con una `BatchNorm1d` scritta a mano fra il layer
lineare e la tanh.

1. quanto sono gaussiane le pre-attivazioni con Kaiming, e perche' "quanto" e'
   la parola sbagliata
2. `(x - mean) / std` lungo la dimensione degli esempi, ed e' differenziabile:
   il gradiente ci passa attraverso
3. `bngain` e `bnbias`, perche' la distribuzione deve *partire* gaussiana, non
   restarci
4. il prezzo: gli esempi del batch non sono piu' indipendenti. Lo stesso
   esempio in 500 batch diversi (`03_out_bn_jitter.png`), e perche' quel bug e' anche
   un regolarizzatore
5. l'inferenza su un esempio solo: calibrazione esplicita o media mobile
   durante il training, e perche' valutare in modalita' training e' sbagliato
6. l'epsilon, e il bias del layer precedente che diventa inutile
   (`bias=False`), srotolato passo per passo in `03_out_batchnorm_bias.md`
7. i cinque argomenti di `torch.nn.BatchNorm1d`, uno per uno

Risultato: **2.1095** contro **2.1070** senza. Cioe' niente, ed e' il punto —
con un layer nascosto la scala giusta si calcola a mano, con cinquanta no.

## 08_gain_gradients.py

Lo stesso esperimento di `07_gain_depth.py` guardando indietro: la
distribuzione dei gradienti, layer per layer, al variare del gain. La rete e'
identica, ma invece di una loss vera iniettiamo in cima un gradiente gaussiano
di deviazione standard 1 e misuriamo cosa arriva a ogni profondita'.

```sh
uv run 03_mlp/08_gain_gradients.py
```

|  gain | attivazione, layer 1 → 6 | gradiente, layer 1 → 6 |
| --- | --- | --- |
| 0.5 | 0.42 → 0.01  (x0.03) | 0.03 → 1.00  (x32) |
| 1 | 0.63 → 0.30  (x0.48) | 0.50 → 1.00  (x2.0) |
| 5/3 | 0.76 → 0.66  (x0.86) | 1.45 → 1.00  (x0.69) |
| 3 | 0.86 → 0.84  (x0.97) | 4.67 → 1.00  (x0.21) |

Le due colonne si muovono in verso opposto lungo la profondita', e 5/3 e'
l'unico valore che le tiene entrambe vicine a 1 nello stesso momento. Il
grafico e' in `08_out_gain_gradients.png`. C'e' anche la trappola: il gradiente
sui *pesi* resta quasi piatto anche quando la rete e' mal calibrata, perche' e'
il prodotto di due errori che si compensano — per questo la lezione guarda le
attivazioni e i loro gradienti separatamente.

E' il quadro completo del perche' prima delle normalizzazioni allenare reti
profonde fosse "balancing a pencil on your finger": una manopola sola che muove
forward e backward insieme, in versi opposti.

## 09_learning_speed.py

Il terzo grafico diagnostico, e l'unico che si guarda *durante* il training:
non le attivazioni, non i gradienti, ma di quanto si spostano davvero i pesi a
ogni passo, in rapporto a quanto sono grandi.

```sh
uv run 03_mlp/09_learning_speed.py
```

    update:data = std(learning_rate * p.grad) / std(p)

In `log10` dovrebbe stare intorno a **-3**: un millesimo di sé stesso per
passo. Molto sotto vuol dire learning rate troppo basso, molto sopra vuol dire
pesi ribaltati a ogni passo. Tre scenari, una curva per matrice di pesi:

| scenario | cosa si vede |
| --- | --- |
| `lr = 0.1` | tutti i tensori fra -2.9 e -2.3, piatti. L'ultimo layer parte a -0.9 e scende: e' quello moltiplicato per 0.1 all'init, quindi ogni update e' grande rispetto ai suoi valori |
| `lr = 0.001` | tutto a -4.8/-5.4, due decadi e mezza sotto. I gradienti sono gli stessi: cambia solo il moltiplicatore, e nessun grafico di gradienti se ne accorgerebbe |
| init senza `/sqrt(fan_in)` | dispersione di 1.80 decadi contro 0.60, le curve si aprono a ventaglio: layer dello stesso modello imparano a velocita' diverse di un fattore sessanta |

Il grafico e' in `09_out_learning_speed.png`.

## 10_batchnorm_robustness.py

Le quattro diagnostiche di `07`, `08` e `09` messe una accanto all'altra, la
stessa rete con e senza batchnorm, quattro gain diversi. Risponde a una
domanda sola: con una normalizzazione in mezzo, cosa resta da calibrare a mano?

```sh
uv run 03_mlp/10_batchnorm_robustness.py
```

Quanto si sposta ogni misura passando da `gain 0.5` a `gain 3.0` (un fattore 6):

| misura | senza batchnorm | con batchnorm |
| --- | --- | --- |
| attivazioni | fino a x36 | **x1.00** |
| gradiente sulle attivazioni | fino a x55 | **x1.00** |
| gradiente sui pesi | da x11 a x5, irregolare | x0.17, cioe' esattamente 1/6 |
| update:data | +0.28 decadi (quasi invariato) | -1.56 decadi, cioe' 1/6² |

Le prime due la batchnorm le azzera come problema, ed esattamente: non "molto
meglio", invariante. Le altre due si muovono, ma seguono una legge pulita.

C'e' anche una colonna con la distribuzione delle attivazioni all'ultimo layer
nascosto, nello spirito di `02_out_init_activations.png`, perche' la sola `std`
non dice *come* la rete si rompe: con gain 0.5 e' un picco stretto sullo zero,
con gain 3 sono due mucchi a ±1. La percentuale di saturi (`|h| > 0.97`) lo
riassume in un numero:

| | gain 0.5 | gain 1 | gain 5/3 | gain 3 |
| --- | --- | --- | --- | --- |
| senza batchnorm | 0.0% | 0.0% | 5.6% | 40.6% |
| con batchnorm | 3.2% | 3.2% | 3.2% | 3.2% |

Il ribaltamento e' nell'ultima riga: senza batchnorm l'update:data e' quasi
insensibile al gain, quindi su quella misura la rete non normalizzata *sembra*
piu' robusta. E' la trappola gia' vista in `08`: il rapporto e' il prodotto di
due errori che si compensano, e la compensazione nasconde una rete che intanto
ha le attivazioni morte. Il grafico e' in `10_out_batchnorm_robustness.png`.

## 03_out_batchnorm_bias.md

Un allegato, non uno script: perche' il bias del layer lineare, davanti a una
batchnorm, non fa niente e prende gradiente zero. E' la sezione 8 di
`03_batchnorm.py` srotolata, con le shape scritte a ogni passaggio e un
esempio in miniatura (4 esempi x 3 neuroni invece di 32 x 200).

## 04_mlp_idiomatic.py e 05_mlp_bare.py

La stessa rete delle tre tappe, scritta come si scrive davvero: le astrazioni
di PyTorch al posto di tutto quello che finora era tenuto a mano.

```sh
uv run 03_mlp/04_mlp_idiomatic.py
```

| a mano, nelle tappe 01-03 | qui |
| --- | --- |
| dict di tensori + `forward()` | `nn.Sequential` |
| `C[X]` | `nn.Embedding` |
| `emb.view(N, -1)` | `nn.Flatten` |
| `W1` senza `b1` | `nn.Linear(..., bias=False)` |
| `W1 * (5/3)/sqrt(fan_in)` | `nn.init.kaiming_normal_(..., nonlinearity="tanh")` |
| `bngain`, `bnbias`, `running_*` | `nn.BatchNorm1d` |
| `stats="batch"` / `"running"` | `model.train()` / `model.eval()` |
| `F.cross_entropy` | `nn.CrossEntropyLoss` |
| `torch.randint` per i batch | `DataLoader` |
| `p.grad = None` | `optimizer.zero_grad(set_to_none=True)` |
| `p -= lr * p.grad` | `optimizer.step()` |
| `0.1 if i < steps//2 else 0.01` | `lr_scheduler.MultiStepLR` |

Quello che nessuna libreria scrive al posto tuo resta l'inizializzazione: il
default di `nn.Linear` e' tre volte piu' stretto di quello che serve per la
tanh, e l'ultimo layer va rimpicciolito a parte. Sul secondo punto il default
se la cava meglio di quanto ci si aspetti — la loss 26 delle tappe precedenti
era colpa del `torch.randn` puro, non di PyTorch — e nel file c'e' il perche'
con i numeri. In fondo, le differenze che non sono solo cosmetiche.

`05_mlp_bare.py` e' lo stesso file senza commenti.

## 06_modules_from_scratch.py

Il giro di ritorno: invece di usare le interfacce di PyTorch, le ricostruiamo
con il codice sparso delle tappe 01-03. E' il bonus con cui si chiude la
lezione 3 (`02_out_lecture.srt`, dal minuto 1:18), dove Karpathy "pytorchifica" il
codice per poter impilare i layer in una lista.

```sh
uv run 03_mlp/06_modules_from_scratch.py
```

Riscritti da zero: `nn.Module` (la raccolta dei parametri e il flag
`train`/`eval` ricorsivo), `nn.Linear`, `nn.BatchNorm1d`, `nn.Tanh`,
`nn.Embedding`, `nn.Flatten`, `nn.Sequential`, `F.cross_entropy`, `optim.SGD`,
`MultiStepLR`, `DataLoader`.

Poi la controprova, che e' il motivo per cui vale la pena scriverlo: la stessa
rete costruita due volte, una con i moduli veri e una con i nostri, gli stessi
pesi copiati dentro, e gli output confrontati numero per numero — in `train()`,
in `eval()`, su un batch da uno, e sui buffer dopo un passo. In fondo al file,
cosa manca davvero a queste classi rispetto a `nn.Module` (la registrazione
automatica, `state_dict()`, `.to(device)`, gli hook), che non e' il calcolo.

## 07_gain_depth.py

Perche' il gain della tanh e' 5/3, visto dove si vede: su una rete profonda.
Sei layer lineare+tanh impilati, nessun training, un batch che li attraversa, e
la distribuzione delle attivazioni misurata all'uscita di ogni tanh — due
volte, cambiando solo il gain nell'inizializzazione.

```sh
uv run 03_mlp/07_gain_depth.py
```

| gain = 1 | gain = 5/3 |
| --- | --- |
| dev.std. 0.63 → 0.30 in sei layer, e continua a scendere | si assesta su 0.65 e ci resta anche a quaranta layer |
| saturazione a zero: la tanh lavora solo nel tratto lineare | saturazione ferma intorno al 6% |

Il grafico e' in `07_out_gain_depth.png`: gli istogrammi per layer nei due
casi, piu' un pannello con la deviazione standard in funzione della
profondita'. E' l'esperimento che rende visibile perche' `02_optimizations.py`
moltiplica per 5/3 — su un layer solo quella costante quasi non si nota, ed e'
anche il motivo per cui la lezione a questo punto smette di usare la rete a un
layer nascosto.

Il dataset `names.txt` è quello di `02_bigram/`.
