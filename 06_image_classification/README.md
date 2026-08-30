# image classification

Il problema e i dati del capitolo 1 di [Neural Networks and Deep
Learning](http://neuralnetworksanddeeplearning.com/chap1.html) di Michael
Nielsen — 784 pixel in ingresso, dieci cifre in uscita, una MLP con un layer
nascosto — scritti però con quello che le cartelle precedenti hanno stabilito.

Non è una ricostruzione. Il capitolo è del 2015 e ci sono dentro scelte che
`03_mlp/` e `05_gpt/` ci hanno insegnato a non fare più: qui si fa direttamente
la versione buona, e i motivi sono nel file.

```sh
uv run 06_image_classification/01_network.py    # ~7 minuti
uv run 06_image_classification/mnist.py         # guarda i dati e basta
open 06_image_classification/02_demo.html       # disegna una cifra e falla riconoscere
```

## I dati

`data/mnist.pkl.gz` (16 MB) è il file del repository di Nielsen preso così
com'è: 70.000 cifre scritte a mano, 28x28 in scala di grigi, divise in

    50.000   train        ci si allena
    10.000   validation   ci si decide tutto
    10.000   test         si guarda una volta sola, alla fine

`mnist.py` è l'unico modulo condiviso. Lanciato da solo stampa le forme e
disegna la prima immagine in ASCII, che è un modo veloce per convincersi che i
dati sono quelli giusti:

```
                                ==++**..**@@@@==
                ....--****@@@@@@@@@@%%**@@@@##::
              ..@@@@@@@@@@@@@@@@@@@@------::..
                %%@@@@@@@@@@####@@@@
                --**==@@@@%%    ..**
                      **@@--
```

## 01_network.py

    784 ingressi     i pixel, 28x28 appiattiti
    800 nascosti     tanh
     10 uscite       logits, nessuna attivazione sopra

636.010 parametri, 60 epoche, sette minuti su CPU. **99.19% su test**, 81 errori
su 10.000, contro il 95.42% che riporta il capitolo con la stessa architettura.

Il punto è che l'ultimo punto percentuale — quello che porta dal 98% al 99% —
**non viene dalla rete, viene dai dati**. L'architettura resta quella del
capitolo: un layer nascosto, `tanh`, logits. Quello che si aggiunge sono le
distorsioni elastiche, ed è il pezzo che vale di più di tutti gli altri messi
insieme.

Lo schema è in `01_out_architecture.png`: il flusso al centro è la rete che il
file costruisce, le note a destra dicono cosa faceva il capitolo nello stesso
punto, quelle a sinistra come parte ogni layer.

### Cosa cambia rispetto al libro, e quanto vale

Misurato a parte, aggiungendo un pezzo per volta alla rete del libro, 30 epoche
ciascuna, accuratezza su validation:

| | val |
| --- | --- |
| la rete del libro | 95.37% |
| + cross-entropy al posto del costo quadratico | 95.87% |
| + init di Kaiming sul nascosto, `std=0.01` sull'uscita | 95.67% |
| + AdamW e minibatch 32 al posto di SGD a mano | 96.23% |
| + tanh al posto della sigmoide | 96.52% |
| + 200 neuroni nascosti al posto di 30 | **98.14%** |

**Come si legge questa tabella, e come non si legge.** Ogni riga è un solo seme,
e il rumore da seme non è trascurabile: la configurazione della terza riga,
rifatta con tre semi, dà 95.67% / 95.74% / 96.01%, cioè 0.34 punti di ampiezza.
Quindi il calo apparente alla riga 3 non è un calo, è rumore — l'init
dell'ultimo layer non è ancora il vincolo che conta, con la sigmoide e `SGD` a
`lr=3.0` davanti. Le righe che si possono leggere sono quelle da mezzo punto in
su, e soprattutto l'ultima.

L'altro effetto delle correzioni è che il risultato smette di essere un
sorteggio, ed è il guadagno che non si vede in tabella. La configurazione finale
su tre semi dà 98.14% / 98.15% / 98.23%: **0.09 punti di ampiezza**. La stessa
misura sulla rete del libro con 100 neuroni nascosti dà 85.57% / 87.37% /
96.73%, undici punti — di cui il capitolo si accorge solo in una nota a piè di
pagina (*"some training runs give results quite a bit worse"*).

**La cross-entropy.** Il costo quadratico moltiplica il gradiente per
`σ'(z)`, che è quasi zero quando la sigmoide è satura: un'uscita
*convintamente sbagliata* è quella che impara più lentamente, esattamente al
contrario di quello che serve. E dieci sigmoidi indipendenti non sono una
distribuzione su dieci classi — la softmax le accoppia, così alzare la
probabilità del 7 abbassa quella delle altre.

**L'inizializzazione.** Sul layer nascosto `gain / sqrt(fan_in)` con gain 5/3
per la tanh, da [`03_mlp/02_optimizations.py`](../03_mlp) e
[`03_mlp/07_gain_depth.py`](../03_mlp). Sull'ultimo layer **no**: lì Kaiming non
c'entra, perché il gain serve a compensare lo schiacciamento di una tanh e sopra
i logits non ce n'è nessuna. Si usa `std=0.01` diretto, come
[`03_mlp/04_mlp_idiomatic.py`](../03_mlp), così i logits partono piccoli — la
rete non ha opinioni invece di averne di sbagliate e forti — ma non nulli.

Si verifica in un numero solo, e lo script lo misura: la loss al passo 0 deve
valere `-log(1/10) = 2.3026`, e viene **2.3068**. Vicina, non identica, ed è
giusto così: azzerare del tutto i pesi darebbe esattamente 2.3026, ma è proprio
la cosa che la lezione sconsiglia — serve un po' di entropia.

**L'ottimizzatore.** Da solo compra +0.05, praticamente niente. Ma è quello che
rende sicuro il passo dopo: la tanh sotto l'`SGD lr=3.0` del libro *peggiora* la
rete di dieci punti, perché la sua derivata in zero vale 1 contro lo 0.25 della
sigmoide e il passo effettivo quadruplica. Un ottimizzatore adattivo scollega la
scelta dell'attivazione da quella del learning rate — ed è per questo che nella
tabella viene prima.

### Le distorsioni elastiche, che sono il salto vero

Fin qui si è sistemata la rete. Per passare il 99% bisogna smettere di
sistemare la rete e cominciare a fabbricare dati.

L'idea è di [Simard, Steinkraus e Platt
(ICDAR 2003)](https://www.microsoft.com/en-us/research/wp-content/uploads/2003/08/icdar03.pdf).
Una deformazione è un **campo di spostamenti**: per ogni pixel dell'immagine di
uscita, da dove andarlo a leggere in quella di partenza. Se il campo è una
funzione lineare della posizione — sei numeri per tutta l'immagine — si ottengono
le trasformazioni *affini*: rotazione, scala, shear. Le rette restano rette.

Le distorsioni *elastiche* danno al campo due numeri per **ogni** pixel, e poi
lo **sfocano con una gaussiana** di deviazione σ. È la sfocatura a fare tutto:
senza, ogni pixel se ne va per conto suo e l'immagine si sbriciola; con troppa,
tutti si spostano insieme e resta una traslazione. In mezzo il tratto si *piega*,
che è una cosa che nessuna trasformazione affine sa fare. Poi si moltiplica per
α, che decide di quanti pixel, e si legge l'originale nei punti spostati
interpolando bilinearmente.

I due numeri sono quelli del paper, `SIGMA = 4.0` e `ALPHA = 34.0`. La ricetta
sta in `distort()`, una trentina di righe senza dipendenze nuove, e
`01_out_distortions.png` mostra cosa produce: la stessa cifra otto volte, tutte
diverse e tutte ancora leggibili.

**Il campo si rigenera a ogni epoca**, come in [Cireșan et al.
(2010)](https://arxiv.org/abs/1003.0358). Costa 2.5 secondi per le 50.000
immagini — più di quanto costi allenarcisi sopra, 2.4 secondi — e in 60 epoche
la rete non vede mai due volte la stessa immagine. È la risposta di quel paper
alla domanda di come facciano reti enormi a generalizzare da cinquantamila
esempi: non generalizzano da cinquantamila esempi, ne vedono un milione e mezzo.

Misurato a parte, 60 epoche ciascuna tranne la prima riga:

| | val | test |
| --- | --- | --- |
| 200 nascosti, 30 epoche, niente | 98.12% | 98.06% |
| 800 nascosti, niente | 98.53% | 98.35% |
| 200 nascosti, elastiche | 98.96% | 98.85% |
| 800 nascosti, affini | 98.98% | 98.99% |
| 800 nascosti, **elastiche** | **99.23%** | **99.20%** |
| 800 nascosti, affini + elastiche | 99.15% | 99.24% |

La prima riga è la configurazione dell'ultima riga della tabella precedente,
rimisurata qui: dà 98.12% invece di 98.14% perché il banco di prova estrae i
minibatch in un altro ordine. Due centesimi, cioè meno del rumore da seme — ma
è il motivo per cui le sei righe qui sopra si confrontano solo fra loro.

**Servono tutte e due le cose.** Le deformazioni da sole, sulla rete a 200
nascosti, si fermano a 98.96%: sotto la soglia. Allargare la rete da sola porta
a 98.53%, ancora più sotto. Solo insieme si passa il 99%.

Il motivo si vede a posteriori. Con un training set fisso la capacità in più
non serviva a niente — la rete già imparava le 50.000 immagini a memoria, e
allargarla la faceva solo arrivarci prima. E senza capacità in più i dati nuovi
non hanno dove entrare. Ognuno dei due toglie il vincolo che rendeva inutile
l'altro.

**Le affini non pagano**, né da sole né aggiunte. Non è un caso: sono il caso
limite delle elastiche per σ grande — Simard lo scrive, *"if σ is large, the
displacements become close to affine"* — quindi non aggiungono un ingrediente
nuovo. Sulla riga finale la scelta è fatta su validation (99.23% contro 99.15%);
sul test l'ordine si inverte, ma il test non partecipa alle decisioni.

**Il rumore da seme è 0.04 punti**, misurato rifacendo la configurazione finale
su tre semi: validation 99.23% / 99.25% / 99.21%, test 99.20% / 99.24% / 99.28%.
Molto stretto — è lo stesso effetto già visto sopra, per cui la rete scritta bene
smette di essere un sorteggio.

**Il confronto con la fonte.** Nella tabella storica di MNIST curata da LeCun,
Simard riporta quattro righe ottenute con *questa* architettura — 784-800-10,
cross-entropy, un solo layer nascosto — e le prime tre le abbiamo rifatte:

| | Simard | qui (test) |
| --- | --- | --- |
| niente distorsioni | 98.4% | 98.35% |
| distorsioni affini | 98.9% | 98.99% |
| distorsioni elastiche | 99.3% | 99.20% |
| la sua convnet, elastiche | 99.6% | — |

Tre righe indipendenti entro un decimo di punto. Vale la pena dirlo al
contrario: se avessimo sbagliato σ, α, o il margine sul campo casuale, la terza
riga si sarebbe staccata dalle altre due. Lo scarto residuo sull'ultima —
sei centesimi rispetto alla media dei tre semi — è dell'ordine del rumore, e per
il resto si spiega col fatto che Simard allena per centinaia di epoche (learning
rate 0.005 moltiplicato per 0.3 ogni 100) contro le nostre 60.

E la riga che non abbiamo rifatto è quella che dà la misura della cosa: **le
distorsioni elastiche, senza toccare l'architettura, valgono quasi quanto
passare a una rete convoluzionale.**

### Il metodo, che è la correzione più importante

Il libro valuta sul test a ogni epoca e riporta l'epoca migliore. Quello è
**scegliere sul test**: il numero che ne esce non stima più niente, perché il
test ha partecipato alla decisione. Il `.pkl.gz` contiene un validation set di
10.000 immagini che il capitolo lascia inutilizzato, ed è lì esattamente per
questo.

Qui tutto — quante epoche, quanti neuroni, quale attivazione, le sei righe della
tabella sopra — si decide su validation. Il test compare in una riga di codice
sola, alla fine. E i pesi che si tengono sono quelli dell'epoca con la
validation migliore, non quelli dell'ultima: è la regola di
[`05_gpt/06_gpt.py`](../05_gpt).

Che poi validation (99.13%) e test (99.19%) cadano a sei centesimi l'una
dall'altra è il segno che la selezione non ha barato.

### Le due diagnostiche

`01_out_training.png`, terzo pannello: **update:data**, da
[`03_mlp/09_learning_speed.py`](../03_mlp) — di che frazione di sé stesso si
sposta ogni tensore a ogni passo, in log10.

| | 1a epoca | ultima |
| --- | --- | --- |
| layer 0 `(800, 784)` | −2.51 | −6.07 |
| layer 1 `(10, 800)` | −2.14 | −5.95 |

La regola pratica è −3, e nella prima epoca ci siamo. Alla fine è molto più in
basso, e **è quello che deve succedere**: il cosine annealing ha portato il
learning rate quasi a zero. Un `update:data` che *restasse* a −5 fin dalla prima
epoca sarebbe tutt'altra cosa — un learning rate troppo basso. Per questo i due
numeri si guardano insieme, mai solo l'ultimo.

Primo e secondo pannello: **l'overfitting non c'è più**, ed è la differenza più
visibile rispetto alla versione senza deformazioni. Quella finiva con train
accuracy a 100.00%, loss di training a 0.0001 e loss di validation che risaliva
dopo l'epoca 9. Qui la train accuracy si ferma a 99.43%, la loss di training a
0.0192, e la loss di validation tocca il minimo all'epoca 56 su 60 — cioè sta
ancora scendendo quando il training finisce. Le due curve non si separano mai.

Non è che si sia regolarizzato meglio: la rete è **quattro volte più grossa** di
prima, e non c'è né dropout né weight decay in più. È che non esiste più un
training set da imparare a memoria.

### Cose provate e non messe, e una conclusione da correggere

`nn.BatchNorm1d` fra il lineare e la tanh porta la validation da 96.52% a
**96.11%**, ed è la conclusione a cui era già arrivato
[`03_mlp/03_batchnorm.py`](../03_mlp): con *un* layer nascosto la scala giusta
si calcola a mano, il problema che la batchnorm risolve nasce a cinquanta.

Le distorsioni **affini** danno 98.98% da sole e 99.15% aggiunte alle elastiche,
contro 99.23%: sono il caso limite delle elastiche, e non aggiungono niente.

**Il dropout, e la conclusione sbagliata che ci avevamo tirato.** Sulla versione
senza deformazioni il dropout a 0.2 abbassava la validation, 97.96% contro
98.14%, e qui c'era scritto che quel divario fra train e validation *"è quasi
tutto irriducibile"*. Era la conclusione sbagliata da un esperimento giusto.

Il dropout non chiudeva il divario perché la regolarizzazione lavora sempre
sulle stesse 50.000 immagini: mette vincoli, non aggiunge informazione. Le
deformazioni invece aggiungono informazione vera — sanno una cosa sul mondo che
la rete non poteva dedurre dai dati, e cioè che **un 5 storto resta un 5**. Il
divario era irriducibile *a dati fissi*, non in assoluto.

È anche il motivo per cui adesso il dropout non serve: non perché non
regolarizzi, ma perché non c'è più niente da regolarizzare.

`01_out_hidden_neurons.png` mostra i 784 pesi di 40 degli 800 neuroni nascosti
rimessi in forma 28x28. Il capitolo ipotizza che imparino a riconoscere pezzi di
cifra; quello che si vede sono macchie con una struttura addosso, ognuna
sensibile a una zona diversa. La versione onesta dell'euristica.

### Il diagramma

`01_out_architecture.tex` è un documento `standalone` autosufficiente, disegnato
in TikZ nello stesso stile — e con la stessa palette — dei diagrammi di
[`05_gpt/`](../05_gpt). La nota tratteggiata in fondo elenca quello che *non*
c'è e perché, che in questa cartella è informativo quanto quello che c'è.

Per rigenerarlo serve un TeX. Il più leggero è
[tectonic](https://tectonic-typesetting.github.io/), un singolo eseguibile che
si scarica da sé solo i pacchetti che servono, più ImageMagick per il PNG:

```sh
brew install tectonic imagemagick
cd 06_image_classification
tectonic 01_out_architecture.tex
magick -density 200 01_out_architecture.pdf -background white -alpha remove -depth 8 01_out_architecture.png
```

## 02_demo.html

La rete che gira nel browser: si disegna una cifra col mouse e le dieci
probabilità si aggiornano mentre si traccia.

`01_network.py` scrive i pesi in `01_out_weights.js` — i quattro tensori in
fila, float32 little-endian, 2.544.040 byte che diventano 3.392.056 caratteri in
base64. Esce un `.js` e non un `.json` per un motivo pratico: un `<script>` si
carica anche con `file://`, un `fetch()` no, lo blocca la CORS policy. Così la
pagina si apre col doppio click, senza tirare su un server.

Il file pesa quattro volte quello della rete a 200 nascosti, ed è il prezzo
degli 800 neuroni. In cambio la demo eredita una cosa che le serve più
dell'accuratezza sul test: una rete allenata su tratti deformati è robusta al
tratto storto, che è esattamente la condizione in cui la mette un mouse.

L'inferenza è quindici righe di JavaScript, nessuna libreria: due
matrice-per-vettore e una tanh. Verificata contro PyTorch sulle prime 200
immagini di test, lo scarto massimo sulle 2.000 probabilità è **1.55e-6**, e le
200 predizioni sono le stesse — la differenza è solo fra sommare float32 in un
ordine o nell'altro. Con la rete a 200 nascosti lo scarto era 2.5e-7: è
cresciuto perché il secondo layer somma 800 termini invece di 200, e l'errore di
accumulazione va con la lunghezza della somma.

### Il preprocessing è tutto il lavoro

MNIST non è "un disegno 28x28". Misurando le 50.000 immagini di training:

- il lato lungo del bounding box dell'inchiostro è **20 pixel in tutte e
  50.000** — media 20.00, massimo 20. Non "circa venti";
- il centro di massa cade a **(14.0, 14.0)** con deviazione standard 0.29: le
  cifre sono ricentrate sul baricentro dell'inchiostro, non sul centro del loro
  bounding box;
- l'aspect ratio è preservato — altezza media 19.73, larghezza 15.69. Un `1`
  resta stretto;
- il tratto è spesso circa 3 pixel su un box da 20, il **15%** del lato lungo
  della cifra. Da lì il pennello da 30 pixel su un canvas da 280, che sembra
  grosso ed è giusto.

`preprocess()` rifà quella normalizzazione. La demo mostra anche `downscale()`,
il canvas schiacciato a 28x28 e basta, che è quello che verrebbe spontaneo
scrivere. Prendendo 300 cifre del test set e ridisegnandole sul canvas da 280 a
scala e posizione casuali (lato fra 120 e 270 pixel):

| | accuratezza |
| --- | --- |
| pixel MNIST originali, la baseline | 99.3% |
| ridisegnate + `preprocess()` | **99.3%** |
| ridisegnate + `downscale()` | 40.7% |

La normalizzazione ricompra tutto: il giro attraverso un canvas dieci volte più
grande, a una scala e in un punto che la rete non ha mai visto, non costa un
decimo di punto. Senza, si perdono cinquantanove punti.

Il motivo è quello del pannello dei neuroni nascosti: questa MLP non ha
**nessuna invarianza per traslazione**, i suoi 200 neuroni sono macchie ognuna
sensibile a una zona precisa, e un 7 spostato di due pixel ne accende altri.
Una convoluzionale perdonerebbe, questa no. Quando la demo sbaglia sulla
colonna non normalizzata non è la rete a essere debole: le si sta mostrando una
distribuzione che non ha mai visto.
