# image classification

Il problema e i dati del capitolo 1 di [Neural Networks and Deep
Learning](http://neuralnetworksanddeeplearning.com/chap1.html) di Michael
Nielsen — 784 pixel in ingresso, dieci cifre in uscita, una MLP con un layer
nascosto — scritti però con quello che le cartelle precedenti hanno stabilito.

Non è una ricostruzione. Il capitolo è del 2015 e ci sono dentro scelte che
`03_mlp/` e `05_gpt/` ci hanno insegnato a non fare più: qui si fa direttamente
la versione buona, e i motivi sono nel file.

```sh
uv run 06_image_classification/01_network.py    # ~1 minuto
uv run 06_image_classification/mnist.py         # guarda i dati e basta
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
    200 nascosti     tanh
     10 uscite       logits, nessuna attivazione sopra

159.010 parametri, 30 epoche, un minuto su CPU. **98.11% su test**, 189 errori
su 10.000, contro il 95.42% che riporta il capitolo con la stessa architettura.

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

Che poi validation (98.14%) e test (98.11%) cadano a tre centesimi l'una
dall'altra è il segno che la selezione non ha barato.

### Le due diagnostiche

`01_out_training.png`, terzo pannello: **update:data**, da
[`03_mlp/09_learning_speed.py`](../03_mlp) — di che frazione di sé stesso si
sposta ogni tensore a ogni passo, in log10.

| | 1a epoca | ultima |
| --- | --- | --- |
| layer 0 `(200, 784)` | −2.56 | −5.59 |
| layer 1 `(10, 200)` | −2.54 | −5.81 |

La regola pratica è −3, e nella prima epoca ci siamo. Alla fine è molto più in
basso, e **è quello che deve succedere**: il cosine annealing ha portato il
learning rate quasi a zero e la loss di training è a 0.0001, quindi i gradienti
sono minuscoli. Un `update:data` che *restasse* a −5 fin dalla prima epoca
sarebbe tutt'altra cosa — un learning rate troppo basso. Per questo i due numeri
si guardano insieme, mai solo l'ultimo.

Primo e secondo pannello: **la rete overfitta**, e si vede. Train accuracy a
100.00%, loss di training a 0.0001, e la loss di validation che tocca il minimo
all'epoca 9 e poi risale. Ma l'accuratezza di validation resta piatta al 98%:
le due misure divergono perché la rete diventa sempre più sicura di sé anche
dove sbaglia — la loss se ne accorge, l'accuratezza no.

### Due cose provate e non messe

`nn.BatchNorm1d` fra il lineare e la tanh porta la validation da 96.52% a
**96.11%**, ed è la conclusione a cui era già arrivato
[`03_mlp/03_batchnorm.py`](../03_mlp): con *un* layer nascosto la scala giusta
si calcola a mano, il problema che la batchnorm risolve nasce a cinquanta.

Il dropout a 0.2 la abbassa, **97.96% contro 98.14%** — più sorprendente, visto
che la rete overfitta eccome. Vuol dire che su MNIST quel
divario fra train e validation è quasi tutto irriducibile. Metterli lo stesso
sarebbe stato culto del cargo.

`01_out_hidden_neurons.png` mostra i 784 pesi di 40 dei 200 neuroni nascosti
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
