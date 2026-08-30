# CLAUDE.md

Ricostruzioni commentate di lezioni e capitoli sulle reti neurali. Ogni cartella
è un esempio a sé: script numerati in ordine di lettura, i dati, e il materiale
di partenza (`.srt` per le lezioni video).

## Lingua

**La prosa è in italiano, il codice è in inglese.** Vale sempre, senza
eccezioni:

- **in italiano**: docstring, commenti, testo stampato a schermo, README.
- **in inglese**: nomi di variabili, funzioni, classi, costanti, argomenti,
  chiavi dei dizionari. E i nomi dei file e delle cartelle, output compresi
  (`01_wavenet.py`, `01_out_loss_curves.png`, `06_image_classification/`).

Quindi `def train(net, x, y)` con la docstring in italiano, mai `def allena`.
Nelle stringhe stampate a schermo gli accenti si scrivono con l'apostrofo
(`perche'`, `e'`), nelle docstring e nei README si usano gli accenti veri
(`perché`, `è`).

## Come si scrive un esempio

- Un docstring lungo in testa a ogni script che spiega **l'idea**, non il
  codice: cosa si sta cercando di fare, perché quella scelta e non un'altra,
  cosa lascia aperto per lo script successivo. È la parte che conta.
- Il corpo diviso in sezioni numerate con separatori
  `# ---------------------------------------------------------------------------`,
  e ogni sezione stampa quello che ha capito. Gli script si leggono anche
  guardandone l'output.
- Ogni script è autonomo e ridefinisce quello che gli serve. Si condividono
  solo i dati e i moduli di caricamento (`mnist.py`, `names.txt`).
- I file generati si chiamano `NN_out_*` con lo stesso `NN` dello script che li
  produce, e stanno nella stessa cartella.
- I percorsi sono relativi al file (`HERE = Path(__file__).parent`), così gli
  script si lanciano da qualsiasi directory.
- Seed fissato e dichiarato in cima.

## I numeri

**Non si scrivono numeri a occhio.** Ogni cifra che compare in un docstring, in
un README o in un commento — accuratezze, tempi, deviazioni standard, conteggi
di parametri — deve venire da un'esecuzione vera dello script. Se un numero
cambia dopo una modifica, si aggiorna il testo.

Quando un risultato **non riproduce** quello della fonte, lo si dice e si va a
capire perché, invece di aggiustare il testo per farlo tornare. La discrepanza è
quasi sempre più istruttiva del numero.

E prima di leggere una differenza, misurare il rumore: con un solo seme, mezzo
punto di accuratezza può non voler dire niente. Se una tabella confronta
configurazioni, o le si rifà su tre semi, o si dice esplicitamente quanto è
ampio il rumore (vedi la tabella dell'ablation in
`06_image_classification/README.md`).

## Come si lancia

Solo [uv](https://docs.astral.sh/uv/), niente venv da attivare a mano:

```sh
uv run 06_image_classification/01_network.py
```

Le dipendenze stanno in `pyproject.toml`: al momento solo `torch` e
`matplotlib`. Prima di aggiungerne una, valutare se l'esempio si può scrivere
senza.

`uv run` va lanciato **dalla radice del repo**, dov'è il `pyproject.toml`: da
un'altra directory `uv` costruisce un ambiente vuoto e `import torch` fallisce.
Il percorso dello script può essere assoluto, non cambia niente.

## Aspettare un training lungo

Gli script qui allenano per minuti. Vanno lanciati in background, e il loro
completamento **arriva da solo come notifica**: non serve nessun processo di
sorveglianza, e metterne uno è solo un modo per sbagliare.

**Mai questo pattern:**

```sh
while pgrep -f "train.py" > /dev/null; do sleep 20; done; cat log   # NO
```

`pgrep -f` cerca dentro la riga di comando completa, e la riga di comando di
*questa stessa shell* contiene `train.py`. Il watcher trova sé stesso, la
condizione resta vera per sempre, e il loop continua a girare anche molto dopo
che il training è finito. Sono shell zombie che si accumulano una per attesa.

Se un'attesa esplicita serve davvero, si aspetta su qualcosa che non possa
auto-matcharsi — un PID, o un file che comparirà:

```sh
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
until [ -f out.png ]; do sleep 5; done
```

Ultima cosa: quando lo stdout è rediretto su file, Python lo bufferizza e il log
resta **vuoto fino alla fine del processo**. Un log a zero byte non vuol dire
che il training è morto, di solito vuol dire che sta girando.
