# wavenet

Lo stesso modello di linguaggio a livello di carattere, con il contesto fuso
un pezzo alla volta invece che tutto insieme: otto caratteri diventano quattro
bigrammi, poi due quadrigrammi, poi un vettore solo. È la struttura ad albero
di [WaveNet](https://arxiv.org/abs/1609.03499).

Uno script e una lezione. `01_wavenet.py` ricostruisce [Building makemore Part
5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0), la cui
trascrizione è in `01_out_lecture.srt`.

```sh
uv run 04_wavenet/01_wavenet.py
```

Una decina di minuti: due training da 200k passi.

## Due cose che lo script dà per fatte

**I layer sono quelli di `torch.nn`.** La lezione li riscrive a mano, ma quel
giro l'abbiamo già fatto in
[`03_mlp/06_modules_from_scratch.py`](../03_mlp/06_modules_from_scratch.py),
dove le nostre versioni sono state confrontate numero per numero con quelle
vere. Qui `Embedding`, `Linear`, `Tanh`, `BatchNorm1d`, `Sequential`,
`optim.SGD` e `MultiStepLR` arrivano da PyTorch; quello che resta da scrivere
sono due classi in croce.

**La MLP piatta non c'è.** È la rete della lezione 3 con una costante cambiata:
non c'è niente da vedere nel rifarla. Serve solo come termine di paragone, e il
suo numero sta nella tabella qui sotto.

## Cosa mostra lo script

1. il dataset con `BLOCK_SIZE = 8` invece di 3: una costante, stessi 182625
   esempi, più passato alle spalle
2. le due classi che PyTorch non dà: `FlattenConsecutive` e l'adattatore di
   assi per `BatchNorm1d`
3. `@` accetta più assi di quanti ne servano: moltiplica sull'ultimo e tratta
   tutti quelli davanti come batch — è il motivo per cui nel file non c'è
   nessun ciclo
4. raggruppare a due a due è una `view`: niente copie, niente calcolo,
   verificato contro la concatenazione esplicita di posizioni pari e dispari
5. la rete e le sue shape, layer per layer
6. il training a parità di parametri con la rete piatta (22397 contro 22097)
7. la stessa rete più grande: due numeri cambiati, 76579 parametri
8. il grafico delle loss (`01_out_loss_curves.png`), e perché va mediato
9. cosa farebbe una convoluzione, con la verifica numerica che il modello non
   cambierebbe

## Il registro delle prestazioni

| rete | parametri | dev |
| --- | --- | --- |
| MLP piatta, contesto 3 ([`03_mlp/03_batchnorm.py`](../03_mlp)) | 12k | 2.1095 |
| MLP piatta, contesto 8 † | 22097 | 2.0263 |
| WaveNet a 3 livelli | 22397 | 2.0149 |
| WaveNet a 3 livelli, 24 dimensioni di embedding e 128 canali | 76579 | 1.9908 |

† misurata a parte, con lo stesso `train()` e gli stessi iperparametri: lo
script non la allena.

Le due righe di mezzo sono quelle da leggere insieme: **a parità di parametri
l'albero da solo non regala quasi niente**. Quasi tutto il guadagno rispetto
alla lezione 3 viene dal contesto più lungo.

Quello che l'albero dà non è la loss, è che adesso esiste una manopola per la
profondità: con la rete piatta l'unico modo di crescere era allargare l'unico
layer nascosto, schiacciando comunque tutto il contesto al primo passo. È
l'ultima riga della tabella.

## 01_out_shapes.md

Un allegato, non uno script: tutte le forme dei tensori messe una sotto
l'altra, lo schema dell'albero, e la distinzione fra i tre tipi di asse — gli
esempi, i gruppi dentro un esempio, i canali. I primi due sono la stessa cosa
per i pesi, ed è da lì che discendono sia il broadcasting di `@` sia la
questione degli assi della batchnorm.

## Le due classi

`FlattenConsecutive(n)` fonde `n` elementi consecutivi dell'asse delle
posizioni dentro quello dei canali:

    (B, T, C)  ->  (B, T // n, C * n)

Con `n` uguale al block size è `nn.Flatten` (un gruppo solo, e l'asse delle
posizioni sparisce con uno `squeeze(1)`); con `n = 2` è un livello dell'albero.
In PyTorch non esiste: `nn.Flatten` prende un intervallo di assi e li
appiattisce tutti, non sa raggruppare a `n` a `n`.

`BatchNorm1dNLC` non è una reimplementazione — è `nn.BatchNorm1d`, chiamata con
gli assi messi come li vuole lei. I nostri tensori sono `(N, T, C)`, canali per
ultimi, perché è la forma che vuole `nn.Linear`; `nn.BatchNorm1d` sugli input a
tre assi vuole `(N, C, L)`, canali in mezzo, che è la convenzione delle
convoluzioni. Passarle il nostro tensore così com'è non darebbe nessun errore e
sarebbe sbagliato: terrebbe una statistica per ogni posizione invece che una
per canale.

La strada ovvia è trasporre, ma esce un tensore non contiguo e la `view` del
`FlattenConsecutive` successivo si rifiuta di lavorarci — servirebbe un
`.contiguous()`, cioè una copia di tutte le attivazioni a ogni layer. Meglio
l'altra: le `T` posizioni sono un asse di batch come `N`, quindi si fondono,
`(N, T, C)` → `(N·T, C)`, e si usa il ramo a due assi, dove i canali sono già
in fondo. Stesse statistiche, e `flatten`/`view_as` su un tensore contiguo non
copiano niente.

## E le convoluzioni?

Nel paper l'architettura è fatta di *dilated causal convolutions*, e nello
script non ce n'è nessuna. Non è una semplificazione: il modello è lo stesso,
la convoluzione è il modo di calcolarlo in fretta.

Predire una parola di sette lettere sono otto esempi indipendenti, che noi
mandiamo dentro la rete come otto righe di un batch. Una convoluzione fa
scorrere la rete lungo la sequenza e le calcola in una passata, con il ciclo
dentro un kernel CUDA — e riusando i nodi intermedi, che fra una finestra e la
successiva si ripetono: sulle otto finestre il primo livello calcola 32
bigrammi, ma quelli diversi sono 14.

Del paper resta fuori anche il contenuto di ogni livello: lì dentro non c'è un
`Linear` e una `tanh`, ma una gated linear unit (due rami, uno che calcola e
uno che decide quanto farlo passare), connessioni residue e skip connection.
Sono le cose che servono a far reggere la stessa idea su molti più livelli.

## I nomi

Venti nomi campionati dalla rete grande, con l'asterisco su quelli che
esistono davvero nel dataset:

    dexten*  jaleer  rochetod  mellisten  anjalia
    zakia*  kreezy  bellahi  gotti*  moriella
    kinzor  darek*  emiless  suhaib*  graylynn*
    priscide  viahlan  dasher  anesley  alaiya*

Sette su venti esistono davvero. Non è necessariamente memorizzazione — i
nomi frequenti sono quelli che qualunque modello decente produce più spesso —
ma ricorda che "nome plausibile" e "nome nuovo" sono due cose diverse, e la
loss non misura né l'una né l'altra.
