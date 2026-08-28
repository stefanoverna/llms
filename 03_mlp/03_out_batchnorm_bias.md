# perché `b1` diventa inutile davanti a una batchnorm

Questo è il dettaglio della sezione 8 di `03_batchnorm.py`, srotolato. La tesi è
che il bias del layer lineare, se subito dopo c'è una batch normalization, non
ha nessun effetto sull'uscita della rete, e il suo gradiente è esattamente
zero. Non "quasi zero": zero.

Per arrivarci serve prima mettersi d'accordo su un po' di vocabolario.

---

## 1. shape, assi, riduzioni

Un tensore è una griglia di numeri. La sua **shape** dice quanto è grande in
ogni direzione: `(4, 3)` è una griglia di 4 righe per 3 colonne.

Ogni direzione si chiama **asse** (o dimensione), e gli assi si numerano da
sinistra, partendo da 0. Per raggiungere un numero servono tanti indici quanti
sono gli assi:

```
                    asse 1  (lunghezza 3)
                  ─────────────────────────>
                ┌                          ┐
        asse 0  │    1      0       5      │
   (lunghezza 4)│    3      4       7      │
                │    5     -2       6      │
          │     │    7      2       2      │
          v     └                          ┘
```

`m[2, 1]` vale `-2`: indice 2 sull'asse 0, indice 1 sull'asse 1.

Un tensore con un asse solo — shape `(3,)` — è una lista piatta di 3 numeri.
Attenzione: `(3,)` **non** è la stessa cosa di `(1, 3)`. Il primo ha un asse,
il secondo ne ha due (di cui uno lungo 1, cioè una griglia con una riga sola).
La differenza sembra pedante e invece è tutta la storia di questo documento.

Le operazioni come `.mean()` e `.std()` accettano il numero dell'asse su cui
lavorare, e quell'asse **sparisce dal risultato**: si dice che l'operazione
*riduce* lungo quell'asse. `m.mean(0)` fa la media di ogni colonna scorrendo
in verticale, e da `(4, 3)` si passa a `(3,)`:

```
                ┌                          ┐
                │    1      0       5      │
                │    3      4       7      │      .mean(0)
                │    5     -2       6      │  ──────────────>  [ 4   1   5 ]
                │    7      2       2      │                      shape (3,)
                └──────────────────────────┘
                     4      1       5
```

Con `keepdim=True` l'asse ridotto non sparisce ma resta lungo 1, quindi il
risultato ha shape `(1, 3)` invece di `(3,)`: stessi numeri, una griglia con
una riga sola invece di una lista piatta. `03_batchnorm.py` usa `keepdim=True`.

---

## 2. cosa indicizzano gli assi nella nostra rete

Un tensore non sa cosa rappresenta: siamo noi a decidere cosa vuol dire un
indice. Nel primo layer della rete (`03_batchnorm.py:128-133`) le pre-attivazioni
sono così:

```python
hpreact = emb.view(emb.shape[0], -1) @ net["W1"]   # (32, 200)
hpreact = hpreact + net["b1"]                      # (32, 200)
```

`hpreact` ha shape `(32, 200)`, dove `32` è `BATCH_SIZE` e `200` è `HIDDEN`:

- l'**asse 0 indicizza gli esempi** del minibatch: la riga `i` è tutto quello
  che la rete calcola per l'esempio `i`;
- l'**asse 1 indicizza i neuroni** del layer nascosto: la colonna `j` è il
  neurone `j` valutato su tutti e 32 gli esempi.

Da qui in poi chiamo l'asse 0 "asse degli esempi". È l'unico asse su cui i 32
esempi del batch si parlano fra loro, ed è l'asse su cui la batchnorm lavora.

I parametri, invece, non hanno un asse degli esempi — ed è ovvio: sono gli
stessi per tutti gli esempi, è il senso di avere dei parametri. `b1` ha shape
`(200,)`: un numero per neurone, punto.

| oggetto                                | shape       | asse degli esempi? |
|----------------------------------------|-------------|--------------------|
| `emb.view(...)`                        | `(32, 30)`  | sì, asse 0         |
| `W1`                                   | `(30, 200)` | no                 |
| `emb @ W1`                             | `(32, 200)` | sì, asse 0         |
| `b1`                                   | `(200,)`    | **no**             |
| `hpreact`                              | `(32, 200)` | sì, asse 0         |
| `mean = hpreact.mean(0, keepdim=True)` | `(1, 200)`  | no, **ridotto**    |
| `std = hpreact.std(0, keepdim=True)`   | `(1, 200)`  | no, **ridotto**    |
| `bngain`, `bnbias`                     | `(1, 200)`  | no                 |

Il seguito è tutto contenuto in questa tabella: `b1` non ha l'asse degli
esempi, e l'asse degli esempi è esattamente quello che la batchnorm riduce.

---

## 3. un esempio in miniatura

Continuo con numeri veri ma piccoli: **4 esempi invece di 32, 3 neuroni invece
di 200**. Cambia solo la taglia, non succede niente di diverso.

Chiamo `u` il prodotto `emb @ W1`, cioè le pre-attivazioni *prima* del bias:

```
u  (4,3)                                b1  (3,)
┌                       ┐
│    1      0       5   │               [  10    100     -1  ]
│    3      4       7   │
│    5     -2       6   │
│    7      2       2   │
└                       ┘
 neur.0  neur.1  neur.2
```

Le medie di colonna di `u` — le chiamo `ū` — sono `[4, 1, 5]` (è la riduzione
disegnata nella sezione 1).

---

## 4. la somma del bias: il broadcasting

`u + b1` mette insieme un `(4, 3)` e un `(3,)`. Non hanno la stessa shape,
quindi non si possono sommare numero per numero così come sono. PyTorch applica
il **broadcasting**: allinea le shape *a partire dall'ultimo asse* e allunga
quelle troppo corte.

**Passo A — allineamento a destra.** Gli assi mancanti a sinistra vengono
riempiti con `1`:

```
u  :  (4, 3)
b1 :     (3,)     ->    b1 : (1, 3)
```

```
b1  (1,3)
┌                       ┐
│   10    100      -1   │      una riga sola
└                       ┘
```

**Passo B — replica.** Dove un tensore ha lunghezza 1 e l'altro 4, quello
lungo 1 viene ripetuto 4 volte:

```
b1  (1,3)  ── replicato 4 volte ──>   b1 "espanso"  (4,3)
                                    ┌                       ┐
┌                       ┐           │   10    100      -1   │
│   10    100      -1   │   ===>    │   10    100      -1   │
└                       ┘           │   10    100      -1   │
                                    │   10    100      -1   │
                                    └                       ┘
```

Quest'ultima matrice è **il termine che viene davvero sommato** a `u`. Non
`(3,)`: `(4, 3)`, con le 4 righe tutte identiche. (PyTorch non la costruisce
davvero in memoria, rilegge la stessa riga 4 volte, ma il risultato è questo.)

**Passo C — la somma, elemento per elemento.**

```
      u  (4,3)              b1 espanso  (4,3)          hpreact  (4,3)
┌                    ┐    ┌                    ┐    ┌                    ┐
│   1     0      5   │    │  10    100     -1  │    │  11    100      4  │
│   3     4      7   │ +  │  10    100     -1  │ =  │  13    104      6  │
│   5    -2      6   │    │  10    100     -1  │    │  15     98      5  │
│   7     2      2   │    │  10    100     -1  │    │  17    102      1  │
└                    ┘    └                    ┘    └                    ┘
                           ↑ costante lungo le colonne
```

Guarda la direzione in cui `b1` è costante: **verticale**, cioè lungo l'asse
degli esempi. Il neurone 1 riceve `+10` su tutti e 4 gli esempi; il neurone 2
riceve `+100` su tutti e 4. È l'unica cosa che conta in tutto il ragionamento.

---

## 5. la media raccoglie il bias per intero

La batchnorm comincia riducendo lungo l'asse degli esempi
(`03_batchnorm.py:147`):

```python
mean = hpreact.mean(0, keepdim=True)     # (4, 3) -> (1, 3)
```

```
┌                       ┐
│  11    100       4    │
│  13    104       6    │
│  15     98       5    │
│  17    102       1    │
└───────────────────────┘
   14    101       4        ->   mean (1,3) = [ 14   101    4 ]
```

Le medie di `u` erano `ū = [4, 1, 5]`. Confronta:

```
   mean    =     ū      +      b1
[14 101 4] = [4  1  5]  +  [10 100 -1]        ✓
```

Il bias entra dentro la media **tutto intero**, non una sua frazione. Il
motivo è il passo C: `b1[j]` compare identico in tutte e 4 le righe della
colonna, quindi sommarlo 4 volte e dividere per 4 lo restituisce com'era.

```
        (10 + 10 + 10 + 10) / 4  =  10
```

È qui che si vede perché contava che `b1` non avesse l'asse degli esempi: se
avesse avuto un valore diverso per ogni esempio, la media ne avrebbe raccolto
solo il valore medio, e il resto sarebbe sopravvissuto.

---

## 6. la sottrazione lo restituisce, tutto

Subito dopo (`03_batchnorm.py:155`) la batchnorm sottrae quella media. Di nuovo
shape diverse, `(4, 3) - (1, 3)`, quindi di nuovo broadcasting: `mean` viene
replicata 4 volte lungo l'asse degli esempi, cioè a tutti e 4 gli esempi di una
colonna viene sottratto lo stesso numero.

```
   hpreact  (4,3)          mean espansa  (4,3)         risultato  (4,3)
┌                    ┐    ┌                    ┐    ┌                    ┐
│  11    100      4  │    │  14    101      4  │    │  -3     -1      0  │
│  13    104      6  │ -  │  14    101      4  │ =  │  -1      3      2  │
│  15     98      5  │    │  14    101      4  │    │   1     -3      1  │
│  17    102      1  │    │  14    101      4  │    │   3      1     -3  │
└                    ┘    └                    ┘    └                    ┘
```

Ora rifai lo stesso calcolo partendo da `u`, cioè da una rete **senza** `b1`:

```
      u  (4,3)              ū espansa  (4,3)           risultato  (4,3)
┌                    ┐    ┌                    ┐    ┌                    ┐
│   1     0      5   │    │   4     1      5   │    │  -3     -1      0  │
│   3     4      7   │ -  │   4     1      5   │ =  │  -1      3      2  │
│   5    -2      6   │    │   4     1      5   │    │   1     -3      1  │
│   7     2      2   │    │   4     1      5   │    │   3      1     -3  │
└                    ┘    └                    ┘    └                    ┘
```

**La stessa identica matrice.** Il neurone 1 con il bias sparava
`100, 104, 98, 102`; senza bias sparava `0, 4, -2, 2`. Dopo la centratura, in
entrambi i casi, `-1, 3, -3, 1`.

In simboli, elemento per elemento:

```
(hpreact - mean)[i,j] = ( u[i,j] + b1[j] ) - ( ū[j] + b1[j] )
                      =   u[i,j] - ū[j]
```

`b1[j]` compare due volte con segno opposto e se ne va. Il suo giro completo è
stato: replicato su 4 righe (passo B), compresso in 1 (sezione 5),
ri-replicato su 4, sottratto. Ritorno esatto al punto di partenza.

---

## 7. anche la deviazione standard è cieca

Resta il denominatore (`03_batchnorm.py:148`):

```python
std = hpreact.std(0, keepdim=True)       # (4, 3) -> (1, 3)
```

La deviazione standard di una colonna si calcola *dagli scarti dalla media di
quella colonna* — cioè dalle matrici della sezione 6, che abbiamo appena visto
essere identiche nei due casi. Detto senza formule: `std` misura quanto i 4
valori sono sparsi fra loro, e spostarli tutti e 4 della stessa quantità non
cambia quanto sono distanti l'uno dall'altro.

```
con b1:      100   104    98   102        std = 2.582
senza b1:      0     4    -2     2        std = 2.582
```

Quindi anche il `(1, 3)` del denominatore non contiene `b1`.

A questo punto della catena le due reti — quella con il bias e quella senza —
hanno in mano gli stessi identici numeri. Tutto quello che viene dopo (`bngain`
e `bnbias`, la `tanh`, il secondo layer, la loss) parte da lì, quindi darà gli
stessi identici risultati. È la controprova stampata in
`03_batchnorm.py:503-508`: logits uguali a meno dell'errore in virgola mobile.

Vale la pena dirlo in modo forte: la loss, che è **uno scalare**, è una
funzione *costante* dei 200 numeri contenuti in `b1`. Puoi metterci dentro
quello che vuoi e non cambia niente.

E quindi il gradiente è zero.

---

## 10. la conseguenza pratica

`b1` sono 200 parametri che non imparano niente e non fanno niente. Non è un
bug lasciarli lì — la rete funziona identica — è solo spreco, e un po' di
confusione per chi legge il codice: sembra un bias, ma la traslazione la fa
`bnbias`.

```python
nn.Conv2d(..., bias=False)
nn.BatchNorm2d(...)
```
