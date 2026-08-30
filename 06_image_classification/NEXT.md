## 1. La catena: da GPT-2 (2019) a Llama (oggi)

Il tuo `05_gpt` è fedele al paper del 2017, che descrive un'architettura pensata
per il training. Tutto quello che trovi qui sotto nasce dopo, quando GPT-2 la
porta su scala e si aprono tre problemi distinti. Conviene tenerli separati,
perché le risposte non si assomigliano:

- **stabilità in profondità** → Pre-LN, RMSNorm, SwiGLU
- **costo dell'inferenza** → KV-cache, MQA, GQA
- **lunghezza del contesto** → RoPE, ALiBi

LLaMA (feb 2023) non inventa nessuna di queste tecniche, le mette insieme. Per
questo è un buon punto d'arrivo.

| tecnica         | anno           | paper                                                   | filone    |
| --------------- | -------------- | ------------------------------------------------------- | --------- |
| clipping        | nov 2012       | Pascanu et al., *On the difficulty of training RNNs*     | training  |
| BPE             | ago 2015       | Sennrich et al., *Neural MT of Rare Words*               | tokenizer |
| weight tying    | ago 2016       | Press & Wolf, *Using the Output Embedding*               | training  |
| cosine schedule | ago 2016       | Loshchilov & Hutter, *SGDR*                              | training  |
| warmup          | giu 2017       | Vaswani et al., *Attention Is All You Need*              | training  |
| top-k           | mag 2018       | Fan et al., *Hierarchical Neural Story Generation*       | sampling  |
| top-p           | apr 2019       | Holtzman et al., *Neural Text Degeneration*              | sampling  |
| RMSNorm         | 2019           | Zhang & Sennrich, *Root Mean Square Layer Normalization*  | stabilità |
| MQA             | nov 2019       | Shazeer, *Fast Transformer Decoding*                     | inferenza |
| SwiGLU          | feb 2020       | Shazeer, *GLU Variants Improve Transformer*              | stabilità |
| Pre-LN          | feb 2020       | Xiong et al., *On Layer Normalization…*                  | stabilità |
| RoPE            | apr 2021       | Su et al., *RoFormer*                                    | contesto  |
| ALiBi           | ago 2021       | Press et al., *Train Short, Test Long*                   | contesto  |
| FlashAttention  | mag 2022       | Dao et al., *FlashAttention*                             | inferenza |
| TinyStories     | mag 2023       | Eldan & Li, *How Small Can Language Models Be*            | corpus    |
| GQA             | mag 2023       | Ainslie et al., *GQA*                                    | inferenza |
| LLaMA, LLaMA 2  | feb / lug 2023 | Touvron et al.                                           | tutto     |

La KV-cache non ha un paper suo: nasce come pratica implementativa ai tempi di
GPT-2. Studiala subito prima di MQA, perché Shazeer apre proprio motivandola, e
parla di memory bandwidth invece che di FLOP.

### Perché si riparte dal tokenizer, e dai dati

Il corpus attuale non regge il piano. Con `block_size=256` e `batch_size=64`,
allo step 3500 (dove la val tocca il minimo) il modello ha visto 57.344.000
token contro uno split di train di 891.017 caratteri: **64 epoche**, che
diventano 92 a fine run. Sono 0.068 token per parametro, quando Chinchilla ne
vuole una ventina. Per fare *una* epoca a 5000 step servirebbero 82M di
caratteri, e tutto il Pirandello rimasto su Gutenberg (novelle, *L'umorismo*, il
teatro siciliano) fa 745K caratteri: porterebbe il corpus da 990K a 1.7M, cioè
da 64 epoche a 37.

A quel punto il modello non sta imparando l'italiano, ripassa 891K caratteri
sessantaquattro volte, e l'overfitting che vedi nel log è la conseguenza
prevista.

Il tokenizer toglie il vincolo. A livello di carattere il corpus deve essere
omogeneo o il vocabolario esplode, ed è la ragione per cui `build_input_it.py`
fa i salti mortali per tenerlo a 75 simboli. Con BPE il vocabolario lo scegli
tu, e a quel punto puoi prendere un corpus grande quanto serve, da dove ti pare.

Corpus standard ancora scaricabili con il solo `urllib`, verificati:

| | |
| --- | --- |
| enwik8 | 36 MB zip, 100 MB di testo |
| enwik9 | 323 MB zip, 1 GB |
| TinyStories | 2.2 GB |
| WikiText-103 | l'URL S3 storico risponde 301 |

Prendi **TinyStories**, più per come è fatto che per la taglia. Eldan & Li
l'hanno costruito perché modelli da una decina di milioni di parametri producano
inglese *coerente*, che è esattamente la tua fascia: ne esce testo leggibile
invece di Wikipedia sgrammaticata, cioè la stessa soddisfazione che ti dà oggi
vedere uscire Pirandello, ma con una loss che scende perché il modello ha
imparato e non perché ha memorizzato. Ne scarichi la fetta che ti serve, intorno
ai 400 MB.

Se preferisci un numero da confrontare con la letteratura invece che sample
belli, la scelta è enwik8.

### La metrica: bits per character

Cambiando tokenizer la loss cambia unità, e i confronti saltano. Misura in
**BPC**, che divide per il numero di caratteri e resta quindi confrontabile fra
tokenizer diversi. La baseline attuale, convertita:

| | nats | BPC |
| --- | --- | --- |
| `big`, val | 1.4025 | **2.0234** |
| `big`, train | 0.8944 | 1.2903 |

Quel divario di 0.73 BPC è la memorizzazione. Tienilo come "dov'eravamo con
Pirandello a caratteri" e non confrontarci il resto: cambiando corpus cambia la
difficoltà del testo, e BPC normalizza il tokenizer, non il testo.

### Il budget, e perché il weight tying diventa portante

A `vocab=75` la testa del modello è lo 0,3% dei parametri, ed è il motivo per
cui avevo scartato il weight tying. Appena tokenizzi il conto cambia, perché
embedding e `lm_head` crescono con il vocabolario e rallentano ogni step.

Stima su 2.5h, ancorata ai 46.4 minuti misurati per 5000 step della config
`big`:

| vocab | tying | emb + head | totale | token in 2.5h | token/parametro |
| ----- | ----- | ---------- | ------ | ------------- | --------------- |
| 4096  | sì    | 1.572.864  | 12.297.216 | 284M      | **23.1**        |
| 8192  | sì    | 3.145.728  | 13.870.080 | 252M      | 18.2            |
| 8192  | no    | 6.291.456  | 17.015.808 | 205M      | 12.1            |
| 16384 | no    | 12.582.912 | 23.307.264 | 150M      | 6.4             |

Parti da **`vocab=4096` con weight tying**: 23.1 token per parametro, in linea
con Chinchilla, e il corpo del modello resta grande quanto oggi. Per TinyStories
4096 simboli sono abbondanti. Il weight tying qui porta il budget di training da
12 a 23 token per parametro, ed è per questo che lo metti nella baseline invece
di ablarlo a parte.

### Un avvertimento sul metodo

Quasi nessuna di queste tecniche nasce per abbassare la loss.

| tecnica    | a cosa serve davvero                      | effetto atteso sulla loss    |
| ---------- | ----------------------------------------- | ---------------------------- |
| BPE        | densità del token, contesto più lungo     | grande, e non confrontabile  |
| sampling   | qualità del testo a modello fermo         | nessuno: non tocca i pesi    |
| clipping   | evitare che il training esploda           | nessuno, se non esplodeva    |
| RMSNorm    | stessa funzione, metà dei calcoli         | nessuno, ed è il punto       |
| SwiGLU     | un feed-forward più espressivo            | piccolo, forse               |
| teste fuse | ridurre l'overhead di lancio              | zero: output identico        |
| KV-cache   | togliere lavoro ridondante in generazione | zero: output identico        |
| RoPE       | contesto più lungo di `block_size`        | nessuno, per costruzione     |
| GQA        | memoria della cache in inferenza          | **peggiora**, ed è accettabile |

Se leggi il piano come una classifica di BPC, l'unica cosa che impari è che
TinyStories è più facile di Pirandello. Per ogni step chiediti invece cosa
diventa possibile, e cosa si rompe se lo togli.

Da qui una conseguenza pratica: **cinque step su dieci non allenano niente.**
01, 03, 06, 07 e 09 girano sulla baseline dello step 02, e le loro misure
(millisecondi, byte, `allclose`) sono deterministiche e immediate.

### Come sono fatti gli step

Un'area per script, un file per step, numerati nell'ordine in cui si leggono,
tutti in `08_modern_gpt/`. Vale la regola del repo: **ogni script è autonomo** e
ridefinisce il GPT intero con la sua modifica dentro, invece di importare quello
dello step prima. Si duplica parecchio codice, ed è voluto: la lezione sta nel
diff fra `05_*.py` e `06_*.py`, e vuoi poterlo leggere tutto insieme.

I dati sono l'eccezione, come già oggi: corpus e tokenizer si caricano da un
modulo condiviso, che è quello che lo step 01 produce.

### Gli step

**`01_tokenizer.py` — BPE da zero, e il corpus nuovo** · *niente training*
Sennrich et al. 2015. Merge iterativo delle coppie più frequenti, `encode`,
`decode`, il vocabolario salvato su disco. Qui dentro scarichi anche la fetta di
TinyStories e la tokenizzi una volta sola, salvando gli id in binario: rifarlo a
ogni run costerebbe più del training. *Cosa impari:* da dove viene buona parte
del comportamento strano degli LLM. Perché sbagliano l'aritmetica, perché non
sanno contare le lettere di una parola, perché uno spazio finale nel prompt
rovina la generazione. *Da misurare:* il *fertility rate*, cioè quanti token per
parola produce lo stesso tokenizer su testi diversi. Hai ancora in casa il banco
di prova: `05_gpt/input_it.txt` (Pirandello) contro `05_gpt/input.txt`
(Shakespeare). Bastano due colonne per spiegare perché un LLM in italiano costa
di più e ha un contesto effettivo più corto.

**`02_baseline.py` — la baseline nuova** · *training, due volte* Press & Wolf
2016 (weight tying). Stesso GPT del `05_gpt`, con `vocab=4096`, weight tying, e
il corpus dello step 01. *Cosa impari:* quanto vale il tokenizer a parità di
minuti. Alleni due volte, char-level e BPE, con lo stesso wall-clock, e metti le
due BPC in tabella. È l'unico numero di tutta la catena attribuibile al
tokenizer e non al testo, ed è il gemello dell'esperimento sul fertility rate.
*Perché il tying sta qui:* non è un'ablation, è una decisione di budget che
prendi una volta. Senza, ogni run successiva parte sottoallenata. *Da non fare:*
confrontare la BPC che ottieni con il 2.0234 di `big`. Corpus diverso,
difficoltà diversa.

**`03_sampling.py` — come si sceglie il token** · *niente training* Fan et al.
2018 (top-k), Holtzman et al. 2019 (top-p). Cambia solo `generate`: temperatura,
top-k e top-p al posto del campionamento dalla distribuzione piena. *Cosa
impari:* che la distribuzione prodotta dal modello e il testo che leggi sono due
cose diverse, e che buona parte di quella che chiami "qualità del modello" è una
scelta fatta dopo, a pesi fermi. A temperatura alta il testo delira, a
temperatura bassa entra in loop: il compromesso si vede a occhio nudo in due
minuti, e top-p esiste per non doverlo tarare a mano. *Verifica a costo zero:*
stesso seme, stesso checkpoint, un parametro alla volta.

**`04_training.py` — l'ottimizzatore** · *training* Pascanu et al. 2012
(clipping), Loshchilov & Hutter 2016 (cosine), Vaswani et al. 2017 (warmup).
`clip_grad_norm_` più warmup lineare e decadimento coseno, al posto del learning
rate costante. *Cosa impari:* che il learning rate va pensato come una
traiettoria lungo tutto il training, e cosa ci fa il warmup all'inizio, quando i
gradienti sono grandi e le medie mobili di AdamW non sanno ancora niente. Lo
vedi nel grafico `update:data` che già produci: la riga a -3 diventa un
bersaglio che puoi spostare con la schedule. *Da non fare:* trasformarlo in una
caccia al decimale. Guarda le curve.

**`05_norm_and_ffwd.py` — dentro il blocco** · *training* Zhang & Sennrich 2019
(RMSNorm), Shazeer 2020 (SwiGLU). Stanno insieme perché sono la stessa mossa
fatta due volte: prendi un pezzo del blocco e lo sostituisci con una versione
che fa quasi la stessa cosa in un modo diverso. *Cosa impari:* da RMSNorm, che
**centrare non serviva**. LayerNorm sottrae la media da dieci anni, e si scopre
che il lavoro lo faceva tutto il riscalamento: smonta un pezzo di rito che
nessuno aveva messo in discussione. Da SwiGLU, che un feed-forward può
moltiplicare fra loro due proiezioni invece di passarne una sola dentro una
non-linearità, e che quella moltiplicazione funziona da valvola: una proiezione
decide quanta parte dell'altra lasciar passare. *Trappola:* SwiGLU ha **tre**
matrici invece di due. L'hidden va a `8/3·d`, non a `4·d`, ed è da lì che viene
il numero strano di LLaMA. Con `4·d` i due modelli hanno taglie diverse e il
confronto non dice niente.

**`06_attention.py` — le teste in un matmul solo** · *niente training* Nessun
paper: viene dalla misura. Le 36 `Head` separate (6 blocchi × 6 teste), ognuna
con le sue tre `Linear` minuscole, diventano un `Linear(n_embed, 3*n_embed)` più
un reshape. *Cosa impari:* quanto poco c'entri il numero di operazioni con il
tempo che ci metti. Un forward della vecchia config `big` a lunghezza di
contesto fissa, 30 ripetizioni dopo warmup:

| T   | MPS     | CPU      |
| --- | ------- | -------- |
| 1   | 6.50 ms | 2.60 ms  |
| 64  | 7.07 ms | 9.60 ms  |
| 128 | 7.18 ms | 13.99 ms |
| 256 | 7.66 ms | 23.95 ms |

Su MPS 256 volte il lavoro costa il **18% in più**: la quadraticità non si vede,
perché il tempo se ne va tutto nell'overhead di lancio dei kernel. Su CPU il
conto torna, 9.2×. Lo stesso codice su due macchine porta a due conclusioni
opposte su cosa convenga ottimizzare. *Perché sta qui:* è il prerequisito degli
step 07 e 09. Finché l'overhead domina il tempo, la cache non è misurabile, e
GQA ha comunque bisogno della dimensione delle teste già esplicita. *Verifica:*
`torch.allclose` fra il vecchio e il nuovo. Se l'output non è identico hai
introdotto un bug.

**`07_kv_cache.py` — la cache, col tetto** · *niente training* Nessun paper; la
motiva l'introduzione di Shazeer 2019. `generate` accumula K e V invece di
rifare il forward su tutto il contesto. Per token si passa da O(T²) a O(T); sul
totale da O(T·B²) a O(T·B), con B = `block_size`. *Cosa impari:* perché prefill
e decode si pagano a prezzi diversi, distinzione su cui poggia buona parte
dell'ingegneria dell'inferenza, e perché il decode è *memory bound* invece che
compute bound, che è poi la premessa dello step 09. *Trappola:* la cache **vale
solo fino a `block_size`**. `generate` ritaglia `idx[:, -block_size:]` e il
forward assegna le posizioni con `torch.arange(T)` (`05_gpt/06_gpt.py:204`),
quindi appena la finestra scorre ogni token vede il proprio indice di posizione
calare di uno, e tutte le K/V salvate diventano stale. Qui fai come GPT-2: cache
più tetto duro, e la generazione si ferma a `block_size`. Il tetto lo toglie lo
step dopo, e vedere prima il problema è il modo più chiaro per capire a cosa
serva RoPE.

**`08_rope.py` — via il tetto** · *training* Su et al. 2021. Sparisce
`position_embedding_table`, e la posizione entra come rotazione di Q e K dentro
ogni testa. *Cosa impari:* che la posizione può essere una proprietà della
*coppia* invece che del singolo token, e che appena lo diventa risolvi in un
colpo il tetto sulla lunghezza, la tabella di posizioni da imparare e il
problema di cache dello step 07. Di tutta la catena è lo step che mostra meglio
cosa comporti scegliere la rappresentazione giusta. *Verifica:* la BPC deve
restare dov'era. Se si sposta parecchio, in un senso o nell'altro, hai sbagliato
l'implementazione. *Nota:* ALiBi (Press et al., ago 2021) è la rivale
contemporanea e si scrive in molte meno righe. Vale la pena leggerla anche solo
per vedere quanto diversamente si può rispondere alla stessa domanda.

**`09_gqa.py` — la memoria della cache** · *niente training per la misura*
Shazeer 2019 (MQA), Ainslie et al. 2023 (GQA). Più teste di query condividono le
stesse K/V: `n_kv_head` scende sotto `n_head`. *Cosa impari:* una scelta di
**inferenza** che ha cambiato l'architettura, pagata con un po' di qualità e
accettata lo stesso. Chi progetta i modelli di oggi tiene al costo di servirli
quanto alla qualità che ottiene allenandoli. *Da misurare:* byte occupati dalla
cache per `n_kv_head ∈ {1, 2, 3, 6}`, da MQA a multi-head pieno. Il conto si fa
con carta e penna e si verifica in una riga. *Trappola:* su un modello da una
decina di milioni di parametri la cache è minuscola in assoluto, e il guadagno
wall-clock qui non lo vedrai. Il numero onesto è il rapporto, più
l'estrapolazione a un modello vero. Scrivilo così, senza gonfiarlo.

**`10_llama.py` — tutto insieme** · *training* Contro la baseline dello step 02:
stesso corpus, stesso tokenizer, stesso seme, stessi minuti. *Cosa impari:* che
un modello moderno vince a parità di BPC, perché si allena stabile e costa un
decimo da servire. Se la BPC finale è più o meno quella dello step 02 il piano
ha funzionato, perché vuol dire che hai comprato lunghezza di contesto, velocità
di decode e memoria senza pagarle in qualità.

### Cosa resta fuori, e perché

**FlashAttention** (Dao et al. 2022) è ortogonale a tutta la catena: non cambia
i FLOP, cambia quante volte passi dalla memoria. In PyTorch è
`F.scaled_dot_product_attention`, quindi ci metti una nota nello step 06 e via.
Scriverlo a mano non insegna niente che il paper non dica meglio.

**Pre-LN** ce l'hai già: `Block.forward` fa `x + self.sa(self.ln1(x))`
(`05_gpt/06_gpt.py:180`). È un risultato del feb 2020 che Karpathy dà per
scontato, ed è il motivo per cui i tuoi 6 blocchi si allenano senza warmup
elaborati. Leggilo per sapere cosa hai già; semmai l'esperimento è rimetterci il
post-LN e guardarlo rompersi.

## 2. L'orizzonte: pesi veri, e post-training

Il passo successivo naturale è caricare i pesi veri di GPT-2 124M **nella tua
implementazione** (è la seconda metà di "Let's reproduce GPT-2"): il codice che
hai scritto smette di essere un esercizio e si mette a eseguire il modello vero.
Arrivarci dopo la catena qui sopra costa poco, perché a quel punto hai già il
tokenizer BPE e il weight tying, che su GPT-2 sono obbligatori: con
`vocab=50257` e `n_embed=768` la testa è 19.298.688 parametri, il 39% del
modello.

Da lì il fine-tuning (SFT su coppie istruzione/risposta, poi DPO) è l'ultimo
pezzo mancante, e risponde alla domanda "perché un modello che predice il token
successivo si comporta come un assistente". Su M2 Pro con MPS è fattibile, ma
costa dipendenze nuove (`transformers`/`datasets`) e va contro la regola del
`pyproject.toml` minimale, quindi lo terrei per ultimo e con gli occhi aperti.
