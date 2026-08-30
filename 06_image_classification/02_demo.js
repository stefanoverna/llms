/**
 * La rete di 01_network.py che gira nel browser, e il lavoro che serve a
 * darle in pasto qualcosa che assomigli a MNIST.
 *
 * L'inferenza e' la parte breve: due matrice-per-vettore e una tanh, il
 * blocco `forward()` piu' sotto. Nessuna libreria, nessun ONNX, nessun
 * TensorFlow.js — 636.010 numeri e due cicli annidati. Tutto il macchinario di
 * `01_network.py` serviva ad allenarla; usarla e' aritmetica delle superiori.
 *
 * La parte lunga, e quella che decide se la demo funziona, e' il
 * preprocessing. MNIST non e' "un disegno 28x28": e' un disegno passato per la
 * normalizzazione di LeCun, e misurando le 50.000 immagini di training si vede
 * esattamente cosa fa.
 *
 *   - il lato lungo del bounding box dell'inchiostro e' **20 pixel in tutte e
 *     50.000**: media 20.00, massimo 20. Non "circa venti", venti.
 *   - il centro di massa cade a (14.0, 14.0) con deviazione standard 0.29,
 *     cioe' le cifre sono ricentrate sul baricentro dell'inchiostro, non sul
 *     centro del loro bounding box.
 *   - l'aspect ratio e' preservato: l'altezza media e' 19.73, la larghezza
 *     15.69. Un `1` resta stretto.
 *
 * Quindi `preprocess()` fa la stessa cosa: ritaglia l'inchiostro, lo scala
 * finche' il lato lungo e' 20, e lo incolla nel 28x28 posizionando il centro
 * di massa a (14, 14). `downscale()` invece e' la versione ingenua, il canvas
 * schiacciato a 28x28 e basta — sta li' per essere guardata mentre si disegna,
 * perche' la differenza fra le due predizioni e' il punto di tutta la demo.
 *
 * Quanto vale, misurato: si prendono 300 cifre del test set, si ridisegnano
 * sul canvas da 280 a scala e posizione casuali (lato fra 120 e 270 pixel), e
 * si guarda quante ne ritrova la rete passando per le due strade.
 *
 *     pixel MNIST originali, la baseline    99.3%
 *     ridisegnate + preprocess()            99.3%
 *     ridisegnate + downscale()             40.7%
 *
 * La normalizzazione **ricompra tutto**: il giro attraverso un canvas dieci
 * volte piu' grande, a una scala e in un punto che la rete non ha mai visto,
 * non costa un decimo di punto. Senza, si perdono cinquantanove punti e la
 * demo sembra rotta.
 *
 * Il motivo per cui conta cosi' tanto e' nel README: questa MLP non ha
 * **nessuna invarianza per traslazione**, e `01_out_hidden_neurons.png` lo
 * mostra — gli 800 neuroni nascosti sono macchie, ognuna sensibile a una zona
 * precisa dell'immagine. Un 7 disegnato due pixel piu' in basso accende
 * neuroni diversi. Una convoluzionale perdonerebbe, questa no. Quando la rete
 * senza normalizzazione sbaglia, non e' la rete a essere debole: e' che le
 * stiamo mostrando una distribuzione che non ha mai visto.
 *
 * Anche il pennello viene da una misura, non dall'occhio: lo spessore del
 * tratto in MNIST e' circa 3 pixel su un box da 20, cioe' il **15%** del lato
 * lungo della cifra. Su un canvas da 280 pixel, per una cifra che lo riempie,
 * sono i 30 di `BRUSH`. Sembra grosso ed e' giusto — le cifre di MNIST hanno
 * il tratto spesso, e disegnare sottile e' un altro modo di uscire dalla
 * distribuzione.
 */

"use strict";

// --- costanti ---------------------------------------------------------------

const CANVAS = 280; // il lato del riquadro su cui si disegna
const BRUSH = 30; // 15% di una cifra che riempie il canvas, come in MNIST
const BOX = 20; // il lato lungo dopo la normalizzazione, come in MNIST
const SIDE = 28; // il lato dell'immagine finale
const CENTER = 14.5; // dove va il centro di massa, in coordinate continue
const INK_THRESHOLD = 0.1; // sotto questa soglia e' frangia di antialiasing

// ---------------------------------------------------------------------------
// 1. i pesi
// ---------------------------------------------------------------------------

/** Da base64 ai quattro tensori, che in JavaScript sono quattro array piatti. */
function decodeWeights(weights) {
  const bytes = Uint8Array.from(atob(weights.data), (c) => c.charCodeAt(0));
  const flat = new Float32Array(bytes.buffer);

  const h = weights.hidden;
  let at = 0;
  const take = (n) => flat.subarray(at, (at += n));

  return {
    hidden: h,
    w1: take(h * 784), // (200, 784) per righe
    b1: take(h),
    w2: take(10 * h), // (10, 200) per righe
    b2: take(10),
  };
}

// ---------------------------------------------------------------------------
// 2. il forward
// ---------------------------------------------------------------------------

/** 784 pixel -> 10 probabilita'. E' tutta qui, la rete. */
function forward(net, x) {
  const h = new Float32Array(net.hidden);
  for (let j = 0; j < net.hidden; j++) {
    let sum = net.b1[j];
    const row = j * 784;
    for (let i = 0; i < 784; i++) sum += net.w1[row + i] * x[i];
    h[j] = Math.tanh(sum);
  }

  const logits = new Float32Array(10);
  for (let k = 0; k < 10; k++) {
    let sum = net.b2[k];
    const row = k * net.hidden;
    for (let j = 0; j < net.hidden; j++) sum += net.w2[row + j] * h[j];
    logits[k] = sum;
  }

  return softmax(logits);
}

/** Si sottrae il massimo prima di esponenziare, se no exp() va a infinito. */
function softmax(logits) {
  const max = Math.max(...logits);
  const exp = logits.map((v) => Math.exp(v - max));
  const total = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / total);
}

function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

// ---------------------------------------------------------------------------
// 3. il preprocessing
// ---------------------------------------------------------------------------

/** L'inchiostro di un canvas come float in [0, 1]: 0 bianco, 1 nero.
 *
 * La convenzione e' quella di `mnist.py`, e va rispettata: si disegna scuro su
 * chiaro perche' e' quello che ci si aspetta da una lavagna, ma la rete ha
 * visto l'opposto, quindi qui si inverte.
 */
function readInk(ctx, width, height) {
  const { data } = ctx.getImageData(0, 0, width, height);
  const ink = new Float32Array(width * height);
  for (let i = 0; i < ink.length; i++) {
    // il canvas ha lo sfondo bianco opaco, quindi basta un canale
    ink[i] = 1 - data[i * 4] / 255;
  }
  return ink;
}

/** Il rettangolo che contiene l'inchiostro, o null se il canvas e' vuoto. */
function inkBounds(ink, width, height) {
  let x0 = width,
    y0 = height,
    x1 = -1,
    y1 = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (ink[y * width + x] <= INK_THRESHOLD) continue;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
  }
  return x1 < 0 ? null : { x0, y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

/** Il centro di massa dell'inchiostro, in coordinate continue del canvas. */
function centerOfMass(ink, width, height) {
  let mass = 0,
    mx = 0,
    my = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = ink[y * width + x];
      mass += v;
      mx += v * (x + 0.5);
      my += v * (y + 0.5);
    }
  }
  return { x: mx / mass, y: my / mass };
}

/** La normalizzazione di MNIST: box da 20, aspect ratio intatto, baricentro al centro.
 *
 * Il ricentraggio si fa in una sola `drawImage()` invece che spostando i pixel
 * dopo: si calcola dove finirebbe il centro di massa una volta scalato, e si
 * sceglie l'origine di destinazione perche' cada su (14.5, 14.5). Traslare
 * dopo, di una quantita' frazionaria, vorrebbe dire interpolare due volte.
 */
function preprocess(source, scratch) {
  const width = source.canvas.width;
  const height = source.canvas.height;
  const ink = readInk(source, width, height);
  const bounds = inkBounds(ink, width, height);
  if (!bounds) return null;

  const scale = BOX / Math.max(bounds.w, bounds.h);
  const mass = centerOfMass(ink, width, height);

  // dove cade il centro di massa se disegno il ritaglio con origine (0, 0)
  const massX = (mass.x - bounds.x0) * scale;
  const massY = (mass.y - bounds.y0) * scale;

  scratch.fillStyle = "#fff";
  scratch.fillRect(0, 0, SIDE, SIDE);
  scratch.imageSmoothingEnabled = true; // e' l'antialiasing che fa i grigi
  scratch.drawImage(
    source.canvas,
    bounds.x0,
    bounds.y0,
    bounds.w,
    bounds.h,
    CENTER - massX,
    CENTER - massY,
    bounds.w * scale,
    bounds.h * scale,
  );

  return readInk(scratch, SIDE, SIDE);
}

/** La versione ingenua: tutto il canvas schiacciato in 28x28, e via. */
function downscale(source, scratch) {
  scratch.fillStyle = "#fff";
  scratch.fillRect(0, 0, SIDE, SIDE);
  scratch.imageSmoothingEnabled = true;
  scratch.drawImage(source.canvas, 0, 0, SIDE, SIDE);
  return readInk(scratch, SIDE, SIDE);
}

// ---------------------------------------------------------------------------
// 4. la pagina
// ---------------------------------------------------------------------------

function init() {
  const net = decodeWeights(WEIGHTS);

  const board = document.getElementById("board").getContext("2d", { willReadFrequently: true });
  const scratch = document
    .createElement("canvas")
    .getContext("2d", { willReadFrequently: true });
  scratch.canvas.width = scratch.canvas.height = SIDE;

  const views = {
    normalized: document.getElementById("view-normalized").getContext("2d"),
    raw: document.getElementById("view-raw").getContext("2d"),
  };
  const verdict = document.getElementById("verdict");
  const confidence = document.getElementById("confidence");
  const rawGuess = document.getElementById("guess-raw");
  const bars = [...document.querySelectorAll(".bar")].map((el) => ({
    fill: el.querySelector(".bar-fill"),
    column: el,
  }));

  document.getElementById("accuracy").textContent = `${WEIGHTS.test_accuracy}%`;
  document.getElementById("params").textContent = WEIGHTS.params.toLocaleString("it-IT");

  function clear() {
    board.fillStyle = "#fff";
    board.fillRect(0, 0, CANVAS, CANVAS);
    board.lineWidth = BRUSH;
    board.lineCap = "round";
    board.lineJoin = "round";
    board.strokeStyle = "#111";
    predict();
  }

  /** Un 28x28 disegnato a pixel grossi, come lo vede la rete. */
  function paint(ctx, x) {
    const cell = ctx.canvas.width / SIDE;
    for (let i = 0; i < SIDE * SIDE; i++) {
      const v = Math.round(255 * (1 - x[i]));
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect((i % SIDE) * cell, Math.floor(i / SIDE) * cell, cell, cell);
    }
  }

  function paintEmpty(ctx) {
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  function predict() {
    const normalized = preprocess(board, scratch);
    const raw = normalized && downscale(board, scratch);

    if (!normalized) {
      paintEmpty(views.normalized);
      paintEmpty(views.raw);
      verdict.textContent = "—";
      verdict.classList.add("empty");
      confidence.textContent = "";
      rawGuess.textContent = "—";
      rawGuess.classList.remove("disagrees");
      bars.forEach((bar) => {
        bar.fill.style.height = "0%";
        bar.column.classList.remove("winner");
      });
      return;
    }

    paint(views.normalized, normalized);
    paint(views.raw, raw);

    const p = forward(net, normalized);
    const q = forward(net, raw);
    const winner = argmax(p);

    verdict.textContent = winner;
    verdict.classList.remove("empty");
    confidence.textContent = `${(p[winner] * 100).toFixed(1)}% sicura`;

    rawGuess.textContent = argmax(q);
    rawGuess.classList.toggle("disagrees", argmax(q) !== winner);

    bars.forEach((bar, digit) => {
      // le barre sono verticali: cresce l'altezza, non la larghezza
      bar.fill.style.height = `${(p[digit] * 100).toFixed(1)}%`;
      bar.column.classList.toggle("winner", digit === winner);
    });
  }

  // --- il pennello ---

  let drawing = false;

  function at(event) {
    const box = board.canvas.getBoundingClientRect();
    const point = event.touches ? event.touches[0] : event;
    return {
      x: ((point.clientX - box.left) / box.width) * CANVAS,
      y: ((point.clientY - box.top) / box.height) * CANVAS,
    };
  }

  function start(event) {
    event.preventDefault();
    drawing = true;
    const { x, y } = at(event);
    board.beginPath();
    board.moveTo(x, y);
    // un punto solo deve lasciare un segno, se no un tap non disegna niente
    board.lineTo(x, y);
    board.stroke();
    predict();
  }

  function move(event) {
    if (!drawing) return;
    event.preventDefault();
    const { x, y } = at(event);
    board.lineTo(x, y);
    board.stroke();
    predict();
  }

  function end() {
    drawing = false;
  }

  board.canvas.addEventListener("mousedown", start);
  board.canvas.addEventListener("mousemove", move);
  board.canvas.addEventListener("touchstart", start, { passive: false });
  board.canvas.addEventListener("touchmove", move, { passive: false });
  window.addEventListener("mouseup", end);
  window.addEventListener("touchend", end);

  document.getElementById("clear").addEventListener("click", clear);

  clear();
}

if (typeof window !== "undefined") window.addEventListener("DOMContentLoaded", init);
