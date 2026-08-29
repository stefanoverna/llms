"""
Costruisce un "tiny-pirandello": l'equivalente italiano di tiny-shakespeare.

Scarica da Project Gutenberg le opere teatrali in italiano di Luigi Pirandello
(pubblico dominio) e le riduce al formato di input.txt:

    PERSONAGGIO:
    testo della battuta

I testi su Gutenberg usano due convenzioni diverse, entrambe gestite qui:

  A) personaggio inline in corsivo, didascalie come blocchi in corsivo
         _Il macchinista._ Che faccio? Inchiodo.
         _Guarderà l'orologio._
     -> il discriminante non e' la forma del nome ma cosa segue lo span:
        se dopo la chiusura c'e' altro testo e' una battuta, se lo span
        occupa l'intero paragrafo e' una didascalia.

  B) personaggio su riga isolata tutta maiuscola, didascalie o rientrate
     di molti spazi oppure inline fra parentesi e corsivo
         TOTO'.
         (_come inorridito_).
         Io?

Le didascalie fra parentesi vengono rimosse ovunque: tiny-shakespeare non
contiene parentesi, quindi toglierle avvicina i due corpus.

Uso:  python build_input_it.py [output.txt]
"""

import re
import sys
import unicodedata
import urllib.request

# Solo teatro in italiano: escluse le opere in siciliano (35804, 31702),
# il saggio "L'umorismo" (56958) e le novelle (56775).
OPERE = {
    18457: "Sei personaggi in cerca d'autore",
    18456: "Enrico IV",
    67417: "L'uomo, la bestia e la virtu",
    64291: "Come prima meglio di prima",
    65627: "La ragione degli altri",
    65798: "La signora Morli una e due",
    65028: "Vestire gli ignudi",
    64680: "Pensaci, Giacomino!",
    65713: "Tutto per bene",
    69997: "La vita che ti diedi",
    65297: "L'innesto",
    64845: "Lumie di Sicilia",
}

URL = "https://www.gutenberg.org/ebooks/{}.txt.utf-8"

START = re.compile(r"\*\*\* ?START OF TH[EI]S? PROJECT GUTENBERG.*?\*\*\*", re.I)
END = re.compile(r"\*\*\* ?END OF TH[EI]S? PROJECT GUTENBERG.*?\*\*\*", re.I)

PAREN = re.compile(r"\([^()]*\)", re.S)
ITALIC_HEAD = re.compile(r"^_(.+?)\._", re.S)

# intestazioni di struttura, non battute
SKIP = re.compile(
    r"^\s*(ATTO|SCENA|PERSONAGGI|I PERSONAGGI|GLI ATTORI|NOTE|INDICE|FINE|"
    r"SIPARIO|TELA|N\.\s?B\.)\b",
    re.I,
)

# parole che restano minuscole nei nomi di personaggio, se non iniziali
MINUSCOLE = {
    "di", "da", "del", "dello", "della", "dei", "degli", "delle", "dal",
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "e", "a",
    "in", "con", "su", "per", "tra", "fra", "al", "allo", "alla",
}


def strip_gutenberg(text: str) -> str:
    if m := START.search(text):
        text = text[m.end():]
    if m := END.search(text):
        text = text[: m.start()]
    return text


def drop_parens(text: str) -> str:
    """Rimuove le didascalie fra parentesi, anche annidate e multi-riga."""
    while True:
        new = PAREN.sub(" ", text)
        if new == text:
            return new
        text = new


def nome(raw: str) -> str:
    raw = raw.replace("_", " ").strip().strip(".,;:").strip()
    raw = " ".join(raw.split())
    parole = []
    for i, w in enumerate(raw.split(" ")):
        low = w.lower()
        parole.append(low if i and low in MINUSCOLE else low.capitalize())
    return " ".join(parole)


def is_speaker_line(line: str) -> bool:
    """Famiglia B: riga isolata tutta maiuscola, breve."""
    s = line.strip().rstrip(".")
    if not (2 <= len(s) <= 38) or s != s.upper():
        return False
    if not sum(c.isalpha() for c in s) >= 2:
        return False
    # numeri romani e simili
    return not set(s) <= set("IVXLCDM .-")


def clean(text: str) -> str:
    text = drop_parens(strip_gutenberg(text))

    out: list[str] = []
    speaker: str | None = None   # personaggio in attesa di battuta
    started = False              # true dopo la prima battuta vera

    for para in re.split(r"\n\s*\n", text):
        para = para.strip("\n")
        if not para.strip():
            continue

        indent = len(para) - len(para.lstrip(" "))
        body = para.strip()

        # didascalia rientrata (famiglia B1)
        if indent >= 12:
            continue
        if SKIP.match(body):
            speaker = None
            continue

        # --- famiglia A: paragrafo che apre in corsivo ---
        if body.startswith("_"):
            m = ITALIC_HEAD.match(body)
            if not m:
                continue                       # corsivo non chiuso: scarto
            resto = body[m.end():].strip()
            if not resto:
                continue                       # didascalia pura
            speaker = nome(m.group(1))
            body = resto

        # --- famiglia B: riga isolata maiuscola ---
        elif is_speaker_line(body) and "\n" not in body:
            speaker = nome(body)
            continue

        righe = [l.strip() for l in body.split("\n")]
        righe = [l for l in righe if l and any(c.isalnum() for c in l)]
        if not righe:
            continue

        if speaker is not None:
            if out:
                out.append("")
            out.append(f"{speaker}:")
            speaker = None
            started = True
        elif not started:
            continue                           # frontespizio / elenco ruoli

        out.extend(righe)

    return "\n".join(out)


def main() -> None:
    dest = sys.argv[1] if len(sys.argv) > 1 else "input_it.txt"
    parts = []
    for book_id, title in OPERE.items():
        with urllib.request.urlopen(URL.format(book_id), timeout=120) as r:
            raw = r.read().decode("utf-8")
        piece = clean(raw)
        print(f"{book_id:>6}  {len(piece)/1024:7.0f} KB  {title}")
        parts.append(piece)

    corpus = "\n\n".join(parts) + "\n"

    # meno simboli nel vocabolario: apostrofi e virgolette in ASCII
    for a, b in [("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                 ("«", '"'), ("»", '"'), ("„", '"'), ("—", "-"),
                 ("–", "-"), ("…", "...")]:
        corpus = corpus.replace(a, b)

    # caratteri con una o due occorrenze (spagnolo e francese di scena, latino
    # liturgico, refusi OCR): ripiegati su lettere gia' nel vocabolario, per
    # non tenere una riga di embedding per un carattere mai imparato
    for a, b in [("Ê", "È"), ("À", "A"), ("Y", "y"), ("ó", "o"),
                 ("í", "i"), ("ï", "i"), ("ç", "c"), ("ñ", "n"),
                 ("æ", "ae")]:
        corpus = corpus.replace(a, b)
    corpus = re.sub(r"[_*\[\]]", "", corpus)
    corpus = re.sub(r"[ \t]+", " ", corpus)
    corpus = re.sub(r" +\n", "\n", corpus)
    corpus = unicodedata.normalize("NFC", corpus)

    with open(dest, "w", encoding="utf-8") as f:
        f.write(corpus)

    vocab = sorted(set(corpus))
    print(f"\n{dest}: {len(corpus):,} caratteri, "
          f"{corpus.count(chr(10)):,} righe, vocab {len(vocab)}")
    print("vocab:", "".join(vocab).replace("\n", "\\n"))


if __name__ == "__main__":
    main()
