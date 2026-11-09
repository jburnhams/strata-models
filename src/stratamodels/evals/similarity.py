"""How alike two outputs are, from 0 (nothing in common) to 1 (the same), whatever the role: the Judge tab shows the most different pairs
first, so a person's time goes where the candidates really disagree, and pairs that say the same thing are never shown.

Kinds (picked from the outputs' shape by `kind_of`, or named in a spec as `similarity = "..."`):
  text    transcripts, answers, captions: the mean of
            word similarity      1 - word edit distance / the longer length, after the evals' text normalisation (metrics.normalise)
            phonetic similarity  the same on a phonetic key of each word (a small Metaphone), so "their" and "there", "for" and "four" count
                                 as the same sound: a pair that differs only in spelling is less worth a listen than one that heard other words
  image   generated pictures: the mean of
            perceptual hash      64-bit DCT hash (pHash) of a 32 x 32 grey copy: 1 - Hamming distance / 64; tolerant of small changes
            structure            SSIM of 64 x 64 grey copies: same layout and edges
            colour               histogram intersection of 8 x 8 x 8 RGB bins: same palette
  audio   spoken outputs (speak): the mean of
            length               shorter duration / longer: pace and pauses
            timbre               cosine similarity of the two recordings' mean log spectrum in 48 bands (PCM WAV, read with the standard
                                 library): the same voice and recording colour score near 1, a different voice lower
  json    anything else: 1 when the outputs are the same JSON (timings left out), else the text similarity of their JSON.
Images are read with Pillow (installed with the GPU stack); without it the image kind falls back to comparing the files' bytes."""

import functools, json, os, re

import numpy as np

from stratamodels.evals import metrics as M

VOLATILE = (
    "seconds",
    "peak_vram_mb",
    "path",
    "scores",
    "scored",
)  # never a difference in themselves (for images the picture behind `path` is compared; the [score] pass's readings are not the output)
IMAGE_EXT = re.compile(r"\.(png|jpe?g|webp)$", re.I)


# ---- text -----------------------------------------------------------------------------------------------------------------------------------
def edit_similarity(a, b):
    """1 - Levenshtein distance / the longer sequence (1 for two empty sequences)."""
    if not a and not b:
        return 1.0
    return 1.0 - M.edits(list(a), list(b)) / max(len(a), len(b))


_RULES = [
    (r"^kn|^gn|^pn|^wr", lambda m: m.group(0)[1]),
    (r"^x", "s"),
    (r"^wh", "w"),
    (r"mb$", "m"),
    (r"sch", "sk"),
    (r"tch", "ch"),
    (r"ph", "f"),
    (r"dg(?=[eiy])", "j"),
    (r"ck", "k"),
    (r"c(?=[iey])", "s"),
    (r"ch", "x"),
    (r"sh", "x"),
    (r"th", "0"),
    (r"gh(?![aeiou])", ""),
    (r"g(?=[iey])", "j"),
    (r"q", "k"),
    (r"c", "k"),
    (r"x", "ks"),
    (r"z", "s"),
    (r"v", "f"),
    (r"(?<=[^aeiou])h(?![aeiou])", ""),
    (r"w(?![aeiou])", ""),
]


@functools.lru_cache(maxsize=65536)
def phonetic(word):
    """A small Metaphone: a key that is the same for words that sound alike in English ("there" and "their" -> "0r", "four" and "for" -> "fr")."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return word.lower()
    if w.isdigit():
        return w
    for pat, rep in _RULES:
        w = re.sub(pat, rep, w)
    w = w[0] + re.sub(r"[aeiouy]", "", w[1:])  # vowels only count at the start
    return re.sub(r"(.)\1+", r"\1", w)  # doubled letters sound once


def text_similarity(a, b):
    """Mean of word and phonetic similarity of two texts, after normalisation."""
    wa, wb = M.normalise(a or ""), M.normalise(b or "")
    return (edit_similarity(wa, wb) + edit_similarity([phonetic(w) for w in wa], [phonetic(w) for w in wb])) / 2


# ---- images ---------------------------------------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=512)
def _image_features(path, mtime):
    from PIL import Image

    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB").resize((64, 64), Image.BILINEAR), dtype=np.float32)
        grey32 = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32)
    n = 32
    k = np.arange(n)
    dct = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))  # DCT-II basis
    low = (dct @ grey32 @ dct.T)[:8, :8].ravel()[1:]  # the 63 lowest frequencies, without the mean
    bits = low > np.median(low)
    grey64 = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    hist = np.histogramdd(rgb.reshape(-1, 3), bins=(8, 8, 8), range=((0, 256),) * 3)[0].ravel()
    hist /= hist.sum()
    return bits, grey64, hist


def _ssim(x, y, c1=(0.01 * 255) ** 2, c2=(0.03 * 255) ** 2):
    """Global SSIM of two grey images of the same size (enough to rank pairs; not the windowed SSIM of the paper)."""
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    return float(((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2)))


def image_similarity(pa, pb):
    """Mean of perceptual-hash, structure and colour similarity of two picture files; 0 when either cannot be read."""
    if pa == pb:
        return 1.0
    try:
        fa = _image_features(pa, os.stat(pa).st_mtime_ns)
        fb = _image_features(pb, os.stat(pb).st_mtime_ns)
    except ImportError:
        try:
            return 1.0 if open(pa, "rb").read() == open(pb, "rb").read() else 0.0
        except OSError:
            return 0.0
    except OSError:
        return 0.0
    phash = 1.0 - float(np.count_nonzero(fa[0] != fb[0])) / fa[0].size
    return (phash + max(_ssim(fa[1], fb[1]), 0.0) + float(np.minimum(fa[2], fb[2]).sum())) / 3


# ---- outputs --------------------------------------------------------------------------------------------------------------------------------
def image_path(o):
    p = o.get("path") if isinstance(o, dict) else None
    return p if isinstance(p, str) and IMAGE_EXT.search(p) else None


AUDIO_EXT = re.compile(r"\.wav$", re.I)


def audio_path(o):
    p = o.get("path") if isinstance(o, dict) else None
    return p if isinstance(p, str) and AUDIO_EXT.search(p) else None


@functools.lru_cache(maxsize=2048)
def _spectrum(path, bands=48):
    """(duration s, mean log power at `bands` log-spaced frequencies from 80 Hz to 8 kHz) of a PCM WAV, or None when it cannot be read."""
    import wave

    try:
        with wave.open(path) as w:
            sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(n)
    except (OSError, EOFError, wave.Error):
        return None
    if sw != 2 or not n:
        return None
    x = np.frombuffer(raw, "<i2").astype(np.float32).reshape(-1, ch).mean(axis=1) / 32768.0
    size = 1024
    if len(x) < size:
        return n / sr, None
    frames = np.lib.stride_tricks.sliding_window_view(x, size)[:: size // 2] * np.hanning(size)
    power = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(size, 1 / sr)
    centres = np.geomspace(
        80, min(8000, sr / 2), bands
    )  # read off the spectrum at log-spaced centres: the same bands at any sample rate
    env = np.interp(centres, freqs, np.log10(power + 1e-10))
    return n / sr, env - env.mean()


def audio_similarity(a, b):
    sa, sb = _spectrum(a), _spectrum(b)
    if sa is None or sb is None:
        return 0.0
    length = min(sa[0], sb[0]) / max(sa[0], sb[0], 1e-9)
    if sa[1] is None or sb[1] is None:
        return length
    ea, eb = sa[1], sb[1]
    cos = float(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-12))
    return float((length + max(cos, 0.0)) / 2)


def kind_of(x, y):
    """'text', 'image', 'audio' or 'json', from the two outputs."""
    if isinstance(x, dict) and isinstance(y, dict):
        if "segments" in x and "segments" in y:
            return "text"
        if image_path(x) and image_path(y):
            return "image"
        if audio_path(x) and audio_path(y):
            return "audio"
    if isinstance(x, str) and isinstance(y, str):
        return "text"
    return "json"


def as_text(o):
    """The words of an output: a transcription's segments, a description's caption, a chat answer's text, or the output itself."""
    if not isinstance(o, dict):
        return "" if o is None else str(o)
    if "segments" in o:
        return M.transcript(o)
    d = o.get("json") if isinstance(o.get("json"), dict) else o
    for k in ("caption", "text"):
        if isinstance(d.get(k), str):
            return d[k]
    return o["text"] if isinstance(o.get("text"), str) else ""


def _strip(o):
    return {k: v for k, v in o.items() if k not in VOLATILE} if isinstance(o, dict) else o


def similarity(x, y, kind=None):
    """How alike two outputs are, 0 to 1."""
    kind = kind or kind_of(x, y)
    if kind == "text":
        return text_similarity(as_text(x), as_text(y))
    if kind == "image":
        pa, pb = image_path(x), image_path(y)
        return image_similarity(pa, pb) if pa and pb else 0.0
    if kind == "audio":
        pa, pb = audio_path(x), audio_path(y)
        return audio_similarity(pa, pb) if pa and pb else 0.0
    if kind != "json":
        raise ValueError(f"no similarity {kind!r} (known: text, image, audio, json)")
    a, b = json.dumps(_strip(x), sort_keys=True), json.dumps(_strip(y), sort_keys=True)
    return 1.0 if a == b else text_similarity(a, b)


KINDS = ("text", "image", "audio", "json")
