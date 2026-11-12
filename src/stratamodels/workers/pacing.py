"""Pace and pauses of synthetic speech: measured the same way everywhere (the TTS worker that shapes a clip, and the eval metrics that score
it), so "150 words a minute" means one thing. Standalone (numpy only): it runs in every worker venv and in the service.

Rate. Words per minute counted on WORDS is noisy line by line ("I took a DNF." and "Organisers were supportive." differ two-fold in
syllables), so the rate is counted in syllables of the words as spoken (numbers read out: "367" is "three hundred and sixty seven") over
the speech's own length (first to last sound, after long pauses are shortened), and given back as words a minute at English's average
1.4 syllables a word. A line said at an even pace then reads the same whatever its words.

Pauses. Speech is the 20 ms frames within 40 dB of the clip's loudest; a pause is a run of quieter frames between the first and last
sound. Pauses longer than `max_pause` are cut down to it (the middle is removed, with 10 ms fades), and the silence before and after the
speech is cut to `edge` seconds."""

import re

import numpy as np

SYLLABLES_PER_WORD = 1.4
FRAME_S = 0.02
FLOOR_DB = 40.0

_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


LETTERS = dict(W="double you", H="aitch", Y="why")  # a letter is one syllable, except these


def number_words(n):
    """A whole number as British English says it (367: three hundred and sixty seven)."""
    n = int(n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " and " + number_words(n % 100))
    if n < 1_000_000:
        return number_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + number_words(n % 1000))
    return " ".join(_ONES[int(d)] for d in str(n))


def spoken(text):
    """The words as they are said: digits and abbreviations in capitals read out, separators dropped."""
    t = re.sub(
        r"\b[A-Z]{2,5}\b", lambda m: " ".join(LETTERS.get(c, c) for c in m.group()), text or ""
    )  # DNF is said "dee en eff"
    t = re.sub(r"\d+", lambda m: " " + number_words(m.group()) + " ", t.lower().replace(",", ""))
    return re.findall(r"[a-z']+", t)


def syllables(text):
    """Syllables of a line as spoken (vowel groups per word, a silent final e dropped; at least one a word)."""
    n = 0
    for w in spoken(text):
        k = len(re.findall(r"[aeiouy]+", w)) - (1 if w.endswith("e") and not w.endswith("le") and len(w) > 2 else 0)
        n += max(1, k)
    return n


def target_seconds(text, wpm):
    """How long the speech of `text` lasts at `wpm` (words a minute, of 1.4 syllables)."""
    return syllables(text) / (float(wpm) * SYLLABLES_PER_WORD / 60.0)


def frames(x, sr):
    """(frame length in samples, level of each 20 ms frame in dB, whether it is speech)."""
    f = max(int(sr * FRAME_S), 1)
    n = len(x) // f
    if n == 0:
        return f, np.zeros(0), np.zeros(0, bool)
    db = 20 * np.log10(np.sqrt((x[: n * f].reshape(n, f).astype(np.float64) ** 2).mean(axis=1)) + 1e-9)
    return f, db, db > db.max() - FLOOR_DB


def pauses(x, sr):
    """(start s, end s of the speech, [pause lengths in s between them])."""
    f, _, sp = frames(x, sr)
    idx = np.flatnonzero(sp)
    if not len(idx):
        return 0.0, len(x) / sr, []
    a, b = int(idx[0]), int(idx[-1])
    out, run = [], 0
    for s in sp[a : b + 1]:
        if not s:
            run += 1
        elif run:
            out.append(run * f / sr)
            run = 0
    return a * f / sr, (b + 1) * f / sr, out


def tighten(x, sr, max_pause=0.45, edge=0.05):
    """The clip with every pause longer than `max_pause` cut to it (its middle removed) and `edge` seconds of silence kept before and after
    the speech; every cut gets 10 ms fades so it does not click."""
    x = np.asarray(x, np.float32)
    f, _, sp = frames(x, sr)
    idx = np.flatnonzero(sp)
    if not len(idx):
        return x
    a, b = int(idx[0]), int(idx[-1])
    keep = [[max(a * f - int(edge * sr), 0), min((b + 1) * f + int(edge * sr), len(x))]]
    if max_pause is not None:
        half, i = int(max_pause * sr / 2), a
        while i <= b:
            if sp[i]:
                i += 1
                continue
            j = i
            while j <= b and not sp[j]:
                j += 1
            if (j - i) * f > max_pause * sr:
                end = keep[-1][1]
                keep[-1][1] = i * f + half
                keep.append([j * f - half, end])
            i = j
    fade = max(int(0.01 * sr), 1)
    out = []
    for k, (s0, s1) in enumerate(keep):
        seg = x[s0:s1].copy()
        n = min(fade, len(seg))
        if k and n:
            seg[:n] *= np.linspace(0, 1, n, dtype=np.float32)
        if k < len(keep) - 1 and n:
            seg[-n:] *= np.linspace(1, 0, n, dtype=np.float32)
        out.append(seg)
    return np.concatenate(out)


def measure(text, x, sr):
    """{speech_s, rate_wpm, max_pause_s, long_pauses (over 0.6 s)} of a clip."""
    a, b, ps = pauses(np.asarray(x, np.float32), sr)
    speech = max(b - a, 1e-3)
    return dict(
        speech_s=round(speech, 3),
        rate_wpm=round(syllables(text) / speech * 60.0 / SYLLABLES_PER_WORD, 1),
        max_pause_s=round(max(ps, default=0.0), 3),
        long_pauses=sum(1 for p in ps if p > 0.6),
    )


def read_wav(path):
    """(float32 mono samples, sample rate) of a 16-bit PCM WAV (standard library only)."""
    import wave

    with wave.open(path) as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        raw = w.readframes(n)
    return np.frombuffer(raw, "<i2").astype(np.float32).reshape(-1, ch).mean(axis=1) / 32768.0, sr
