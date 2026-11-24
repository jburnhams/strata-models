"""Pure consonant sounds for phonics ("mmm", not "muh"), cut from speech, and the measures that check them. Standalone (numpy only): the TTS
worker cuts with it and the eval metrics score with it, so "no added vowel" means one thing in both.

Why cut. Kokoro asked for a lone consonant in IPA (/ʃ/) gives about 20 ms of it and then a vowel: it was trained on words, never on a sound
alone. Said at the start of a syllable (a carrier, /ʃɑː/), the consonant comes out whole. So the worker says the carrier, finds where the
vowel starts and keeps what comes before it:

  vowel onset  The carrier's loudest frames (within 3 dB of its peak) are its vowel. Each 10 ms frame's spectrum (24 log bands, 80 Hz to 11 kHz) is compared with the
               vowel's own; the onset is the first frame from which three in a row sit nearer the vowel than halfway between the
               consonant's distance and the vowel's, and within 12 dB of the clip's loudest. The cut is made one frame before it, so the
               vowel's first glottal pulses stay out (a stop is cut at it: its burst can be that frame). Within that, a hiss keeps its
               longest high-frequency run and a nasal or liquid its longest low-frequency one (span()), dropping the breath and the low
               thump Kokoro can make before a sound.
  hold         A sound that can be held (fricatives, nasals, l and r) is lengthened to `hold_s` from the middle of its span (its steady
               part) by overlap-adding 40 ms grains: at whole pitch periods for a voiced sound, so the waveform keeps its phase and does
               not buzz; at random places for noise (s, sh, f, th, h), which has no phase to keep.
  stops        p t k b d g and the affricates ch, j: the burst alone, cut before the voicing of the vowel, as phonics teaching wants.
  brief        w, y and the pairs x (/ks/) and qu (/kw/): the span as said, with no hold, since none of them is a sound held on its own.

Measures (measure()): high (share of the energy above 2.5 kHz: most of a fricative's), low (below 1 kHz), voiced (share of the sound's
frames with a clear pitch), and vowel_leak: the share of the sound's energy in frames that look like a vowel (voiced, with over 40 % of
the frame's energy in 500 Hz to 3 kHz, where a vowel's first two formants are; a nasal or a voiced fricative keeps its voiced energy
lower, or noise higher). It is rough: a close vowel such as /iː/, with its first formant under 500 Hz, scores about 0.2. pure()
turns the measures into a pass or fail per kind of sound, for the eval's sound_pure."""

import itertools

import numpy as np

HOP_S, WIN_S = 0.01, 0.025

# Letters and Sounds, Phases 2 and 3: the 25 consonant sounds, in the order taught, with how each is said. (IPA as Kokoro takes it: ɹ for
# the English r, ɡ the IPA g.)  kind: hiss (voiceless fricative, held), buzz (voiced fricative, held), nasal and liquid (held), stop and
# affricate (the burst only), brief (said as is, not held).
CONSONANTS = {
    "s": ("s", "hiss"),
    "t": ("t", "stop"),
    "p": ("p", "stop"),
    "n": ("n", "nasal"),
    "m": ("m", "nasal"),
    "d": ("d", "stop"),
    "g": ("ɡ", "stop"),
    "c/k/ck": ("k", "stop"),
    "r": ("ɹ", "liquid"),
    "h": ("h", "hiss"),
    "b": ("b", "stop"),
    "f/ff": ("f", "hiss"),
    "l/ll": ("l", "liquid"),
    "j": ("dʒ", "affricate"),
    "v": ("v", "buzz"),
    "w": ("w", "brief"),
    "x": ("ks", "brief"),
    "y": ("j", "brief"),
    "z/zz": ("z", "buzz"),
    "qu": ("kw", "brief"),
    "ch": ("tʃ", "affricate"),
    "sh": ("ʃ", "hiss"),
    "th (thin)": ("θ", "hiss"),
    "th (then)": ("ð", "buzz"),
    "ng": ("ŋ", "nasal"),
}
KINDS = dict(CONSONANTS.values())
HELD = {"hiss", "buzz", "nasal", "liquid"}
VOWELS = set("aeiouəɪʊɛæɑɒɔʌɜɐ")


def kind(sound):
    """How a sound is said: a kind from CONSONANTS, "vowel" for a sound with a vowel in it (said as it is), or "other" for a consonant not
    in the table (cut like a brief sound)."""
    s = sound.replace("ː", "").replace("ˈ", "").strip()
    if s in KINDS:
        return KINDS[s]
    return "vowel" if any(c in VOWELS for c in s) else "other"


def bands(x, sr):
    """(log band energies per frame [frames, 24] in dB, power spectra per frame, their frequencies)."""
    h, w = int(sr * HOP_S), int(sr * WIN_S)
    if len(x) < w:
        x = np.pad(x, (0, w - len(x)))
    win = np.hanning(w)
    n = 1 + (len(x) - w) // h
    f = np.stack([np.abs(np.fft.rfft(x[i * h : i * h + w] * win)) ** 2 for i in range(n)])
    fr = np.fft.rfftfreq(w, 1 / sr)
    edges = np.geomspace(80, min(sr / 2, 11000), 25)
    b = np.stack([f[:, (fr >= a) & (fr < c)].sum(1) for a, c in itertools.pairwise(edges)], 1)
    return 10 * np.log10(b + 1e-10), f, fr


def onset(x, sr):
    """(start, cut, loud) in samples of a consonant + vowel carrier: where its sound starts, where to cut before the vowel, and the vowel's
    first frame within 3 dB of the carrier's peak. None when the clip has no clear vowel to find (too short, or silent)."""
    lb, f, _ = bands(x, sr)
    h = int(sr * HOP_S)
    e = 10 * np.log10(f.sum(1) + 1e-10)
    on = np.where(e > e.max() - 35)[0]
    if len(on) < 8:
        return None
    s0 = int(on[0])
    loud = np.where(e > e.max() - 3)[
        0
    ]  # the vowel is the loudest part of the carrier (Kokoro's tail after it can be long and quiet)
    loud = loud[loud > s0 + 1]
    if len(loud) < 2:
        return None
    va = int(loud[0])
    vowel = loud[1:] if len(loud) > 2 else loud
    d = np.sqrt(((lb - lb[vowel].mean(0)) ** 2).mean(1))  # each frame's distance from the vowel's own spectrum, in dB
    dc, dv = float(np.median(d[s0 : s0 + max(3, (va - s0) // 3)])), float(np.median(d[vowel]))
    if dc <= dv:
        return None  # the start sounds no different from the vowel: nothing to cut
    thr = (dc + dv) / 2
    near = (d < thr) & (
        e > e.max() - 12
    )  # the vowel is loud: a stop's burst can sit near its spectrum, never at its level
    k = s0 + 1
    while k < va and not near[k : k + 3].all():
        k += 1
    return s0 * h, k * h, va * h


def period(y, sr):
    """The pitch period in samples (70 to 400 Hz), or None for a sound with no clear pitch (noise): the lag of the highest normalised
    cross-correlation of the clip with itself, when it is over 0.6."""
    y = np.asarray(y, np.float64) - np.mean(y)
    n = len(y)
    lo, hi = int(sr / 400), min(int(sr / 70), n // 2)
    if hi <= lo or not y.any():
        return None
    c = np.correlate(y, y, "full")[n - 1 :]
    e = np.cumsum(y**2)
    head = e[n - 1 - np.arange(n)]  # energy of y[: n - t]
    tail = e[-1] - np.concatenate([[0.0], e[:-1]])  # energy of y[t:]
    r = c / np.sqrt(np.maximum(head * tail, 1e-20))
    # every multiple of the period correlates about as well as the period itself (a steady tone: 0.99999999999999 at 160 and at 320
    # samples), so a plain argmax picks either by rounding, which differs by platform (macOS CI found 320): the shortest lag within 1 %
    # of the best, among the peaks (the lags beside a peak are within 1 % of it too), is the period
    w = r[lo - 1 : hi + 1] if lo > 0 else np.concatenate([[-np.inf], r[: hi + 1]])
    peak = (w[1:-1] >= w[:-2]) & (w[1:-1] >= w[2:])
    best = float(np.max(w[1:-1]))
    ok = peak & (w[1:-1] >= 0.99 * best)
    t = lo + int(np.argmax(ok) if ok.any() else np.argmax(w[1:-1]))  # no peak inside the range: its best end, as before
    return t if r[t] > 0.6 else None


def hold(y, sr, secs, seed=0, t=None):
    """Lengthen a steady sound to `secs` by overlap-adding 40 ms Hann grains of it (see the module doc). t: its pitch period in samples
    (found here when not given; None for noise)."""
    t = t or period(y, sr)
    if t and t * 2 > len(y):
        t = None
    g = int(sr * 0.04)
    if t:
        g = max(2 * t, (g // t) * t)
    g = min(g, len(y))
    g -= g % 2
    if g < 4:
        return np.zeros(int(secs * sr), np.float32)
    hop = max(t, (g // 2 // t) * t) if t else g // 2
    n = int(secs * sr)
    out, norm, w = np.zeros(n + g), np.zeros(n + g), np.hanning(g)
    span, rng = len(y) - g, np.random.default_rng(seed)
    pos, step = 0, t or 1
    for o in range(0, n, hop):
        if t:  # walk back and forth through the span a period at a time
            p = pos
            if not 0 <= pos + step <= span:
                step = -step
            pos = min(max(pos + step, 0), span)
        else:
            p = int(rng.integers(0, span + 1))
        out[o : o + g] += y[p : p + g] * w
        norm[o : o + g] += w
    return (out[:n] / np.maximum(norm[:n], 1e-3)).astype(np.float32)


def fade(y, sr, fin=0.015, fout=0.04):
    y = np.array(y, np.float32)
    a, b = min(int(sr * fin), len(y) // 2), min(int(sr * fout), len(y) // 2)
    if a:
        y[:a] *= np.linspace(0, 1, a)
    if b:
        y[len(y) - b :] *= np.linspace(1, 0, b)
    return y


def _longest(mask, lo, hi):
    """(first, last + 1) of the longest run of True in mask[lo:hi], or None."""
    best, i = None, lo
    while i < hi:
        if mask[i]:
            j = i
            while j < hi and mask[j]:
                j += 1
            if best is None or j - i >= best[1] - best[0]:  # ties go to the later run: the one next to the vowel
                best = (i, j)
            i = j
        else:
            i += 1
    return best


def span(x, sr, sound):
    """(start, end) in samples of the consonant in a carrier clip (the sound, then a vowel). The vowel onset bounds it (see the module doc);
    within that a hiss keeps its longest run of frames with most of their energy above 2.5 kHz, and a nasal or liquid its longest run
    with most below 500 Hz, which drops the breath and the low thump Kokoro can make before a sound starts."""
    k = kind(sound)
    c = onset(x, sr)
    if c is None:
        raise ValueError(f"no vowel found to cut /{sound}/ from")
    a, b, va = c
    h = int(sr * HOP_S)
    if k not in ("stop", "affricate"):
        b = max(
            a + h, b - h
        )  # one frame short of the vowel, so its first glottal pulses stay out; a stop keeps its burst, which can sit there
    _, f, fr = bands(x, sr)
    tot = f.sum(1) + 1e-12
    e = 10 * np.log10(tot)
    loud = e > e.max() - 30
    run = None
    if k == "hiss":
        run = _longest(loud & (f[:, fr > 2500].sum(1) / tot > 0.4), a // h, b // h)
    elif k in (
        "nasal",
        "liquid",
    ):  # a nasal fades into the vowel: its low run can carry on past the spectral onset, up to the loud vowel
        run = _longest(loud & (f[:, fr < 500].sum(1) / tot > 0.5), a // h, va // h)
        if run:
            run = (run[0], max(run[0] + 1, run[1] - 1))  # one frame short of the vowel
    if run and run[1] - run[0] >= 3:
        a, b = run[0] * h, run[1] * h if k != "hiss" else min(b, run[1] * h)
    return a, b


def cut(x, sr, sound, hold_s=0.5, seed=0):
    """(the pure sound, info) from a carrier clip (the sound then a vowel). info: {kind, span_s (the consonant as said), held_s}."""
    k = kind(sound)
    a, b = span(x, sr, sound)
    y = np.asarray(x[a:b], np.float32)
    info = dict(kind=k, span_s=round(len(y) / sr, 3))
    if k in HELD and hold_s and len(y) >= int(0.03 * sr):
        n = len(y)
        core = y[int(0.2 * n) : int(0.8 * n)] if n >= int(0.06 * sr) else y
        t = None
        if (
            k != "hiss"
        ):  # a voiced sound: its pitch, or the vowel's that follows it (a short span gives too few periods to find one)
            t = period(core, sr) or period(np.asarray(x[b : b + int(0.08 * sr)]), sr)
        z = hold(core, sr, hold_s, seed, t)
        z = z * (np.sqrt(np.mean(y**2)) / max(float(np.sqrt(np.mean(z**2))), 1e-9))  # the level it was said at
        info["held_s"] = hold_s
        return fade(z, sr), info
    if k in ("stop", "affricate"):
        y = y[: int((0.15 if k == "stop" else 0.25) * sr)]
        return fade(y, sr, fin=0.002, fout=0.01), info
    return fade(y, sr, fin=0.005, fout=0.02), info


def measure(x, sr):
    """{sound_s, high, low, voiced, vowel_leak} of a sound clip (see the module doc); the sound is the frames within 35 dB of its loudest."""
    x = np.asarray(x, np.float64)
    if not len(x) or not np.abs(x).max():
        return dict(sound_s=0.0, high=0.0, low=0.0, voiced=0.0, vowel_leak=0.0)
    _, f, fr = bands(x, sr)
    h, w = int(sr * HOP_S), int(sr * WIN_S)
    tot = f.sum(1) + 1e-12
    e = 10 * np.log10(tot)
    on = e > e.max() - 35
    hi, lo = f[:, fr > 2500].sum(1), f[:, fr < 1000].sum(1)
    mid = f[:, (fr >= 500) & (fr <= 3000)].sum(1) / tot
    voiced = np.array([bool(period(x[i * h : i * h + 2 * w], sr)) for i in range(len(f))]) & on
    vowel = voiced & (mid > 0.4)
    return dict(
        sound_s=round(float(on.sum()) * HOP_S, 3),
        high=round(float(hi[on].sum() / tot[on].sum()), 3),
        low=round(float(lo[on].sum() / tot[on].sum()), 3),
        voiced=round(float(voiced.sum() / max(on.sum(), 1)), 3),
        vowel_leak=round(float(tot[vowel].sum() / tot[on].sum()), 3),
    )


def pure(sound, m):
    """(whether a clip's measures m look like `sound` said on its own, why not). Rough by design: a check that catches the gross failures
    (a vowel after the sound, a hiss with no hiss, a nasal with no voice, nothing at all), not a judge of quality, which is the listening's
    job. A vowel passes when it is voiced."""
    k = kind(sound)
    if m["sound_s"] < 0.01:
        return False, "no sound"
    if k == "vowel":
        return (m["voiced"] >= 0.5, "not voiced")
    if m["vowel_leak"] > 0.1:
        return False, "a vowel in it"
    if k == "hiss":
        return (m["high"] > 0.5 and m["voiced"] < 0.3, "no hiss")
    if k in ("nasal", "liquid"):
        return (m["voiced"] > 0.5 and m["high"] < 0.2, "not voiced")
    if (
        k == "buzz"
    ):  # Kokoro's z, v and th (then) can come out unvoiced at the start of a syllable: a voice or a hiss will do
        return (m["voiced"] > 0.3 or m["high"] > 0.5, "neither voice nor hiss")
    if k in ("stop", "affricate", "brief", "other"):
        return (m["sound_s"] <= 0.3, "too long for a stop")
    return True, ""
