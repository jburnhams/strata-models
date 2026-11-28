"""Metrics: how one output is scored against its item, and how the scores of a run add up. Each metric says which way is better, so the analysis
can say "better" or "worse" without a person remembering that WER goes down.

Three shapes:
  per-item mean    a number per item (latency_s), averaged; compared with paired_diff.
  per-item binary  right or wrong per item (exact_match, parsed), a share; Wilson interval, compared with McNemar and paired_diff.
  ratio            (errors, size) per item, added up over the run (wer: all word errors over all reference words); compared with paired_ratio_diff.
  dataset          computed from all items together (verification AUC); interval by resampling items, comparison by paired bootstrap.
`score(item, output)` returns what the shape needs, or None when the metric does not apply to that item (no reference, the job failed: a failed
job counts as wrong for binary metrics and as all errors for ratios, never silently skipped)."""

import collections
import dataclasses
import functools
import itertools
import json
import math
import re
import unicodedata

import numpy as np

from stratamodels.evals import stats as S
from stratamodels.evals import vector as _vec


@dataclasses.dataclass(frozen=True)
class Metric:
    name: str
    shape: str  # mean | binary | ratio | dataset
    better: str  # lower | higher
    description: str
    unit: str = ""
    score: object = None  # (item, output, error) -> value (mean/binary), (num, den) (ratio), or what `dataset` reads
    dataset: object = None  # (items, outputs) -> value, for shape 'dataset'
    prepare: object = (
        None  # optional, shape 'dataset': (items, outputs) -> state read once, so a bootstrap sample is only `on`
    )
    on: object = (
        None  # (state, indices) -> the value on that sample of the items (with repeats), the same as `dataset` on it
    )

    def sampler(self, items, outputs):
        """A function of item indices (a bootstrap sample) -> the metric's value on those items."""
        if self.prepare is not None:
            st = self.prepare(items, outputs)
            return lambda idx: self.on(st, idx)
        return lambda idx: self.dataset([items[i] for i in idx], [outputs[i] for i in idx])

    def to_json(self):
        return dict(name=self.name, shape=self.shape, better=self.better, description=self.description, unit=self.unit)


# ---- text ---------------------------------------------------------------------------------------------------------------------------------
def normalise(text):
    """Lower case, accents kept, punctuation out (apostrophes inside words kept), whitespace collapsed: the same for every transcription eval, so
    numbers from different runs compare. Number words are not unified ("ten" vs "10" counts as an error): written in every report."""
    t = unicodedata.normalize("NFKC", text or "").lower().replace("’", "'")
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"(?<!\w)'|'(?!\w)", " ", t)
    return t.split()


def edits(ref, hyp):
    """Levenshtein distance between two token lists (substitutions, deletions, insertions all cost 1). One numpy pass per row of ref: the
    insertion chain cur[j] = min(t[j], cur[j-1] + 1) is a running minimum of t[j] - j, so a row has no Python loop (~20x faster on a long
    description than the plain double loop, the same numbers)."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    import numpy as np

    ids = {}
    h = np.array([ids.setdefault(t, len(ids)) for t in hyp])
    r = [ids.get(t, -1) for t in ref]
    js = np.arange(len(hyp) + 1)
    prev = js.copy()
    for i, x in enumerate(r, 1):
        t = np.empty_like(prev)
        t[0] = i
        t[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (h != x))
        prev = np.minimum.accumulate(t - js) + js
    return int(prev[-1])


def transcript(output):
    """The text of a transcription output (segments joined), or of a reading with no segments (an `ocr` page, a vision-language model's
    answer): its `text`."""
    if not output:
        return ""
    if not isinstance(output, dict):
        return str(output)
    if "segments" not in output:
        return str(output.get("text") or "")
    return " ".join(s.get("text", "") for s in output["segments"])


def _ref_text(item):
    return (item.get("reference") or {}).get("text")


def _wer(item, output, error):
    ref = _ref_text(item)
    if ref is None:
        return None
    r = normalise(ref)
    return (len(r) if error else edits(r, normalise(transcript(output))), len(r))


def _cer(item, output, error):
    ref = _ref_text(item)
    if ref is None:
        return None
    r = list(" ".join(normalise(ref)))
    return (len(r) if error else edits(r, list(" ".join(normalise(transcript(output))))), len(r))


def exact_chars(text):
    """The characters of a transcription with case and punctuation kept (what `normalise` throws away: small capitals read as capitals, a full
    stop read as a comma): only the typography that no reader can be held to is unified (curly quotes and apostrophes, the dashes, ligatures by
    NFKC) and runs of whitespace, line breaks included, are one space."""
    t = unicodedata.normalize("NFKC", text or "")
    t = t.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u2013": "-",
                "\u2014": "-",
                "\u2012": "-",
                "\u00ad": None,
            }
        )
    )
    t = re.sub(r"\s+([.,;:!?)\]])", r"\1", " ".join(t.split()))  # "nose !" is how 1828 set it, not a reading error
    return list(t)


def _cer_exact(item, output, error):
    ref = _ref_text(item)
    if ref is None:
        return None
    r = exact_chars(ref)
    return (len(r) if error else edits(r, exact_chars(transcript(output))), len(r))


LINE_FOUND = 0.5  # a reference line is found when this share of its words is in the reading


def found_words(ref, hyp):
    """Which words of ref (a token list) a reading hyp has: matched in order first (difflib's longest matching blocks), then the rest by bag,
    each reading word used once, so a line read out of order (a column, a side note) still counts and a common word is not credited twice."""
    import difflib

    hit = [False] * len(ref)
    used = [False] * len(hyp)
    for a, b, n in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_matching_blocks():
        for k in range(n):
            hit[a + k] = used[b + k] = True
    left = collections.Counter(t for t, u in zip(hyp, used) if not u)
    for i, t in enumerate(ref):
        if not hit[i] and left[t] > 0:
            left[t] -= 1
            hit[i] = True
    return hit


def _line_recall(item, output, error):
    """The reference's printed lines (one a line of reference.text) the reading found: at least LINE_FOUND of a line's words, after
    normalise(). cer counts a missed line as some characters among thousands; this counts it as a line, which is what a rebuilt page lacks
    (the advert's last line on ward1874-leaf034)."""
    ref = _ref_text(item)
    if ref is None:
        return None
    lines = [w for w in (normalise(ln) for ln in ref.splitlines()) if w]
    if error or not lines:
        return (0, len(lines)) if lines else None
    hit = found_words([t for ln in lines for t in ln], normalise(transcript(output)))
    found, k = 0, 0
    for ln in lines:
        found += sum(hit[k : k + len(ln)]) >= LINE_FOUND * len(ln)
        k += len(ln)
    return (found, len(lines))


def _wer_bag(item, output, error):
    """Word errors with the order ignored: the reference words a reading lacks or adds beyond it (whichever is more), over the reference words.
    Beside `wer` it separates misread words from lines read in the wrong order (a title page read column by column)."""
    ref = _ref_text(item)
    if ref is None:
        return None
    r = collections.Counter(normalise(ref))
    n = sum(r.values())
    if error:
        return (n, n)
    h = collections.Counter(normalise(transcript(output)))
    return (max(sum((r - h).values()), sum((h - r).values())), n)


# OCR junk inside a word: a symbol no word holds, a digit among letters, a capital after a small letter, a run of five consonants, a letter
# doubled three times. Old spellings and names pass all of these; "tbe", "wbich" (b for h) need a dictionary or the proofreading model.
_JUNK = re.compile(
    r"[|~^_<>{}\\»«¢§¥©®@#*=+\[\]]|\d\D*[^\W\d_]|[^\W\d_]\D*\d|[a-z][A-Z]|[bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ]{5}|(\w)\1\1"
)
_LONE = re.compile(r"^[^\W\d_]$")
_ORDINAL = re.compile(
    r"^\d+(st|nd|rd|th|d)$", re.I
)  # 3rd, 2d (old style), 21st: digits and letters that belong together


def garbled_words(text):
    """(junk words, words) in a reading: tokens on whitespace, outer punctuation stripped; junk is a word `_JUNK` matches, or a lone letter that
    is not a, I, O or &."""
    words = [w.strip(".,;:!?\"'()-\u2018\u2019\u201c\u201d") for w in (text or "").split()]
    words = [w for w in words if w]
    bad = sum(
        1
        for w in words
        if (_JUNK.search(w) and not _ORDINAL.match(w)) or (_LONE.match(w) and w not in ("a", "A", "I", "O"))
    )
    return bad, len(words)


def _garbled(item, output, error):
    if error or not output:
        return None
    bad, n = garbled_words(transcript(output))
    return (bad, n) if n else None


UNCLEAR_CONF = 60.0  # Tesseract's word confidence (0-100) below which a word counts as unclear


def _unclear(item, output, error):
    """(words Tesseract was unsure of, words) on an `ocr` reading with word confidences; None for a reading without them (a vision-language
    model's text)."""
    if error or not isinstance(output, dict) or not output.get("lines"):
        return None
    confs = [w.get("conf") for ln in output["lines"] for w in (ln.get("words") or []) if w.get("conf") is not None]
    return (sum(1 for c in confs if c < UNCLEAR_CONF), len(confs)) if confs else None


# ---- speech (text to speech) -----------------------------------------------------------------------------------------------------------------
def _tts_wer(item, output, error):
    """Intelligibility: the words the [score] pass heard in the audio (a transcription) against the text that was to be spoken. A failed job, or a
    reading that failed, is all errors; an output not read yet does not count."""
    ref = _ref_text(item) or (item.get("input") or {}).get("text")
    if ref is None:
        return None
    r = normalise(ref)
    if error:
        return (len(r), len(r))
    sc = (output or {}).get("scored")
    if sc is None:
        return None
    return (len(r), len(r)) if sc.get("error") else (edits(r, normalise(transcript(sc.get("result")))), len(r))


def _rtf(item, output, error):
    if error or not isinstance(output, dict) or not output.get("duration_s"):
        return None
    return float(output.get("seconds") or item.get("_run_s") or 0) / float(output["duration_s"])


def pace_of(item, output, error):
    """{rate_wpm, max_pause_s, long_pauses, speech_s} of a speech output: from the output when the worker measured it, else measured here from
    its file (outputs made before the worker measured pace) with the same code (stratamodels.workers.pacing)."""
    if error or not isinstance(output, dict):
        return None
    if output.get("rate_wpm") is not None:
        return output
    path, text = output.get("path"), (item.get("input") or {}).get("text")
    if not isinstance(path, str) or not text:
        return None
    from stratamodels.workers import pacing

    try:
        x, sr = pacing.read_wav(path)
    except (OSError, EOFError, ValueError):
        return None
    return pacing.measure(text, x, sr)


def _pace_spread(items, outputs):
    """Coefficient of variation of the speaking rate across the items (spread / mean): 0 is every line at the same pace."""
    r = [m["rate_wpm"] for it, (o, e) in zip(items, outputs) if (m := pace_of(it, o, e))]
    return float(np.std(r) / np.mean(r)) if len(r) > 1 and np.mean(r) > 0 else float("nan")


def _long_pauses(item, output, error):
    m = pace_of(item, output, error)
    return None if m is None else float(m["long_pauses"])


def _max_pause(item, output, error):
    m = pace_of(item, output, error)
    return None if m is None else float(m["max_pause_s"])


# ---- phonics sounds (speak-consonants: workers/phonics.py, shared with the TTS worker that cuts them) -------------------------------------
def sound_of(item, output, error):
    """(the sound asked for, the measures of the clip made) for a phonics sound output: measured here from its file, with the same code the
    worker used, so a recorded person's clip is measured exactly as a model's."""
    snd = (item.get("input") or {}).get("sound")
    path = output.get("path") if isinstance(output, dict) and not error else None
    if not snd or not isinstance(path, str):
        return None
    from stratamodels.workers import pacing, phonics

    try:
        x, sr = pacing.read_wav(path)
    except (OSError, EOFError, ValueError):
        return None
    return snd, phonics.measure(x, sr)


def _sound_pure(item, output, error):
    if error:
        return False if (item.get("input") or {}).get("sound") else None
    got = sound_of(item, output, error)
    if got is None:
        return None
    from stratamodels.workers import phonics

    return phonics.pure(*got)[0]


def _vowel_leak(item, output, error):
    got = sound_of(item, output, error)
    return None if got is None else got[1]["vowel_leak"]


def _fit_err(item, output, error):
    if error or not isinstance(output, dict) or not output.get("target_s"):
        return None
    return abs(float(output["speech_s"]) / float(output["target_s"]) - 1)


# ---- faces --------------------------------------------------------------------------------------------------------------------------------
def iou(a, b):
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def faces_of(output):
    """Every face of a faces output, whether it came as one image, or views of one image."""
    if not output:
        return []
    if "views" in output:
        return [f for v in output["views"] for f in v["faces"]]
    return output.get("faces", [])


def best_face(item, output, min_iou=0.3):
    """The output face that is the item's reference face (reference.box, [x1, y1, x2, y2]), or None when none overlaps it enough."""
    box = (item.get("reference") or {}).get("box")
    if box is None:
        return None
    scored = [(iou(f["bbox"], box), f) for f in faces_of(output) if "bbox" in f]
    best = max(scored, key=lambda t: t[0], default=(0, None))
    return best[1] if best[0] >= min_iou else None


def _face_found(item, output, error):
    if (item.get("reference") or {}).get("box") is None:
        return None
    return False if error else best_face(item, output, 0.5) is not None


def _faces_recall(item, output, error):
    boxes = (item.get("reference") or {}).get("boxes")
    if boxes is None:
        return None
    if error:
        return (len(boxes), len(boxes))
    preds = [f["bbox"] for f in faces_of(output) if "bbox" in f]
    used = set()
    hit = 0
    for g in boxes:
        j = max(((iou(g, p), k) for k, p in enumerate(preds) if k not in used), default=(0, None))
        if j[1] is not None and j[0] >= 0.5:
            used.add(j[1])
            hit += 1
    return (len(boxes) - hit, len(boxes))  # misses over gold faces: lower is better, like an error rate


CUT_TOLERANCE_S = 0.5  # the spec's tightest tolerance (strata-reels spec S5: 0.5, 2 and 5 s): a camcorder cut on tape is one frame, so 0.5 s is generous


def match_cuts(ref, got, tol=CUT_TOLERANCE_S):
    """How many reference cuts (seconds) are found by `got`, one to one, each within `tol` seconds: the most pairs there can be. Nearest
    pairs first is not that: with cuts [1.0, 1.65] and answers [0.6, 1.25], taking (1.0, 1.25) first leaves both others alone. On a line,
    walking both in order and pairing each reference cut with the earliest answer still in reach is the maximum matching."""
    ref, got = sorted(ref), sorted(got)
    i = j = hit = 0
    while i < len(ref) and j < len(got):
        if got[j] < ref[i] - tol:
            j += 1  # too early for this cut, so for every later one
        elif got[j] > ref[i] + tol:
            i += 1  # nothing left in reach of this cut
        else:
            hit, i, j = hit + 1, i + 1, j + 1
    return hit


def _cuts_of(output):
    return [float(c["time"]) for c in (output or {}).get("cuts") or [] if isinstance(c, dict) and "time" in c]


def _ref_cuts(item):
    ref = (item.get("reference") or {}).get("cuts")
    return None if ref is None else [float(t) for t in ref], float(
        (item.get("reference") or {}).get("tolerance_s") or CUT_TOLERANCE_S
    )


def _cut_miss(item, output, error):
    ref, tol = _ref_cuts(item)
    if ref is None:
        return None
    return (len(ref), len(ref)) if error else (len(ref) - match_cuts(ref, _cuts_of(output), tol), len(ref))


def _cut_false(item, output, error):
    ref, tol = _ref_cuts(item)
    if ref is None or error:
        return None  # a failed job found nothing, so it made no false cut: cut_miss and failed count it
    got = _cuts_of(output)
    return (len(got) - match_cuts(ref, got, tol), len(got))


def _cut_f1(item, output, error):
    ref, tol = _ref_cuts(item)
    if ref is None:
        return None
    if error:
        return 0.0  # a failed job did not say the stretch is one shot, even when it is
    got = _cuts_of(output)
    if not ref and not got:
        return 1.0  # one shot, and nothing claimed: right
    hit = match_cuts(ref, got, tol)
    return 2 * hit / (len(ref) + len(got))


# ---- face tracks (role face-tracks) --------------------------------------------------------------------------------------------------------
# An item's reference is a clip labelled at regular moments (scripts/evals/face_tracks_labels.py): `times` (every moment labelled, so a face
# answered then that no label matches is a false one) and `people`: {name: [[time, x0, y0, x1, y1], ...]}, and `ignore` boxes (faces nobody
# named: an answered face there that no label takes is not counted at all, MOT's "don't care" regions). A track is looked up at a labelled
# moment by interpolating its boxes, so candidates that detect at different rates are compared at the same moments.
TRACK_IOU = 0.5


def track_box_at(boxes, t, fps):
    """A track's box at time t (its boxes: [[time, x0, y0, x1, y1, ...]] in time order): between two of its boxes, the line between them;
    within half a frame of its first or last, that box; else None (the track is not on screen then)."""
    if not boxes:
        return None
    half = 0.5 / fps if fps else 0.1
    if t < boxes[0][0] - half - 1e-9 or t > boxes[-1][0] + half + 1e-9:
        return None
    for a, b in itertools.pairwise(boxes):
        if a[0] <= t <= b[0]:
            w = 0.0 if b[0] == a[0] else (t - a[0]) / (b[0] - a[0])
            return [a[k] + w * (b[k] - a[k]) for k in range(1, 5)]
    return list((boxes[0] if t < boxes[0][0] else boxes[-1])[1:5])


def track_matches(item, output):
    """What a face-tracks answer matched against the item's labels: (pairs, labelled, answered), pairs being (person, track id, time) of each
    labelled face matched one to one (IoU >= TRACK_IOU) with a track's box at that moment; labelled and answered count faces at the moments."""
    from scipy.optimize import linear_sum_assignment

    ref = item.get("reference") or {}
    fps = float((output or {}).get("fps") or 0)
    tracks = (output or {}).get("tracks") or []
    by_time = {}
    for name, obs in (ref.get("people") or {}).items():
        for o in obs:
            by_time.setdefault(round(float(o[0]), 3), []).append((name, o[1:5]))
    ignore = {}
    for o in ref.get("ignore") or []:
        ignore.setdefault(round(float(o[0]), 3), []).append(o[1:5])
    pairs, labelled, answered = [], 0, 0
    for t in [round(float(x), 3) for x in ref.get("times") or []]:
        gold = by_time.get(t, [])
        hyp = [(tr["id"], b) for tr in tracks if (b := track_box_at(tr.get("boxes") or [], t, fps)) is not None]
        taken = set()
        if gold and hyp:
            m = np.array([[iou(g[1], h[1]) for h in hyp] for g in gold])
            rows, cols = linear_sum_assignment(-m)
            for r, c in zip(rows, cols, strict=True):
                if m[r, c] >= TRACK_IOU:
                    pairs.append((gold[r][0], hyp[c][0], t))
                    taken.add(int(c))
        dont_care = sum(
            1 for c, h in enumerate(hyp) if c not in taken and any(iou(h[1], z) >= TRACK_IOU for z in ignore.get(t, []))
        )
        labelled, answered = labelled + len(gold), answered + len(hyp) - dont_care
    return pairs, labelled, answered


def _shot_of(item):
    """A labelled moment's shot number, from the item's `cuts` (input): 0 before the first cut, 1 after it, ..."""
    cuts = sorted(float(c) for c in (item.get("input") or {}).get("cuts") or [])
    return lambda t: sum(1 for c in cuts if t >= c - 1e-6)


def _identity_tp(pairs):
    """Labelled faces in their person's track, people and tracks paired one to one so the most fall in their pair (Ristani et al. 2016)."""
    from scipy.optimize import linear_sum_assignment

    if not pairs:
        return 0
    people = sorted({p for p, _, _ in pairs})
    tracks = sorted({t for _, t, _ in pairs})
    c = np.zeros((len(people), len(tracks)))
    for p, t, _ in pairs:
        c[people.index(p), tracks.index(t)] += 1
    rows, cols = linear_sum_assignment(-c)
    return int(c[rows, cols].sum())


def _id_f1_scored(item, output, error, per_shot):
    if (item.get("reference") or {}).get("people") is None:
        return None
    if error:
        return 0.0
    pairs, labelled, answered = track_matches(item, output)
    if not labelled and not answered:
        return 1.0
    if per_shot:
        shot = _shot_of(item)
        by_shot = {}
        for pr in pairs:
            by_shot.setdefault(shot(pr[2]), []).append(pr)
        idtp = sum(_identity_tp(ps) for ps in by_shot.values())
    else:
        idtp = _identity_tp(pairs)
    return 2 * idtp / (labelled + answered)


def _id_f1(item, output, error):
    """The identity F1 of multi-object tracking (Ristani et al. 2016) within each shot: in every shot (the item's `cuts`), people and tracks
    are paired one to one so the most labelled faces fall in their person's track; a face in another track, or in none, is missed, and an
    answered face not in its pair is false. Per shot because face-tracks ends every track at a cut by design (one track per person per
    shot), so a person seen in five shots is five tracks, all of which can count. A track that runs across a cut is paired in each shot
    apart; if it changes person there, track_mixed says so."""
    return _id_f1_scored(item, output, error, per_shot=True)


def _id_f1_clip(item, output, error):
    """The identity F1 over the whole clip (the first face-tracks-home run's id_f1, 2026-11-27): one track per person for every shot, so a
    person seen in several shots can count only one of their tracks. Kept to compare with the per-shot id_f1, and for a tracker that is
    meant to follow a person across cuts."""
    return _id_f1_scored(item, output, error, per_shot=False)


def _track_mixed(item, output, error):
    """Tracks that hold two people (at least two labelled faces of each), over the tracks that match anyone."""
    if (item.get("reference") or {}).get("people") is None or error:
        return None
    pairs, _, _ = track_matches(item, output)
    counts = {}
    for p, t, _ in pairs:
        counts.setdefault(t, {}).setdefault(p, 0)
        counts[t][p] += 1
    return (sum(1 for c in counts.values() if sum(1 for n in c.values() if n >= 2) >= 2), len(counts))


def _track_split(item, output, error):
    """Extra tracks per person per shot: a person seen in one shot by three tracks counts two, over the person-shots labelled. Shots come from
    the item's `cuts` (input)."""
    if (item.get("reference") or {}).get("people") is None or error:
        return None
    shot = _shot_of(item)
    pairs, _, _ = track_matches(item, output)
    seen = {}
    for p, t, when in pairs:
        seen.setdefault((p, shot(when)), set()).add(t)
    labelled = {(name, shot(float(o[0]))) for name, obs in item["reference"]["people"].items() for o in obs}
    return (sum(len(v) - 1 for v in seen.values()), len(labelled))


def _face_recall_tracks(item, output, error):
    """Labelled faces that some track holds at that moment, over the labelled faces: a miss rate, lower is better."""
    if (item.get("reference") or {}).get("people") is None:
        return None
    n = sum(len(v) for v in item["reference"]["people"].values())
    if error:
        return (n, n)
    pairs, labelled, _ = track_matches(item, output)
    return (labelled - len(pairs), labelled)


def _avg_ranks(x):
    """1-based ranks of x with ties sharing their average rank (vectorised: verification is recomputed for every bootstrap sample)."""
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + 1 + ends) / 2.0)[inv]


def verification_prepare(items, outputs):
    """What verification reads from each item, once: the reference face's embedding (None when not found), the person, and the similarity of
    every two items' embeddings. A bootstrap sample then only picks rows and columns."""
    emb, who = [], []
    for it, out in zip(items, outputs):
        p = (it.get("reference") or {}).get("person")
        f = best_face(it, out[0]) if p is not None and out[1] is None else None
        emb.append(np.asarray(f["embedding"], np.float32) if f and "embedding" in f else None)
        who.append(p)
    dim = next((len(e) for e in emb if e is not None), 512)
    E = (
        np.stack([e if e is not None else np.zeros(dim, np.float32) for e in emb])
        if emb
        else np.zeros((0, dim), np.float32)
    )
    ok = np.array([e is not None for e in emb], bool)
    labelled = np.array([w is not None for w in who], bool)
    _, codes = (
        np.unique(np.array([str(w) for w in who], dtype=object), return_inverse=True)
        if who
        else (None, np.zeros(0, int))
    )
    return dict(sim=E @ E.T, ok=ok, labelled=labelled, who=np.asarray(codes))


def verification_on(st, idx=None, far=0.01):
    """verification() on items idx (with repeats) of a prepared set, all of them when idx is None."""
    idx = np.arange(len(st["ok"])) if idx is None else np.asarray(idx)
    idx = idx[st["labelled"][idx]]
    n = len(idx)
    if n < 2:
        return dict(auc=float("nan"), tar=float("nan"), threshold=float("nan"), pairs=0)
    iu = np.triu_indices(n, 1)
    a, b = idx[iu[0]], idx[iu[1]]
    s = st["sim"][a, b].astype(np.float32)
    s[~(st["ok"][a] & st["ok"][b])] = -1.0
    same = st["who"][a] == st["who"][b]
    pos, neg = s[same], s[~same]
    if pos.size == 0 or neg.size == 0:
        return dict(auc=float("nan"), tar=float("nan"), threshold=float("nan"), pairs=int(s.size))
    ranks = _avg_ranks(np.concatenate([pos, neg]))  # tied scores share their average rank (a missed face ties at -1)
    auc = (ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
    thr = float(np.quantile(neg, 1 - far))
    tar = float((pos > thr).mean())
    return dict(auc=float(auc), tar=tar, threshold=thr, pairs=int(s.size))


def verification(items, outputs, far=0.01):
    """Face verification from the embeddings of the reference faces: every pair of items is same-person or not (reference.person). Returns
    dict(auc, tar, threshold, pairs): ROC AUC, and the share of same-person pairs accepted at the threshold that lets `far` of different-person
    pairs through. Items whose face was not found count as rejected (distance 2)."""
    return verification_on(verification_prepare(items, outputs), None, far)


# ---- picture descriptions (structured, against a reference description) -------------------------------------------------------------------
def description_of(output):
    """The parsed description of a vision-language output (its `json`), or a recorded one given whole."""
    if not isinstance(output, dict):
        return None
    d = output.get("json", output if "caption" in output else None)
    return d if isinstance(d, dict) else None


def _g(d, path):
    for k in path.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _bucket(n):
    if not isinstance(n, (int, float)) or isinstance(n, bool):
        return None
    return 0 if n <= 0 else 1 if n == 1 else 2 if n == 2 else "3-5" if n <= 5 else "6+"


def _main_light(d):
    src = [s for s in (_g(d, "lighting.sources") or []) if isinstance(s, dict)]
    return (
        max(src, key=lambda s: s.get("share") if isinstance(s.get("share"), (int, float)) else 0).get("type")
        if src
        else None
    )


# The fields an app filters or sorts on, each read the same way from both descriptions. Free text (caption, place, event) is judged blind instead.
def _either(d, nested, flat):
    """A field from a full description (schema-v1, nested) or from the flat photo-fields preset."""
    v = _g(d, nested)
    return d.get(flat) if v is None and isinstance(d, dict) else v


def _children(d):
    n = _either(d, "people.children", "children")
    return None if not isinstance(n, int) or isinstance(n, bool) else n > 0


def _has_text(d):
    t = _g(d, "text_visible")
    return len(t) > 0 if isinstance(t, list) else t if isinstance(t, bool) else None


DESCRIPTION_FIELDS = {
    "indoor_outdoor": lambda d: _either(d, "setting.indoor_outdoor", "indoor_outdoor"),
    "time_of_day": lambda d: _either(d, "setting.time_of_day", "time_of_day"),
    "weather": lambda d: _either(d, "setting.weather", "weather"),
    "people": lambda d: _bucket(_either(d, "people.count", "people_count")),
    "children": _children,
    "person_subject": lambda d: _either(d, "people.main_subject_is_person", "main_subject_is_person"),
    "main_light": lambda d: _main_light(d) or (d.get("main_light") if isinstance(d, dict) else None),
    "light_level": lambda d: _either(d, "lighting.level", "light_level"),
    "sharpness": lambda d: _either(d, "exposure.sharpness", "sharpness"),
    "skin_visible": lambda d: _either(d, "colour.skin_visible", "skin_visible"),
    "has_text": _has_text,
}


def _ref_desc(item):
    return description_of((item.get("reference") or {}).get("description"))


def field_agreement(ref, got):
    """{field: True/False} for the fields the reference fills (a missing answer is a disagreement)."""
    out = {}
    for f, read in DESCRIPTION_FIELDS.items():
        want = read(ref)
        if want is None:
            continue
        have = read(got) if got else None
        out[f] = have is not None and str(have).lower() == str(want).lower()
    return out


def _desc_fields(item, output, error):
    ref = _ref_desc(item)
    if ref is None:
        return None
    a = field_agreement(ref, None if error else description_of(output))
    return sum(a.values()) / len(a) if a else None


def _tagset(d):
    return {" ".join(str(t).lower().split()) for t in (d or {}).get("tags") or [] if str(t).strip()}


def _tag_f1(item, output, error):
    ref = _ref_desc(item)
    if ref is None:
        return None
    r = _tagset(ref)
    g = set() if error else _tagset(description_of(output))
    if not r:
        return None
    tp = len(r & g)
    return 0.0 if not tp else 2 * tp / (len(r) + len(g))


def _desc_valid(item, output, error):
    if error:
        return False
    d = description_of(output)
    return bool(
        d
        and isinstance(d.get("caption"), str)
        and d["caption"].strip()
        and isinstance(d.get("tags"), list)
        and d["tags"]
    )


# ---- recorded sound (denoise-audio: the [score] pass reads each output with the audio-quality role, workers/audio_quality.py) -------------
def reading(output, group, key):
    """One number from the [score] reading of an output (scored.result.<group>.<key>), or None: not read yet, the reading failed (a failed
    measurement says nothing about the sound; `failed` counts failed jobs), or the measure did not apply (no reference for an intrusive one)."""
    sc = (output or {}).get("scored") if isinstance(output, dict) else None
    if not isinstance(sc, dict) or sc.get("error"):
        return None
    v = ((sc.get("result") or {}).get(group) or {}).get(key)
    return float(v) if isinstance(v, (int, float)) and v == v else None


def _reader(group, key):
    return lambda item, output, error: None if error else reading(output, group, key)


def _removed_db(item, output, error):
    v = output.get("removed_db") if isinstance(output, dict) and not error else None
    return float(v) if isinstance(v, (int, float)) else None


AUDIO_METRICS = [
    (
        "dnsmos_ovrl",
        "dnsmos",
        "ovrl",
        "DNSMOS P.835 overall quality (1-5): Microsoft's model of what listeners would score, no reference",
        "",
    ),
    (
        "dnsmos_sig",
        "dnsmos",
        "sig",
        "DNSMOS P.835 speech quality (1-5): how undistorted the speech itself sounds (denoisers that eat words score low)",
        "",
    ),
    (
        "dnsmos_bak",
        "dnsmos",
        "bak",
        "DNSMOS P.835 background (1-5): how little the noise intrudes (5 = none heard; keeping the place's sound costs here)",
        "",
    ),
    (
        "squim_stoi",
        "squim",
        "stoi",
        "SQUIM's estimate of STOI (0-1, intelligibility) without a reference (torchaudio SQUIM objective)",
        "",
    ),
    ("squim_pesq", "squim", "pesq", "SQUIM's estimate of PESQ (1-4.5) without a reference", ""),
    ("squim_si_sdr", "squim", "si_sdr", "SQUIM's estimate of SI-SDR without a reference", "dB"),
    (
        "si_sdr",
        "intrusive",
        "si_sdr",
        "Scale-invariant signal-to-distortion ratio against the clean speech the item was mixed from",
        "dB",
    ),
    ("pesq", "intrusive", "pesq", "PESQ (wide band, 1-4.5) against the clean speech the item was mixed from", ""),
    (
        "estoi",
        "intrusive",
        "estoi",
        "Extended STOI (0-1, intelligibility) against the clean speech the item was mixed from",
        "",
    ),
]


# ---- rebuilt pages (page-build and the readers that fill its page description: reference.page is the cleaned page, as a description) ------
def _page_ref(item):
    return (item.get("reference") or {}).get("page")


def _page_out(output):
    return output.get("page") if isinstance(output, dict) and isinstance(output.get("page"), dict) else None


def _xyxy(box):
    x, y, w, h = (float(v) for v in box)
    return [x, y, x + w, y + h]


def _area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def _pieces(whole, parts, min_share=0.1):
    """Indices of the `parts` that lie mostly inside `whole` (over half of each inside) and are not slivers of it (at least `min_share` of
    its area): two or more such parts are `whole` split, or, read the other way, merged into it."""
    w = _xyxy(whole["box"])
    out = []
    for k, p in enumerate(parts):
        b = _xyxy(p["box"])
        a = _area(b)
        inter = max(0.0, min(w[2], b[2]) - max(w[0], b[0])) * max(0.0, min(w[3], b[3]) - max(w[1], b[1]))
        if a > 0 and inter / a > 0.5 and a >= min_share * _area(w):
            out.append(k)
    return out


def match_boxes(ref, got, min_iou=0.5):
    """Pairs (i, j) of reference and output parts (each a dict with an [x, y, w, h] `box`) that are the same part: greedily by best IoU first,
    each used once, at least `min_iou`. A reference part the output split (two or more output parts mostly inside it) and an output part that
    merged reference parts match nothing, whatever their IoU: an even split has IoU 0.5 with each half, an uneven one more, and either would
    otherwise be scored as a found block when it is the segmentation error these metrics measure."""
    split = {i for i, r in enumerate(ref) if len(_pieces(r, got)) > 1}
    merged = {j for j, g in enumerate(got) if len(_pieces(g, ref)) > 1}
    scored = sorted(
        (
            (iou(_xyxy(r["box"]), _xyxy(g["box"])), i, j)
            for i, r in enumerate(ref)
            if i not in split
            for j, g in enumerate(got)
            if j not in merged
        ),
        reverse=True,
    )
    used_r, used_g, out = set(), set(), []
    for v, i, j in scored:
        if v < min_iou:
            break
        if i not in used_r and j not in used_g:
            used_r.add(i)
            used_g.add(j)
            out.append((i, j))
    return sorted(out)


def _parts_f1(key):
    def score(item, output, error):
        ref = _page_ref(item)
        if ref is None:
            return None
        r = ref.get(key) or []
        g = [] if error else (_page_out(output) or {}).get(key) or []
        if key == "figures" and isinstance(output, dict):
            # rules and unread print page-build cut from the scan itself (its `rules` and `unread`) are not a reader's finds: scoring them
            # took the full rebuild's figure F1 from 29 % to 13 % on the same reader's figures (page-rebuild, 2026-11-23)
            cut = set(output.get("scan_figures") or [])
            g = [f for i, f in enumerate(g) if i not in cut]
        if not r and not g:
            return None  # nothing to find and nothing found: says nothing about the reader
        return 2 * len(match_boxes(r, g)) / (len(r) + len(g))

    return score


def _matched_blocks(item, output, error):
    ref, got = _page_ref(item), None if error else _page_out(output)
    if ref is None or got is None:
        return None
    r, g = ref.get("blocks") or [], got.get("blocks") or []
    return [(r[i], g[j], i, j) for i, j in match_boxes(r, g)]


def _role_acc(item, output, error):
    m = _matched_blocks(item, output, error)
    if not m:
        return None
    return sum((a.get("role") or "body") == (b.get("role") or "body") for a, b, _, _ in m) / len(m)


def _reading_rank(blocks):
    """Each block's place in reading order, as strata_python.pages reads it: numbered blocks by number, then the rest as listed."""
    idx = sorted(range(len(blocks)), key=lambda k: (blocks[k].get("order") is None, blocks[k].get("order") or 0, k))
    return {k: n for n, k in enumerate(idx)}


def _order_tau(item, output, error):
    m = _matched_blocks(item, output, error)
    if not m or len(m) < 2:
        return None
    ref, got = _page_ref(item)["blocks"], _page_out(output)["blocks"]
    rr, gr = _reading_rank(ref), _reading_rank(got)
    pairs = [(rr[i], gr[j]) for _, _, i, j in m]
    s = sum(np.sign(a[0] - b[0]) * np.sign(a[1] - b[1]) for k, a in enumerate(pairs) for b in pairs[k + 1 :])
    return float(s) / (len(pairs) * (len(pairs) - 1) / 2)  # Kendall's tau (no ties: ranks are distinct)


@functools.lru_cache(maxsize=32)
def _page_binarised(image):
    """(the scan's ink, dpi): the page binarised as the font eval does (not deskewed: a rebuild draws in the scan's own pixels). Seconds a page,
    so kept for every page metric of every candidate (the masks are cheap and made per call, so only the ink is held: a bool a pixel)."""
    from PIL import Image

    from stratamodels.evals import ink as INK

    im = Image.open(image)
    found = im.info.get("dpi")
    dpi = round(float(found[0]), 1) if found and found[0] > 1 else 300.0
    full, _ = INK.binarise(im, dpi, deskew=False)
    return full, dpi


def _page_scan_ink(image, boxes):
    """(the scan's ink, the reference text blocks as a mask, dpi) for page_ink_f1: the union of the cleaned page's text blocks (4 px around),
    so borders, rules, pictures, library stamps and show-through from the next leaf count for no candidate."""
    from stratamodels.evals import ink as INK

    full, dpi = _page_binarised(image)
    mask = np.zeros_like(full)
    for x, y, w, h in boxes:
        mask |= INK.box_mask(full.shape, [x - 4, y - 4, w + 8, h + 8])
    return full, mask, dpi


def _page_ink_f1(item, output, error):
    """F1 of the rebuilt page's ink (page-build's raster at the scan's dpi) against the scan's within the reference text blocks
    (evals/ink.py): how closely the PDF's text lands on the print, whatever the reader got right about the layout."""
    ref = _page_ref(item)
    image = (item.get("input") or {}).get("image")
    if ref is None or not image or not ref.get("blocks"):
        return None
    if error or not isinstance(output, dict) or not output.get("png"):
        return 0.0 if error else None
    from PIL import Image

    from stratamodels.evals import ink as INK

    boxes = tuple(tuple(float(v) for v in b["box"]) for b in ref["blocks"])
    try:
        scan, mask, dpi = _page_scan_ink(image, boxes)
        got = np.asarray(Image.open(output["png"]).convert("L")) < 128
    except (OSError, ValueError, TypeError):
        return None
    if got.shape != scan.shape:
        return 0.0  # a page drawn at another size lands on none of the print
    return INK.ink_f1(scan, got, tol_px=max(1, round(dpi / 250)), mask=mask)["f1"]


def _print_read(item, output, error):
    """The share of the scan's ink in the reference text blocks that lies in the reading's line boxes (3 px around): print the reader found and
    read, whatever it read it as. Needs boxes, so a reading with no lines (a vision model's text) has none; a reading of an upscaled picture is
    scaled back by its width."""
    ref = _page_ref(item)
    image = (item.get("input") or {}).get("image")
    if ref is None or not image or not ref.get("blocks"):
        return None
    if error:
        return 0.0
    if not isinstance(output, dict) or not isinstance(output.get("lines"), list):
        return None
    from stratamodels.evals import ink as INK

    boxes = tuple(tuple(float(v) for v in b["box"]) for b in ref["blocks"])
    try:
        scan, mask, _ = _page_scan_ink(image, boxes)
    except (OSError, ValueError, TypeError):
        return None
    ink = scan & mask
    if not ink.any():
        return None
    k = scan.shape[1] / float(output.get("width") or scan.shape[1])
    read = np.zeros_like(scan)
    for ln in output["lines"]:
        x, y, w, h = (float(v) * k for v in ln["box"])
        read |= INK.box_mask(scan.shape, [x - 3, y - 3, w + 6, h + 6])
    return float((ink & read).sum() / ink.sum())


PRINT_REACH = 0.06  # page_ink_all reads this fraction of the page's size around the labelled print (its border), as page-build's rules do


def _page_print_ink(image, boxes):
    """(the scan's ink, the print area as a mask, dpi) for page_ink_all: the box round every labelled block and figure, grown by PRINT_REACH
    of the page's size, so the border round the page counts and the book's edge and cover beside the leaf do not."""
    from stratamodels.evals import ink as INK

    full, dpi = _page_binarised(image)
    h, w = full.shape
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    rx, ry = PRINT_REACH * w, PRINT_REACH * h
    mask = INK.box_mask(full.shape, [x0 - rx, y0 - ry, x1 - x0 + 2 * rx, y1 - y0 + 2 * ry])
    return full, mask, dpi


def _page_ink_all(item, output, error):
    """F1 of the rebuilt page's ink against the scan's over the whole printed area (text, pictures, ornaments, rules and the border): how much
    the rebuild looks like the page, where page_ink_f1 reads the text alone. A rebuild pasting the whole scan in as a picture would score 1:
    pictures cut from the scan are for figures and rules, which the judge sees."""
    ref = _page_ref(item)
    image = (item.get("input") or {}).get("image")
    if ref is None or not image or not ref.get("blocks"):
        return None
    if error or not isinstance(output, dict) or not output.get("png"):
        return 0.0 if error else None
    from PIL import Image

    from stratamodels.evals import ink as INK

    boxes = tuple(tuple(float(v) for v in b["box"]) for b in [*ref["blocks"], *(ref.get("figures") or [])])
    try:
        scan, mask, dpi = _page_print_ink(image, boxes)
        got = np.asarray(Image.open(output["png"]).convert("L")) < 128
    except (OSError, ValueError, TypeError):
        return None
    if got.shape != scan.shape:
        return 0.0
    return INK.ink_f1(scan, got, tol_px=max(1, round(dpi / 250)), mask=mask)["f1"]


# ---- fonts (role font-match) ----------------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=256)
def _scan_ink(image, ocr):
    """(the item's ink, the area of its words) for ink_f1: the picture binarised as runtimes/fontfit.py does, and the union of Tesseract's
    word boxes (3 px around), so rules, ornaments and specks outside the words count for no candidate."""
    from PIL import Image

    from stratamodels.evals import ink as INK

    im = Image.open(image)
    found = im.info.get("dpi")
    dpi = float(found[0]) if found and found[0] > 1 else 300.0
    full, _ = INK.binarise(im, dpi, deskew=False)
    mask = np.zeros_like(full)
    try:
        with open(ocr, encoding="utf8") as f:
            lines = json.load(f).get("lines") or []
    except (OSError, ValueError, TypeError):
        lines = []
    for ln in lines:
        for w in ln.get("words") or []:
            x, y, bw, bh = w["box"]
            mask |= INK.box_mask(full.shape, [x - 3, y - 3, bw + 6, bh + 6])
    return full, (mask if mask.any() else None), dpi


def _ink_f1(item, output, error):
    """F1 of the rendering's ink against the scan's within the words (evals/ink.py), measured here from the two pictures, not taken from the
    runtime's own score: every candidate is held to the same reading of the scan."""
    if error or not isinstance(output, dict) or not output.get("path"):
        return 0.0 if error else None
    from PIL import Image

    from stratamodels.evals import ink as INK

    inp = item.get("input") or {}
    try:
        ref, mask, dpi = _scan_ink(inp.get("image"), inp.get("ocr"))
        got = np.asarray(Image.open(output["path"]).convert("L")) < 128
    except (OSError, ValueError, TypeError):
        return None
    if got.shape != ref.shape:
        return 0.0
    return INK.ink_f1(ref, got, tol_px=max(1, round(dpi / 250)), mask=mask)["f1"]


def _ref_classes(item):
    r = item.get("reference") or {}
    return list(r.get("classes") or ([r["class"]] if r.get("class") else []))


def _font_class_hit(item, output, error):
    want = _ref_classes(item)
    if not want:
        return None
    return False if error or not isinstance(output, dict) else output.get("class") in want


def _font_class_close(item, output, error):
    from stratamodels.runtimes.fontfit import closeness

    want = _ref_classes(item)
    if not want:
        return None
    if error or not isinstance(output, dict):
        return 0.0
    return max(closeness(output.get("class"), c) for c in want)


def _font_face(n):
    def score(item, output, error):
        want = (item.get("reference") or {}).get("face")
        if not want:
            return None
        if error or not isinstance(output, dict):
            return False
        ranked = [f.get("face") for f in output.get("faces") or []] or [output.get("face")]
        return want in ranked[:n] if n > 1 else output.get("face") == want

    return score


def _font_style(item, output, error):
    want = (item.get("reference") or {}).get("style")
    if not want:
        return None
    return False if error or not isinstance(output, dict) else output.get("style") == want


def _font_size_err(item, output, error):
    want = (item.get("reference") or {}).get("size_pt")
    if not want or error or not isinstance(output, dict) or not output.get("size_pt"):
        return None
    return abs(float(output["size_pt"]) - float(want)) / float(want)


# ---- generic ------------------------------------------------------------------------------------------------------------------------------
def _latency(item, output, error):
    """The job's run time; for outputs recorded elsewhere (runtime `recorded`: a replay takes no time) the time they took when made, if it
    was recorded (`seconds`), else none: a replay must not read as an instant model."""
    if error:
        return None
    if isinstance(output, dict) and (
        output.get("recorded") or output.get("cli")
    ):  # `cli`: recorded by scripts/evals/record_cloud.py
        s = output.get("seconds")
        return float(s) if isinstance(s, (int, float)) else None
    return item.get("_run_s")


def own_cost(o):
    """The dollars one answer says it cost: its `cost_usd`, or for a runtime that asked a cloud model on the way (font-match's proposer) that
    call's cost."""
    if isinstance(o.get("cost_usd"), (int, float)):
        return float(o["cost_usd"])
    p = o.get("proposer")
    if isinstance(p, dict) and isinstance(p.get("cost_usd"), (int, float)):
        return float(p["cost_usd"])
    return None


def _cost(item, output, error):
    """US dollars the item cost: a cloud API model's `cost_usd` (from the ledger's price, runtimes/cloud.py), 0 for a model run here (its cost
    is GPU time: latency_s), none for outputs recorded through a subscription CLI (paid in quota, not dollars) or a failed item. A pipeline adds
    the cost of each step in its trace."""
    if error or not isinstance(output, dict):
        return None
    # a pipeline's steps (a cloud model's layout, then a local drawing) each cost what their answer says: the item cost all of them
    steps = [t for t in output.get("pipeline") or [] if isinstance(t, dict)][
        :-1
    ]  # the last entry is the candidate's own job: `output` itself
    total = 0.0
    for o in [*steps, output]:
        c = own_cost(o)
        if c is not None:
            total += c
        elif o.get("recorded") or o.get("cli"):
            return None
    return total


def _exact(item, output, error):
    want = (item.get("reference") or {}).get("answer")
    if want is None:
        return None
    if error:
        return False
    got = output.get("answer") if isinstance(output, dict) else output
    return " ".join(normalise(str(got))) == " ".join(normalise(str(want)))


def _rank_of_answer(item, output, error):
    """1-based rank of the item's right document among the scored pool (ties count against the answer: the worst rank of equals), or None
    when the item has no `reference.answer` (an index into the pool)."""
    want = (item.get("reference") or {}).get("answer")
    if want is None:
        return None
    scores = output.get("scores") if isinstance(output, dict) else None
    if error or not scores or not 0 <= int(want) < len(scores):
        return math.inf
    s = scores[int(want)]
    return 1 + sum(1 for i, x in enumerate(scores) if i != int(want) and x >= s)


def _hit_at(k):
    def score(item, output, error):
        r = _rank_of_answer(item, output, error)
        return None if r is None else r <= k

    return score


def _mrr(item, output, error):
    r = _rank_of_answer(item, output, error)
    return None if r is None else (0.0 if r == math.inf else 1.0 / r)


def _parsed(item, output, error):
    if error:
        return False
    text = output.get("text") if isinstance(output, dict) else output
    if not isinstance(text, str):
        return isinstance(output, (dict, list))
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def _failed(item, output, error):
    return error is not None


def _peak_vram(item, output, error):
    v = output.get("peak_vram_mb") if isinstance(output, dict) and not error else None
    return None if not v else v / 1024


def _quality(key):
    """A reading the video worker measured on its result (output.quality), for the video evals: fidelity, flicker, sharpness, noise."""

    def score(item, output, error):
        q = output.get("quality") if isinstance(output, dict) and not error else None
        v = q.get(key) if isinstance(q, dict) else None
        return float(v) if isinstance(v, (int, float)) else None

    return score


# ---- image-kind and vectorise ---------------------------------------------------------------------------------------------------------------
def _kind_ref(item):
    return (item.get("reference") or {}).get("kind")


def _kind_got(output):
    return output.get("kind") if isinstance(output, dict) else None


def _kind_right(item, output, error):
    """The answer's kind is the labelled one (a kind the label also accepts, `reference.also`, counts: a print can fairly be two)."""
    if error:
        return False
    ref = item.get("reference") or {}
    return _kind_got(output) in [ref.get("kind"), *(ref.get("also") or [])]


def _group_right(item, output, error):
    """The answer's broad group (illustration, photo, other) is the labelled kind's."""
    from stratamodels.runtimes.imagekind import kind_of

    if error or _kind_ref(item) is None:
        return False
    return isinstance(output, dict) and output.get("group") == kind_of(str(_kind_ref(item)))["group"]


def _line_right(item, output, error):
    """The answer says line art exactly when the labelled kind is: the decision strata-scan makes on it (vectorise or keep the picture)."""
    from stratamodels.runtimes.imagekind import kind_of

    if error or _kind_ref(item) is None:
        return False
    return isinstance(output, dict) and bool(output.get("line_art")) == kind_of(str(_kind_ref(item)))["line_art"]


def _macro_f1(items, outputs, key):
    """Mean F1 over the labelled classes (kind or group): a rare class counts as much as the 56 engravings."""
    from stratamodels.runtimes.imagekind import kind_of

    def lab(k):
        return None if k is None else (k if key == "kind" else kind_of(str(k))["group"])

    pairs = [
        (lab(_kind_ref(it)), lab(_kind_got(o[0])) if o and o[1] is None else None) for it, o in zip(items, outputs)
    ]
    classes = sorted({r for r, _ in pairs if r is not None})
    f1s = []
    for c in classes:
        tp = sum(1 for r, g in pairs if r == c and g == c)
        fp = sum(1 for r, g in pairs if r != c and g == c)
        fn = sum(1 for r, g in pairs if r == c and g != c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / len(f1s) if f1s else None


METRICS = {
    m.name: m
    for m in [
        Metric(
            "wer",
            "ratio",
            "lower",
            "Word error rate: word substitutions, deletions and insertions over the reference words, after normalise()",
            "%",
            _wer,
        ),
        Metric("cer", "ratio", "lower", "Character error rate, the same on characters", "%", _cer),
        Metric(
            "hit_at_1",
            "binary",
            "higher",
            "The right document ranks first among the item's pool (retrieval)",
            "%",
            _hit_at(1),
        ),
        Metric(
            "hit_at_5",
            "binary",
            "higher",
            "The right document is among the first 5 of the item's pool (retrieval)",
            "%",
            _hit_at(5),
        ),
        Metric(
            "mrr",
            "mean",
            "higher",
            "Mean reciprocal rank of the right document in the item's pool (retrieval)",
            "",
            _mrr,
        ),
        Metric(
            "cer_exact",
            "ratio",
            "lower",
            "Character error rate with case and punctuation kept (only curly quotes, dashes and whitespace unified): small capitals, italics read as other letters, a wrong stop all count",
            "%",
            _cer_exact,
        ),
        Metric(
            "line_recall",
            "ratio",
            "higher",
            f"Printed lines of the reference the reading found (at least {LINE_FOUND:.0%} of a line's words, in any order): a missed line counts whole, where cer counts its characters",
            "%",
            _line_recall,
        ),
        Metric(
            "print_read",
            "mean",
            "higher",
            "Share of the scan's ink in the reference text blocks inside the reading's line boxes: print found and read (as anything); none for a reading without boxes",
            "%",
            _print_read,
        ),
        Metric(
            "wer_bag",
            "ratio",
            "lower",
            "Word error rate with word order ignored: beside wer, what is left of it is misread or missing words, not lines in the wrong order",
            "%",
            _wer_bag,
        ),
        Metric(
            "garbled",
            "ratio",
            "lower",
            "Words that look like OCR junk (a stray symbol, a digit among letters, a capital inside a word, five consonants running, a lone letter): needs no reference",
            "%",
            _garbled,
        ),
        Metric(
            "unclear",
            "ratio",
            "lower",
            f"Words Tesseract itself was unsure of (confidence below {UNCLEAR_CONF:g}): needs no reference; none for a vision-language model's text",
            "%",
            _unclear,
        ),
        Metric(
            "fidelity_ssim",
            "mean",
            "higher",
            "SSIM of the video, resized back to the prepared size, against the tape (the same deinterlace and crop): low means invented or lost detail, or colour drift",
            "",
            _quality("ssim"),
        ),
        Metric(
            "fidelity_psnr",
            "mean",
            "higher",
            "PSNR of the video, resized back, against the tape",
            "dB",
            _quality("psnr"),
        ),
        Metric(
            "flicker_ratio",
            "mean",
            "lower",
            "Frame-to-frame luma change of the video over the tape's own: over 1 shimmers or flickers more than the tape (per-frame models), under 1 is steadier",
            "",
            _quality("flicker_ratio"),
        ),
        Metric(
            "sharpness",
            "mean",
            "higher",
            "Variance of the luma Laplacian on stills at the prepared size: falls when detail is blurred (read it with luma_noise: noise raises it too)",
            "",
            _quality("sharpness"),
        ),
        Metric(
            "luma_noise",
            "mean",
            "lower",
            "Luma noise (Immerkaer sigma) on stills at the prepared size: grain",
            "",
            _quality("luma_noise"),
        ),
        Metric(
            "chroma_noise",
            "mean",
            "lower",
            "Colour noise (Immerkaer sigma of Cb and Cr) on stills at the prepared size: VHS's blotchy colour",
            "",
            _quality("chroma_noise"),
        ),
        Metric(
            "faces_missed",
            "ratio",
            "lower",
            "Share of labelled faces with no detected face at IoU 0.5",
            "%",
            _faces_recall,
        ),
        Metric("face_found", "binary", "higher", "The labelled face was detected (IoU 0.5)", "%", _face_found),
        Metric(
            "cut_miss",
            "ratio",
            "lower",
            "Shot cuts in the reference that no answered cut lies within 0.5 s of (or the item's tolerance_s), matched one to one",
            "%",
            _cut_miss,
        ),
        Metric(
            "cut_false",
            "ratio",
            "lower",
            "Answered shot cuts that match no reference cut: cuts that are not there",
            "%",
            _cut_false,
        ),
        Metric(
            "cut_f1",
            "mean",
            "higher",
            "F1 of the answered cuts against the reference per item (1 when both say one shot)",
            "",
            _cut_f1,
        ),
        Metric(
            "id_f1",
            "mean",
            "higher",
            "Face tracks: identity F1 within each shot, the labelled faces in their person's track (people and tracks paired one to one per shot) against all labelled and answered faces",
            "",
            _id_f1,
        ),
        Metric(
            "id_f1_clip",
            "mean",
            "higher",
            "Face tracks: identity F1 over the whole clip (one track per person across every shot), the score before 2026-11-27, to compare",
            "",
            _id_f1_clip,
        ),
        Metric(
            "track_mixed",
            "ratio",
            "lower",
            "Face tracks that hold two labelled people (two faces or more of each)",
            "%",
            _track_mixed,
        ),
        Metric(
            "track_split",
            "ratio",
            "lower",
            "Extra face tracks per labelled person per shot (one person in three tracks counts two)",
            "%",
            _track_split,
        ),
        Metric(
            "track_face_miss",
            "ratio",
            "lower",
            "Labelled faces that no face track holds at that moment",
            "%",
            _face_recall_tracks,
        ),
        Metric(
            "block_f1",
            "mean",
            "higher",
            "Text blocks found where the cleaned page has them: F1 of blocks matched at IoU 0.5 (a merged or split block matches nothing)",
            "%",
            _parts_f1("blocks"),
        ),
        Metric(
            "figure_f1",
            "mean",
            "higher",
            "Pictures found where the cleaned page has them: F1 of figures matched at IoU 0.5",
            "%",
            _parts_f1("figures"),
        ),
        Metric(
            "role_acc",
            "mean",
            "higher",
            "Of the blocks found, the share given the right role (body, heading, caption, footnote, running head, folio...)",
            "%",
            _role_acc,
        ),
        Metric(
            "order_tau",
            "mean",
            "higher",
            "Reading order of the blocks found against the cleaned page's: Kendall's tau (1 the same order, -1 reversed)",
            "",
            _order_tau,
        ),
        Metric(
            "page_ink_f1",
            "mean",
            "higher",
            "The rebuilt page's ink against the scan's within the page's text blocks: F1 with a pixel or two of tolerance (evals/ink.py)",
            "%",
            _page_ink_f1,
        ),
        Metric(
            "page_ink_all",
            "mean",
            "higher",
            "The rebuilt page's ink against the scan's over the whole printed area (text, pictures, rules, the border): F1 with a pixel or two of tolerance",
            "%",
            _page_ink_all,
        ),
        Metric(
            "verification_auc",
            "dataset",
            "higher",
            "ROC AUC of same-person against different-person pairs of the labelled faces",
            "",
            None,
            lambda it, out: verification(it, out)["auc"],
            verification_prepare,
            lambda st, idx: verification_on(st, idx)["auc"],
        ),
        Metric(
            "tar_at_far1",
            "dataset",
            "higher",
            "Same-person pairs accepted when 1 % of different-person pairs are",
            "%",
            None,
            lambda it, out: verification(it, out)["tar"],
            verification_prepare,
            lambda st, idx: verification_on(st, idx)["tar"],
        ),
        Metric(
            "desc_fields",
            "mean",
            "higher",
            "Share of the reference description's structured fields (indoor/outdoor, time of day, weather, people, children, person subject, main light, light level, sharpness, skin, text) given the same answer",
            "%",
            _desc_fields,
        ),
        Metric(
            "tag_f1",
            "mean",
            "higher",
            "F1 of the search tags against the reference description's tags (lower case, exact words)",
            "%",
            _tag_f1,
        ),
        Metric(
            "desc_valid", "binary", "higher", "The output is a description with a caption and tags", "%", _desc_valid
        ),
        Metric(
            "exact_match", "binary", "higher", "The answer equals the reference answer after normalise()", "%", _exact
        ),
        Metric("parsed", "binary", "higher", "The output is valid JSON", "%", _parsed),
        Metric(
            "ink_f1",
            "mean",
            "higher",
            "F1 of the fitted text's ink against the scan's within the words, a pixel or two of tolerance (evals/ink.py): how closely the face, size, width and weight reproduce the print; needs no reference",
            "",
            _ink_f1,
        ),
        Metric(
            "font_class",
            "binary",
            "higher",
            "The face's class (old-style, modern, clarendon...) is one the reference accepts",
            "%",
            _font_class_hit,
        ),
        Metric(
            "font_close",
            "mean",
            "higher",
            "Closeness of the face's class to the reference's: 1 the same, 0.5 a neighbouring class (old-style/transitional, modern/fat-face, clarendon/slab...), 0 otherwise",
            "",
            _font_class_close,
        ),
        Metric(
            "font_top1",
            "binary",
            "higher",
            "The face is the reference face (synthetic items, whose face is known)",
            "%",
            _font_face(1),
        ),
        Metric(
            "font_top3", "binary", "higher", "The reference face is among the three best ranked", "%", _font_face(3)
        ),
        Metric(
            "font_style",
            "binary",
            "higher",
            "The main style (roman, italic, bold, small capitals) is the reference's",
            "%",
            _font_style,
        ),
        Metric(
            "font_size_err",
            "mean",
            "lower",
            "Relative error of the size in points (synthetic items)",
            "%",
            _font_size_err,
        ),
        Metric("failed", "binary", "lower", "The job failed", "%", _failed),
        Metric(
            "peak_vram_gb",
            "mean",
            "lower",
            "Highest VRAM the job's own process used (image generation runs as its own process)",
            "GB",
            _peak_vram,
        ),
        Metric(
            "tts_wer",
            "ratio",
            "lower",
            "Intelligibility of speech made from text: the [score] model transcribes the audio back, word error rate against the text",
            "%",
            _tts_wer,
        ),
        Metric(
            "rtf",
            "mean",
            "lower",
            "Real-time factor: seconds to make the audio over the audio's length (under 1 is faster than speaking it)",
            "",
            _rtf,
        ),
        Metric(
            "pace_spread",
            "dataset",
            "lower",
            "How much the speaking rate varies from line to line: standard deviation over mean of each clip's rate (syllables of the words as said, over the speech's length; workers/pacing.py)",
            "%",
            None,
            _pace_spread,
        ),
        Metric(
            "long_pauses",
            "mean",
            "lower",
            "Pauses over 0.6 s between the first and last sound, per clip",
            "",
            _long_pauses,
        ),
        Metric("max_pause_s", "mean", "lower", "The longest pause between words in a clip", "s", _max_pause),
        Metric(
            "sound_pure",
            "binary",
            "higher",
            "A phonics sound said on its own, with no vowel added: a hiss that hisses, a nasal that is voiced, a stop that stays short (workers/phonics.py pure(): a check of gross failures, not of quality; a failed job fails it)",
            "%",
            _sound_pure,
        ),
        Metric(
            "vowel_leak",
            "mean",
            "lower",
            "Share of a phonics sound's energy in frames that look like a vowel (voiced, mostly in 500 Hz to 3 kHz; workers/phonics.py)",
            "%",
            _vowel_leak,
        ),
        Metric(
            "fit_err",
            "mean",
            "lower",
            "How far the speech's length is from the length asked for (wpm or fit_s), as a share",
            "%",
            _fit_err,
        ),
        *[Metric(n, "mean", "higher", d, u, _reader(g, k)) for n, g, k, d, u in AUDIO_METRICS],
        Metric(
            "removed_db",
            "mean",
            "higher",
            "How much quieter the denoiser left the sound (before levelling): what it took out. Context, not quality: more is not better",
            "dB",
            _removed_db,
        ),
        Metric(
            "kind_right",
            "binary",
            "higher",
            "The picture's kind (engraving, woodcut, halftone...) is the labelled one",
            "%",
            _kind_right,
        ),
        Metric(
            "group_right",
            "binary",
            "higher",
            "The picture's group (illustration, photo, other) is the labelled kind's",
            "%",
            _group_right,
        ),
        Metric(
            "line_right",
            "binary",
            "higher",
            "Line art (vectorise it) or not, as the labelled kind is",
            "%",
            _line_right,
        ),
        Metric(
            "kind_f1",
            "dataset",
            "higher",
            "Mean F1 over the labelled kinds: a rare kind counts as much as a common one",
            "%",
            None,
            lambda it, out: _macro_f1(it, out, "kind"),
        ),
        Metric(
            "group_f1",
            "dataset",
            "higher",
            "Mean F1 over the three groups (illustration, photo, other)",
            "%",
            None,
            lambda it, out: _macro_f1(it, out, "group"),
        ),
        Metric(
            "vec_ink_f1",
            "mean",
            "higher",
            "The drawing's ink against the picture's, F1 within 1 px at 1600 px (evals/vector.py)",
            "",
            _vec.reader("ink_f1"),
        ),
        Metric(
            "vec_ssim",
            "mean",
            "higher",
            "Structural similarity of drawing and picture in grey at 1024 px, slightly blurred: alike at reading size",
            "",
            _vec.reader("ssim"),
        ),
        Metric(
            "vec_tone",
            "mean",
            "lower",
            "Mean grey difference at an eighth of the size (0..1): light and dark where they were",
            "",
            _vec.reader("tone"),
        ),
        Metric(
            "vec_colour",
            "mean",
            "lower",
            "Mean colour difference (CIE76 delta E) at an eighth of the size",
            "",
            _vec.reader("colour"),
        ),
        Metric("vec_kb", "mean", "lower", "Size of the SVG", "KB", _vec.size_kb),
        Metric("vec_nodes", "mean", "lower", "Path commands in the SVG: what an editor would move", "", _vec.nodes),
        Metric(
            "judged",
            "judged",
            "higher",
            "Blind human judgments: share of head-to-head pairs against the baseline the candidate won (a tie is half a win)",
            "%",
        ),
        Metric(
            "latency_s",
            "mean",
            "lower",
            "Seconds the model ran per item (model already loaded; load time is reported apart)",
            "s",
            _latency,
        ),
        Metric(
            "cost_usd",
            "mean",
            "lower",
            "US dollars per item on a cloud API (Claude, Google AI Studio's paid key), priced from the call's tokens; 0 for a model run here, none for a subscription CLI's recordings",
            "$",
            _cost,
        ),
    ]
}


def summarise(metric, items, outputs, seed=None):
    """One candidate's value of `metric` with its 95 % interval: dict(value, low, high, n). outputs[i] is (output, error), items carry _run_s."""
    m = METRICS[metric]
    if m.shape == "dataset":
        on = m.sampler(items, outputs)
        n = len(items)
        val = on(np.arange(n))
        rng = np.random.default_rng(S.SEED if seed is None else seed)
        boots = []
        for _ in range(200):
            idx = rng.integers(0, n, n)
            v = on(idx)
            if not math.isnan(v):
                boots.append(v)
        lo, hi = (
            (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
            if boots
            else (float("nan"), float("nan"))
        )
        return dict(value=val, low=lo, high=hi, n=n)
    vals = [m.score(it, o, e) for it, (o, e) in zip(items, outputs)]
    vals = [v for v in vals if v is not None]
    if m.shape == "ratio":
        v, lo, hi = S.bootstrap_ratio([a for a, _ in vals], [b for _, b in vals], seed=seed)
    elif m.shape == "binary":
        v, lo, hi = S.wilson(sum(bool(x) for x in vals), len(vals))
    else:
        v, lo, hi = S.bootstrap_mean(vals, seed=seed)
    out = dict(value=v, low=lo, high=hi, n=len(vals))
    if m.shape == "mean" and vals:
        out["p95"] = S.quantile(vals, 0.95)
    return out


def compare(metric, items, base_outputs, cand_outputs, seed=None):
    """Candidate against baseline on the same items: dict(n, mean_a, mean_b, diff (candidate minus baseline), low, high, p, improves (True when the
    interval lies wholly on the better side, False when wholly on the worse side, None when it straddles 0), and for binary the McNemar counts)."""
    m = METRICS[metric]
    if m.shape == "dataset":
        on_a, on_b = m.sampler(items, base_outputs), m.sampler(items, cand_outputs)
        n = len(items)
        a, b = on_a(np.arange(n)), on_b(np.arange(n))
        rng = np.random.default_rng(S.SEED if seed is None else seed)
        d = []
        for _ in range(200):
            idx = rng.integers(0, n, n)
            x, y = on_a(idx), on_b(idx)
            if not (math.isnan(x) or math.isnan(y)):
                d.append(y - x)
        lo, hi = (float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))) if d else (float("nan"), float("nan"))
        p = (
            float((min(np.sum(np.array(d) <= 0), np.sum(np.array(d) >= 0)) * 2 + 1) / (len(d) + 1))
            if d
            else float("nan")
        )
        r = dict(n=n, mean_a=a, mean_b=b, diff=b - a, low=lo, high=hi, p=min(p, 1.0))
    else:
        pairs = [(m.score(it, *ab), m.score(it, *cb)) for it, ab, cb in zip(items, base_outputs, cand_outputs)]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if m.shape == "ratio":
            r = S.paired_ratio_diff(
                [x[0] for x, _ in pairs], [y[0] for _, y in pairs], [x[1] for x, _ in pairs], seed=seed
            )
        else:
            r = S.paired_diff([float(x) for x, _ in pairs], [float(y) for _, y in pairs], seed=seed)
            if m.shape == "binary":
                mc = S.mcnemar([bool(x) for x, _ in pairs], [bool(y) for _, y in pairs])
                r.update(only_baseline=mc["only_a"], only_candidate=mc["only_b"], p=mc["p"])
    sign = -1 if m.better == "lower" else 1
    r["improves"] = (
        None
        if any(math.isnan(r[k]) for k in ("low", "high"))
        else (
            True
            if sign * r["low"] > 0 and sign * r["high"] > 0
            else False
            if sign * r["low"] < 0 and sign * r["high"] < 0
            else None
        )
    )
    return r
