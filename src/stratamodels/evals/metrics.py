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
import json
import math
import re
import unicodedata

import numpy as np

from stratamodels.evals import stats as S


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


@functools.lru_cache(maxsize=64)
def _page_scan_ink(image, boxes):
    """(the scan's ink, the reference text blocks as a mask, dpi) for page_ink_f1: the page binarised as the font eval does (not deskewed: a
    rebuild draws in the scan's own pixels), and the union of the cleaned page's text blocks (4 px around), so borders, rules, pictures, library
    stamps and show-through from the next leaf count for no candidate."""
    from PIL import Image

    from stratamodels.evals import ink as INK

    im = Image.open(image)
    found = im.info.get("dpi")
    dpi = round(float(found[0]), 1) if found and found[0] > 1 else 300.0
    full, _ = INK.binarise(im, dpi, deskew=False)
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
        if isinstance(o.get("cost_usd"), (int, float)):
            total += float(o["cost_usd"])
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
            "cer_exact",
            "ratio",
            "lower",
            "Character error rate with case and punctuation kept (only curly quotes, dashes and whitespace unified): small capitals, italics read as other letters, a wrong stop all count",
            "%",
            _cer_exact,
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
