"""Metrics: how one output is scored against its item, and how the scores of a run add up. Each metric says which way is better, so the analysis
can say "better" or "worse" without a person remembering that WER goes down.

Three shapes:
  per-item mean    a number per item (latency_s), averaged; compared with paired_diff.
  per-item binary  right or wrong per item (exact_match, parsed), a share; Wilson interval, compared with McNemar and paired_diff.
  ratio            (errors, size) per item, added up over the run (wer: all word errors over all reference words); compared with paired_ratio_diff.
  dataset          computed from all items together (verification AUC); interval by resampling items, comparison by paired bootstrap.
`score(item, output)` returns what the shape needs, or None when the metric does not apply to that item (no reference, the job failed: a failed
job counts as wrong for binary metrics and as all errors for ratios, never silently skipped)."""

import dataclasses, json, math, re, unicodedata

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
    """Levenshtein distance between two token lists (substitutions, deletions, insertions all cost 1)."""
    if not ref:
        return len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def transcript(output):
    """The text of a transcription output (segments joined)."""
    if not output:
        return ""
    return " ".join(s.get("text", "") for s in output.get("segments", [])) if isinstance(output, dict) else str(output)


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


def verification(items, outputs, far=0.01):
    """Face verification from the embeddings of the reference faces: every pair of items is same-person or not (reference.person). Returns
    dict(auc, tar, threshold, pairs): ROC AUC, and the share of same-person pairs accepted at the threshold that lets `far` of different-person
    pairs through. Items whose face was not found count as rejected (distance 2)."""
    emb, who = [], []
    for it, out in zip(items, outputs):
        p = (it.get("reference") or {}).get("person")
        if p is None:
            continue
        f = best_face(it, out[0]) if out[1] is None else None
        emb.append(np.asarray(f["embedding"], np.float32) if f and "embedding" in f else None)
        who.append(p)
    n = len(emb)
    if n < 2:
        return dict(auc=float("nan"), tar=float("nan"), threshold=float("nan"), pairs=0)
    dim = next((len(e) for e in emb if e is not None), 512)
    E = np.stack([e if e is not None else np.zeros(dim, np.float32) for e in emb])
    ok = np.array([e is not None for e in emb])
    sim = E @ E.T
    iu = np.triu_indices(n, 1)
    s = sim[iu]
    s[~(ok[iu[0]] & ok[iu[1]])] = -1.0
    same = np.array(who)[iu[0]] == np.array(who)[iu[1]]
    pos, neg = s[same], s[~same]
    if pos.size == 0 or neg.size == 0:
        return dict(auc=float("nan"), tar=float("nan"), threshold=float("nan"), pairs=int(s.size))
    ranks = _avg_ranks(np.concatenate([pos, neg]))  # tied scores share their average rank (a missed face ties at -1)
    auc = (ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
    thr = float(np.quantile(neg, 1 - far))
    tar = float((pos > thr).mean())
    return dict(auc=float(auc), tar=tar, threshold=thr, pairs=int(s.size))


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
DESCRIPTION_FIELDS = {
    "indoor_outdoor": lambda d: _g(d, "setting.indoor_outdoor"),
    "time_of_day": lambda d: _g(d, "setting.time_of_day"),
    "weather": lambda d: _g(d, "setting.weather"),
    "people": lambda d: _bucket(_g(d, "people.count")),
    "children": lambda d: None if not isinstance(_g(d, "people.children"), int) else _g(d, "people.children") > 0,
    "person_subject": lambda d: _g(d, "people.main_subject_is_person"),
    "main_light": _main_light,
    "light_level": lambda d: _g(d, "lighting.level"),
    "sharpness": lambda d: _g(d, "exposure.sharpness"),
    "skin_visible": lambda d: _g(d, "colour.skin_visible"),
    "has_text": lambda d: None if not isinstance(_g(d, "text_visible"), list) else len(_g(d, "text_visible")) > 0,
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


# ---- generic ------------------------------------------------------------------------------------------------------------------------------
def _latency(item, output, error):
    return None if error else item.get("_run_s")


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
            "faces_missed",
            "ratio",
            "lower",
            "Share of labelled faces with no detected face at IoU 0.5",
            "%",
            _faces_recall,
        ),
        Metric("face_found", "binary", "higher", "The labelled face was detected (IoU 0.5)", "%", _face_found),
        Metric(
            "verification_auc",
            "dataset",
            "higher",
            "ROC AUC of same-person against different-person pairs of the labelled faces",
            "",
            None,
            lambda it, out: verification(it, out)["auc"],
        ),
        Metric(
            "tar_at_far1",
            "dataset",
            "higher",
            "Same-person pairs accepted when 1 % of different-person pairs are",
            "%",
            None,
            lambda it, out: verification(it, out)["tar"],
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
    ]
}


def summarise(metric, items, outputs, seed=None):
    """One candidate's value of `metric` with its 95 % interval: dict(value, low, high, n). outputs[i] is (output, error), items carry _run_s."""
    m = METRICS[metric]
    if m.shape == "dataset":
        val = m.dataset(items, outputs)
        rng = np.random.default_rng(S.SEED if seed is None else seed)
        n = len(items)
        boots = []
        for _ in range(200):
            idx = rng.integers(0, n, n)
            v = m.dataset([items[i] for i in idx], [outputs[i] for i in idx])
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
        a = m.dataset(items, base_outputs)
        b = m.dataset(items, cand_outputs)
        rng = np.random.default_rng(S.SEED if seed is None else seed)
        n = len(items)
        d = []
        for _ in range(200):
            idx = rng.integers(0, n, n)
            it = [items[i] for i in idx]
            x, y = m.dataset(it, [base_outputs[i] for i in idx]), m.dataset(it, [cand_outputs[i] for i in idx])
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
