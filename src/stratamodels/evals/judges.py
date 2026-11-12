"""Model judges: a model (local, through the queue, or a cloud model through its subscription CLI) judges the same blind pairs people judge, and
its verdicts are kept beside theirs, so how far each judge agrees with people can be measured before it is trusted with more.

Judges come in kinds, by what they must read, and each kind is a registry role with its own chosen model (models.toml), as `transcribe` has:
  text     judge-text: transcripts, captions, JSON answers; sees no pictures
  vision   judge-vision: photo descriptions, generated pictures; sees the item's pictures and pictures the candidates made
  audio    judge-audio: recordings (cleaned sound, speech): hears the item's own recording, then X's and Y's, each as 16 kHz mono WAV (what
           the models hear anyway: hiss above 8 kHz is lost on them, people judge that); a local model whose mmproj hears (Qwen3-Omni,
           Gemma 4 E4B), or Gemini through agy (codex has no audio)
Which model a kind uses is decided like any model choice: by an eval, here on agreement with people (the `judges` part of the analysis).

A spec names its judges in `[[judges]]`:

    [[judges]]
    name = "vision"                   # how it appears beside the people (unique in the spec)
    kind = "vision"                   # text | vision | audio (default vision): a local judge runs on role judge-<kind>, through the queue like any job
    model = "gemma-4-12b-vl"          # optional for a local judge: a model serving that role, to trial it; default the role's chosen model
    params = { think = false }        # added to the judging job (temperature defaults to 0)

    [[judges]]
    name = "gemini-flash"
    cli = "agy"                       # a cloud model on a plan: codex (ChatGPT) | agy (Antigravity); `model` is then the CLI's model slug
    model = "gemini-3.8-flash-low"
    effort = "low"
    pairs = "judged"                  # judged (default): only pairs people judged, the ones agreement is measured on | all: every pair the Judge tab offers
    limit = 20                        # at most this many pairs per run: a dear judge gets few calls, like a sampled candidate
    orders = 2                        # ask twice with X and Y swapped (default); a judge that changes its mind with the order gives a tie
    auto = true                       # the service also judges each pair a person judges, right after they do
    criteria = "..."                  # what makes one answer better (default: a general rule for the task)

A verdict is an eval_judgments row like a person's, with `judge` set (the hash of everything that decides the verdict: model, role or CLI,
effort, params, criteria, orders, the prompt's version) and `detail` holding each order's answer and reason. A changed judge config is a new
judge: its old verdicts stop counting, as a person's do when an output changes. People's judgments alone decide `judged` experiments and order the
Judge tab; judges' verdicts only go into the agreement analysis (`judges` in the analysis: share agreeing with people, Cohen's kappa, position
consistency, and what each judge alone would conclude about each candidate against the baseline).

Run them with `./strata-models eval judge NAME [--judge J] [--limit N]` (local judges need the service: their jobs go through its queue; cloud
judges call the official CLI from that process, which must be signed in) or POST /api/evals/{name}/judges/{judge}/run. A judge that is also a
candidate (Gemini judging Gemini's descriptions) tends to prefer its own: compare its agreement on those pairs before trusting it there."""

import hashlib
import json
import time

from sqlalchemy import select

from stratamodels.db.models import EvalJudgment
from stratamodels.evals import core as C
from stratamodels.evals import stats as S

PROMPT_VERSION = 1  # bump when the prompt below changes: every judge's verdicts are then a new judge's
KINDS = ("text", "vision", "audio")
JUDGE_KEYS = {
    "name",
    "kind",
    "model",
    "role",
    "cli",
    "effort",
    "params",
    "criteria",
    "pairs",
    "limit",
    "orders",
    "auto",
    "timeout",
    "note",
}
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")
HIDDEN = {  # output fields a judge never sees: they would tell the candidates apart, or say nothing about quality
    "model",
    "pipeline",
    "seconds",
    "tokens_in",
    "tokens_out",
    "peak_vram_mb",
    "finish",
    "scores",
    "scored",
    "cli",
    "cli_model",
    "run_s",
    "load_s",
}
CRITERIA = "Judge which answer does the task better: correct and faithful to the input first, then complete, then clear. Length is not quality."
SYSTEM = (
    "You judge two answers to the same task, blind: you are not told which model wrote either. Read the task and its input, then both "
    "answers, and decide which is better by the criteria given. If they are equally good (or equally bad), answer tie."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "description": "one or two sentences: the difference that decided it"},
        "winner": {"type": "string", "enum": ["X", "Y", "tie"]},
    },
    "required": ["reason", "winner"],
    "additionalProperties": False,
}


class JudgeError(RuntimeError):
    """One verdict could not be had (a failed job, an answer that is not a verdict): left out, tried again on the next run."""


def parse_judge(d):
    if not isinstance(d, dict) or not d.get("name"):
        raise ValueError("a judge needs a name")
    if d.get("cli") and not d.get("model"):
        raise ValueError(f"judge {d['name']!r}: a cloud judge names its model (the CLI's model slug)")
    kind = str(d.get("kind") or "vision")
    if kind not in KINDS:
        raise ValueError(f"judge {d['name']!r}: kind is {', '.join(KINDS[:-1])} or {KINDS[-1]}")
    if kind == "audio" and d.get("cli") == "codex":
        raise ValueError(f"judge {d['name']!r}: codex cannot hear recordings (an audio judge in the cloud is agy)")
    extra = set(d) - JUDGE_KEYS
    if extra:
        raise ValueError(
            f"judge {d['name']!r} has unknown fields {sorted(extra)} (known: {', '.join(sorted(JUDGE_KEYS))})"
        )
    cli = str(d["cli"]) if d.get("cli") else None
    if cli and cli not in ("codex", "agy"):
        raise ValueError(f"judge {d['name']!r}: cli is codex or agy")
    pairs = str(d.get("pairs", "judged"))
    if pairs not in ("judged", "all"):
        raise ValueError(f"judge {d['name']!r}: pairs is judged or all")
    orders = int(d.get("orders", 2))
    if orders not in (1, 2):
        raise ValueError(f"judge {d['name']!r}: orders is 1 or 2")
    limit = d.get("limit")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError(f"judge {d['name']!r}: limit is a number of pairs (at least 1)")
    return dict(
        name=str(d["name"]),
        kind=kind,
        model=str(d["model"]) if d.get("model") else None,
        role=None if cli else str(d.get("role") or f"judge-{kind}"),
        cli=cli,
        effort=str(d["effort"]) if d.get("effort") else None,
        params=dict(d.get("params") or {}),
        criteria=str(d.get("criteria") or CRITERIA),
        pairs=pairs,
        limit=limit,
        orders=orders,
        auto=bool(d.get("auto", False)),
        timeout=int(d.get("timeout", 600)),
        note=str(d.get("note", "")),
    )


def resolve(engine, j):
    """The model judge j runs on: its own, else its role's chosen one in the registry (None when the registry cannot say)."""
    if j["model"] or j["cli"]:
        return j["model"]
    from stratamodels.registry import Registry

    try:
        return Registry.from_db(engine).resolve(j["role"]).key
    except (
        Exception
    ):  # RegistryError, or no registry in this database (scripts, some tests): the role stands for its model
        return None


def judge_key(j, model=None):
    """Everything that decides a judge's verdicts (not how many it is given, or when). `model`: the model it resolves to, so a new choice for a
    kind's role is a new judge, and the old one's verdicts stay its own."""
    keep = {k: j[k] for k in ("kind", "role", "cli", "effort", "params", "criteria", "orders")}
    return C._h(C._canon(dict(keep, model=model or j["model"], v=PROMPT_VERSION)), 16)


def find(spec, name):
    j = next((j for j in spec.judges if j["name"] == name), None)
    if j is None:
        raise C.EvalError(
            f"{spec.name}: no judge {name!r} (judges: {', '.join(j['name'] for j in spec.judges) or 'none'})"
        )
    return j


# ---- what a judge is shown ---------------------------------------------------------------------------------------------------------------
def _is_image(p):
    return isinstance(p, str) and p.lower().endswith(IMAGE_EXT)


def _is_audio(p):
    return isinstance(p, str) and p.lower().endswith(AUDIO_EXT)


def _wav16(path, clip=None, cut=None):
    """A recording (or its clip) as 16 kHz mono WAV, made once into <data>/judges/audio/ (keyed by the file, its size and time, and the clip):
    every audio judge hears the same format, local or cloud."""
    import os

    from stratamodels import settings
    from stratamodels.evals import cloud

    st = os.stat(path)
    key = C._h(C._canon(dict(p=os.path.abspath(path), s=st.st_size, m=int(st.st_mtime), c=clip)), 16)
    out = os.path.join(settings.data_dir(), "judges", "audio", f"{key}.wav")
    if not os.path.isfile(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        (cut or cloud.cut_audio)(path, clip, out + ".part.wav")
        os.replace(out + ".part.wav", out)
    return out


def _item_images(item):
    inp = item.get("input") or {}
    imgs = list(inp.get("images") or []) + ([inp["image"]] if inp.get("image") else [])
    if _is_image(inp.get("path")):
        imgs.append(inp["path"])
    return [str(i) for i in imgs]


def shown(output):
    """An output as a judge reads it: its JSON answer if it has one, else its text (a transcript's segments joined), else the rest of it; never
    the fields that name the model."""
    o = C.without_pipeline(output)
    if not isinstance(o, dict):
        return json.dumps(o, ensure_ascii=False)
    if o.get("json") is not None:
        return json.dumps(o["json"], ensure_ascii=False, indent=1)
    if isinstance(o.get("text"), str):
        return o["text"]
    if isinstance(o.get("segments"), list):
        return " ".join(str(s.get("text", "")).strip() for s in o["segments"] if isinstance(s, dict)).strip()
    return json.dumps({k: v for k, v in o.items() if k not in HIDDEN}, ensure_ascii=False, indent=1)


def prompt(spec, j, item, x, y, cut=None):
    """(system, text, files) for one verdict: the task, the item's input (and reference, if it has one), answers X and Y, and the files: for
    a vision judge the pictures (the item's, then X's and Y's when the answers are pictures), for an audio judge the recordings (the item's,
    then X's and Y's, as 16 kHz WAV)."""
    raw = item.get("input") or {}
    vision, audio = j["kind"] == "vision", j["kind"] == "audio"
    inp = {k: v for k, v in raw.items() if k not in ("path", "image", "images", *(("clip",) if audio else ()))}
    imgs = _item_images(item) if vision else []
    if audio and _is_audio(raw.get("path")):
        imgs = [_wav16(str(raw["path"]), raw.get("clip"), cut)]
    parts = [f"Task: {spec.question}", f"Criteria: {j['criteria']}"]
    if imgs and audio:
        parts.append("Recording 1 is the input (the original recording). Listen to every recording before you judge.")
    elif imgs:
        parts.append(f"The input picture{'s are' if len(imgs) > 1 else ' is'} attached first ({len(imgs)}).")
    if inp:
        parts.append("Input:\n" + json.dumps(inp, ensure_ascii=False, indent=1))
    if item.get("reference"):
        parts.append(
            "Reference (what a correct answer says):\n" + json.dumps(item["reference"], ensure_ascii=False, indent=1)
        )
    for label, o in (("X", x), ("Y", y)):
        p = (C.without_pipeline(o) or {}).get("path") if isinstance(o, dict) else None
        if vision and _is_image(p):
            imgs.append(str(p))
            parts.append(f"Answer {label}: the picture attached as number {len(imgs)}.")
        elif audio and _is_audio(p):
            imgs.append(_wav16(str(p), None, cut))
            parts.append(f"Answer {label}: recording {len(imgs)}.")
        else:
            parts.append(f"Answer {label}:\n{shown(o)}")
    parts.append('Which answer is better? Reply with JSON: {"reason": "...", "winner": "X" | "Y" | "tie"}.')
    return SYSTEM, "\n\n".join(parts), imgs


def verdict(answer):
    """X, Y or tie from a judge's answer ({json} or {text}); JudgeError if it gave none."""
    from stratamodels.evals.cloud import parse_json

    d = answer.get("json") if isinstance(answer, dict) else None
    if not isinstance(d, dict):
        d = parse_json((answer or {}).get("text") if isinstance(answer, dict) else None)
    w = str((d or {}).get("winner", "")).strip().lower()
    if w not in ("x", "y", "tie"):
        raise JudgeError(f"no verdict in the answer: {str(answer)[:200]}")
    return {"x": "X", "y": "Y", "tie": "tie"}[w], str((d or {}).get("reason", ""))[:400]


# ---- asking ------------------------------------------------------------------------------------------------------------------------------
class QueueAsker:
    """A local judge: a job on its role through the queue (executor as in core.run)."""

    def __init__(self, executor):
        self.ex = executor

    def __call__(self, j, system, text, images, schema):
        params = {
            "temperature": 0,
            "max_tokens": 1024,
            **j["params"],
            "system": system,
            "prompt": text,
            "schema": schema,
        }
        if images:
            params["audios" if j["kind"] == "audio" else "images"] = images
        r = self.ex.wait(self.ex.submit(j["role"], params, j.get("resolved") or j["model"], None))
        if r["status"] != "done":
            raise JudgeError(f"judging job {r['status']}: {r.get('error') or ''}".strip())
        return r.get("result") or {}


class CliAsker:
    """A cloud judge: the vendor's own CLI on the plan (evals.cloud). A spent allowance raises cloud.QuotaError, which stops the run."""

    def __init__(self, run=None):
        self.run = run  # subprocess.run stand-in for tests

    def __call__(self, j, system, text, images, schema):
        import subprocess

        from stratamodels.evals import cloud

        kw = dict(model=j["model"], effort=j["effort"], timeout=j["timeout"])
        if self.run is not None:
            kw["run"] = self.run
        try:
            r = cloud.CLIS[j["cli"]](system, text, images, schema, **kw)
        except (cloud.CallError, subprocess.TimeoutExpired) as e:
            raise JudgeError(str(e))
        return dict(text=r["text"])


def asker_for(j, executor=None):
    if j["cli"]:
        return CliAsker()
    if executor is None:
        raise C.EvalError(f"judge {j['name']} is a local model: its jobs need the service's queue")
    return QueueAsker(executor)


# ---- which pairs -------------------------------------------------------------------------------------------------------------------------
def _rows(engine, experiment):
    with C.session(engine) as s:
        return list(
            s.scalars(select(EvalJudgment).where(EvalJudgment.experiment == experiment).order_by(EvalJudgment.id))
        )


def _pair_id(item_id, ka, kb):
    return (item_id, frozenset((ka, kb)))


def _current(spec, ds):
    return {(c.name, it["id"]): C.result_key(spec, c, it) for c in spec.candidates for it in ds.items}


def _counts(r, current):
    """Whether a judgment is of outputs that are still the candidates'."""
    return (
        bool(r.a_key and r.b_key)
        and current.get((r.a, r.item_id)) == r.a_key
        and current.get((r.b, r.item_id)) == r.b_key
    )


def todo(spec, engine, j, only=None):
    """The pairs judge j has still to judge, as (item index, a, b): pairs people judged (pairs = judged, in the order they judged them) or every
    pair the Judge tab would offer (all); never one it already judged with the outputs as they are. `only`: (item id, a, b), one pair."""
    ds, outs, _ = C.outputs_for(spec, engine)
    current = _current(spec, ds)
    index = {it["id"]: i for i, it in enumerate(ds.items)}
    rows = [r for r in _rows(engine, spec.name) if _counts(r, current)]
    jk = judge_key(j, resolve(engine, j))
    done = {_pair_id(r.item_id, r.a_key, r.b_key) for r in rows if r.judge == jk}

    def ok(i, n):
        return outs[n][i] is not None and not outs[n][i][1]

    cands = []
    if only is not None:
        iid, a, b = only
        if iid in index:
            cands = [(index[iid], a, b)]
    elif j["pairs"] == "judged":
        seen = set()
        for r in rows:
            if r.judge is None and r.item_id in index and _pair_id(r.item_id, r.a_key, r.b_key) not in seen:
                seen.add(_pair_id(r.item_id, r.a_key, r.b_key))
                cands.append((index[r.item_id], r.a, r.b))
    else:
        from stratamodels.evals import similarity as SIM

        names = [c.name for c in spec.candidates]
        for i in range(len(ds.items)):
            here = [n for n in names if ok(i, n)]
            for k, a in enumerate(here):
                for b in here[k + 1 :]:
                    if spec.primary == "judged" and spec.baseline not in (a, b):
                        continue  # as the Judge tab: head to head with the baseline only
                    oa, ob = C.without_pipeline(outs[a][i][0]), C.without_pipeline(outs[b][i][0])
                    if SIM.similarity(oa, ob, spec.similarity or None) < 0.9999:  # identical answers teach nothing
                        cands.append((i, a, b))
    out = []
    for i, a, b in cands:
        iid = ds.items[i]["id"]
        if (
            a in outs
            and b in outs
            and ok(i, a)
            and ok(i, b)
            and _pair_id(iid, current[(a, iid)], current[(b, iid)]) not in done
        ):
            out.append((i, a, b))
    return ds, outs, current, out


def _first(iid, ka, kb):
    """Which output is X when asked once: fixed by the pair (reproducible) and even across pairs (no side favoured)."""
    return hashlib.sha256(f"{iid}|{min(ka, kb)}|{max(ka, kb)}".encode()).digest()[0] % 2 == 0


def judge_pair(spec, j, item, a, oa, b, ob, ask):
    """One verdict on a pair: (winner as a candidate's name or 'tie', detail). Asked in one order, or both (X and Y swapped) when orders = 2: a
    judge whose choice follows the position rather than the answer gives a tie, and detail.consistent says so."""
    asks = []
    for x, ox, y, oy in [(a, oa, b, ob), (b, ob, a, oa)][: j["orders"]]:
        system, text, images = prompt(spec, j, item, ox, oy)
        t0 = time.perf_counter()
        w, why = verdict(ask(j, system, text, images, SCHEMA))
        asks.append(
            dict(
                x=x,
                y=y,
                said=w,
                winner=x if w == "X" else y if w == "Y" else "tie",
                reason=why,
                seconds=round(time.perf_counter() - t0, 2),
            )
        )
    wins = {r["winner"] for r in asks}
    winner = asks[0]["winner"] if len(wins) == 1 else "tie"
    return winner, dict(asks=asks, consistent=len(wins) == 1 if len(asks) > 1 else None)


def run_judge(spec, engine, name, ask, limit=None, only=None, log=print):
    """Judge what judge `name` has still to judge (at most `limit`, else the judge's own limit), storing each verdict as it comes. Returns
    dict(judged, failed, left, stopped): stopped names why it stopped early (a spent plan allowance)."""
    from stratamodels.evals import cloud

    j = find(spec, name)
    j = dict(j, resolved=resolve(engine, j))
    ds, outs, current, pairs = todo(spec, engine, j, only)
    cap = limit or j["limit"]
    left = len(pairs)
    if cap:
        pairs = pairs[:cap]
    log(f"{spec.name}: judge {name}: {left} pairs to judge, {len(pairs)} now")
    jk, judged, failed, stopped = judge_key(j, j["resolved"]), 0, 0, None
    for n, (i, a, b) in enumerate(pairs, 1):
        it = ds.items[i]
        ka, kb = current[(a, it["id"])], current[(b, it["id"])]
        if not _first(it["id"], ka, kb):
            a, b, ka, kb = b, a, kb, ka
        try:
            winner, detail = judge_pair(spec, j, it, a, outs[a][i][0], b, outs[b][i][0], ask)
        except cloud.QuotaError as e:
            stopped = f"quota: {e}"
            log(f"  quota: stopping after {n - 1} pairs; run again later. {e}")
            break
        except JudgeError as e:
            failed += 1
            log(f"  [{n}/{len(pairs)}] {it['id']} {a} v {b}: failed: {e}")
            continue
        with C.session(engine) as s:
            s.add(
                EvalJudgment(
                    experiment=spec.name,
                    item_id=it["id"],
                    a=a,
                    b=b,
                    winner=winner,
                    rater=j["name"][:64],
                    at=time.time(),
                    note=(detail["asks"][0]["reason"] or None) and detail["asks"][0]["reason"][:500],
                    a_key=ka,
                    b_key=kb,
                    judge=jk,
                    detail=detail,
                )
            )
            s.commit()
        judged += 1
        log(
            f"  [{n}/{len(pairs)}] {it['id']} {a} v {b}: {winner}{'' if detail['consistent'] is not False else ' (changed with the order)'}"
        )
    return dict(judged=judged, failed=failed, left=left - judged, stopped=stopped)


# ---- agreement ---------------------------------------------------------------------------------------------------------------------------
def kappa(pairs):
    """Cohen's kappa of two raters over the same pairs ([(person's category, judge's category)]): agreement beyond what their own habits (how
    often each says tie, or the first) would give by chance. 1 = always agree, 0 = chance, below 0 = worse than chance."""
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(1 for p, q in pairs if p == q) / n
    cats = {c for pq in pairs for c in pq}
    pe = sum((sum(1 for p, _ in pairs if p == c) / n) * (sum(1 for _, q in pairs if q == c) / n) for c in cats)
    return 1.0 if pe >= 1 else (po - pe) / (1 - pe)


def _share(k, n):
    v, lo, hi = S.wilson(k, n) if n else (float("nan"),) * 3
    return dict(value=v, low=lo, high=hi, n=n, k=k)


def agreement(people, verdicts):
    """How far a judge agrees with people on the pairs both judged. `people`, `verdicts`: [(pair id, a, b, winner, at)] (a person may judge a
    pair more than once: their latest counts). Categories are relative to the pair (its first candidate by name wins, the second, or tie)."""
    latest = {}
    for pid, a, b, w, _at in sorted(people, key=lambda r: r[4]):
        latest[pid] = (a, b, w)
    mine = {}
    for pid, a, b, w, _at in sorted(verdicts, key=lambda r: r[4]):
        mine[pid] = (a, b, w)

    def cat(a, b, w):
        return "tie" if w == "tie" else "first" if w == min(a, b) else "second"

    both = [(cat(*latest[p]), cat(*mine[p])) for p in mine if p in latest]
    decided = [(p, q) for p, q in both if p != "tie" and q != "tie"]
    tally = dict(
        same_winner=sum(1 for p, q in decided if p == q),
        opposite=sum(1 for p, q in decided if p != q),
        judge_tie=sum(1 for p, q in both if q == "tie" and p != "tie"),
        person_tie=sum(1 for p, q in both if p == "tie" and q != "tie"),
        both_tie=sum(1 for p, q in both if p == q == "tie"),
    )
    return dict(
        n=len(both),
        agree=_share(sum(1 for p, q in both if p == q), len(both)),
        decided=_share(
            tally["same_winner"], len(decided)
        ),  # of the pairs where both picked a winner, the share where they picked the same
        kappa=kappa(both),
        tally=tally,
    )


def summary(spec, engine, current, names, base):
    """The `judges` part of the analysis: per judge in the spec, its verdicts on current outputs, agreement with people, position consistency,
    and what it alone would conclude about each candidate against the baseline (core.judged_vs on its verdicts)."""
    rows = [r for r in _rows(engine, spec.name) if _counts(r, current)]
    people = [(_pair_id(r.item_id, r.a_key, r.b_key), r.a, r.b, r.winner, r.at) for r in rows if r.judge is None]
    pwin = {}
    for pid, _a, _b, w, _at in sorted(people, key=lambda r: r[4]):
        pwin[pid] = w
    out = {}
    for j in spec.judges:
        model = resolve(engine, j)
        jk = judge_key(j, model)
        mine = [r for r in rows if r.judge == jk]
        trip = [(r.a, r.b, r.winner) for r in mine]
        cons = [
            r.detail.get("consistent")
            for r in mine
            if isinstance(r.detail, dict) and r.detail.get("consistent") is not None
        ]
        ag = agreement(people, [(_pair_id(r.item_id, r.a_key, r.b_key), r.a, r.b, r.winner, r.at) for r in mine])
        disagree = [
            dict(item_id=r.item_id, a=r.a, b=r.b, person=pwin[pid], judge=r.winner, reason=r.note or "")
            for r in mine
            for pid in [_pair_id(r.item_id, r.a_key, r.b_key)]
            if pid in pwin and pwin[pid] != r.winner
        ]
        out[j["name"]] = dict(
            kind=j["kind"],
            model=model or j["model"],
            via=j["cli"] or "queue",
            role=j["role"],
            key=jk,
            judged=len(mine),
            with_people=ag["n"],
            agree=ag["agree"],
            decided=ag["decided"],
            kappa=ag["kappa"],
            tally=ag["tally"],
            consistent=_share(sum(1 for c in cons if c), len(cons)),
            comparisons={n: C.judged_vs(trip, n, base) for n in names if n != base},
            disagreements=disagree[:10],
        )
    return out


def verdicts_on(engine, spec, item_id):
    """Every judgment of one item, people's and judges', newest first (the judging page shows them after a person has judged). `current`: of
    the outputs as they are now and, for a judge's, by a judge in the spec as it is now (an old model's verdict on other answers is history)."""
    with C.session(engine) as s:
        rows = list(
            s.scalars(
                select(EvalJudgment)
                .where(EvalJudgment.experiment == spec.name, EvalJudgment.item_id == item_id)
                .order_by(EvalJudgment.id.desc())
            )
        )
    item = [it for it in C.load_dataset(spec.dataset, engine=engine).items if it["id"] == item_id]
    current = {(c.name, item_id): C.result_key(spec, c, item[0]) for c in spec.candidates} if item else {}
    active = {judge_key(j, resolve(engine, j)) for j in spec.judges}
    return [
        dict(
            a=r.a,
            b=r.b,
            winner=r.winner,
            rater=r.rater,
            judge=r.judge,
            at=r.at,
            note=r.note,
            detail=r.detail,
            a_key=r.a_key,
            b_key=r.b_key,
            current=_counts(r, current) and (r.judge is None or r.judge in active),
        )
        for r in rows
    ]
