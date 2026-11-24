"""Printed text with boxes from a page picture, role `ocr`: a reader that gives boxes (docTR, Tesseract) and then a vision-language model that
corrects each of its lines against the page, so the answer has the reader's boxes and the model's text. CPU lane (vram_gb = 0): the reader
named here runs on the CPU in this process, the corrector is a cloud call.

Why (2026-11-24, eval ocr-lines on 20 hard pages): the Claude readers read best (Sonnet found every printed line, 0.01 % character errors;
Haiku 99.8 % of lines, 0.3 %) but give no boxes, which a rebuilt page and a searchable PDF need; docTR gives boxes round 99 % of the print
with 1.1 % character errors. This is both in one answer, so an app asks `ocr` once and the evals choose.

options (models.toml):
  reader      the model that reads the page first, a model of the registry with role `ocr` (e.g. doctr-db-parseq-cpu)
  fixer       the vision-language model that corrects its lines (e.g. claude-haiku-5-5-api)
  preset      the fixer's preset (default page-text-fix-v1: shown the page and the numbered lines, it answers {lines: [{n, text}]} for the
              lines it changes; an empty text means no print there)
  max_tokens  the fixer's answer budget (default 8192)

params:
  path or image   the page picture
  ocr             an `ocr` answer to correct instead of reading the page (as JSON, JSON text or a path): Tesseract's, say, which the
                  page-layout eval dataset carries
  anything else is the reader's (`straight` for docTR, `psm` for Tesseract)
Answer: the reader's answer (lines with boxes, blocks, unread...) with each corrected line's text and words replaced: a line keeps its box;
its words keep theirs when the number of words is unchanged, else the line's box is shared among the new words by their length. A line the
fixer emptied is left out (its box joins `unread`, conf 0). Beside it: `reader` (its engine), `fixer` ({model, seconds, cost_usd, changed}
or {model, seconds, error}: a failed call answers the reader's text as it is, never fails the job), cost_usd and seconds."""

import json
import os
import re
import time

from stratamodels.runtimes import RunError

PRESET = "page-text-fix-v1"


def load(model, models_dir):
    opts = dict(model.options or {})
    for k in ("reader", "fixer"):
        if not opts.get(k):
            raise RunError(f"model {model.key} names no {k} in models.toml (options.{k})")
    return dict(key=model.key, options=opts, models_dir=models_dir, loaded={})


def unload(handle):
    for rt, h in (handle.get("loaded") or {}).values():
        rt.unload(h)
    handle.clear()


def _model(handle, key):
    """(runtime, handle) of a registry model, loaded once in this process as page-build loads its layout model."""
    if key not in handle["loaded"]:
        from stratamodels import registry, runtimes, settings

        eng = settings.engine()
        reg = registry.Registry.from_db(eng) if eng is not None else registry.Registry.load()
        m = reg.models.get(key)
        if m is None:
            raise RunError(f"the model {key!r} is not in the registry")
        rt = runtimes.get(m.runtime)
        handle["loaded"][key] = (rt, rt.load(m, handle["models_dir"]))
    return handle["loaded"][key]


def _json(v, what):
    if isinstance(v, dict):
        return v
    if not isinstance(v, str) or not v.strip():
        raise RunError(f"no {what}")
    t = v.strip()
    if t.startswith("{"):
        try:
            return json.loads(t)
        except ValueError as e:
            raise RunError(f"{what} is not valid JSON: {e}") from e
    if not os.path.isfile(t):
        raise RunError(f"no such {what}: {t[:200]!r}")
    with open(t, encoding="utf8") as f:
        return json.load(f)


def numbered(ocr):
    """The reader's lines as page-text-fix-v1 is shown them: number, block, box [x0, y0, x1, y1] in thousandths of the page, text."""
    w, h = float(ocr.get("width") or 0), float(ocr.get("height") or 0)
    if not w or not h:
        raise RunError("the ocr answer has no page size")
    out = []
    for n, ln in enumerate(ocr.get("lines") or [], 1):
        x, y, bw, bh = (float(v) for v in ln["box"])
        box = ",".join(str(round(v)) for v in (x * 1000 / w, y * 1000 / h, (x + bw) * 1000 / w, (y + bh) * 1000 / h))
        out.append(f"{n} b{int(ln.get('block') or 0)} [{box}] {str(ln.get('text') or '').strip()}")
    return "\n".join(out)


def corrections(answer):
    """{line number: text} from the fixer's {lines: [{n, text}]} (first answer for a number wins; a number that is not whole is skipped)."""
    out = {}
    for c in (answer or {}).get("lines") or []:
        if not isinstance(c, dict):
            continue
        try:
            n = int(c.get("n"))
        except (TypeError, ValueError):
            continue
        out.setdefault(n, " ".join(str(c.get("text") or "").split()))
    return out


def respell(line, text):
    """The line with `text`: its words' boxes kept when the count is the same, else the line's box shared by the new words' lengths (a space
    counted as a letter), each at the line's confidence."""
    new = text.split()
    old = line.get("words") or []
    if len(new) == len(old):
        words = [dict(wd, text=t) for wd, t in zip(old, new)]
    else:
        x, y, w, h = (float(v) for v in line["box"])
        total = sum(len(t) for t in new) + max(0, len(new) - 1)
        words, at = [], 0
        for t in new:
            x0 = x + w * at / total
            at += len(t)
            words.append(
                dict(
                    text=t,
                    box=[round(x0), round(y), max(1, round(w * len(t) / total)), round(h)],
                    conf=line.get("conf", 0.0),
                )
            )
            at += 1
    return dict(line, text=text, words=words)


def apply(ocr, fixes):
    """The ocr answer with the fixer's corrections: (answer, lines changed)."""
    lines, unread, changed = [], list(ocr.get("unread") or []), 0
    for n, ln in enumerate(ocr.get("lines") or [], 1):
        if n not in fixes or fixes[n] == " ".join(str(ln.get("text") or "").split()):
            lines.append(ln)
            continue
        changed += 1
        if not fixes[n]:
            unread.append(dict(box=list(ln["box"]), conf=0.0))
            continue
        lines.append(respell(ln, fixes[n]))
    return dict(ocr, lines=lines, unread=unread, text="\n".join(ln["text"] for ln in lines)), changed


def run(handle, params):
    t0 = time.monotonic()
    image = params.get("path") or params.get("image")
    if not isinstance(image, str) or not os.path.isfile(image):
        raise RunError(f"no such picture: {image!r}")
    opts = handle["options"]
    if params.get("ocr"):
        ocr = _json(params["ocr"], "ocr answer")
    else:
        rt, h = _model(handle, opts["reader"])
        rest = {k: v for k, v in params.items() if k not in ("path", "image", "ocr")}
        ocr = rt.run(h, dict(rest, path=image))
    if not isinstance(ocr, dict) or not isinstance(ocr.get("lines"), list):
        raise RunError("the reader gave no lines")
    fixer, t1 = str(opts["fixer"]), time.monotonic()
    if not ocr["lines"]:
        out, info = ocr, dict(model=fixer, seconds=0.0, changed=0)
    else:
        try:
            frt, fh = _model(handle, fixer)
            r = frt.run(
                fh,
                dict(
                    preset=str(opts.get("preset") or PRESET),
                    context=numbered(ocr),
                    image=image,
                    max_tokens=int(opts.get("max_tokens", 8192)),
                ),
            )
            r = r if isinstance(r, dict) else {"text": r}
            ans = r.get("json") if isinstance(r.get("json"), dict) else _answer(str(r.get("text") or ""))
            out, changed = apply(ocr, corrections(ans))
            info = dict(
                model=fixer, seconds=round(time.monotonic() - t1, 2), cost_usd=r.get("cost_usd"), changed=changed
            )
        except Exception as e:  # budget, network, a bad answer: the reader's text as it is
            out, info = (
                ocr,
                dict(model=fixer, seconds=round(time.monotonic() - t1, 2), error=f"{type(e).__name__}: {e}"[:500]),
            )
    return dict(
        out,
        reader=ocr.get("engine"),
        fixer=info,
        engine=f"{ocr.get('engine')} + {fixer}",
        cost_usd=info.get("cost_usd") or 0.0,
        seconds=round(time.monotonic() - t0, 3),
        model=handle["key"],
    )


def _answer(text):
    t = re.sub(r"^```\w*\n|```\s*$", "", text.strip())
    try:
        return json.loads(t)
    except ValueError as e:
        raise RunError(f"the fixer's answer is not JSON: {text[:200]!r}") from e
