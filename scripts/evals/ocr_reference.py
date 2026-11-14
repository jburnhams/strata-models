"""Have one strong model read a few chosen pages once, as the reference the other readers are scored against (Jonathan, 2026-11-14: "a very
limited number of powerful calls (ie opus) to get high quality reading and then compare others to that"). Typing pages by hand does not scale
past the ten in ocr_pages/; a strong vision model's reading of the worst pages is close enough to rank the others, and two hand-typed pages
read the same way say how close (its cer_exact against the typed text, in SOURCES.json).

    uv run python scripts/evals/ocr_reference.py --strata-scan http://127.0.0.1:8368 --book ward1874=1 --book collier1828=2 \\
        --item collier1828-p088=misread --item ward1874-p019=check ... [--model claude-opus-5-5-api] [--max-usd 1.0]

Each page is fetched from strata-scan, read through strata-models (role vision-language, preset ocr-read-v1, the same instructions the local readers get)
and written to scripts/evals/ocr_hard/<item>.txt; SOURCES.json keeps, per item, the kind of error that got it picked, the model, its cost and
the date. A page that already has a reading is not read again; a reading cut off at max_tokens is not saved, so it is tried again next time. An empty
reading is kept: the right answer for a page of ornaments read as text is no text. Stops before the next call once --max-usd has been spent. Then build the
dataset with `ocr_pages.py --refs scripts/evals/ocr_hard --name ocr-hard` (evals/specs/ocr-hard.toml)."""

import argparse
import datetime
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "client"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strata_models_client as sm
from ocr_pages import HERE as TYPED
from ocr_pages import from_strata_scan

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_hard")
TRUNCATED = (
    "max_tokens",
    "MAX_TOKENS",
    "length",
)  # stop reasons for a cut-off answer: Claude, Gemini, OpenAI-compatible (NIM)
PARAMS = dict(system_preset="ocr-read-v1", prompt="Transcribe this page.", max_tokens=4096, effort="high")


LIMIT = 10 * 1024 * 1024  # the Claude API's largest image, counted as base64


def fit(path, long_edge=2576):
    """A copy of the page small enough to send: under the API's 10 MB (base64) and no longer than `long_edge` on its long side (the API scales
    bigger pictures down anyway, so nothing it would read is lost). A page that already fits is sent as it is."""
    if not os.path.exists(path) or os.path.getsize(path) * 4 / 3 < LIMIT * 0.95:
        return path
    from PIL import Image

    im = Image.open(path).convert("L")
    if max(im.size) > long_edge:
        k = long_edge / max(im.size)
        im = im.resize((round(im.width * k), round(im.height * k)), Image.Resampling.LANCZOS)
    out = path[: -len(".png")] + ".api.png"
    im.save(out, optimize=True)
    if os.path.getsize(out) * 4 / 3 >= LIMIT * 0.95:
        out = out[: -len(".png")] + ".jpg"
        im.save(out, quality=92)
    return out


def exact_cer(ref, hyp):
    """cer_exact of a reading against a hand-typed page (the eval's own metric, imported late: the script runs without the service's code too)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
    from stratamodels.evals import metrics as M

    r = M.exact_chars(ref)
    return M.edits(r, M.exact_chars(hyp)) / len(r) if r else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--strata-scan", required=True)
    ap.add_argument(
        "--book", action="append", default=[], help="NAME=ID: which strata-scan book is ward1874 / collier1828"
    )
    ap.add_argument(
        "--item", action="append", default=[], help="ITEM=KIND: a page to read (ward1874-p019) and why it was picked"
    )
    ap.add_argument("--model", default="claude-opus-5-5-api")
    ap.add_argument(
        "--max-usd", type=float, default=1.0, help="stop before the next call once this much has been spent"
    )
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    src_path = os.path.join(a.out, "SOURCES.json")
    sources = json.load(open(src_path, encoding="utf8")) if os.path.exists(src_path) else {}
    want = dict(i.split("=", 1) if "=" in i else (i, "") for i in a.item)
    todo = [i for i in want if not os.path.exists(os.path.join(a.out, f"{i}.txt"))]
    spent = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        pages = from_strata_scan(a.strata_scan.rstrip("/"), dict(b.split("=", 1) for b in a.book), tmp, items=todo)
        for item in todo:
            if item not in pages:
                print(f"{item}: no such page in strata-scan", file=sys.stderr)
                continue
            if spent >= a.max_usd:
                print(f"stopping: ${spent:.4f} spent, the cap is ${a.max_usd:.2f}", file=sys.stderr)
                break
            # role vision-language: the cloud runtime reads the same params under either role, and every cloud model serves it
            r = sm.run(
                "vision-language",
                dict(PARAMS, image=fit(pages[item][0])),
                app="evals",
                priority="normal",
                model=a.model,
            )
            text = (r.get("text") or "").strip()
            spent += r.get("cost_usd") or 0.0
            if (
                r.get("finish") in TRUNCATED
            ):  # a partial page as the reference would count the rest of a full reading as insertions
                print(
                    f"{item}: the reading stopped at max_tokens ({r.get('finish')}); not saved, run again with a higher limit",
                    file=sys.stderr,
                )
                continue
            with open(os.path.join(a.out, f"{item}.txt"), "w", encoding="utf8") as f:
                f.write(text + "\n")
            typed = os.path.join(TYPED, f"{item}.txt")
            check = exact_cer(open(typed, encoding="utf8").read(), text) if os.path.exists(typed) else None
            sources[item] = dict(
                kind=want[item],
                model=a.model,
                cost_usd=r.get("cost_usd"),
                tokens_in=r.get("tokens_in"),
                tokens_out=r.get("tokens_out"),
                finish=r.get("finish"),
                date=datetime.date.today().isoformat(),
                **({"cer_exact_vs_typed": round(check, 4)} if check is not None else {}),
            )
            with open(src_path, "w", encoding="utf8") as f:
                json.dump(dict(sorted(sources.items())), f, indent=1)
            print(
                f"{item}: {len(text)} chars, ${r.get('cost_usd') or 0:.4f}"
                + (f", cer_exact {check:.2%} against the typed page" if check is not None else "")
            )
    print(f"spent ${spent:.4f} on {a.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
