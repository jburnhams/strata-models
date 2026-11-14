"""Printed text from a page picture with Tesseract 5 (the `tesseract` command, LSTM engine), role `ocr`. Asked for by strata-scan, which turns
scanned books into text: a page needs its lines and words with boxes (to draw the text over the picture, to find what is not text, to correct a
line by hand), and Tesseract gives those on the CPU in a second or two a page, so a book never waits for the GPU (vram_gb = 0: the CPU lane).
A vision-language reader may read old type better but gives no boxes; that is an eval to run against this, not a reason to wait for one.

name in models.toml is the language list (`eng`, `eng+fra`: traineddata files Tesseract has); options: psm (page segmentation, default 3 =
automatic), oem (default 1 = LSTM), binary (the command, else `tesseract` on PATH or Windows' default install folder), tessdata (a folder of
traineddata files, else Tesseract's own).

params:
  path     the page picture (PNG, JPEG, TIFF); `image` is the same (the name vision-language jobs use, so one eval dataset serves both)
  clean    steps to clean the page before reading it, in order: a list or "flatten,upscale" (runtimes/pageclean.py: flatten, contrast,
           sharpen, denoise, otsu, adaptive, upscale). Boxes are then in the cleaned picture's pixels (twice the size after upscale), and
           `width`/`height` say so; a `pdf` shows the cleaned picture
  lang     overrides the model's languages for this page
  psm      overrides the page segmentation mode (6 = one block of text, 4 = one column of varying sizes, 11 = sparse text)
  dpi      the scan's resolution, when the file does not say (Tesseract guesses otherwise, and small type suffers)
  pdf      a .pdf path to write as well: the picture with the text under it, invisible but searchable and selectable (a book's pages are
           joined into one searchable PDF by the app)
Answer: {width, height, text, lines: [{text, box: [x, y, w, h], conf, block, words: [{text, box, conf}]}], engine, lang, seconds}. conf is
0-100 (Tesseract's), a line's is the mean of its words'; boxes are pixels of the picture. Lines come in Tesseract's reading order; an empty
page has no lines and text ""."""

import csv
import io
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any

from PIL import Image

from stratamodels.runtimes import RunError, pageclean  # pyright: ignore[reportMissingTypeStubs]  (the package __init__ is not in the strict list yet)

WINDOWS_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # where the UB Mannheim installer and Chocolatey put it; it is not on PATH


def binary(option: str | None = None) -> str | None:
    """The tesseract command to run, or None when there is none."""
    if option:
        return option if os.path.isfile(option) else shutil.which(option)
    found = shutil.which("tesseract")
    if found:
        return found
    return WINDOWS_DEFAULT if os.name == "nt" and os.path.isfile(WINDOWS_DEFAULT) else None


def load(model: Any, models_dir: str) -> dict[str, Any]:
    opts: dict[str, Any] = dict(model.options)
    exe = binary(opts.get("binary"))
    if not exe:
        raise RunError(
            "tesseract is not installed (pacman -S tesseract tesseract-data-eng; apt install tesseract-ocr; brew install tesseract)"
        )
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30, check=False)
    except OSError as e:
        raise RunError(f"cannot run {exe}: {e}") from e
    version = (out.stdout or out.stderr).splitlines()[0].strip() if (out.stdout or out.stderr) else "tesseract"
    return dict(
        key=model.key,
        exe=exe,
        version=version,
        lang=model.name,
        psm=int(opts.get("psm", 3)),
        oem=int(opts.get("oem", 1)),
        tessdata=opts.get("tessdata"),
    )


def _box(r: dict[str, str]) -> list[int]:
    return [int(r["left"]), int(r["top"]), int(r["width"]), int(r["height"])]


def _union(boxes: list[list[int]]) -> list[int]:
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def parse_tsv(tsv: str) -> dict[str, Any]:
    """Tesseract's TSV (one row per page, block, paragraph, line and word) into lines of words. A line is keyed by (block, paragraph, line),
    because line numbers restart in every paragraph (and a second page is refused, since every number restarts there); words with no text (Tesseract emits empty level-5 rows for picture blocks) are dropped,
    and a line left with no words is dropped with them, so an engraving never turns into an empty line."""
    width = height = 0
    order: list[tuple[int, int, int]] = []
    words: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for r in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        level = int(r["level"] or 0)
        if level == 1:
            if (
                width or height
            ):  # a second page (a multipage TIFF): its numbers restart, so its lines would merge with the first's
                raise RunError("tesseract reads one page at a time: give it a single-page picture")
            width, height = int(r["width"]), int(r["height"])
        if level != 5:
            continue
        text = (r.get("text") or "").strip()
        if not text:
            continue
        key = (int(r["block_num"]), int(r["par_num"]), int(r["line_num"]))
        if key not in words:
            order.append(key)
            words[key] = []
        words[key].append(dict(text=text, box=_box(r), conf=round(max(float(r["conf"]), 0.0), 1)))
    lines: list[dict[str, Any]] = []
    for key in order:
        ws = words[key]
        lines.append(
            dict(
                text=" ".join(w["text"] for w in ws),
                box=_union([w["box"] for w in ws]),
                conf=round(sum(w["conf"] for w in ws) / len(ws), 1),
                block=key[0],
                words=ws,
            )
        )
    return dict(width=width, height=height, lines=lines, text="\n".join(ln["text"] for ln in lines))


def run(handle: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    path = params.get("path") or params.get("image")
    if not isinstance(path, str) or not os.path.isfile(path):
        raise RunError(f"no such picture: {path!r}")
    try:
        steps = pageclean.steps_of(params.get("clean"))
    except ValueError as e:
        raise RunError(str(e)) from e
    lang = str(params.get("lang") or handle["lang"])
    psm = int(params.get("psm") or handle["psm"])
    pdf = params.get("pdf")
    if pdf is not None and (
        not isinstance(pdf, str)
        or not pdf.lower().endswith(".pdf")
        or not os.path.isdir(os.path.dirname(os.path.abspath(pdf)))
    ):
        raise RunError(f"pdf must be a .pdf path in an existing folder: {pdf!r}")
    t0 = time.monotonic()
    dpi: float | None = float(params["dpi"]) if params.get("dpi") else None
    with tempfile.TemporaryDirectory(prefix="stratamodels-ocr-") as tmp:
        base = os.path.join(tmp, "page")
        if steps:
            try:
                with Image.open(path) as im:
                    if (
                        getattr(im, "n_frames", 1) > 1
                    ):  # cleaning would keep the first frame only, and parse_tsv's refusal never sees the rest
                        raise RunError(f"{path}: a picture with more than one page; give one page a job")
                    found = im.info.get("dpi")
                    dpi = dpi or (float(found[0]) if found and found[0] > 1 else None)
                    cleaned, dpi = pageclean.clean(im, steps, dpi)
            except OSError as e:
                raise RunError(f"tesseract failed on {path}: not a picture ({e})") from e
            path = os.path.join(tmp, "clean.png")
            cleaned.save(path, dpi=(dpi, dpi) if dpi else None)
        cmd = [handle["exe"], path, base, "-l", lang, "--psm", str(psm), "--oem", str(handle["oem"])]
        if dpi:
            cmd += ["--dpi", str(round(dpi))]
        if handle.get("tessdata"):
            cmd += ["--tessdata-dir", str(handle["tessdata"])]
        cmd += ["tsv"] + (["pdf"] if pdf else [])
        # One thread a page: the service runs up to cpu_jobs pages at once, and Tesseract's OpenMP threads on top of that oversubscribe
        # the CPU so badly that a page took 80 s instead of 2 (measured with four pages at once on four cores).
        env = dict(os.environ, OMP_THREAD_LIMIT="1")
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False, env=env)
        if out.returncode != 0 or not os.path.isfile(base + ".tsv"):
            raise RunError(f"tesseract failed on {path}: {(out.stderr or '').strip()[-500:]}")
        with open(base + ".tsv", encoding="utf8") as f:
            result = parse_tsv(f.read())
        if pdf:
            shutil.move(base + ".pdf", pdf)
            result["pdf"] = os.path.abspath(pdf)
    return dict(
        result,
        engine=handle["version"],
        lang=lang,
        psm=psm,
        clean=steps,
        seconds=round(time.monotonic() - t0, 3),
        model=handle["key"],
    )


def unload(handle: dict[str, Any]) -> None:
    handle.clear()  # nothing stays in memory between pages: each page is its own process
