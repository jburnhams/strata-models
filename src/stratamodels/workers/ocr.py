"""Printed text with boxes from a page picture by a detect-then-read model, role `ocr`, run as a venv-worker (stratamodels/runtimes/worker.py) in
its own venv (`scripts/setup_worker_venv.sh ocr`: torch and python-doctr, apart from the service's). The answer is the `ocr` role's, the same
shape as Tesseract's (runtimes/tesseract.py), so page-build, font-match and every app read either.

Why: Tesseract finds the lines of a page by its own layout analysis, which misses display type (an advert's last line, ward1874-leaf034,
was never read) and garbles worn capitals. docTR (Mindee, Apache-2.0) finds words with a text detector (DBNet) and reads each with a
recogniser (PARSeq): on that page it read every line, BLITZ where Tesseract read BUOITZ.

engine (models.toml `name`): "doctr", with options det (default db_resnet50) and reco (default parseq), the docTR architectures, and layout
(default false: also run docTR's layout detector, lw_detr_s, about 2 s more a page), and device ("cpu" keeps it off the GPU); weights download into <models>/doctr on first load. On the
GPU when there is one, else the CPU (4-14 s a page).

params: path or image (the page picture); straight (default true: assume the page is not rotated, faster and steadier on scans). Tesseract's
own (psm, clean, lang, dpi, pdf) are refused.
Answer: {width, height, text, lines: [{text, box: [x, y, w, h], conf, block, words: [{text, box, conf}]}], blocks: [{block, box, kind}],
unread: [{box, conf}], regions, engine, seconds}: boxes in the picture's pixels, conf 0-100 (a line's is the mean of its words'), lines in the
model's reading order, block numbered from 1. What the model knows beyond the text is kept, so the app can see what was not read: `blocks` is
docTR's grouping of lines (kind "text") and its artefacts (kind "picture": a mark it found and took for no text); `unread` is every word box
the detector found that the recogniser read as nothing; `regions` (only with the layout option) is the layout detector's [{box, kind, conf}],
kind its DocLayNet class in lower case ("text", "section-header", "picture", "page-header", ...). docTR has no paragraphs: no line has `para`."""

import os
import time

from protocol import ParamError, serve

M = {}


def load(model, models_dir):
    engine = model.get("name") or "doctr"
    if engine != "doctr":
        raise ValueError(f"no ocr engine {engine!r} (doctr)")
    os.environ.setdefault("DOCTR_CACHE_DIR", os.path.join(models_dir, "doctr"))
    import torch
    from doctr.models import ocr_predictor

    opts = model.get("options") or {}
    det, reco = opts.get("det", "db_resnet50"), opts.get("reco", "parseq")
    layout = bool(opts.get("layout", False))
    cuda = (
        torch.cuda.is_available() and opts.get("device") != "cpu"
    )  # device = "cpu": a model run inside a CPU-lane job (runtimes/ocrfix.py)
    M["key"] = model.get("key")
    M["engine"] = f"doctr {det}+{reco}" + ("+layout" if layout else "")
    M["make"] = lambda straight: ocr_predictor(
        det_arch=det, reco_arch=reco, pretrained=True, assume_straight_pages=straight, detect_layout=layout
    ).to("cuda" if cuda else "cpu")
    M["models"] = {True: M["make"](True)}
    return dict(device="cuda" if cuda else "cpu", engine=M["engine"])


def _box(geometry, w, h):
    (x0, y0), (x1, y1) = geometry
    return [round(x0 * w), round(y0 * h), round((x1 - x0) * w), round((y1 - y0) * h)]


def answer(page, engine):
    """A docTR page (its export(): dimensions and blocks of lines of words, geometry in fractions of the page) as the `ocr` answer."""
    h, w = page["dimensions"]
    lines, blocks, unread, n = [], [], [], 0
    for block in page.get("blocks") or []:
        n += 1
        b = n
        blocks.append(dict(block=b, box=_box(block["geometry"], w, h), kind="text"))
        for art in block.get("artefacts") or []:
            n += 1
            blocks.append(dict(block=n, box=_box(art["geometry"], w, h), kind="picture"))
        for ln in block.get("lines") or []:
            unread += [
                dict(box=_box(wd["geometry"], w, h), conf=round(100 * float(wd.get("confidence", 0.0)), 1))
                for wd in ln.get("words") or []
                if not str(wd.get("value") or "").strip()
            ]
            words = [
                dict(
                    text=wd["value"],
                    box=_box(wd["geometry"], w, h),
                    conf=round(100 * float(wd.get("confidence", 0.0)), 1),
                )
                for wd in ln.get("words") or []
                if str(wd.get("value") or "").strip()
            ]
            if not words:
                continue
            lines.append(
                dict(
                    text=" ".join(wd["text"] for wd in words),
                    box=_box(ln["geometry"], w, h),
                    conf=round(sum(wd["conf"] for wd in words) / len(words), 1),
                    block=b,
                    words=words,
                )
            )
    out = dict(
        width=w,
        height=h,
        text="\n".join(ln["text"] for ln in lines),
        lines=lines,
        blocks=blocks,
        unread=unread,
        engine=engine,
    )
    if page.get("layout") is not None:
        out["regions"] = [
            dict(
                box=_box(r["geometry"], w, h),
                kind=str(r.get("type") or "").lower(),
                conf=round(100 * float(r.get("confidence") or 0.0), 1),
            )
            for r in page["layout"]
        ]
    return out


def run(params):
    path = params.get("path") or params.get("image")
    if not path or not os.path.isfile(str(path)):
        raise ParamError(f"no such picture: {path!r}")
    # Tesseract's own params are refused rather than ignored (an eval asking docTR for psm 4 would compare nothing); anything else is an eval
    # item's other input (the page-layout dataset's `ocr`, a Tesseract reading), as the Tesseract runtime ignores it
    theirs = set(params) & {"psm", "oem", "clean", "lang", "dpi", "pdf"}
    if theirs:
        raise ParamError(f"Tesseract's params, which docTR has no use for: {sorted(theirs)}")
    straight = params.get("straight", True) is not False
    if straight not in M["models"]:
        M["models"][straight] = M["make"](straight)
    from doctr.io import DocumentFile

    t0 = time.monotonic()
    result = M["models"][straight](DocumentFile.from_images(str(path)))
    out = answer(result.export()["pages"][0], M["engine"])
    out["seconds"] = round(time.monotonic() - t0, 3)
    return out


if __name__ == "__main__":
    serve(load, run)
