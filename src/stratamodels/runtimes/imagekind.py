"""What kind of picture this is (runtime `image-kind`, role `image-kind`): one of KINDS, fine-grained (engraving, woodcut, halftone, hand-
coloured plate, marbled pattern...), and the broad GROUP it falls in (illustration, photo, or other: a picture that is neither, such as a
binding, a pattern, handwriting or a stamp), with whether it is line art (all ink and paper, nothing in between: the pictures a tracer turns
into a good SVG, see runtimes/vectorise.py) and whether it has colour. strata-scan asks it of every picture it cuts from a book, to decide
what to vectorise and what to keep as a JPEG.

Three engines (model options `engine`), measured against each other by eval image-kind (evals/specs/image-kind.toml):
  rules   numpy only: how much of the picture is mid-tone, colour, a halftone screen's peak in the spectrum, edges. Quick and dull; the baseline.
  clip    zero-shot CLIP (open_clip, as runtimes/clip.py): the picture against a few prompts per kind, the best kind's prompts win. options
          `arch` ("ViT-B-32:laion2b_s34b_b79k"), `device` ("cpu" by default: 150 pictures a minute, and it never takes the card)
  vlm     a vision-language model of the registry (options `vlm`, a model key) answers preset image-kind-v1 (schema-checked JSON), asked
          through runtimes/delegate.py (in this process like page-build's grouper, or as a job on the service from `eval run --local`)
params: image (a path; required); vlm also takes `context` (what the app knows: the book, the page's caption) and `max_tokens`.
Answer: {kind, group, line_art, colour, scores?: {kind: score} (rules, clip), why?: (vlm), seconds, cost_usd? (a cloud vlm)}."""

import time
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np

from stratamodels.runtimes import RunError

# kind: (group, line art, what it is). The kind is how the picture was MADE, so a modern book's halftone of an 1830 engraving is an engraving
# (its lines are still lines), and a photograph of a puppet is a photograph however it was printed. `line` says whether the kind is drawn in
# ink lines on paper with no tone between, which is what a tracer keeps exactly; hand-coloured plates are line art with washes over them.
KINDS: dict[str, tuple[str, bool, str]] = {
    "line-drawing": ("illustration", True, "a pen-and-ink drawing or outline: clean lines, little or no shading"),
    "engraving": (
        "illustration",
        True,
        "a copper or steel engraving or an etching: fine lines, shading by parallel and cross-hatched lines",
    ),
    "wood-engraving": (
        "illustration",
        True,
        "a wood engraving: white lines cut from black, fine tints, as in Victorian books and magazines",
    ),
    "woodcut": ("illustration", True, "a woodcut or chapbook cut: bold, coarse black lines and areas, simple shapes"),
    "hand-coloured": (
        "illustration",
        True,
        "a black-line print (engraving, etching, lithograph) coloured by hand with washes",
    ),
    "lithograph": ("illustration", False, "a lithograph or crayon drawing: soft grainy tones as well as lines"),
    "colour-print": (
        "illustration",
        False,
        "printed in colours: a chromolithograph, colour woodblock or modern colour illustration",
    ),
    "tonal-drawing": ("illustration", False, "a pencil, charcoal, wash or watercolour drawing with continuous tones"),
    "painting": ("illustration", False, "a reproduction of a painting (oil, watercolour, gouache)"),
    "diagram": ("illustration", True, "a diagram, map, plan, chart or technical drawing"),
    "ornament": (
        "illustration",
        True,
        "a printer's ornament: headpiece, tailpiece, border, rule, decorated initial, fleuron",
    ),
    "silhouette": ("illustration", True, "a silhouette or shadow figure: flat black shapes"),
    "photograph": ("photo", False, "a photograph with continuous tones (black and white, sepia or colour)"),
    "halftone": ("photo", False, "a photograph printed through a halftone screen: a pattern of dots visible up close"),
    "pattern": ("other", False, "a repeating or decorative surface: marbled paper, endpaper, textile, wallpaper"),
    "binding": ("other", False, "part of the book itself: cloth or leather cover, spine, page edges"),
    "handwriting": ("other", True, "handwriting: an inscription, signature, note or manuscript"),
    "stamp": ("other", False, "a stamp, bookplate, library label, seal or barcode"),
    "text": ("other", True, "printed text only (a page or block of type, a caption)"),
    "blank": ("other", False, "blank or nearly blank paper"),
}
GROUPS = ("illustration", "photo", "other")
PRESET = "image-kind-v1"


def kind_of(kind: str) -> dict[str, Any]:
    """{kind, group, line_art} of a kind name (an unknown name is `other`, not line art, so a bad answer is never vectorised)."""
    g, line, _ = KINDS.get(kind, ("other", False, ""))
    return dict(kind=kind if kind in KINDS else "unknown", group=g, line_art=line)


def load(model: Any, models_dir: str) -> dict[str, Any]:
    o = cast("Mapping[str, Any]", model.options or {})
    engine = str(o.get("engine", "rules"))
    if engine not in ("rules", "clip", "vlm"):
        raise RunError(f"model {model.key}: engine is rules, clip or vlm, not {engine!r}")
    h: dict[str, Any] = dict(key=model.key, engine=engine, options=dict(o), models_dir=models_dir)
    if engine == "clip":
        from stratamodels.runtimes import imagekind_clip

        imagekind_clip.load(
            h, str(o.get("arch", "ViT-B-32:laion2b_s34b_b79k")), str(o.get("device", "cpu")), models_dir
        )
    if engine == "vlm" and not o.get("vlm"):
        raise RunError(f"model {model.key}: engine vlm names no vision-language model (options.vlm)")
    return h


def unload(handle: dict[str, Any]) -> None:
    undo = handle.get("vlm_unload")
    if callable(undo):
        undo()
    handle.clear()


def run(handle: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(params) - {"image", "context", "max_tokens"}
    if unknown:
        raise RunError(f"unknown params {sorted(unknown)}")
    path = params.get("image")
    import os

    if not isinstance(path, str) or not os.path.isfile(path):
        raise RunError(f"no such image: {path!r}")
    t0 = time.monotonic()
    if handle["engine"] == "rules":
        out = rules(_read(path))
    elif handle["engine"] == "clip":
        from stratamodels.runtimes import imagekind_clip

        out = imagekind_clip.run(handle, path)
    else:
        out = _vlm(handle, path, params)
    return dict(out, seconds=round(time.monotonic() - t0, 3))


# ---- rules --------------------------------------------------------------------------------------------------------------------------------
def _read(path: str, side: int = 1024) -> np.ndarray:
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((side, side), Image.Resampling.LANCZOS)
        return np.asarray(im)


def features(rgb: np.ndarray) -> dict[str, float]:
    """What the rules read, each 0..1-ish: `mid` the share of pixels between ink and paper (tone); `ink` the share of dark pixels; `colour`
    the colourfulness of the picture apart from its paper's own tint; `screen` how strongly the spectrum has the isolated peaks of a halftone
    screen; `edges` the share of strong edges; `spread` the grey levels' standard deviation (a blank page has almost none)."""
    f = rgb.astype(np.float64)
    grey = f @ np.array([0.299, 0.587, 0.114])
    paper = float(np.percentile(grey, 95))
    inkv = float(np.percentile(grey, 3))
    span = max(paper - inkv, 1.0)
    t = (grey - inkv) / span
    flat = span < 24  # no ink to speak of: one tone (a blank leaf) is neither ink nor mid-tone
    mid = 0.0 if flat else float(((t > 0.25) & (t < 0.75)).mean())
    ink = 0.0 if flat else float((t < 0.35).mean())
    # colourfulness (Hasler and Suesstrunk 2003) of the pixels minus the paper's tint: a yellowed page is not a coloured picture
    tint = np.median(f[grey >= np.percentile(grey, 80)], axis=0) if grey.size else np.zeros(3)
    g = f - (tint - tint.mean())
    rg, yb = g[..., 0] - g[..., 1], 0.5 * (g[..., 0] + g[..., 1]) - g[..., 2]
    colour = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())) / 100.0
    # a halftone screen: in the middle of the picture, the strongest non-central spectrum peak against the spectrum's typical level
    h, w = grey.shape
    c = grey[h // 4 : h // 4 + min(256, h // 2), w // 4 : w // 4 + min(256, w // 2)]
    screen = 0.0
    if min(c.shape) >= 64:
        spec = np.abs(np.fft.fftshift(np.fft.fft2(c - c.mean())))
        cy, cx = spec.shape[0] // 2, spec.shape[1] // 2
        yy, xx = np.ogrid[: spec.shape[0], : spec.shape[1]]
        r = np.hypot(yy - cy, xx - cx)
        ring = spec[(r > min(cy, cx) * 0.25) & (r < min(cy, cx) * 0.95)]
        screen = float(ring.max() / (np.median(ring) + 1e-9)) / 100.0 if ring.size else 0.0
    gy, gx = np.gradient(grey)
    edges = float((np.hypot(gx, gy) > 40).mean())
    return dict(mid=mid, ink=ink, colour=colour, screen=screen, edges=edges, spread=float(grey.std()) / 128.0)


def rules(rgb: np.ndarray) -> dict[str, Any]:
    """A kind from features() by thresholds set on the plates of the eval's three books (2026-11-28): a baseline to beat, not a classifier."""
    ft = features(rgb)
    if ft["spread"] < 0.06 or ft["ink"] < 0.003:
        kind = "blank"
    elif ft["colour"] > 0.25:
        kind = (
            "hand-coloured"
            if ft["mid"] < 0.35 and ft["edges"] > 0.08
            else ("colour-print" if ft["edges"] > 0.06 else "photograph")
        )
    elif ft["screen"] > 0.5 and ft["mid"] > 0.3:
        kind = "halftone"
    elif ft["mid"] > 0.45:
        kind = "photograph" if ft["edges"] < 0.05 else "tonal-drawing"
    elif ft["ink"] > 0.35:
        kind = "woodcut"
    else:
        kind = "engraving" if ft["edges"] > 0.12 else "line-drawing"
    return dict(kind_of(kind), colour=ft["colour"] > 0.25, features={k: round(v, 4) for k, v in ft.items()})


# ---- vlm ----------------------------------------------------------------------------------------------------------------------------------
def _delegate(h: dict[str, Any]) -> Callable[[dict[str, Any]], Any]:
    if "vlm" not in h:
        from stratamodels.runtimes import delegate

        h["vlm"], h["vlm_unload"] = delegate.caller(
            str(h["options"]["vlm"]),
            h.get("models_dir", ""),
            h["options"].get("via"),
            h["options"].get("vlm_options"),  # the model's load options changed (a variant: Qwen with a longer context)
        )
    return cast("Callable[[dict[str, Any]], Any]", h["vlm"])


def _vlm(h: dict[str, Any], path: str, params: Mapping[str, Any]) -> dict[str, Any]:
    r = _delegate(h)(
        dict(
            preset=str(h["options"].get("preset", PRESET)),
            context=str(params.get("context") or "none"),
            image=path,
            max_tokens=int(params.get("max_tokens") or h["options"].get("max_tokens", 1024)),
        )
    )
    r = cast("Mapping[str, Any]", r if isinstance(r, Mapping) else {"text": r})
    ans = r.get("json")
    if not isinstance(ans, Mapping):
        import json

        try:
            ans = json.loads(str(r.get("text") or "").strip().removeprefix("```json").removesuffix("```"))
        except ValueError:
            raise RunError(f"the model's answer is not JSON: {str(r.get('text'))[:200]!r}")
    ans = cast("Mapping[str, Any]", ans)
    out = dict(kind_of(str(ans.get("kind", ""))), colour=bool(ans.get("colour")), why=str(ans.get("why", ""))[:500])
    if r.get("cost_usd") is not None:
        out["cost_usd"] = r.get("cost_usd")
    return out
