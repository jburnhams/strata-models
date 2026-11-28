"""How sensitive a picture is: the probability (0-1) that it shows nudity or sexual content, from a small image classifier (role `nsfw`,
default model nsfw-marqo-384: a timm ViT-tiny at 384 px, Apache-2.0, two classes NSFW / SFW; about 40 pictures a second on a 3060 and well
under 1 GB of VRAM). Chosen by hand-checking 341 real photos against NudeNet (the previous pick) and two other classifiers: see docs/progress.md.

Why a probability and not a detector's box score: NudeNet gave 0.45-0.7 to plenty of ordinary pictures (a shirtless swimmer, a baby, a cake)
and 0.38-0.44 to explicit ones, so its number is not a fraction of anything; the classifier's is near 0 or near 1 with the doubtful pictures
(statues, art prints) in between.

model options: repo (the timm hub id, default Marqo/nsfw-image-detection-384), positive (the label that means sensitive, default NSFW),
resize ('crop', the default: the model's own shortest-side centre crop, which is what it was trained on; or 'square': the whole picture
squeezed to the model's size. Squeezing keeps the edges of a wide photo but is worse on the 341 pictures: ordinary pictures reach 0.70 and explicit
ones fall to 0.48, against 0.17 and 0.81 with the crop), batch (default 32). A wide photo's far edges are outside the crop; a sensitive
subject there is missed.
params: paths (a list of image files) or path (one); both answer the same way. A picture that cannot be read does not fail the job: its item
has `error` and no score.
Answer: {items: [{path, sensitive}], model}; with `path` instead of `paths`, {path, sensitive, model}."""

import os

import numpy as np

from stratamodels.runtimes import RunError, clip

REPO = "Marqo/nsfw-image-detection-384"
POSITIVE = "NSFW"
BATCH = 32


def load(model, models_dir):
    import timm
    import timm.data.config as tcfg
    import timm.data.transforms_factory as ttf
    import torch

    repo = str(model.options.get("repo", model.name or REPO))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = timm.create_model(f"hf_hub:{repo}", pretrained=True, cache_dir=os.path.join(models_dir, "nsfw")).eval().to(dev)
    if dev == "cuda":
        m = m.half()
    cfg = tcfg.resolve_data_config({}, model=m)
    names = list(m.pretrained_cfg.get("label_names") or [])
    positive = str(model.options.get("positive", POSITIVE))
    if positive not in names:
        raise RunError(f"{repo} has labels {names}, not {positive!r}: set the model option positive")
    resize = str(model.options.get("resize", "crop"))
    if resize not in ("square", "crop"):
        raise RunError("resize is 'square' or 'crop'")
    return dict(
        model=m,
        dev=dev,
        size=int(cfg["input_size"][-1]),
        mean=np.array(cfg["mean"], np.float32),
        std=np.array(cfg["std"], np.float32),
        index=names.index(positive),
        resize=resize,
        batch=int(model.options.get("batch", BATCH)),
        repo=repo,
        transform=ttf.create_transform(**cfg) if resize == "crop" else None,
    )


def _prepare(h, path):
    """One picture as a float32 (3, S, S) array for the model, or raises."""
    from PIL import Image, ImageOps

    if not isinstance(path, str) or not os.path.isfile(path):
        raise RunError(f"no such image: {path!r}")
    with Image.open(path) as im:
        im.draft("RGB", (h["size"] * 2, h["size"] * 2))  # JPEG: decode at a reduced size
        im = ImageOps.exif_transpose(im).convert("RGB")
        if h["transform"] is not None:
            return h["transform"](im).numpy()
        im = im.resize((h["size"], h["size"]), Image.Resampling.BICUBIC)
    a = np.asarray(im, np.float32) / 255.0
    return ((a - h["mean"]) / h["std"]).transpose(2, 0, 1)


def score(h, batch):
    """(N, 3, S, S) float32 -> N probabilities of the sensitive label."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(batch)).to(h["dev"])
    if h["dev"] == "cuda":
        x = x.half()
    with torch.no_grad():
        p = h["model"](x).float().softmax(-1)[:, h["index"]]
    return p.cpu().numpy()


def run(handle, params):
    unknown = set(params) - {"paths", "path"}
    if unknown:
        raise RunError(f"unknown params {sorted(unknown)}")
    single = params.get("path") is not None
    paths = [params["path"]] if single else params.get("paths")
    if not isinstance(paths, list) or not paths:
        raise RunError("give path, or paths (a list of image files)")
    items: list[dict] = []
    ok: list[tuple[int, np.ndarray]] = []
    for p in paths:
        try:
            ok.append((len(items), _prepare(handle, p)))
            items.append(dict(path=p))
        except RunError:
            raise
        except Exception as e:  # corrupt or unsupported file: that picture only
            items.append(dict(path=p, error=f"{type(e).__name__}: {e}"[:200]))
    for i in range(0, len(ok), handle["batch"]):
        chunk = ok[i : i + handle["batch"]]
        for (k, _), s in zip(chunk, score(handle, np.stack([a for _, a in chunk]))):
            items[k]["sensitive"] = round(float(s), 4)
    if single:
        return dict(**items[0], model=handle["repo"])
    return dict(items=items, model=handle["repo"])


def unload(handle):
    clip.unload(handle)
