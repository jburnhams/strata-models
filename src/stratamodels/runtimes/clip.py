"""Image and text embeddings with open_clip (CLIP ViT-B/32 laion2b, ViT-bigG, SigLIP, ...): one vector per picture or prompt, L2-normalised, so
a dot product is a cosine similarity. Zero-shot labels are a softmax over the similarities to a list of prompts.

name in models.toml is "<architecture>:<pretrained tag>" (e.g. "ViT-B-32:laion2b_s34b_b79k"); weights are fetched once into <models>/open_clip.

params (one or more of):
  paths       image files
  array       {path, key?, bgr?}: an .npy / .npz file holding images as uint8 (N, H, W, 3) (strata-stills keeps its previews like that);
              key names the array in an .npz (default the first), bgr true when the channels are OpenCV's order
  texts       prompts to embed
  resize      'crop' (the model's own preprocessing: shortest side, centre crop; the default) or 'square' (the whole picture squeezed to the
              model's size, as strata-stills' lr_clip.py always did)
  out         an .npz file to write {images, texts} into instead of answering with the numbers (thousands of 512-float vectors are too big
              for a job row); the answer then says the shapes
Answer: {images: [[...]], texts: [[...]], dim} or, with out, {out, images: n, texts: n, dim}."""

import logging
import os

import numpy as np

from stratamodels.runtimes import RunError

log = logging.getLogger(__name__)

BATCH = 128


def load(model, models_dir):
    import open_clip
    import torch

    arch, _, tag = model.name.partition(":")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, _, pre = open_clip.create_model_and_transforms(
        arch, pretrained=tag or None, cache_dir=os.path.join(models_dir, "open_clip")
    )
    m = m.to(dev).eval()
    if dev == "cuda" and model.options.get("half", True):
        m = m.half()
    size = pre.transforms[0].size if hasattr(pre.transforms[0], "size") else 224
    size = size if isinstance(size, int) else size[0]
    norm = next(t for t in pre.transforms if t.__class__.__name__ == "Normalize")
    return dict(
        model=m,
        pre=pre,
        tok=open_clip.get_tokenizer(arch),
        dev=dev,
        size=size,
        mean=list(norm.mean),
        std=list(norm.std),
    )


def _array(spec):
    p = spec.get("path") if isinstance(spec, dict) else None
    if not isinstance(p, str) or not os.path.isfile(p):
        raise RunError(f"no such array file: {p!r}")
    z = np.load(p)
    if isinstance(z, np.lib.npyio.NpzFile):
        key = spec.get("key") or z.files[0]
        if key not in z.files:
            raise RunError(f"{p} has no array {key!r} (it has {', '.join(z.files)})")
        a = z[key]
    else:
        a = z
    if a.ndim != 4 or a.shape[-1] != 3 or a.dtype != np.uint8:
        raise RunError(f"the array must be uint8 (N, H, W, 3), not {a.dtype} {a.shape}")
    return a[..., ::-1] if spec.get("bgr") else a


def _square(h, rgb_u8):
    """The whole picture squeezed to the model's size with bicubic interpolation, then the model's normalisation (strata-stills' way)."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(rgb_u8)).permute(0, 3, 1, 2).float().div(255).to(h["dev"])
    x = torch.nn.functional.interpolate(x, size=h["size"], mode="bicubic", align_corners=False).clamp(0, 1)
    mean = torch.tensor(h["mean"], device=h["dev"]).view(1, 3, 1, 1)
    std = torch.tensor(h["std"], device=h["dev"]).view(1, 3, 1, 1)
    return (x - mean) / std


def _crop(h, rgb_u8):
    import torch
    from PIL import Image

    return torch.stack([h["pre"](Image.fromarray(a)) for a in rgb_u8]).to(h["dev"])


def _encode_images(h, rgb_u8, resize):
    import torch

    out = []
    dtype = next(h["model"].parameters()).dtype
    for s in range(0, len(rgb_u8), BATCH):
        x = (_square if resize == "square" else _crop)(h, rgb_u8[s : s + BATCH]).to(dtype)
        with torch.no_grad():
            out.append(torch.nn.functional.normalize(h["model"].encode_image(x).float(), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def _read_rgb(path):
    if not isinstance(path, str) or not os.path.isfile(path):
        raise RunError(f"no such image: {path!r}")
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def run(handle, params):
    import torch

    unknown = set(params) - {"paths", "array", "texts", "resize", "out"}
    if unknown:
        raise RunError(f"unknown params {sorted(unknown)}")
    resize = params.get("resize", "crop")
    if resize not in ("crop", "square"):
        raise RunError("resize is 'crop' or 'square'")
    imgs = np.zeros((0, 0), np.float32)
    if params.get("array") is not None:
        imgs = _encode_images(handle, _array(params["array"]), resize)
    elif params.get("paths") is not None:
        if not isinstance(params["paths"], list):
            raise RunError("paths is a list of image files")
        if resize == "square":
            pics = [_read_rgb(p) for p in params["paths"]]
            imgs = np.concatenate([_encode_images(handle, a[None], "square") for a in pics]) if pics else imgs
        else:
            imgs = _encode_images(handle, [_read_rgb(p) for p in params["paths"]], "crop")
    texts = np.zeros((0, 0), np.float32)
    if params.get("texts") is not None:
        if not isinstance(params["texts"], list) or not all(isinstance(t, str) for t in params["texts"]):
            raise RunError("texts is a list of strings")
        with torch.no_grad():
            texts = (
                torch.nn.functional.normalize(
                    handle["model"].encode_text(handle["tok"](params["texts"]).to(handle["dev"])).float(), dim=-1
                )
                .cpu()
                .numpy()
            )
    if not imgs.size and not texts.size:
        raise RunError("give paths, array or texts")
    dim = int((imgs if imgs.size else texts).shape[1])
    if params.get("out"):
        out = params["out"]
        if (
            not isinstance(out, str)
            or not out.endswith(".npz")
            or not os.path.isdir(os.path.dirname(os.path.abspath(out)))
        ):
            raise RunError("out is an .npz path in an existing folder")
        np.savez(out, images=imgs.astype(np.float32), texts=texts.astype(np.float32))
        return dict(out=out, images=len(imgs) if imgs.size else 0, texts=len(texts) if texts.size else 0, dim=dim)
    return dict(
        images=[[round(float(v), 6) for v in r] for r in imgs] if imgs.size else [],
        texts=[[round(float(v), 6) for v in r] for r in texts] if texts.size else [],
        dim=dim,
    )


def unload(handle):
    handle.clear()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # freeing is best effort, but not silent
        log.warning("could not empty the CUDA cache", exc_info=True)
