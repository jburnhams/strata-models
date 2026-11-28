"""image-kind's CLIP engine (open_clip and torch, the gpu group): each kind's prompts averaged into one text vector, a picture's kind the
nearest by cosine. Apart from imagekind.py so the unit suite's coverage leaves it out, as it does faces.py and whisper.py; it runs in
tests/gpu (test_image_kind_clip_on_a_drawn_engraving)."""

from typing import Any

import numpy as np

from stratamodels.runtimes.imagekind import _read, features, kind_of  # pyright: ignore[reportPrivateUsage]

# ---- clip ---------------------------------------------------------------------------------------------------------------------------------
# A few ways of saying each kind: CLIP was taught on captions, so the words a caption would use (an 'engraving', 'antique print') matter
# more than the definition. The best-scoring kind is the one whose prompts' mean embedding is nearest.
PROMPTS: dict[str, list[str]] = {
    "line-drawing": ["a pen and ink line drawing", "a simple black outline drawing", "a sketch in ink lines"],
    "engraving": [
        "an antique copper engraving",
        "an etching with cross-hatching",
        "a 19th century engraved illustration",
    ],
    "wood-engraving": [
        "a Victorian wood engraving",
        "a wood engraved book illustration",
        "a black and white wood engraving from a magazine",
    ],
    "woodcut": ["a crude woodcut", "a chapbook woodcut with bold black lines", "a woodblock print in black ink"],
    "hand-coloured": [
        "a hand-coloured engraving",
        "an antique print coloured with watercolour washes",
        "a hand-tinted etching",
    ],
    "lithograph": ["a lithograph", "a crayon lithograph with soft tones", "a tinted lithographic print"],
    "colour-print": [
        "a chromolithograph",
        "a colourful printed poster illustration",
        "a colour printed book illustration",
    ],
    "tonal-drawing": ["a pencil drawing", "a charcoal sketch", "a watercolour wash drawing"],
    "painting": ["an oil painting", "a reproduction of a painting", "a watercolour painting"],
    "diagram": ["a diagram", "a map", "a technical drawing"],
    "ornament": ["a decorative printer's ornament", "a decorative border", "an ornamental headpiece"],
    "silhouette": ["a black silhouette", "a shadow figure cut out in black"],
    "photograph": ["a black and white photograph", "a photograph", "a sepia photograph"],
    "halftone": [
        "a halftone photograph printed in a book",
        "a printed photo with a visible dot screen",
        "a newspaper photograph",
    ],
    "pattern": ["marbled paper", "a marbled endpaper", "a repeating decorative pattern"],
    "binding": ["a cloth book cover", "the spine of an old book", "a leather book binding"],
    "handwriting": ["handwriting", "a handwritten inscription", "a signature in ink"],
    "stamp": ["a library stamp", "a bookplate", "a printed label with a barcode"],
    "text": ["a page of printed text", "a paragraph of text in a book"],
    "blank": ["a blank sheet of paper", "an empty page"],
}


def load(h: dict[str, Any], arch: str, device: str, models_dir: str) -> None:
    import os

    import open_clip  # pyright: ignore[reportMissingImports]
    import torch  # pyright: ignore[reportMissingImports]

    name, _, tag = arch.partition(":")
    dev = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    m, _, pre = open_clip.create_model_and_transforms(
        name, pretrained=tag or None, cache_dir=os.path.join(models_dir, "open_clip")
    )
    m = m.to(dev).eval()
    tok = open_clip.get_tokenizer(name)
    kinds = list(PROMPTS)
    with torch.no_grad():
        means = []
        for k in kinds:
            e = torch.nn.functional.normalize(m.encode_text(tok(PROMPTS[k]).to(dev)).float(), dim=-1).mean(0)
            means.append(torch.nn.functional.normalize(e, dim=-1))
        text = torch.stack(means)
    h.update(clip=m, pre=pre, dev=dev, kinds=kinds, text=text, scale=float(m.logit_scale.exp().item()))


def run(h: dict[str, Any], path: str) -> dict[str, Any]:
    import torch  # pyright: ignore[reportMissingImports]
    from PIL import Image

    rgb = _read(path, 768)
    with torch.no_grad():
        x = h["pre"](Image.fromarray(rgb)).unsqueeze(0).to(h["dev"])
        e = torch.nn.functional.normalize(h["clip"].encode_image(x).float(), dim=-1)
        p = (h["scale"] * e @ h["text"].T).softmax(-1)[0].cpu().numpy()
    order = np.argsort(-p)
    kind = h["kinds"][int(order[0])]
    return dict(
        kind_of(kind),
        colour=features(rgb)["colour"] > 0.25,
        scores={h["kinds"][int(i)]: round(float(p[i]), 4) for i in order[:5]},
    )
