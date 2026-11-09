"""What kind of face a detected face is: a real person, or a statue, a picture (painting, print, poster, screen, cartoon), a toy (doll,
mannequin, mask) or an animal. Face detectors find all of these; an app that judges skin colour (strata-stills) or matches people
(bright-horizons) wants only the real ones.

How: the face cut out with a margin, embedded by CLIP (the open-clip runtime's model), then a small trained head (logistic regression on
the embedding) gives the kind. Without a head file the kind comes from CLIP's zero-shot similarity to a few prompts per kind (weaker).
Measured on 447 hand-labelled faces from strata-stills' catalogue (114 not real), ViT-B/32, 5-fold: real-or-not AUC 0.99 with the head
(at 0.5: keeps 98.8 % of real faces, drops 88 % of the rest), 0.92 zero-shot; the detector's own score gives 0.69.

model options: head (an .npz under <models>/face-kind/: W (K, D), b (K,), classes (K,), trained by scripts/face_kind_train.py), margin.
params: path (an image) + boxes ([[x1, y1, x2, y2], ...] in its pixels, as the faces role gives them), or array ({path, key?, bgr?}: face
crops as uint8 (N, H, W, 3), already cut). Answer: {faces: [{real, kind, probs: {kind: p}}], head: 'trained <file>' | 'zero-shot'}."""
import os

import numpy as np

from stratamodels.runtimes import RunError, clip

KINDS = ('real', 'statue', 'picture', 'toy', 'animal')
PROMPTS = {'real': ["a photo of a real person's face", 'a close-up photo of a human face'], 'statue': ['a statue or sculpture of a face', 'a stone or bronze bust'],
           'picture': ['a painting or drawing of a face', 'a printed poster or photo of a face', 'a cartoon face'], 'toy': ['a doll, toy or mannequin face', 'a mask'],
           'animal': ['the face of an animal']}
MARGIN = 0.3


def load(model, models_dir):
    h = clip.load(model, models_dir); h['margin'] = float(model.options.get('margin', MARGIN)); h['head'] = None
    f = model.options.get('head')
    if f:
        p = f if os.path.isabs(f) else os.path.join(models_dir, 'face-kind', f)
        if os.path.isfile(p): z = np.load(p, allow_pickle=False); h['head'] = dict(W=z['W'].astype(np.float32), b=z['b'].astype(np.float32), classes=[str(c) for c in z['classes']], file=os.path.basename(p))
    if h['head'] is None:
        import torch
        with torch.no_grad():
            T = [torch.nn.functional.normalize(torch.nn.functional.normalize(h['model'].encode_text(h['tok'](PROMPTS[k]).to(h['dev'])).float(), dim=-1).mean(0), dim=0).cpu().numpy() for k in KINDS]
        h['text'] = np.stack(T)
    return h


def crops(rgb, boxes, margin=MARGIN):
    """Each box ([x1, y1, x2, y2] pixels) cut from the picture with `margin` of its size on every side, kept inside the picture."""
    H, W = rgb.shape[:2]; out = []
    for b in boxes:
        try: x1, y1, x2, y2 = (float(v) for v in b)
        except (TypeError, ValueError): raise RunError(f'a box is [x1, y1, x2, y2], not {b!r}')
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0: raise RunError(f'empty box {b!r}')
        X1, Y1, X2, Y2 = int(max(0, x1 - margin * w)), int(max(0, y1 - margin * h)), int(min(W, x2 + margin * w)), int(min(H, y2 + margin * h))
        if X2 <= X1 or Y2 <= Y1: raise RunError(f'box {b!r} is outside the {W}x{H} picture')
        out.append(rgb[Y1:Y2, X1:X2])
    return out


def classify(h, emb):
    """(N, D) unit embeddings -> (N, K) probabilities over the head's classes (or KINDS, zero-shot) and the class names."""
    if h.get('head'):
        z = emb @ h['head']['W'].T + h['head']['b']; names = h['head']['classes']
    else: z = 100.0 * emb @ h['text'].T; names = list(KINDS)
    z = z - z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True), names


def run(handle, params):
    unknown = set(params) - {'path', 'boxes', 'array'}
    if unknown: raise RunError(f'unknown params {sorted(unknown)}')
    if params.get('array') is not None: faces = list(clip._array(params['array']))
    elif params.get('path') is not None:
        if not isinstance(params.get('boxes'), list): raise RunError('boxes is a list of [x1, y1, x2, y2]')
        faces = crops(clip._read_rgb(params['path']), params['boxes'], handle.get('margin', MARGIN))
    else: raise RunError('give path and boxes, or array')
    if not faces: return dict(faces=[], head=_head_name(handle))
    emb = np.concatenate([clip._encode_images(handle, [np.ascontiguousarray(f)], 'crop') for f in faces])
    p, names = classify(handle, emb); real = names.index('real')
    return dict(faces=[dict(real=round(float(r[real]), 4), kind=names[int(r.argmax())], probs={n: round(float(v), 4) for n, v in zip(names, r)}) for r in p], head=_head_name(handle))


def _head_name(h): return f"trained {h['head']['file']}" if h.get('head') else 'zero-shot'


def unload(handle): clip.unload(handle)
