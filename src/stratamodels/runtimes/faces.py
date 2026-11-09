"""insightface model packs (buffalo_l): face boxes, five keypoints, detection score, the ArcFace embedding (L2-normalised, 512 floats) and, when the
pack's landmark_3d_68 module is loaded, the 68 3D landmarks and head pose. One loaded pack serves every app; params choose what comes back.

params: path (an image file), or paths (several, answered in order); max_side (shrink larger pictures first, the boxes are given back in the
original's pixels); det_thresh; fields (subset of bbox, kps, det_score, embedding, landmark_3d_68, pose; default all but landmark_3d_68);
views (one image only): a list of ways to look at it again, each {crop: [x, y, w, h], rotate: 90 | 270 (degrees clockwise, applied after the
crop)}, answered as `views: [{faces}, ...]` with every box and point given back in the original picture's pixels. bright-horizons uses them to
look harder for children lying down or small in the background, without sending the image itself."""

import collections, os

from stratamodels.runtimes import RunError

DEFAULT_FIELDS = ("bbox", "kps", "det_score", "embedding", "pose")
ALL_FIELDS = DEFAULT_FIELDS + ("landmark_3d_68",)


def load(model, models_dir):
    from insightface.app import FaceAnalysis

    o = model.options
    fa = FaceAnalysis(
        name=model.name,
        root=os.path.join(models_dir, "insightface"),
        allowed_modules=o.get("modules"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    fa.prepare(ctx_id=0, det_size=tuple(o.get("det_size", (640, 640))))
    return dict(fa=fa, det_thresh=fa.det_model.det_thresh)


_recent = collections.OrderedDict()  # (path, mtime_ns, size) -> decoded image: an app that looks at a photo twice (bright-horizons' harder looks) decodes it once
RECENT = 4


def _read(path):
    import cv2

    if not isinstance(path, str) or not os.path.isfile(path):
        raise RunError(f"no such image: {path!r}")
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if key in _recent:
        _recent.move_to_end(key)
        return _recent[key]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RunError(f"not an image OpenCV can read: {path}")
    _recent[key] = img
    while len(_recent) > RECENT:
        _recent.popitem(last=False)
    return img


def _face(f, scale, fields):
    out = {}
    if "bbox" in fields:
        out["bbox"] = [round(float(v) / scale, 2) for v in f.bbox]
    if "kps" in fields and getattr(f, "kps", None) is not None:
        out["kps"] = [[round(float(x) / scale, 2), round(float(y) / scale, 2)] for x, y in f.kps]
    if "det_score" in fields:
        out["det_score"] = round(float(f.det_score), 4)
    if "embedding" in fields and getattr(f, "normed_embedding", None) is not None:
        out["embedding"] = [round(float(v), 6) for v in f.normed_embedding]
    if "landmark_3d_68" in fields and getattr(f, "landmark_3d_68", None) is not None:
        out["landmark_3d_68"] = [
            [round(float(p[0]) / scale, 2), round(float(p[1]) / scale, 2), round(float(p[2]) / scale, 2)]
            for p in f.landmark_3d_68
        ]
    if "pose" in fields and getattr(f, "pose", None) is not None:
        out["pose"] = [round(float(v), 2) for v in f.pose]
    return out


def detect(handle, img, max_side=None, det_thresh=None, fields=DEFAULT_FIELDS):
    """Faces in one BGR image (numpy array), largest-side limit applied first."""
    import cv2

    fa = handle["fa"]
    h, w = img.shape[:2]
    scale = 1.0
    if max_side and max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    fa.det_model.det_thresh = float(det_thresh) if det_thresh is not None else handle["det_thresh"]
    try:
        faces = fa.get(img)
    finally:
        fa.det_model.det_thresh = handle["det_thresh"]
    return dict(width=w, height=h, faces=[_face(f, scale, fields) for f in faces])


def _view(img, v):
    """The image as the view sees it, and the function that takes a point of that view back to the original."""
    import cv2

    H, W = img.shape[:2]
    ox = oy = 0
    if v.get("crop"):
        try:
            x, y, w, h = (int(round(float(a))) for a in v["crop"])
        except (TypeError, ValueError):
            raise RunError(f"crop must be [x, y, w, h], not {v['crop']!r}")
        x, y = max(x, 0), max(y, 0)
        w, h = min(w, W - x), min(h, H - y)
        if w <= 0 or h <= 0:
            raise RunError(f"crop {v['crop']} is outside the {W}x{H} picture")
        img, ox, oy = img[y : y + h, x : x + w], x, y
    h, w = img.shape[:2]
    rot = int(v.get("rotate") or 0) % 360
    if rot == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        back = lambda px, py: (py + ox, h - 1 - px + oy)
    elif rot == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        back = lambda px, py: (w - 1 - py + ox, px + oy)
    elif rot == 0:
        back = lambda px, py: (px + ox, py + oy)
    else:
        raise RunError(f"rotate must be 90 or 270, not {v.get('rotate')!r}")
    return img, back


def _unview(face, back):
    """A face found in a view, with its box (the corners' bounding box) and points in the original's pixels."""
    out = dict(face)
    if "bbox" in face:
        (ax, ay), (bx, by) = back(face["bbox"][0], face["bbox"][1]), back(face["bbox"][2], face["bbox"][3])
        out["bbox"] = [round(min(ax, bx), 2), round(min(ay, by), 2), round(max(ax, bx), 2), round(max(ay, by), 2)]
    if "kps" in face:
        out["kps"] = [[round(c, 2) for c in back(px, py)] for px, py in face["kps"]]
    if "landmark_3d_68" in face:
        out["landmark_3d_68"] = [[*(round(c, 2) for c in back(p[0], p[1])), p[2]] for p in face["landmark_3d_68"]]
    return out


def run(handle, params):
    fields = tuple(params.get("fields") or DEFAULT_FIELDS)
    bad = [f for f in fields if f not in ALL_FIELDS]
    if bad:
        raise RunError(f"unknown fields {bad} (known: {', '.join(ALL_FIELDS)})")
    if params.get("views") is not None:
        if not isinstance(params["views"], list) or params.get("paths") is not None:
            raise RunError("views is a list, for one image (path)")
        img = _read(params.get("path"))
        out = []
        for v in params["views"]:
            if not isinstance(v, dict):
                raise RunError(f"a view is {{crop?, rotate?}}, not {v!r}")
            vimg, back = _view(img, v)
            found = detect(handle, vimg, params.get("max_side"), params.get("det_thresh"), fields)
            out.append(dict(view=v, faces=[_unview(f, back) for f in found["faces"]]))
        return dict(path=params["path"], width=img.shape[1], height=img.shape[0], views=out)
    paths = params.get("paths") if params.get("paths") is not None else [params.get("path")]
    if not isinstance(paths, list) or not paths:
        raise RunError("give path (one image) or paths (a list)")
    res = [
        dict(path=p, **detect(handle, _read(p), params.get("max_side"), params.get("det_thresh"), fields))
        for p in paths
    ]
    return res[0] if params.get("paths") is None else dict(images=res)


def unload(handle):
    handle.clear()
    _recent.clear()
