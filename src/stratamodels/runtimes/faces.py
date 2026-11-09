"""insightface model packs (buffalo_l): face boxes, five keypoints, detection score, the ArcFace embedding (L2-normalised, 512 floats) and, when the
pack's landmark_3d_68 module is loaded, the 68 3D landmarks and head pose. One loaded pack serves every app; params choose what comes back.

params: path (an image file), or paths (several, answered in order); max_side (shrink larger pictures first, the boxes are given back in the
original's pixels); det_thresh; fields (subset of bbox, kps, det_score, embedding, landmark_3d_68, pose; default all but landmark_3d_68)."""
import os

from stratamodels.runtimes import RunError

DEFAULT_FIELDS = ('bbox', 'kps', 'det_score', 'embedding', 'pose')
ALL_FIELDS = DEFAULT_FIELDS + ('landmark_3d_68',)


def load(model, models_dir):
    from insightface.app import FaceAnalysis
    o = model.options
    fa = FaceAnalysis(name=model.name, root=os.path.join(models_dir, 'insightface'), allowed_modules=o.get('modules'), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    fa.prepare(ctx_id=0, det_size=tuple(o.get('det_size', (640, 640))))
    return dict(fa=fa, det_thresh=fa.det_model.det_thresh)


def _read(path):
    import cv2
    if not isinstance(path, str) or not os.path.isfile(path): raise RunError(f'no such image: {path!r}')
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise RunError(f'not an image OpenCV can read: {path}')
    return img


def _face(f, scale, fields):
    out = {}
    if 'bbox' in fields: out['bbox'] = [round(float(v) / scale, 2) for v in f.bbox]
    if 'kps' in fields and getattr(f, 'kps', None) is not None: out['kps'] = [[round(float(x) / scale, 2), round(float(y) / scale, 2)] for x, y in f.kps]
    if 'det_score' in fields: out['det_score'] = round(float(f.det_score), 4)
    if 'embedding' in fields and getattr(f, 'normed_embedding', None) is not None: out['embedding'] = [round(float(v), 6) for v in f.normed_embedding]
    if 'landmark_3d_68' in fields and getattr(f, 'landmark_3d_68', None) is not None: out['landmark_3d_68'] = [[round(float(p[0]) / scale, 2), round(float(p[1]) / scale, 2), round(float(p[2]) / scale, 2)] for p in f.landmark_3d_68]
    if 'pose' in fields and getattr(f, 'pose', None) is not None: out['pose'] = [round(float(v), 2) for v in f.pose]
    return out


def detect(handle, img, max_side=None, det_thresh=None, fields=DEFAULT_FIELDS):
    """Faces in one BGR image (numpy array), largest-side limit applied first."""
    import cv2
    fa = handle['fa']; h, w = img.shape[:2]; scale = 1.0
    if max_side and max(h, w) > max_side: scale = max_side / max(h, w); img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    fa.det_model.det_thresh = float(det_thresh) if det_thresh is not None else handle['det_thresh']
    try: faces = fa.get(img)
    finally: fa.det_model.det_thresh = handle['det_thresh']
    return dict(width=w, height=h, faces=[_face(f, scale, fields) for f in faces])


def run(handle, params):
    fields = tuple(params.get('fields') or DEFAULT_FIELDS)
    bad = [f for f in fields if f not in ALL_FIELDS]
    if bad: raise RunError(f'unknown fields {bad} (known: {", ".join(ALL_FIELDS)})')
    paths = params.get('paths') if params.get('paths') is not None else [params.get('path')]
    if not isinstance(paths, list) or not paths: raise RunError('give path (one image) or paths (a list)')
    res = [dict(path=p, **detect(handle, _read(p), params.get('max_side'), params.get('det_thresh'), fields)) for p in paths]
    return res[0] if params.get('paths') is None else dict(images=res)


def unload(handle): handle.clear()
