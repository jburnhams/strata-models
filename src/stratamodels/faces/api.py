"""The face store's HTTP API, mounted by server/app.py (the client is strata_python.clients.faces).

  POST /api/faces/images                 {app, ref, path?, url?, tags?, taken_at?, priority?, det_thresh?, force?, faces?, model?, width?, height?} index one picture:
                                         run the `faces` role on `path` (or take `faces` an app already has, made with `model`), keep a 1600 px
                                         WebP copy, store faces and embeddings. Indexing the same ref with the same model and settings again
                                         answers the stored picture without a job (unless `force`).
  GET  /api/faces/images?app=&ref=&tag=&include=&before=&limit=   indexed pictures, newest first
  GET  /api/faces/images/{id}            one picture with its faces and each face's person
  GET  /api/faces/images/{id}/picture    the kept medium copy (to show a face in its picture)
  DELETE /api/faces/images/{id}          one picture out of the store, with its faces, embeddings, decisions and hints
  POST /api/faces/images/delete          {ids?, app?, tag?, dry_run?} many at once (pictures matching every one given; dry_run, the default,
                                         only counts): {images, faces, decisions, confirmed, hints, deleted}
  GET  /api/faces/{id}                   one face; GET /api/faces/{id}/crop?px=160&pad=1.6 its crop (cached WebP)
  POST /api/faces/search                 {app, face_id | embedding | path, k?, tags?, include?, named?, max_distance?, model?} nearest faces
  POST /api/faces/cluster                {app, tags?, include?, threshold?, min_size?, limit?, min_score?, model?, suggested?} undecided faces, grouped
                                         (`suggested` false: not the faces waiting on a suggestion)
  POST /api/faces/assign                 {face_ids, identity_id, app, status?, source?, confidence?, note?} -> {batch}
  POST /api/faces/batches/{id}/undo        an assign batch, a hints batch or a propose run
  POST /api/faces/hints                  {app, hints: [{image_id, app_ref, face_id?, name?, status?}], note?} an app's possible labels -> {batch, added, skipped}
  GET  /api/faces/hints?app=&image_id=&app_ref=&limit=   an app's hints
  POST /api/faces/propose                {app, dry_run?, groups?, tags?, include?, limit?, model?, min_seeds?, min_purity?, max_distance?, margin?,
                                         min_score?, group_share?, group_min?} turn the app's hints into suggested faces (dry_run, the default,
                                         only counts); nothing is ever confirmed here
  GET  /api/faces/tags; PUT /api/faces/tags/{tag} {opt_in, note?}
  GET  /api/identities?app=&q=&app_ref=&limit=   people (as `app` sees them; `faces` confirmed, `suggested` waiting); POST /api/identities {name?, fields?, app?, app_fields?, app_ref?}
  GET/PATCH /api/identities/{id}         one person (?app=); {name?, fields?} change the global ones
  PUT  /api/identities/{id}/apps/{app}   {fields, app_ref?} an app's own fields for a person
  GET  /api/identities/{id}/review?limit=48&min_likeness=   the next round of their suggested faces, most certain first, each with its `distance` to
                                         the person (their confirmed faces once there are 3, else the tightest group of the suggestions)
  GET  /api/identities/{id}/faces?status=confirmed&limit=&offset=   a person's faces in that state (`suggested`: waiting for a person), best first
  POST /api/identities/{id}/merge {into, app}; POST /api/identities/{id}/unmerge; POST /api/identities/not-same {a, b}
  GET  /api/identities/merge-suggestions?limit=&model=

The `model` of a search or clustering defaults to the `faces` role's model: embeddings of different models are never compared. Every face is
stored, but search, clustering, review and propose only see faces over the store's bar (settings faces_min_score and faces_min_px), so a
caller never has to leave out tiny or doubtful faces itself; `min_score` on a clustering can only raise the bar."""

import hashlib
import os
import time
from typing import Any

from fastapi import HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from stratamodels import settings
from stratamodels.faces.store import FaceStore, NotFound

ROLE = "faces"
FIELDS = ["bbox", "kps", "det_score", "embedding", "pose"]
MEDIUM_PX, MEDIUM_QUALITY = 1600, 80  # the kept copy: about 200 KB a photo as WebP (decided 2026-11-25)
INDEX_TIMEOUT_S = 600


class IndexIn(BaseModel):
    app: str
    ref: str
    path: str | None = None
    url: str | None = None
    tags: list[str] = Field(default_factory=list)
    taken_at: float | None = None
    priority: str = "low"
    det_thresh: float | None = None
    force: bool = False
    faces: list[dict] | None = None
    model: str | None = None
    width: int | None = None
    height: int | None = None


class SearchIn(BaseModel):
    app: str
    face_id: int | None = None
    embedding: list[float] | None = None
    path: str | None = None
    k: int = 20
    tags: list[str] | None = None
    include: list[str] = Field(default_factory=list)
    named: bool | None = None
    max_distance: float | None = None
    model: str | None = None


class ClusterIn(BaseModel):
    app: str
    tags: list[str] | None = None
    include: list[str] = Field(default_factory=list)
    threshold: float | None = None
    min_size: int = 2
    limit: int = 5000
    min_score: float | None = None
    model: str | None = None
    suggested: bool = True


class AssignIn(BaseModel):
    face_ids: list[int]
    identity_id: int | None
    app: str
    status: str = "confirmed"
    source: str = "human"
    confidence: float | None = None
    note: str | None = None


class HintsIn(BaseModel):
    app: str
    hints: list[dict]
    note: str | None = None


class ProposeIn(BaseModel):
    app: str
    dry_run: bool = True
    groups: bool = False
    tags: list[str] | None = None
    include: list[str] = Field(default_factory=list)
    limit: int | None = None
    model: str | None = None
    min_seeds: int | None = None
    min_purity: float | None = None
    max_distance: float | None = None
    margin: float | None = None
    min_score: float | None = None
    group_share: float | None = None
    group_min: int | None = None


class DeleteIn(BaseModel):
    ids: list[int] | None = None
    app: str | None = None
    tag: str | None = None
    dry_run: bool = True


class TagIn(BaseModel):
    opt_in: bool
    note: str | None = None


class IdentityIn(BaseModel):
    name: str | None = None
    fields: dict = Field(default_factory=dict)
    app: str | None = None
    app_fields: dict | None = None
    app_ref: str | None = None


class IdentityPatch(BaseModel):
    name: str | None = None
    fields: dict | None = None


class AppFieldsIn(BaseModel):
    fields: dict = Field(default_factory=dict)
    app_ref: str | None = None


class MergeIn(BaseModel):
    into: int
    app: str


class NotSameIn(BaseModel):
    a: int
    b: int


def faces_dir():
    return os.path.join(settings.data_dir(), "faces")


def picture_size(path):
    """(width, height) of the picture as the faces runtime and the crops see it (upright: a phone photo taken on its side has its stored
    sides swapped), in any format strata-python's images reads, else (0, 0). Imported faces' boxes are in these pixels, and a crop from
    the kept copy scales them by the copy's width over this."""
    from strata_python import images

    try:
        return images.size_upright(path)
    except (OSError, ValueError, SyntaxError):
        return 0, 0


def detection_hash(settings_hash, det_thresh):
    """The stored signature of an indexing: the model's settings plus the detection threshold asked for, so asking again at another
    threshold runs again instead of answering the faces found at the old one."""
    if det_thresh is None:
        return settings_hash
    return hashlib.sha256(f"{settings_hash}|det_thresh={float(det_thresh)}".encode()).hexdigest()[:16]


def medium_copy(path):
    """The kept 1600 px WebP of a picture, relative to the data folder; None when Pillow cannot read it (a raw, a video: the app's own copy is
    then the only one, and the face store shows crops without their picture)."""
    from strata_python import images

    try:
        out = images.thumbnail(
            path, MEDIUM_PX, cache_dir=os.path.join(faces_dir(), "medium"), fmt="webp", quality=MEDIUM_QUALITY
        )
    except (OSError, ValueError, SyntaxError):  # PIL.UnidentifiedImageError is an OSError
        return None
    return os.path.relpath(out, settings.data_dir())


def mount(app, scheduler, store: FaceStore | None = None):
    s = scheduler
    fs = store or FaceStore(s.store.engine)
    app.state.faces = fs

    @app.exception_handler(NotFound)
    def not_found(_req: Request, e: Exception):
        return JSONResponse(status_code=404, content=dict(detail=str(e.args[0] if e.args else e)))

    def bad(e):
        raise HTTPException(400, str(e)) from None

    def role_model(model=None, app_name=None):
        m = s.registry.resolve(ROLE, model, app_name)
        return m.key, m.settings_hash

    def detect(path, app_name, priority="normal", det_thresh=None, model=None, timeout=INDEX_TIMEOUT_S):
        """Run the faces role on one picture and wait for it: (result, model key, settings hash)."""
        params = dict(path=path, fields=FIELDS)
        if det_thresh is not None:
            params["det_thresh"] = float(det_thresh)
        j = s.submit(ROLE, params, app_name, priority if priority in ("high", "normal", "low") else "low", model)
        end = time.monotonic() + timeout
        while j["status"] not in ("done", "failed", "cancelled") and time.monotonic() < end:
            j = s.wait(j["id"], min(30.0, max(end - time.monotonic(), 0.0)))
        if j["status"] == "failed":
            raise HTTPException(422, j.get("error") or "the faces job failed")
        if j["status"] != "done":
            raise HTTPException(504, f"faces job {j['id']} is still {j['status']}")
        return j["result"], j["model"], j.get("settings_hash")

    @app.post("/api/faces/images")
    def post_image(b: IndexIn):
        try:
            if b.faces is not None:
                if not b.model:
                    raise ValueError("faces made elsewhere need the model that made their embeddings")
                faces = [dict(f, found_by=f.get("found_by") or "import") for f in b.faces]
                here = bool(b.path and os.path.isfile(b.path))
                medium = medium_copy(b.path) if here else None
                w, h = (b.width, b.height) if b.width and b.height else picture_size(b.path) if here else (0, 0)
                return fs.save_image(
                    app=b.app,
                    ref=b.ref,
                    faces=faces,
                    model=b.model,
                    settings_hash=None,
                    path=b.path,
                    url=b.url,
                    tags=b.tags,
                    taken_at=b.taken_at,
                    width=w,
                    height=h,
                    medium=medium,
                )
            if not b.path or not os.path.isfile(b.path):
                raise ValueError(f"no such picture: {b.path!r}")
            key, sh = role_model(b.model, b.app)
            if not b.force and fs.is_current(b.app, b.ref, key, detection_hash(sh, b.det_thresh)):
                return fs.find_image(b.app, b.ref)
        except ValueError as e:
            bad(e)
        r, key, sh = detect(b.path, b.app, b.priority, b.det_thresh, b.model)
        return fs.save_image(
            app=b.app,
            ref=b.ref,
            faces=r.get("faces") or [],
            model=key,
            settings_hash=detection_hash(sh, b.det_thresh),
            path=b.path,
            url=b.url,
            tags=b.tags,
            taken_at=b.taken_at,
            width=r.get("width") or 0,
            height=r.get("height") or 0,
            medium=medium_copy(b.path),
        )

    @app.get("/api/faces/images")
    def get_images(
        app_name: str | None = Query(None, alias="app"),
        ref: str | None = None,
        tag: list[str] | None = Query(None),
        include: list[str] | None = Query(None),
        before: int | None = None,
        limit: int = 50,
    ):
        if app_name and ref:
            one = fs.find_image(app_name, ref)
            return dict(items=[one] if one else [], next=None)
        items = fs.images(app_name, tag, include or (), before, limit)
        return dict(items=items, next=items[-1]["id"] if len(items) >= max(1, min(limit, 500)) else None)

    @app.get("/api/faces/images/{image_id:int}")
    def get_image(image_id: int):
        return fs.image(image_id)

    @app.get("/api/faces/images/{image_id:int}/picture")
    def get_picture(image_id: int):
        img = fs.image(image_id)
        p = os.path.join(settings.data_dir(), img["medium"]) if img["medium"] else None
        if p is None or not os.path.isfile(p):
            raise HTTPException(404, "no copy of this picture is kept")
        return FileResponse(p, media_type="image/webp", headers={"Cache-Control": "max-age=86400"})

    def delete(ids=None, app_name=None, tag=None, dry_run=True):
        try:
            out: dict[str, Any] = fs.delete_images(ids=ids, app=app_name, tag=tag, dry_run=dry_run)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        media = out.pop("media")
        if out["deleted"]:
            root = os.path.realpath(faces_dir())
            for m in media:
                p = os.path.realpath(os.path.join(settings.data_dir(), m))
                if p.startswith(root + os.sep) and os.path.isfile(p):
                    os.remove(p)
        return out

    @app.delete("/api/faces/images/{image_id:int}")
    def delete_image(image_id: int):
        fs.image(image_id)  # 404 when there is no such picture
        return delete(ids=[image_id], dry_run=False)

    @app.post("/api/faces/images/delete")
    def delete_images(b: DeleteIn):
        return delete(b.ids, b.app, b.tag, b.dry_run)

    @app.get("/api/faces/tags")
    def get_tags():
        return dict(items=fs.tags())

    @app.put("/api/faces/tags/{tag}")
    def put_tag(tag: str, b: TagIn):
        try:
            return fs.set_tag(tag, b.opt_in, b.note)
        except ValueError as e:
            bad(e)

    @app.get("/api/faces/{face_id:int}")
    def get_face(face_id: int):
        return fs.face(face_id)

    @app.get("/api/faces/{face_id:int}/crop")
    def get_crop(face_id: int, px: int = 160, pad: float = 1.6):
        """A face's crop, from the original when it is readable here, else from the kept copy (its box scaled to it)."""
        from strata_python import images

        info: dict[str, Any] = fs.image_for_face(face_id)
        pad = min(max(float(pad), 1.0), 4.0)
        cache = os.path.join(faces_dir(), "crops")
        x0, y0, x1, y1 = (float(v) for v in info["box"])
        src, box = info["path"], (x0, y0, x1, y1)
        try:
            if not (src and os.path.isfile(src)):
                raise OSError("original not here")
            out = images.crop(src, box, px, cache_dir=cache, pad=pad)
        except (OSError, ValueError, SyntaxError):
            if not info["medium"]:
                raise HTTPException(404, "neither the picture nor a copy of it is here") from None
            med = os.path.join(settings.data_dir(), info["medium"])
            k = 1.0
            if info["width"] and info["height"]:
                from PIL import Image

                with Image.open(med) as im:
                    k = im.width / info["width"]
            try:
                out = images.crop(med, (x0 * k, y0 * k, x1 * k, y1 * k), px, cache_dir=cache, pad=pad)
            except (OSError, ValueError, SyntaxError) as e:
                raise HTTPException(404, f"cannot cut this face: {e}") from None
        return FileResponse(out, headers={"Cache-Control": "max-age=604800"})

    def query_vector(b, model):
        if b.face_id is not None:
            return fs.embedding(b.face_id, model), [b.face_id]
        if b.embedding is not None:
            return b.embedding, []
        if b.path:
            if not os.path.isfile(b.path):
                raise ValueError(f"no such picture: {b.path!r}")
            r, _, _ = detect(b.path, b.app, "high", None, model)
            faces = [f for f in r.get("faces") or [] if f.get("embedding")]
            if not faces:
                raise ValueError("no face found in that picture")
            big = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
            return big["embedding"], []
        raise ValueError("give face_id, embedding or path")

    @app.post("/api/faces/search")
    def post_search(b: SearchIn):
        try:
            model = b.model or role_model(None, b.app)[0]
            v, exclude = query_vector(b, model)
            hits = fs.search(
                v,
                model,
                k=b.k,
                tags=b.tags,
                include=b.include,
                named=b.named,
                max_distance=b.max_distance,
                exclude=exclude,
            )
        except ValueError as e:
            bad(e)
        return dict(model=model, hits=hits)

    @app.post("/api/faces/cluster")
    def post_cluster(b: ClusterIn):
        model = b.model or role_model(None, b.app)[0]
        return dict(
            model=model,
            clusters=fs.cluster(
                model,
                tags=b.tags,
                include=b.include,
                threshold=b.threshold,
                min_size=b.min_size,
                limit=b.limit,
                min_score=b.min_score,
                suggested=b.suggested,
            ),
        )

    @app.post("/api/faces/assign")
    def post_assign(b: AssignIn):
        try:
            return dict(batch=fs.assign(b.face_ids, b.identity_id, b.app, b.status, b.source, b.confidence, b.note))
        except ValueError as e:
            bad(e)

    @app.post("/api/faces/batches/{batch_id:int}/undo")
    def post_undo(batch_id: int):
        try:
            return fs.undo(batch_id)
        except ValueError as e:
            bad(e)

    @app.post("/api/faces/hints", status_code=201)
    def post_hints(b: HintsIn):
        try:
            return fs.add_hints(b.app, b.hints, b.note)
        except ValueError as e:
            bad(e)

    @app.get("/api/faces/hints")
    def get_hints(
        app_name: str = Query(..., alias="app"),
        image_id: int | None = None,
        app_ref: str | None = None,
        limit: int = 500,
    ):
        return dict(items=fs.hints(app_name, image_id, app_ref, limit))

    @app.post("/api/faces/propose")
    def post_propose(b: ProposeIn):
        model = b.model or role_model(None, b.app)[0]
        params = b.model_dump(
            include={"min_seeds", "min_purity", "max_distance", "margin", "min_score", "group_share", "group_min"}
        )
        try:
            return fs.propose(
                b.app,
                model,
                dry_run=b.dry_run,
                groups=b.groups,
                tags=b.tags,
                include=b.include,
                limit=b.limit,
                **params,
            )
        except (ValueError, TypeError) as e:
            bad(e)

    @app.get("/api/identities")
    def get_identities(
        app_name: str | None = Query(None, alias="app"),
        q: str | None = None,
        app_ref: str | None = None,
        limit: int = 200,
    ):
        return dict(items=fs.identities(app_name, q, app_ref, limit))

    @app.post("/api/identities", status_code=201)
    def post_identity(b: IdentityIn):
        try:
            return fs.create_identity(b.name, b.fields, b.app, b.app_fields, b.app_ref)
        except ValueError as e:
            bad(e)

    @app.get("/api/identities/merge-suggestions")
    def get_suggestions(limit: int = 50, model: str | None = None):
        model = model or role_model()[0]
        return dict(model=model, items=fs.merge_suggestions(model, limit))

    @app.post("/api/identities/not-same")
    def post_not_same(b: NotSameIn):
        try:
            return fs.not_same(b.a, b.b)
        except ValueError as e:
            bad(e)

    @app.get("/api/identities/{identity_id:int}")
    def get_identity(identity_id: int, app_name: str | None = Query(None, alias="app")):
        return fs.identity(identity_id, app_name)

    @app.patch("/api/identities/{identity_id:int}")
    def patch_identity(identity_id: int, b: IdentityPatch):
        return fs.update_identity(identity_id, b.name, b.fields)

    @app.put("/api/identities/{identity_id:int}/apps/{app_name}")
    def put_app_fields(identity_id: int, app_name: str, b: AppFieldsIn):
        try:
            return fs.set_app_fields(identity_id, app_name, b.fields, b.app_ref)
        except ValueError as e:
            bad(e)

    @app.get("/api/identities/{identity_id:int}/faces")
    def get_identity_faces(identity_id: int, status: str = "confirmed", limit: int = 500, offset: int = 0):
        return dict(items=fs.faces_of(identity_id, status, limit, offset))

    @app.get("/api/identities/{identity_id:int}/review")
    def get_review(identity_id: int, limit: int = 48, model: str | None = None, min_likeness: float | None = None):
        return fs.review(identity_id, model or role_model()[0], limit, min_likeness=min_likeness)

    @app.post("/api/identities/{identity_id:int}/merge")
    def post_merge(identity_id: int, b: MergeIn):
        try:
            return fs.merge(identity_id, b.into, b.app)
        except ValueError as e:
            bad(e)

    @app.post("/api/identities/{identity_id:int}/unmerge")
    def post_unmerge(identity_id: int):
        return fs.unmerge(identity_id)

    return fs
