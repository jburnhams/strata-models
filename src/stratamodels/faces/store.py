"""The face store: one database of faces, embeddings and people for every app (tables in db/models.py, the face section).

What lives here is record keeping and search; what a match means stays in the calling app (auto-approve thresholds, review workflow,
consent). Decided with the owner 2026-11-25:
  - one global identity per person; each app may add its own fields (identity_app_metadata), which win when that app reads the person;
  - every picture carries its app as a tag; search and clustering cover every tag except opt-in ones (face_tags.opt_in, e.g. other people's
    children), which count only when the caller names them;
  - merges across apps are suggested here and made by a person, never automatically; undone by clearing `merged_into`;
  - decisions about faces are append-only (face_assignments), the newest not undone wins, and a batch undoes as one.

Every call opens its own session (as store.Store does), so the API's threads never share one. Times are epoch seconds."""

import dataclasses
import time
from collections import defaultdict

import numpy as np
import sqlalchemy as sa
from sqlalchemy.orm import Session
from strata_python.db.vectors import nearest

from stratamodels.db.models import (
    Face,
    FaceAssignment,
    FaceBatch,
    FaceEmbedding,
    FaceImage,
    FaceImageTag,
    FaceTag,
    Identity,
    IdentityAppMetadata,
    IdentityNotSame,
)

STATUSES = ("confirmed", "rejected", "suggested")
CLUSTER_THRESHOLD = 0.55  # cosine distance on ArcFace embeddings: conservative (strata-360's `who`; bright-horizons' unknown groups used 0.55)
MAX_CLUSTER = 5000  # faces grouped at once: average linkage keeps an n x n distance matrix (5000: 100 MB of float32)
SUGGEST_MAX_DISTANCE = 0.6  # identity centroids further apart than this are not worth a person's look
EXEMPLARS = 30  # confirmed faces per identity used for its centroid in merge suggestions (the best detected)


@dataclasses.dataclass(frozen=True)
class State:
    """A face's current state: its newest decision not undone, with the person read through merges."""

    face_id: int
    status: str
    identity_id: int | None
    app: str
    source: str
    confidence: float | None
    at: float


class NotFound(KeyError):
    """No such picture, face, identity or batch (a 404 through strata_python.web's handlers)."""


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def face_row(f, state=None):
    out = dict(
        id=f.id,
        image_id=f.image_id,
        box=[f.x0, f.y0, f.x1, f.y1],
        det_score=f.det_score,
        kps=f.kps or [],
        pose=f.pose or [],
        found_by=f.found_by,
        identity_id=None,
        status=None,
    )
    if state is not None:
        out.update(identity_id=state.identity_id, status=state.status)
    return out


class FaceStore:
    def __init__(self, engine, clock=time.time):
        self.engine, self.clock = engine, clock

    def session(self):
        return Session(self.engine, expire_on_commit=False)

    # ---- tags --------------------------------------------------------------------------------------------------------------------------
    def _scope(self, s, tags=None, include=()):
        """A condition on FaceImage.id: pictures with any of `tags` (when given), never one with an opt-in tag that is not in `include` or
        `tags` (naming a tag in `tags` counts as asking for it)."""
        conds = []
        if tags:
            conds.append(FaceImage.id.in_(sa.select(FaceImageTag.image_id).where(FaceImageTag.tag.in_(list(tags)))))
        allowed = set(include or ()) | set(tags or ())
        hidden = [t for t in s.scalars(sa.select(FaceTag.tag).where(FaceTag.opt_in.is_(True))) if t not in allowed]
        if hidden:
            conds.append(FaceImage.id.not_in(sa.select(FaceImageTag.image_id).where(FaceImageTag.tag.in_(hidden))))
        return sa.and_(sa.true(), *conds)

    def tags(self):
        with self.session() as s:
            counts = dict(s.execute(sa.select(FaceImageTag.tag, sa.func.count()).group_by(FaceImageTag.tag)).all())
            opt = {t.tag: t for t in s.scalars(sa.select(FaceTag))}
            return [
                dict(
                    tag=t,
                    images=counts.get(t, 0),
                    opt_in=bool(opt[t].opt_in) if t in opt else False,
                    note=opt[t].note if t in opt else "",
                )
                for t in sorted(set(counts) | set(opt))
            ]

    def set_tag(self, tag, opt_in, note=None):
        tag = str(tag).strip()
        if not tag:
            raise ValueError("a tag needs a name")
        with self.session() as s:
            row = s.get(FaceTag, tag)
            if row is None:
                row = FaceTag(tag=tag, updated_at=self.clock())
                s.add(row)
            row.opt_in, row.updated_at = bool(opt_in), self.clock()
            if note is not None:
                row.note = str(note)[:500]
            s.commit()
        return dict(tag=tag, opt_in=bool(opt_in))

    # ---- pictures and faces --------------------------------------------------------------------------------------------------------------
    def _current(self, s, face_ids=None):
        """face id -> its newest assignment not undone (a FaceAssignment), for the given faces (all when None)."""
        live = sa.or_(
            FaceAssignment.batch_id.is_(None),
            FaceAssignment.batch_id.in_(sa.select(FaceBatch.id).where(FaceBatch.undone_at.is_(None))),
        )
        newest = sa.select(sa.func.max(FaceAssignment.id)).where(live).group_by(FaceAssignment.face_id)
        if face_ids is not None:
            if not face_ids:
                return {}
            newest = newest.where(FaceAssignment.face_id.in_(list(face_ids)))
        out = {}
        roots = {}
        for a in s.scalars(sa.select(FaceAssignment).where(FaceAssignment.id.in_(newest))):
            who = a.identity_id
            if who is not None:  # read through merges (a copy: the row itself is never changed)
                who = roots[who] if who in roots else roots.setdefault(who, self._root(s, who))
            out[a.face_id] = State(a.face_id, a.status, who, a.app, a.source, a.confidence, a.at)
        return out

    def _image_dict(self, s, img):
        faces = list(s.scalars(sa.select(Face).where(Face.image_id == img.id).order_by(Face.id)))
        state = self._current(s, [f.id for f in faces])
        tags = list(
            s.scalars(sa.select(FaceImageTag.tag).where(FaceImageTag.image_id == img.id).order_by(FaceImageTag.tag))
        )
        return dict(
            id=img.id,
            app=img.app,
            ref=img.ref,
            path=img.path,
            url=img.url,
            tags=tags,
            width=img.width,
            height=img.height,
            taken_at=img.taken_at,
            medium=img.medium,
            model=img.model,
            settings_hash=img.settings_hash,
            created_at=img.created_at,
            faces=[face_row(f, state.get(f.id)) for f in faces],
        )

    def find_image(self, app, ref):
        with self.session() as s:
            img = s.scalar(sa.select(FaceImage).where(FaceImage.app == app, FaceImage.ref == ref))
            return None if img is None else self._image_dict(s, img)

    def image(self, image_id):
        with self.session() as s:
            img = s.get(FaceImage, int(image_id))
            if img is None:
                raise NotFound(f"no such picture: {image_id}")
            return self._image_dict(s, img)

    def images(self, app=None, tags=None, include=(), before=None, limit=50):
        with self.session() as s:
            q = (
                sa.select(FaceImage)
                .where(self._scope(s, tags, include))
                .order_by(FaceImage.id.desc())
                .limit(max(1, min(int(limit), 500)))
            )
            if app:
                q = q.where(FaceImage.app == app)
            if before:
                q = q.where(FaceImage.id < int(before))
            return [self._image_dict(s, i) for i in s.scalars(q)]

    def is_current(self, app, ref, model, settings_hash):
        """Whether `app`'s `ref` was already indexed with this model and settings (an index call then needs no job)."""
        img = self.find_image(app, ref)
        return img is not None and img["model"] == model and img["settings_hash"] == settings_hash

    def save_image(
        self,
        *,
        app,
        ref,
        faces,
        model,
        settings_hash=None,
        path=None,
        url=None,
        tags=(),
        taken_at=None,
        width=0,
        height=0,
        medium=None,
    ):
        """Store one indexed picture and its faces. `faces` are the detector's faces: {bbox | box, det_score, kps?, pose?, embedding?,
        found_by?}. Indexing the same (app, ref) again with another model adds that model's embeddings to the faces it finds again (the box
        overlapping by IoU 0.5 or more), so decisions about them stand; faces it no longer finds stay if anyone decided about them."""
        app, ref = str(app)[:64], str(ref)[:512]
        if not app or not ref:
            raise ValueError("app and ref are needed")
        now = self.clock()
        with self.session() as s:
            img = s.scalar(sa.select(FaceImage).where(FaceImage.app == app, FaceImage.ref == ref))
            if img is None:
                img = FaceImage(app=app, ref=ref, created_at=now)
                s.add(img)
            img.path, img.url = path or img.path, url or img.url
            img.width, img.height = int(width or img.width or 0), int(height or img.height or 0)
            img.taken_at = taken_at if taken_at is not None else img.taken_at
            img.medium = medium or img.medium
            img.model, img.settings_hash = model, settings_hash
            s.flush()
            for t in {app, *(str(t).strip()[:128] for t in tags if str(t).strip())}:
                if s.get(FaceImageTag, (img.id, t)) is None:
                    s.add(FaceImageTag(image_id=img.id, tag=t))
            old = list(s.scalars(sa.select(Face).where(Face.image_id == img.id)))
            decided = (
                set(s.scalars(sa.select(FaceAssignment.face_id).where(FaceAssignment.face_id.in_([f.id for f in old]))))
                if old
                else set()
            )
            kept = set()
            for d in faces:
                box = [float(v) for v in (d.get("box") or d.get("bbox"))[:4]]
                match = max(old, key=lambda f: iou(box, (f.x0, f.y0, f.x1, f.y1)), default=None)
                if (
                    match is not None
                    and match.id not in kept
                    and iou(box, (match.x0, match.y0, match.x1, match.y1)) >= 0.5
                ):
                    f = match
                else:
                    f = Face(
                        image_id=img.id,
                        x0=box[0],
                        y0=box[1],
                        x1=box[2],
                        y1=box[3],
                        det_score=float(d.get("det_score") or 0.0),
                        created_at=now,
                    )
                    s.add(f)
                f.x0, f.y0, f.x1, f.y1 = box
                f.det_score = float(d.get("det_score") or f.det_score or 0.0)
                f.kps, f.pose = d.get("kps") or f.kps, d.get("pose") or f.pose
                f.found_by = str(d.get("found_by") or f.found_by or "scan")[:32]
                s.flush()
                kept.add(f.id)
                if d.get("embedding") is not None:
                    e = s.scalar(
                        sa.select(FaceEmbedding).where(FaceEmbedding.face_id == f.id, FaceEmbedding.model == model)
                    )
                    if e is None:
                        s.add(
                            FaceEmbedding(
                                face_id=f.id,
                                model=model,
                                vector=[float(v) for v in d["embedding"]],
                                created_at=now,
                                settings_hash=settings_hash,
                            )
                        )
                    else:
                        e.vector, e.settings_hash, e.created_at = [float(v) for v in d["embedding"]], settings_hash, now
            for f in old:
                if f.id not in kept and f.id not in decided:
                    s.delete(f)
            s.commit()
            return self._image_dict(s, img)

    def face(self, face_id):
        with self.session() as s:
            f = s.get(Face, int(face_id))
            if f is None:
                raise NotFound(f"no such face: {face_id}")
            img = s.get(FaceImage, f.image_id)
            return dict(face_row(f, self._current(s, [f.id]).get(f.id)), app=img.app, ref=img.ref)

    def embedding(self, face_id, model):
        with self.session() as s:
            v = s.scalar(
                sa.select(FaceEmbedding.vector).where(
                    FaceEmbedding.face_id == int(face_id), FaceEmbedding.model == model
                )
            )
            if v is None:
                if s.get(Face, int(face_id)) is None:
                    raise NotFound(f"no such face: {face_id}")
                raise ValueError(f"face {face_id} has no {model} embedding yet")
            return v

    def image_for_face(self, face_id):
        with self.session() as s:
            f = s.get(Face, int(face_id))
            if f is None:
                raise NotFound(f"no such face: {face_id}")
            img = s.get(FaceImage, f.image_id)
            return dict(
                path=img.path, medium=img.medium, width=img.width, height=img.height, box=[f.x0, f.y0, f.x1, f.y1]
            )

    # ---- search and clustering -----------------------------------------------------------------------------------------------------------
    def search(self, vector, model, *, k=20, tags=None, include=(), named=None, max_distance=None, exclude=()):
        """The faces nearest `vector` among `model`'s embeddings, in scope (tags, opt-in), closest first. `named` True/False keeps faces with or
        without a confirmed person. Asks the index for more than `k` when it must filter after it (named), so the answer is still k long."""
        k = max(1, min(int(k), 500))
        with self.session() as s:
            stmt = (
                sa.select(FaceEmbedding)
                .join(Face, Face.id == FaceEmbedding.face_id)
                .join(FaceImage, FaceImage.id == Face.image_id)
                .where(FaceEmbedding.model == model, self._scope(s, tags, include))
            )
            if exclude:
                stmt = stmt.where(FaceEmbedding.face_id.not_in([int(x) for x in exclude]))
            want = k if named is None else k * 5 + 50
            hits = nearest(s, FaceEmbedding.vector, vector, k=want, select=stmt, max_distance=max_distance)
            state = self._current(s, [e.face_id for e, _ in hits])
            out = []
            for e, d in hits:
                st = state.get(e.face_id)
                is_named = st is not None and st.status == "confirmed" and st.identity_id is not None
                if named is not None and bool(named) != is_named:
                    continue
                f = s.get(Face, e.face_id)
                img = s.get(FaceImage, f.image_id)
                out.append(dict(face=face_row(f, st), distance=round(float(d), 6), app=img.app, ref=img.ref))
                if len(out) >= k:
                    break
            return out

    def cluster(self, model, *, tags=None, include=(), threshold=None, min_size=2, limit=MAX_CLUSTER, min_score=None):
        """Faces nobody has decided about (no confirmed person, not rejected), grouped by average linkage on cosine distance, largest group first.
        The newest `limit` faces in scope are grouped."""
        from scipy.cluster.hierarchy import fcluster, linkage

        t = CLUSTER_THRESHOLD if threshold is None else float(threshold)
        limit = max(2, min(int(limit), MAX_CLUSTER))
        with self.session() as s:
            q = (
                sa.select(FaceEmbedding.face_id)
                .join(Face, Face.id == FaceEmbedding.face_id)
                .join(FaceImage, FaceImage.id == Face.image_id)
                .where(FaceEmbedding.model == model, self._scope(s, tags, include))
                .order_by(Face.id.desc())
            )
            if min_score is not None:
                q = q.where(Face.det_score >= float(min_score))
            state = self._current(s)
            free = [f for f in s.scalars(q) if f not in state or state[f].status == "suggested"][
                :limit
            ]  # an undone decision frees the face
            rows = (
                s.execute(
                    sa.select(FaceEmbedding.face_id, FaceEmbedding.vector).where(
                        FaceEmbedding.model == model, FaceEmbedding.face_id.in_(free)
                    )
                ).all()
                if free
                else []
            )
        if len(rows) < max(2, min_size):
            return []
        ids = np.array([r[0] for r in rows])
        E = np.array([r[1] for r in rows], dtype=np.float32)
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
        lab = fcluster(linkage(E, "average", metric="cosine"), t, "distance")
        groups = defaultdict(list)
        for i, c in zip(ids, lab, strict=True):
            groups[int(c)].append(int(i))
        out = [dict(faces=sorted(g), size=len(g)) for g in groups.values() if len(g) >= min_size]
        out.sort(key=lambda g: (-g["size"], g["faces"][0]))
        return out

    # ---- people ----------------------------------------------------------------------------------------------------------------------------
    def _root(self, s, identity_id):
        """The identity a merged one now lives in (following merges; a loop is impossible, merge refuses one)."""
        seen = set()
        i = int(identity_id)
        while i not in seen:
            seen.add(i)
            row = s.get(Identity, i)
            if row is None or row.merged_into is None:
                return i
            i = row.merged_into
        return i

    def _members(self, s, root_id):
        """A person and everyone merged into them (directly or through another merge), the person first."""
        out, todo = [root_id], [root_id]
        while todo:
            kids = list(s.scalars(sa.select(Identity.id).where(Identity.merged_into.in_(todo))))
            todo = [k for k in kids if k not in out]
            out += todo
        return out

    def _identity_dict(self, s, row, app=None, counts=None):
        # An app's fields and id for a merged person stay on the merged row (so an unmerge gives them back) and are read
        # through the survivor; the survivor's own win where both have them.
        members = self._members(s, row.id)
        rank = {m: i for i, m in enumerate(members)}
        metas = sorted(
            s.scalars(sa.select(IdentityAppMetadata).where(IdentityAppMetadata.identity_id.in_(members))),
            key=lambda m: -rank[m.identity_id],
        )
        apps = {}
        for m in metas:
            had = apps.get(m.app, dict(app_ref=None, fields={}))
            apps[m.app] = dict(app_ref=m.app_ref or had["app_ref"], fields=dict(had["fields"], **(m.fields or {})))
        glob = dict(row.fields or {})
        fields = dict(glob, **({"name": row.name} if row.name else {}))
        if app and app in apps:
            fields.update(apps[app]["fields"])
        if counts is None:
            counts = self._face_counts(s, [row.id])
        return dict(
            id=row.id,
            name=fields.get("name"),
            fields=fields,
            global_fields=glob,
            apps=apps,
            merged_into=row.merged_into,
            merged=sorted(s.scalars(sa.select(Identity.id).where(Identity.merged_into == row.id))),
            faces=counts.get(row.id, 0),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def _face_counts(self, s, identity_ids=None):
        """identity id -> confirmed faces, counting the faces of identities merged into it."""
        out = defaultdict(int)
        for a in self._current(s).values():
            if a.status == "confirmed" and a.identity_id is not None:
                out[a.identity_id] += 1
        return out

    def identity(self, identity_id, app=None):
        with self.session() as s:
            row = s.get(Identity, self._root(s, identity_id))
            if row is None:
                raise NotFound(f"no such person: {identity_id}")
            return self._identity_dict(s, row, app)

    def identities(self, app=None, q=None, app_ref=None, limit=200):
        with self.session() as s:
            limit = max(1, min(int(limit), 2000))
            stmt = sa.select(Identity).where(Identity.merged_into.is_(None)).order_by(Identity.id)
            if app_ref is not None:  # the app's id may sit on a person since merged into another: answer the survivor
                owners = s.scalars(
                    sa.select(IdentityAppMetadata.identity_id).where(
                        IdentityAppMetadata.app == app, IdentityAppMetadata.app_ref == str(app_ref)
                    )
                )
                stmt = stmt.where(Identity.id.in_({self._root(s, o) for o in owners}))
            if (
                not q
            ):  # a text search reads every person and limits after matching, so a match past the first `limit` is found
                stmt = stmt.limit(limit)
            rows = list(s.scalars(stmt))
            counts = self._face_counts(s)
            out = [self._identity_dict(s, r, app, counts) for r in rows]
            if q:
                needle = str(q).lower()
                out = [
                    i
                    for i in out
                    if any(
                        needle in str(v).lower()
                        for v in [i["name"], *i["fields"].values(), *(m["app_ref"] for m in i["apps"].values())]
                        if v
                    )
                ][:limit]
            return out

    def create_identity(self, name=None, fields=None, app=None, app_fields=None, app_ref=None):
        now = self.clock()
        with self.session() as s:
            row = Identity(
                created_at=now,
                updated_at=now,
                name=(str(name).strip()[:300] or None) if name else None,
                fields=dict(fields or {}),
            )
            s.add(row)
            s.flush()
            if app and (app_fields is not None or app_ref is not None):
                self._set_app(s, row.id, app, app_fields or {}, app_ref)
            s.commit()
            return self._identity_dict(s, row, app)

    def update_identity(self, identity_id, name=None, fields=None):
        with self.session() as s:
            row = s.get(Identity, self._root(s, identity_id))
            if row is None:
                raise NotFound(f"no such person: {identity_id}")
            if name is not None:
                row.name = str(name).strip()[:300] or None
            if fields is not None:
                merged = dict(row.fields or {})
                for k, v in fields.items():
                    if v is None:
                        merged.pop(k, None)
                    else:
                        merged[k] = v
                row.fields = merged
            row.updated_at = self.clock()
            s.commit()
            return self._identity_dict(s, row)

    def _set_app(self, s, identity_id, app, fields, app_ref):
        app = str(app)[:64]
        if app_ref is not None:
            taken = s.scalar(
                sa.select(IdentityAppMetadata).where(
                    IdentityAppMetadata.app == app, IdentityAppMetadata.app_ref == str(app_ref)
                )
            )
            if taken is not None and taken.identity_id != identity_id:
                if self._root(s, taken.identity_id) != self._root(s, identity_id):
                    raise ValueError(f"{app} already calls person {taken.identity_id} {app_ref!r}")
                app_ref = None  # a person merged into this one holds it, and it is already read through this one
        m = s.get(IdentityAppMetadata, (identity_id, app))
        if m is None:
            m = IdentityAppMetadata(identity_id=identity_id, app=app, updated_at=self.clock())
            s.add(m)
        m.fields, m.updated_at = dict(fields or {}), self.clock()
        if app_ref is not None:
            m.app_ref = str(app_ref)[:256]

    def set_app_fields(self, identity_id, app, fields, app_ref=None):
        with self.session() as s:
            row = s.get(Identity, self._root(s, identity_id))
            if row is None:
                raise NotFound(f"no such person: {identity_id}")
            self._set_app(s, row.id, app, fields, app_ref)
            s.commit()
            return self._identity_dict(s, row, app)

    def faces_of(self, identity_id, status="confirmed", limit=500):
        """The faces assigned to a person (and to anyone merged into them) whose current state is `status`, best detected first."""
        with self.session() as s:
            root = self._root(s, identity_id)
            if s.get(Identity, root) is None:
                raise NotFound(f"no such person: {identity_id}")
            ids = [a.face_id for a in self._current(s).values() if a.identity_id == root and a.status == status]
            faces = list(
                s.scalars(
                    sa.select(Face)
                    .where(Face.id.in_(ids))
                    .order_by(Face.det_score.desc(), Face.id)
                    .limit(max(1, min(int(limit), 5000)))
                )
            )
            state = self._current(s, [f.id for f in faces])
            return [face_row(f, state.get(f.id)) for f in faces]

    # ---- decisions -------------------------------------------------------------------------------------------------------------------------
    def assign(self, face_ids, identity_id, app, status="confirmed", source="human", confidence=None, note=None):
        if status not in STATUSES:
            raise ValueError(f"status is one of {', '.join(STATUSES)}, not {status!r}")
        if identity_id is None and status != "rejected":
            raise ValueError("only a rejection may name no person (the face is not a face)")
        face_ids = sorted({int(f) for f in face_ids})
        if not face_ids:
            raise ValueError("no faces given")
        now = self.clock()
        with self.session() as s:
            root = None
            if identity_id is not None:
                root = self._root(s, identity_id)
                if s.get(Identity, root) is None:
                    raise NotFound(f"no such person: {identity_id}")
            found = set(s.scalars(sa.select(Face.id).where(Face.id.in_(face_ids))))
            missing = [f for f in face_ids if f not in found]
            if missing:
                raise NotFound(f"no such face: {missing[0]}")
            b = FaceBatch(kind="assign", app=str(app)[:64], at=now, note=(str(note)[:500] if note else None))
            s.add(b)
            s.flush()
            for f in face_ids:
                s.add(
                    FaceAssignment(
                        face_id=f,
                        status=status,
                        app=b.app,
                        at=now,
                        identity_id=root,
                        source=str(source)[:16],
                        batch_id=b.id,
                        confidence=confidence,
                    )
                )
            s.commit()
            return b.id

    def undo(self, batch_id):
        with self.session() as s:
            b = s.get(FaceBatch, int(batch_id))
            if b is None:
                raise NotFound(f"no such batch: {batch_id}")
            if b.kind != "assign":
                raise ValueError(f"batch {batch_id} is a {b.kind}: undo a merge with unmerge")
            if b.undone_at is None:
                b.undone_at = self.clock()
                s.commit()
            return dict(id=b.id, undone_at=b.undone_at)

    def merge(self, identity_id, into, app):
        """Merge one person into another: a person's decision (the service only suggests). Their faces are read through `into` from now on."""
        with self.session() as s:
            a, b = self._root(s, identity_id), self._root(s, into)
            if s.get(Identity, a) is None or s.get(Identity, b) is None:
                raise NotFound(f"no such person: {identity_id if s.get(Identity, a) is None else into}")
            if a == b:
                raise ValueError("they are already the same person")
            row = s.get(Identity, a)
            row.merged_into, row.updated_at = b, self.clock()
            s.add(FaceBatch(kind="merge", app=str(app)[:64], at=self.clock(), note=f"{a} into {b}"))
            s.commit()
            return self._identity_dict(s, s.get(Identity, b), app)

    def unmerge(self, identity_id):
        with self.session() as s:
            row = s.get(Identity, int(identity_id))
            if row is None:
                raise NotFound(f"no such person: {identity_id}")
            row.merged_into, row.updated_at = None, self.clock()
            s.commit()
            return self._identity_dict(s, row)

    def not_same(self, a, b):
        a, b = sorted((int(a), int(b)))
        if a == b:
            raise ValueError("a person is the same as themself")
        with self.session() as s:
            for i in (a, b):
                if s.get(Identity, i) is None:
                    raise NotFound(f"no such person: {i}")
            if s.get(IdentityNotSame, (a, b)) is None:
                s.add(IdentityNotSame(a=a, b=b, at=self.clock()))
                s.commit()
        return dict(a=a, b=b)

    def merge_suggestions(self, model, limit=50, max_distance=SUGGEST_MAX_DISTANCE):
        """Pairs of people whose confirmed faces sit close: the distance between their centroids (of up to EXEMPLARS best faces each), closest
        first, leaving out pairs a person said are different. For a person to confirm; nothing is merged here."""
        with self.session() as s:
            by_person = defaultdict(list)
            for a in self._current(s).values():
                if a.status == "confirmed" and a.identity_id is not None:
                    by_person[a.identity_id].append(a.face_id)
            cent = {}
            for pid, fids in by_person.items():
                rows = s.execute(
                    sa.select(FaceEmbedding.vector)
                    .join(Face, Face.id == FaceEmbedding.face_id)
                    .where(FaceEmbedding.model == model, FaceEmbedding.face_id.in_(fids))
                    .order_by(Face.det_score.desc())
                    .limit(EXEMPLARS)
                ).all()
                if rows:
                    E = np.array([r[0] for r in rows], dtype=np.float64)
                    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
                    c = E.mean(0)
                    cent[pid] = c / max(np.linalg.norm(c), 1e-9)
            apart = {(r.a, r.b) for r in s.scalars(sa.select(IdentityNotSame))}
            apps = defaultdict(set)
            for m in s.scalars(sa.select(IdentityAppMetadata)):
                apps[m.identity_id].add(m.app)
        ids = sorted(cent)
        if len(ids) < 2:
            return []
        C = np.array([cent[i] for i in ids])
        D = 1.0 - C @ C.T
        out = []
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                if (a, b) in apart or D[x, y] > max_distance:
                    continue
                out.append(
                    dict(a=a, b=b, distance=round(float(D[x, y]), 6), apps_a=sorted(apps[a]), apps_b=sorted(apps[b]))
                )
        out.sort(key=lambda p: p["distance"])
        return out[: max(1, min(int(limit), 500))]
