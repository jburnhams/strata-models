"""The face store: one database of faces, embeddings and people for every app (tables in db/models.py, the face section).

What lives here is record keeping and search; what a match means stays in the calling app (auto-approve thresholds, review workflow,
consent). Decided with the owner 2026-11-25:
  - one global identity per person; each app may add its own fields (identity_app_metadata), which win when that app reads the person;
  - every picture carries its app as a tag; search and clustering cover every tag except opt-in ones (face_tags.opt_in, e.g. other people's
    children), which count only when the caller names them; an opt-in tag with share_named (2026-11-28) still lends its faces a person
    confirmed for someone to every search, so naming a face on a family tape helps recognise that person in other apps;
  - merges across apps are suggested here and made by a person, never automatically; undone by clearing `merged_into`;
  - decisions about faces are append-only (face_assignments), the newest not undone wins, and a batch undoes as one;
  - one global layer of decisions (2026-11-26): a confirmation counts for every app, its app kept as the source. An app's possible labels
    (face_hints: names on a picture, or on a face) are not decisions; `propose` turns hints that agree into `suggested` faces for a person.

Every call opens its own session (as store.Store does), so the API's threads never share one. Times are epoch seconds."""

import dataclasses
import time
from collections import defaultdict
from typing import cast

import numpy as np
import sqlalchemy as sa
from sqlalchemy.orm import Session
from strata_python.db.vectors import nearest

from stratamodels.db.models import (
    Face,
    FaceAssignment,
    FaceBatch,
    FaceEmbedding,
    FaceHint,
    FaceImage,
    FaceImageTag,
    FaceTag,
    Identity,
    IdentityAppMetadata,
    IdentityNotSame,
)
from stratamodels.faces import propose as plan_hints

STATUSES = ("confirmed", "rejected", "suggested")
HINT_STATUSES = ("possible", "confirmed")
CHUNK = 5000
REVIEW_MIN_CONFIRMED = 3  # confirmed faces before a person's own faces, not an app's labels, rank their suggestions
REVIEW_POOL = 2000  # most confident suggestions grouped for the first round
REVIEW_MAX = 10000  # faces in one answer: a person's whole list above a likeness, at once
REVIEW_CLUSTER = 0.35  # tight: the first round should be faces that are surely one person  # ids per IN (...) when reading many pictures' faces
CLUSTER_THRESHOLD = 0.55  # cosine distance on ArcFace embeddings: conservative (strata-360's `who`; bright-horizons' unknown groups used 0.55)
MAX_CLUSTER = 5000  # faces grouped at once: average linkage keeps an n x n distance matrix (5000: 100 MB of float32)
SUGGEST_MAX_DISTANCE = 0.6  # identity centroids further apart than this are not worth a person's look
TRACK_SUGGEST = (
    10,
    0.5,
    0.6,
)  # a group's nearest confirmed faces, how near, and the share one person must hold to be suggested (the face-tracks eval's labelling, 2026-11-27)
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


def default_bar():
    """(min_score, min_px) from the service's settings: faces_min_score and faces_min_px."""
    from stratamodels import settings

    return settings.faces_min_score(), settings.faces_min_px()


class FaceStore:
    def __init__(self, engine, clock=time.time, bar=default_bar):
        """`bar` gives (min_score, min_px): the faces worth a person's look or a match, read on every query so a changed setting applies at
        once. Faces under it are kept (a lower bar later brings them back) but never searched, grouped, suggested or proposed."""
        self.engine, self.clock, self.bar = engine, clock, bar

    def usable(self, min_score=None):
        """The store's one bar for a face worth matching: the detector's score at least faces_min_score (or `min_score` when higher), and the
        box's shorter side at least faces_min_px of the picture's pixels. Applied here, in every search, grouping, review and proposal, so
        no app has to filter tiny or doubtful faces itself."""
        score, px = self.bar()
        if min_score is not None:
            score = max(score, float(min_score))
        conds = []
        if score > 0:
            conds.append(Face.det_score >= score)
        if px > 0:
            conds += [Face.x1 - Face.x0 >= px, Face.y1 - Face.y0 >= px]
        return sa.and_(sa.true(), *conds)

    def session(self):
        return Session(self.engine, expire_on_commit=False)

    # ---- tags --------------------------------------------------------------------------------------------------------------------------
    def _scope(self, s, tags=None, include=(), named_too=False):
        """A condition on FaceImage.id: pictures with any of `tags` (when given), never one with an opt-in tag that is not in `include` or
        `tags` (naming a tag in `tags` counts as asking for it). `named_too` lets in the pictures of opt-in tags that share their named
        faces; the caller must then keep only their confirmed faces (`_shared_named`)."""
        conds = []
        if tags:
            conds.append(FaceImage.id.in_(sa.select(FaceImageTag.image_id).where(FaceImageTag.tag.in_(list(tags)))))
        hidden = self._hidden(s, tags, include)
        if named_too:
            hidden = [t for t in hidden if t not in self._sharing(s)]
        if hidden:
            conds.append(FaceImage.id.not_in(sa.select(FaceImageTag.image_id).where(FaceImageTag.tag.in_(hidden))))
        return sa.and_(sa.true(), *conds)

    def _hidden(self, s, tags=None, include=()):
        """The opt-in tags the caller did not ask for."""
        allowed = set(include or ()) | set(tags or ())
        return [t for t in s.scalars(sa.select(FaceTag.tag).where(FaceTag.opt_in.is_(True))) if t not in allowed]

    def _sharing(self, s):
        """The opt-in tags whose named faces count everywhere."""
        return set(s.scalars(sa.select(FaceTag.tag).where(FaceTag.opt_in.is_(True), FaceTag.share_named.is_(True))))

    def _shared_named(self, s, tags=None, include=()):
        """With `_scope(named_too=True)`: a condition on Face.id that keeps, of the pictures in scope only through a sharing tag, just the
        faces confirmed for someone, so the nearest-neighbour limit counts only faces that can be answered. None when no tag shares."""
        partial = [t for t in self._hidden(s, tags, include) if t in self._sharing(s)]
        if not partial:
            return None
        lent = sa.select(FaceImageTag.image_id).where(FaceImageTag.tag.in_(partial))
        named = []
        for part in _chunks(list(s.scalars(sa.select(Face.id).where(Face.image_id.in_(lent)).order_by(Face.id)))):
            named += [
                f for f, a in self._current(s, part).items() if a.status == "confirmed" and a.identity_id is not None
            ]
        return sa.or_(Face.image_id.not_in(lent), Face.id.in_(named)) if named else Face.image_id.not_in(lent)

    def tags(self):
        with self.session() as s:
            counts = dict(s.execute(sa.select(FaceImageTag.tag, sa.func.count()).group_by(FaceImageTag.tag)).all())
            opt = {t.tag: t for t in s.scalars(sa.select(FaceTag))}
            return [
                dict(
                    tag=t,
                    images=counts.get(t, 0),
                    opt_in=bool(opt[t].opt_in) if t in opt else False,
                    share_named=bool(opt[t].share_named) if t in opt else False,
                    note=opt[t].note if t in opt else "",
                )
                for t in sorted(set(counts) | set(opt))
            ]

    def set_tag(self, tag, opt_in, note=None, share_named=None):
        """Sets a tag's opt-in (and, when given, whether its named faces count everywhere; left as it was when None)."""
        tag = str(tag).strip()
        if not tag:
            raise ValueError("a tag needs a name")
        with self.session() as s:
            row = s.get(FaceTag, tag)
            if row is None:
                row = FaceTag(tag=tag, updated_at=self.clock())
                s.add(row)
            row.opt_in, row.updated_at = bool(opt_in), self.clock()
            if share_named is not None:
                row.share_named = bool(share_named)
            if note is not None:
                row.note = str(note)[:500]
            s.commit()
            return dict(tag=tag, opt_in=bool(opt_in), share_named=bool(row.share_named))

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

    def delete_images(self, *, ids=None, app=None, tag=None, dry_run=True):
        """Remove pictures from the store with everything hanging on them: their faces, embeddings, decisions (assignments) and hints. Pick
        them by `ids`, `app`, `tag` (pictures matching every one given; `ids=[]` matches none); at least one is needed, so nothing removes the whole store.
        `dry_run` (the default) only counts. Answers {images, faces, decisions, confirmed, hints, deleted, media}: `confirmed` is how many of
        the faces a person had named (their decisions go too), `media` the kept copies no other picture uses (for the caller to remove)."""
        if ids is None and not app and not tag:
            raise ValueError("say which pictures: ids, app or tag")
        with self.session() as s:
            q = sa.select(FaceImage.id)
            if (
                ids is not None
            ):  # an empty list matches nothing: it never widens the other selectors to all their pictures
                q = q.where(FaceImage.id.in_([int(i) for i in ids]))
            if app:
                q = q.where(FaceImage.app == str(app))
            if tag:
                q = q.where(FaceImage.id.in_(sa.select(FaceImageTag.image_id).where(FaceImageTag.tag == str(tag))))
            images = list(s.scalars(q))
            faces = []
            for part in _chunks(images):
                faces += list(s.scalars(sa.select(Face.id).where(Face.image_id.in_(part))))
            state = self._current(s, faces) if faces else {}
            decisions = hints = 0
            for part in _chunks(faces):
                decisions += s.scalar(sa.select(sa.func.count()).where(FaceAssignment.face_id.in_(part))) or 0
            for part in _chunks(images):
                hints += s.scalar(sa.select(sa.func.count()).where(FaceHint.image_id.in_(part))) or 0
            media = set()
            for part in _chunks(images):
                media.update(m for m in s.scalars(sa.select(FaceImage.medium).where(FaceImage.id.in_(part))) if m)
            if media:  # a kept copy is named by its content: another picture may share it
                media -= set(
                    s.scalars(
                        sa.select(FaceImage.medium).where(
                            FaceImage.medium.in_(sorted(media)), FaceImage.id.not_in(images)
                        )
                    )
                )
            out = dict(
                images=len(images),
                faces=len(faces),
                decisions=decisions,
                confirmed=sum(1 for a in state.values() if a.status == "confirmed"),
                hints=hints,
                deleted=False,
                media=sorted(media),
            )
            if dry_run or not images:
                return out
            for part in _chunks(faces):  # by hand, not by ON DELETE CASCADE: SQLite enforces it only when asked to
                s.execute(sa.delete(FaceAssignment).where(FaceAssignment.face_id.in_(part)))
                s.execute(sa.delete(FaceHint).where(FaceHint.face_id.in_(part)))
                s.execute(sa.delete(FaceEmbedding).where(FaceEmbedding.face_id.in_(part)))
            for part in _chunks(images):
                s.execute(sa.delete(FaceHint).where(FaceHint.image_id.in_(part)))
                s.execute(sa.delete(Face).where(Face.image_id.in_(part)))
                s.execute(sa.delete(FaceImageTag).where(FaceImageTag.image_id.in_(part)))
                s.execute(sa.delete(FaceImage).where(FaceImage.id.in_(part)))
            s.commit()
            out["deleted"] = True
            return out

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
        without a confirmed person. The confirmed faces of an opt-in tag that shares them are in scope too (never its other faces, which are
        left out before the index's limit, so they never crowd out a face that can be answered). Asks the index for more than `k` when it
        must filter after it (named), so the answer is still k long."""
        k = max(1, min(int(k), 500))
        with self.session() as s:
            stmt = (
                sa.select(FaceEmbedding)
                .join(Face, Face.id == FaceEmbedding.face_id)
                .join(FaceImage, FaceImage.id == Face.image_id)
                .where(
                    FaceEmbedding.model == model,
                    self._scope(s, tags, include, named_too=named is not False),
                    self.usable(),
                )
            )
            if exclude:
                stmt = stmt.where(FaceEmbedding.face_id.not_in([int(x) for x in exclude]))
            lent = self._shared_named(s, tags, include) if named is not False else None
            if lent is not None:
                stmt = stmt.where(lent)
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

    def cluster(
        self,
        model,
        *,
        tags=None,
        include=(),
        threshold=None,
        min_size=2,
        limit=MAX_CLUSTER,
        min_score=None,
        suggested=True,
    ):
        """Faces nobody has decided about (no confirmed person, not rejected), grouped by average linkage on cosine distance, largest group first.
        The newest `limit` faces in scope are grouped. `suggested` False leaves out faces waiting on a suggestion too (they are decided on
        their person's page, not as unnamed)."""
        from scipy.cluster.hierarchy import fcluster, linkage

        t = CLUSTER_THRESHOLD if threshold is None else float(threshold)
        limit = max(2, min(int(limit), MAX_CLUSTER))
        with self.session() as s:
            q = (
                sa.select(FaceEmbedding.face_id)
                .join(Face, Face.id == FaceEmbedding.face_id)
                .join(FaceImage, FaceImage.id == Face.image_id)
                .where(FaceEmbedding.model == model, self._scope(s, tags, include), self.usable(min_score))
                .order_by(Face.id.desc())
            )
            state = self._current(s)
            free = [f for f in s.scalars(q) if f not in state or (suggested and state[f].status == "suggested")][
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

    def group_tracks(self, model, tracks, *, apart=(), threshold=None, tags=None, include=(), suggest=TRACK_SUGGEST):
        """A video's face tracks (strata_python face-tracks answers: [{key, embedding}]) grouped into people, for an app to name a group at a
        time instead of a track: average linkage on the tracks' embeddings (`group_apart`), never putting two tracks in one group when
        `apart` says they were seen together (two faces in one frame are two people). Each group gets the known person its mean embedding
        is nearest, when the `suggest` (k, max distance, share) nearest confirmed faces in scope mostly agree; it is only a suggestion, and
        nothing is stored. Largest group first. Nothing here reads the tracks back: an app keeps them, and asks again after a change.
        A track that names its stored stills (`faces`) and has none over the store's bar is not grouped but answered in `hidden`, so an app
        never has to judge tiny or doubtful faces itself. Returns {groups, hidden}."""
        tracks = list(tracks)
        named = [int(f) for t in tracks for f in (t.get("faces") or [])]
        if named:
            with self.session() as s:
                ok = set(s.scalars(sa.select(Face.id).where(Face.id.in_(named), self.usable())))
            hidden = [str(t["key"]) for t in tracks if t.get("faces") and not ok & {int(f) for f in t["faces"]}]
            tracks = [t for t in tracks if str(t["key"]) not in set(hidden)]
        else:
            hidden = []
        keys = [str(t["key"]) for t in tracks]
        if len(set(keys)) != len(keys):
            raise ValueError("every track needs a key of its own")
        if len(keys) > MAX_CLUSTER:
            raise ValueError(f"at most {MAX_CLUSTER} tracks are grouped at once, not {len(keys)}")
        if not keys:
            return dict(groups=[], hidden=hidden)
        E = np.array([t["embedding"] for t in tracks], dtype=np.float32)
        if E.ndim != 2:
            raise ValueError("every track needs an embedding of the same length")
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
        index = {k: i for i, k in enumerate(keys)}
        pairs = [(index[str(a)], index[str(b)]) for a, b in apart if str(a) in index and str(b) in index]
        t = CLUSTER_THRESHOLD if threshold is None else float(threshold)
        k, max_distance, share = suggest
        out = []
        for members in group_apart(E, pairs, t):
            mean = E[members].mean(axis=0)
            mean /= max(float(np.linalg.norm(mean)), 1e-9)
            hits = self.search(
                [float(x) for x in mean], model, k=k, tags=tags, include=include, named=True, max_distance=max_distance
            )
            out.append(dict(tracks=[keys[i] for i in members], size=len(members), suggested=self._vote(hits, share)))
        out.sort(key=lambda g: (-g["size"], index[g["tracks"][0]]))
        return dict(groups=out, hidden=hidden)

    def _vote(self, hits, share):
        """The person most of `hits` (confirmed faces) are, with their name and share, or None when no one holds `share` of them (and two)."""
        with self.session() as s:
            votes = defaultdict(int)
            for h in hits:
                votes[self._root(s, h["face"]["identity_id"])] += 1
            if not votes:
                return None
            who, n = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))
            if n < max(2, share * len(hits)):
                return None
            return dict(
                identity_id=who,
                name=self._identity_dict(s, s.get(Identity, who))["name"],
                share=round(n / len(hits), 3),
            )

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
            faces=counts.get(row.id, (0, 0))[0],
            suggested=counts.get(row.id, (0, 0))[1],
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def _tagged(self, s, face_ids, tag):
        """The faces among `face_ids` whose picture carries `tag` (a source: an app's name, or a tag it gave)."""
        out = set()
        for part in _chunks(sorted(face_ids)):
            out.update(
                s.scalars(
                    sa.select(Face.id)
                    .join(FaceImageTag, FaceImageTag.image_id == Face.image_id)
                    .where(Face.id.in_(part), FaceImageTag.tag == str(tag))
                )
            )
        return out

    def _face_counts(self, s, identity_ids=None, tag=None):
        """identity id -> [confirmed, suggested] faces, counting the faces of identities merged into it; with `tag`, only faces of pictures
        carrying it."""
        out = defaultdict(lambda: [0, 0])
        state = self._current(s)
        if tag:
            keep = self._tagged(s, [f for f, a in state.items() if a.identity_id is not None], tag)
            state = {f: a for f, a in state.items() if f in keep}
        for a in state.values():
            if a.identity_id is not None and a.status in ("confirmed", "suggested"):
                out[a.identity_id][0 if a.status == "confirmed" else 1] += 1
        return out

    def identity(self, identity_id, app=None):
        with self.session() as s:
            row = s.get(Identity, self._root(s, identity_id))
            if row is None:
                raise NotFound(f"no such person: {identity_id}")
            return self._identity_dict(s, row, app)

    def identities(self, app=None, q=None, app_ref=None, limit=200, tag=None):
        """People, as `app` sees them. With `tag` (a source), only people with a face in pictures carrying it, their counts of those faces."""
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
            counts = self._face_counts(s, tag=tag)
            if tag:  # a source's people: every one with a face there, however far down the list
                stmt = stmt.where(Identity.id.in_([i for i, c in counts.items() if c[0] or c[1]]))
            rows = list(s.scalars(stmt))
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

    def faces_of(self, identity_id, status="confirmed", limit=500, offset=0):
        """The faces assigned to a person (and to anyone merged into them) whose current state is `status`, best detected first; `offset`
        skips that many (pages of a long list)."""
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
                    .offset(max(0, int(offset)))
                    .limit(max(1, min(int(limit), 5000)))
                )
            )
            state = self._current(s, [f.id for f in faces])
            return [face_row(f, state.get(f.id)) for f in faces]

    def review(
        self,
        identity_id,
        model,
        limit=48,
        min_confirmed=REVIEW_MIN_CONFIRMED,
        pool=REVIEW_POOL,
        threshold=REVIEW_CLUSTER,
        min_likeness=None,
        tag=None,
    ):
        """The next round of a person's suggested faces to decide, most certain first, scored afresh on every call (so each round a person
        confirms makes the next one surer). The model of the person: with `min_confirmed` confirmed faces, the centre of their densest bunch
        (up to EXEMPLARS*10 best detected); before that, the largest group (average linkage at `threshold`) among their `pool` most confident
        suggestions, every face of which an app already says may be them. A suggested face nearer a face rejected for this person than the
        model is left out of the round (it looks like the one that was not them). Answers {basis, basis_faces, suggested, confirmed, near
        (within 0.3 / 0.4 / 0.5), counts (how many are at least each likeness 0.50-0.95, likeness = 1 - distance), left_out, items: faces with
        `distance`, closest first}. `min_likeness` keeps only the faces at least that alike (with a large `limit`: all of them at once); `tag`
        only the suggestions from pictures carrying it (one source, such as a video eval's stills)."""
        from scipy.cluster.hierarchy import fcluster, linkage

        centre = None
        with self.session() as s:
            root = self._root(s, identity_id)
            if s.get(Identity, root) is None:
                raise NotFound(f"no such person: {identity_id}")
            state = self._current(s)
            mine = {f: a for f, a in state.items() if a.identity_id == root}
            sug = [f for f, a in mine.items() if a.status == "suggested"]
            if (
                tag and sug
            ):  # one source's suggestions only (the model of the person is still built from all their faces)
                keep = self._tagged(s, sug, tag)
                sug = [f for f in sug if f in keep]
            if (
                sug
            ):  # a suggestion under the store's bar is not offered (it stays, and comes back if the bar is lowered)
                ok = set()
                for part in _chunks(sorted(sug)):
                    ok.update(s.scalars(sa.select(Face.id).where(Face.id.in_(part), self.usable())))
                sug = [f for f in sug if f in ok]
            conf = [f for f, a in mine.items() if a.status == "confirmed"]
            rej = [f for f, a in mine.items() if a.status == "rejected"]
            out = dict(basis=None, basis_faces=0, suggested=len(sug), confirmed=len(conf), near={}, counts={}, items=[])
            if not sug:
                return out
            emb = {}
            for part in _chunks(sorted(sug + rej)):
                for fid, vec in s.execute(
                    sa.select(FaceEmbedding.face_id, FaceEmbedding.vector).where(
                        FaceEmbedding.model == model, FaceEmbedding.face_id.in_(part)
                    )
                ):
                    emb[fid] = plan_hints.unit(np.array(vec, dtype=np.float64))
            if len(conf) >= min_confirmed:
                C = [
                    plan_hints.unit(np.array(v, dtype=np.float64))
                    for (v,) in s.execute(
                        sa.select(FaceEmbedding.vector)
                        .join(Face, Face.id == FaceEmbedding.face_id)
                        .where(FaceEmbedding.model == model, FaceEmbedding.face_id.in_(conf))
                        .order_by(Face.det_score.desc())
                        .limit(EXEMPLARS * 10)
                    )
                ]
                if C:
                    centre = plan_hints.centroid(np.array(C))
                    out.update(basis="confirmed", basis_faces=len(C))
        cand = [f for f in sug if f in emb]
        if not cand:
            return out
        if centre is None:
            best = sorted(cand, key=lambda f: -(mine[f].confidence or 0.0))[: max(2, int(pool))]
            X = np.array([emb[f] for f in best])
            if len(best) > 1:
                lab = fcluster(linkage(X, "average", metric="cosine"), float(threshold), "distance")
                top = max(sorted(set(lab.tolist())), key=lambda c: int((lab == c).sum()))
                group = X[lab == top]
            else:
                group = X
            centre = plan_hints.unit(group.mean(0))
            out.update(basis="group", basis_faces=len(group))
        A = np.array([emb[f] for f in cand])
        d = 1.0 - A @ centre
        if rej and any(f in emb for f in rej):
            R = np.array([emb[f] for f in rej if f in emb])
            nearest_rej = (1.0 - A @ R.T).min(1)
            keep = nearest_rej >= d
        else:
            keep = np.ones(len(cand), dtype=bool)
        out["near"] = {k: int(((d < float(k)) & keep).sum()) for k in ("0.3", "0.4", "0.5")}
        out["counts"] = {f"{x / 100:.2f}": int(((1.0 - d >= x / 100 - 1e-9) & keep).sum()) for x in range(50, 100, 5)}
        out["left_out"] = int((~keep).sum())
        if min_likeness is not None:
            keep = keep & (1.0 - d >= float(min_likeness) - 1e-9)
        order = [int(i) for i in np.argsort(d) if keep[i]][: max(1, min(int(limit), REVIEW_MAX))]
        with self.session() as s:
            faces = {}
            for part in _chunks([cand[i] for i in order]):
                faces.update({f.id: f for f in s.scalars(sa.select(Face).where(Face.id.in_(part)))})
        out["items"] = [dict(face_row(faces[cand[i]], mine[cand[i]]), distance=round(float(d[i]), 6)) for i in order]
        return out

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
            if b.kind not in ("assign", "hint"):
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
        first, leaving out pairs a person said are different. For a person to confirm; nothing is merged here. A person with no confirmed face
        yet (one `propose` made from an app's hints) is compared by their suggested faces (`basis_a`/`basis_b` say which)."""
        with self.session() as s:
            by_person, suggested = defaultdict(list), defaultdict(list)
            for a in self._current(s).values():
                if a.status == "confirmed" and a.identity_id is not None:
                    by_person[a.identity_id].append(a.face_id)
                elif a.status == "suggested" and a.identity_id is not None:
                    suggested[a.identity_id].append(a.face_id)
            basis = dict.fromkeys(by_person, "confirmed")
            for (
                pid,
                fids,
            ) in suggested.items():  # a person proposed from hints has only suggested faces until someone confirms one
                if pid not in by_person:
                    by_person[pid], basis[pid] = fids, "suggested"
            cent = {}
            for pid, fids in by_person.items():
                rows = s.execute(
                    sa.select(FaceEmbedding.vector)
                    .join(Face, Face.id == FaceEmbedding.face_id)
                    .where(FaceEmbedding.model == model, FaceEmbedding.face_id.in_(fids), self.usable())
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
                    dict(
                        a=a,
                        b=b,
                        distance=round(float(D[x, y]), 6),
                        apps_a=sorted(apps[a]),
                        apps_b=sorted(apps[b]),
                        basis_a=basis[a],
                        basis_b=basis[b],
                    )
                )
        out.sort(key=lambda p: p["distance"])
        return out[: max(1, min(int(limit), 500))]

    # ---- hints: an app's possible labels, and proposing from them --------------------------------------------------------------------------
    def add_hints(self, app, hints, note=None):
        """Store an app's possible labels in one batch (undone as one). Each hint: {image_id, app_ref, face_id?, name?, status?}; `face_id`
        must be a face of that picture. A hint the app already has (same picture, face, app_ref and status, batch not undone) is skipped.
        Returns {batch, added, skipped}."""
        app = str(app or "")[:64]
        if not app:
            raise ValueError("hints need an app")
        rows = []
        for h in hints:
            ref = str(h.get("app_ref") or "").strip()[:256]
            if not ref or h.get("image_id") is None:
                raise ValueError("a hint needs image_id and app_ref")
            status = h.get("status") or "possible"
            if status not in HINT_STATUSES:
                raise ValueError(f"a hint's status is one of {', '.join(HINT_STATUSES)}, not {status!r}")
            face_id = None if h.get("face_id") is None else int(h["face_id"])
            name = (str(h["name"]).strip()[:300] or None) if h.get("name") else None
            rows.append((int(h["image_id"]), face_id, ref, name, status))
        if not rows:
            raise ValueError("no hints given")
        now = self.clock()
        with self.session() as s:
            images = {r[0] for r in rows}
            found = set()
            for part in _chunks(sorted(images)):
                found |= set(s.scalars(sa.select(FaceImage.id).where(FaceImage.id.in_(part))))
            missing = sorted(images - found)
            if missing:
                raise NotFound(f"no such picture: {missing[0]}")
            want = sorted({r[1] for r in rows if r[1] is not None})
            face_img = {}
            for part in _chunks(want):
                face_img.update(s.execute(sa.select(Face.id, Face.image_id).where(Face.id.in_(part))).all())
            for img, f, *_ in rows:
                if f is not None and face_img.get(f) != img:
                    raise ValueError(f"face {f} is not in picture {img}")
            have = {(h.image_id, h.face_id, h.app_ref, h.status) for h in self._live_hints(s, app, images)}
            b = FaceBatch(kind="hint", app=app, at=now, note=(str(note)[:500] if note else None))
            s.add(b)
            s.flush()
            added = 0
            for img, f, ref, name, status in rows:
                if (img, f, ref, status) in have:
                    continue
                have.add((img, f, ref, status))
                s.add(
                    FaceHint(
                        app=app, image_id=img, app_ref=ref, at=now, face_id=f, name=name, status=status, batch_id=b.id
                    )
                )
                added += 1
            s.commit()
            return dict(batch=b.id, added=added, skipped=len(rows) - added)

    def _live_hints(self, s, app, images=None):
        live = sa.or_(
            FaceHint.batch_id.is_(None),
            FaceHint.batch_id.in_(sa.select(FaceBatch.id).where(FaceBatch.undone_at.is_(None))),
        )
        q = sa.select(FaceHint).where(FaceHint.app == app, live).order_by(FaceHint.id)
        if images is None:
            return list(s.scalars(q))
        out = []
        for part in _chunks(sorted(images)):
            out += list(s.scalars(q.where(FaceHint.image_id.in_(part))))
        return out

    def hints(self, app, image_id=None, app_ref=None, limit=500):
        """An app's hints (batches not undone), oldest first, optionally for one picture or one of its people."""
        with self.session() as s:
            rows = self._live_hints(s, str(app), None if image_id is None else [int(image_id)])
            if app_ref is not None:
                rows = [h for h in rows if h.app_ref == str(app_ref)]
            return [
                dict(
                    id=h.id,
                    app=h.app,
                    image_id=h.image_id,
                    face_id=h.face_id,
                    app_ref=h.app_ref,
                    name=h.name,
                    status=h.status,
                    batch=h.batch_id,
                    at=h.at,
                )
                for h in rows[: max(1, min(int(limit), 5000))]
            ]

    def _faces_with_embeddings(self, s, model, images):
        out = defaultdict(list)
        for part in _chunks(sorted(images)):
            for fid, img, score, vec in s.execute(
                sa.select(Face.id, Face.image_id, Face.det_score, FaceEmbedding.vector)
                .join(FaceEmbedding, FaceEmbedding.face_id == Face.id)
                .where(FaceEmbedding.model == model, Face.image_id.in_(part), self.usable())
                .order_by(Face.id)
            ):
                out[img].append((fid, plan_hints.unit(np.array(vec, dtype=np.float64)), float(score)))
        return dict(out)

    def propose(self, app, model, *, dry_run=True, groups=False, tags=None, include=(), limit=None, **params):
        """Turn an app's hints into suggested faces (faces/propose.py says how). `dry_run` (the default) only counts. Otherwise each person
        who qualifies gets the global person their app_ref points at (identity_app_metadata), or a new one named from the hint, and their
        suggestions are written as one batch (`suggested`, source `hint`, confidence 1 - distance), undone as one. `groups` also weighs the
        unnamed groups of FaceStore.cluster (tags, include, limit: its scope). Returns {people: [...], qualifying, suggested, batch}."""
        p = plan_hints.Params(**{k: v for k, v in params.items() if v is not None})
        app = str(app)
        group_list = []
        if groups:
            found = self.cluster(
                model,
                tags=tags,
                include=include,
                min_size=p.group_min,
                min_score=p.min_score or None,
                limit=MAX_CLUSTER if limit is None else limit,
            )
            group_list = [list(cast(list[int], g["faces"])) for g in found]
        with self.session() as s:
            rows = self._live_hints(s, app)
            images = {h.image_id for h in rows}
            if group_list:
                members = sorted({f for g in group_list for f in g})
                for part in _chunks(members):
                    images |= set(s.scalars(sa.select(Face.image_id).where(Face.id.in_(part))))
            faces = self._faces_with_embeddings(s, model, images)
            state = self._current(s)
            mapped = {}
            for m in s.scalars(sa.select(IdentityAppMetadata).where(IdentityAppMetadata.app == app)):
                if m.app_ref:
                    mapped[m.app_ref] = self._root(s, m.identity_id)
            by_person = defaultdict(list)
            for a in state.values():
                if a.status == "confirmed" and a.identity_id is not None:
                    by_person[a.identity_id].append(a.face_id)
            confirmed = {}
            for ref, pid in mapped.items():
                fids = by_person.get(pid)
                if fids:
                    confirmed[ref] = [
                        plan_hints.unit(np.array(v, dtype=np.float64))
                        for (v,) in s.execute(
                            sa.select(FaceEmbedding.vector)
                            .join(Face, Face.id == FaceEmbedding.face_id)
                            .where(FaceEmbedding.model == model, FaceEmbedding.face_id.in_(fids))
                            .order_by(Face.det_score.desc())
                            .limit(200)
                        )
                    ]
        hints = [plan_hints.Hint(h.image_id, h.app_ref, h.face_id, h.name, h.status) for h in rows]
        people = plan_hints.plan(hints, faces, set(state), confirmed, group_list, p)
        out = []
        for ref, who in people.items():
            row = who.summary()
            row["identity_id"] = mapped.get(ref)
            out.append(row)
        result = dict(
            app=app,
            model=model,
            dry_run=bool(dry_run),
            params=dataclasses.asdict(p),
            hinted_pictures=len({h.image_id for h in rows}),
            people=out,
            qualifying=sum(1 for w in people.values() if w.qualifies),
            suggested=sum(len(w.suggestions) for w in people.values()),
            groups=len(group_list),
            batch=None,
            created=0,
        )
        if dry_run or not result["suggested"]:
            return result
        now = self.clock()
        with self.session() as s:
            b = FaceBatch(kind="assign", app=app[:64], at=now, note=f"propose from {app} hints")
            s.add(b)
            s.flush()
            for row, who in zip(out, people.values(), strict=True):
                if not who.suggestions:
                    continue
                pid = row["identity_id"]
                if pid is None:
                    ident = Identity(created_at=now, updated_at=now, name=who.name or who.app_ref, fields={})
                    s.add(ident)
                    s.flush()
                    self._set_app(s, ident.id, app, {}, who.app_ref)
                    pid = row["identity_id"] = ident.id
                    result["created"] += 1
                for f, d, _how in who.suggestions:
                    s.add(
                        FaceAssignment(
                            face_id=f,
                            status="suggested",
                            app=app[:64],
                            at=now,
                            identity_id=pid,
                            source="hint",
                            batch_id=b.id,
                            confidence=round(max(0.0, 1.0 - d), 6),
                        )
                    )
            s.commit()
            result["batch"] = b.id
        return result


def _chunks(ids, n=CHUNK):
    for i in range(0, len(ids), n):
        yield ids[i : i + n]


def group_apart(E, apart, threshold):
    """Rows of `E` (unit vectors) grouped by average linkage on cosine distance, cut at `threshold`, with no group holding both rows of a
    pair in `apart`. The whole set is grouped as `cluster` does (scipy); a group that holds such a pair (rare: two people who look alike
    seen together) is grouped again on its own, merging the nearest two parts that may be one person until none is within the threshold,
    so its tracks still join whoever they are nearest. Groups as lists of row numbers, each sorted, in order of their first row."""
    n = len(E)
    if n == 1:
        return [[0]]
    from scipy.cluster.hierarchy import fcluster, linkage

    D = np.clip(1.0 - E @ E.T, 0.0, 2.0)
    labels = fcluster(linkage(D[np.triu_indices(n, 1)], "average"), threshold, "distance")
    groups = defaultdict(list)
    for row, c in enumerate(labels):
        groups[int(c)].append(row)
    partner = defaultdict(set)
    for a, b in apart:
        if a != b:
            partner[a].add(b)
            partner[b].add(a)
    out = []
    for rows in groups.values():
        inside = set(rows)
        if any(partner[r] & inside for r in rows):
            out += _apart(D[np.ix_(rows, rows)], rows, partner, threshold)
        else:
            out.append(sorted(rows))
    return sorted(out, key=lambda g: g[0])


def _apart(D, rows, partner, threshold):
    """Average linkage with cannot-link pairs, step by step (Lance-Williams), on one group's rows: small, so n^3 is nothing."""
    parts = [[r] for r in rows]
    d = D.astype(np.float64).copy()
    np.fill_diagonal(d, np.inf)
    where = {r: i for i, r in enumerate(rows)}
    no = np.zeros(d.shape, dtype=bool)
    for r in rows:
        for p in partner[r]:
            if p in where:
                no[where[r], where[p]] = True
    alive = np.ones(len(rows), dtype=bool)
    while True:
        m = np.where(no | ~alive[:, None] | ~alive[None, :], np.inf, d)
        i, j = divmod(int(np.argmin(m)), len(rows))
        if not m[i, j] <= threshold:
            break
        ni, nj = len(parts[i]), len(parts[j])
        d[i, :] = d[:, i] = (ni * d[i, :] + nj * d[j, :]) / (ni + nj)
        d[i, i] = np.inf
        no[i, :] |= no[j, :]
        no[:, i] = no[i, :]
        parts[i] += parts[j]
        parts[j] = []
        alive[j] = False
    return [sorted(p) for p in parts if p]
