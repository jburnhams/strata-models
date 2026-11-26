"""Turning possible labels (face_hints) into suggested faces: the planning half, pure numpy so it can be run on a copy of the data.

An app's hints say who is in a picture (a Google Photos name tag) or, sometimes, which face is whom (bright-horizons' approved faces). For each
of the app's people (`app_ref`):
  1. seeds: faces that can only be them: a face hinted as them; the one face of a picture whose only hinted person is them; faces already
     confirmed for the global person their app_ref points at;
  2. a centroid of the seeds: the centre of their densest bunch (`centroid`), so a name that is half someone else's faces does not
     average two people into a third;
  3. they qualify when they have `min_seeds` seeds and at least `min_purity` of them lie within `max_distance` of the centroid (an app's name
     on mixed-up faces, or too few photos, is not proposed);
  4. in every picture hinted with a qualifying person, the face clearly closest to their centroid is suggested: within `max_distance`,
     and at least `margin` closer than the picture's next face and than any other person hinted on it (one face, one person; one person,
     one face per picture);
  5. optionally an unnamed group (FaceStore.cluster) is suggested whole for a person when most of its members that sit in this app's hinted
     pictures (`group_share`, at least `group_min` of them) are in pictures hinted with that person, and its centroid is within
     `max_distance` of theirs: the group's faces (each within `max_distance` too) in pictures nobody hinted get a name too.

Faces anyone already decided about (confirmed, rejected or suggested, by a person or an earlier run) are never suggested again, so running
it twice adds nothing. Nothing here is confirmed: the suggestions wait for a person. Measured on the owner's Google Photos names against the
Immich pictures (2026-11-26): seeds of the 12 commonest names sit 88-92 % within 0.5 of their centroid, and other people's labelled
photos have a face that near only 0-1 % of the time (docs/progress.md)."""

import dataclasses
from collections import defaultdict

import numpy as np

MIN_SEEDS = 10
MIN_PURITY = 0.85
MAX_DISTANCE = 0.5  # cosine distance on ArcFace embeddings
MARGIN = 0.05
MIN_SCORE = 0.7  # below it, toddlers' and blurred faces matched the wrong person (0.75 % of faces; 0.05 % at 0.7: docs/progress.md)
GROUP_SHARE = 0.6
GROUP_MIN = 5


@dataclasses.dataclass(frozen=True)
class Params:
    min_seeds: int = MIN_SEEDS
    min_purity: float = MIN_PURITY
    max_distance: float = MAX_DISTANCE
    margin: float = MARGIN
    min_score: float = MIN_SCORE  # faces the detector scored lower are neither seeds nor suggested
    group_share: float = GROUP_SHARE
    group_min: int = GROUP_MIN


@dataclasses.dataclass
class Hint:
    image_id: int
    app_ref: str
    face_id: int | None = None
    name: str | None = None
    status: str = "possible"


@dataclasses.dataclass
class Person:
    app_ref: str
    name: str | None
    pictures: int = 0  # pictures hinted with them
    seeds: int = 0
    purity: float = 0.0
    qualifies: bool = False
    centroid: np.ndarray | None = None
    suggestions: list = dataclasses.field(
        default_factory=list
    )  # (face_id, distance, how): how = picture | face | group

    def summary(self):
        how = defaultdict(int)
        for _, _, h in self.suggestions:
            how[h] += 1
        return dict(
            app_ref=self.app_ref,
            name=self.name,
            pictures=self.pictures,
            seeds=self.seeds,
            purity=round(self.purity, 4),
            qualifies=self.qualifies,
            suggested=len(self.suggestions),
            by=dict(how),
        )


def unit(E):
    E = np.asarray(E, dtype=np.float64)
    return E / np.maximum(np.linalg.norm(E, axis=-1, keepdims=True), 1e-9)


def centroid(E, radius=MAX_DISTANCE, sample=2000):
    """The centre of the densest bunch of unit rows: start at the row with the most others within `radius`, then the mean of the rows
    within `radius` of the current centre, three times. A plain mean of two people's faces sits between them and passes as a third
    person; this settles on the larger one, so a name half on someone else shows up as low purity. (Up to `sample` rows pick the start.)"""
    S = E if len(E) <= sample else E[np.linspace(0, len(E) - 1, sample).astype(int)]
    c = S[int(np.argmax((1.0 - radius < S @ S.T).sum(1)))]
    for _ in range(3):
        keep = E[(1.0 - E @ c) < radius]
        if len(keep) == 0:
            break
        c = unit(keep.mean(0))
    return c


def plan(hints, faces, decided, confirmed=None, groups=(), params=Params()):
    """The suggestions an app's hints support, without writing anything.

    hints: [Hint]; faces: {image_id: [(face_id, unit embedding, det_score)]} for every hinted picture (and every group member's);
    decided: face ids with a current decision; confirmed: {app_ref: [unit embeddings]} of faces already confirmed for the person their
    app_ref maps to; groups: [[face_id]] unnamed groups. Returns {app_ref: Person}, people in order of pictures, most first."""
    p = params
    confirmed = confirmed or {}
    emb, img_of, ok = {}, {}, set()
    for img, fl in faces.items():
        for fid, e, score in fl:
            emb[fid] = np.asarray(e, dtype=np.float64)
            img_of[fid] = img
            if score >= p.min_score:
                ok.add(fid)
    refs_on = defaultdict(set)
    people: dict[str, Person] = {}
    face_hints = defaultdict(set)  # app_ref -> face ids hinted as them
    for h in hints:
        refs_on[h.image_id].add(h.app_ref)
        who = people.setdefault(h.app_ref, Person(h.app_ref, h.name))
        if h.name and not who.name:
            who.name = h.name
        if h.face_id is not None and h.face_id in emb:
            face_hints[h.app_ref].add(h.face_id)
    for refs in refs_on.values():
        for r in refs:
            people[r].pictures += 1
    # 1-3: seeds, centroids, who qualifies
    seeds = defaultdict(list)
    for r, fids in face_hints.items():
        seeds[r] += [emb[f] for f in sorted(fids) if f in ok]
    for img, refs in refs_on.items():
        fl = [f for f, _, _ in faces.get(img, ()) if f in ok]
        if len(refs) == 1 and len(fl) == 1 and len(faces.get(img, ())) == 1:
            (r,) = refs
            if fl[0] not in face_hints[r]:
                seeds[r].append(emb[fl[0]])
    for r, E in confirmed.items():
        if r in people:
            seeds[r] += [np.asarray(e, dtype=np.float64) for e in E]
    for r, who in people.items():
        if not seeds[r]:
            continue
        S = unit(np.array(seeds[r]))
        who.seeds = len(S)
        who.centroid = centroid(S, p.max_distance)
        who.purity = float(np.mean((1.0 - S @ who.centroid) < p.max_distance))
        who.qualifies = who.seeds >= p.min_seeds and who.purity >= p.min_purity
    good = {r for r, w in people.items() if w.qualifies}
    taken = set(decided)
    # 4: one face per qualifying person per picture, clearly the closest both ways
    for img, refs in sorted(refs_on.items()):
        rs = sorted(refs & good)
        fl = [f for f, _, _ in faces.get(img, ()) if f in ok and f not in taken]
        if not rs or not fl:
            continue
        C = np.array([people[r].centroid for r in rs])
        D = 1.0 - np.array([emb[f] for f in fl]) @ C.T  # faces x people
        pairs = sorted((D[i, j], i, j) for i in range(len(fl)) for j in range(len(rs)))
        used_f, used_r = set(), set()
        for d, i, j in pairs:
            if d >= p.max_distance or i in used_f or j in used_r:
                continue
            other_faces = np.delete(D[:, j], i)
            other_refs = np.delete(D[i, :], j)
            if (other_faces.size and other_faces.min() < d + p.margin) or (
                other_refs.size and other_refs.min() < d + p.margin
            ):
                continue
            used_f.add(i)
            used_r.add(j)
            people[rs[j]].suggestions.append((fl[i], float(d), "picture"))
            taken.add(fl[i])
    # a face the app itself put a name on: suggested when it is near that person's centroid (or the app's person confirmed it)
    for r in sorted(face_hints):
        who = people[r]
        for f in sorted(face_hints[r]):
            if f in taken or f not in ok:
                continue
            d = float(1.0 - emb[f] @ who.centroid) if who.centroid is not None else 1.0
            confirmed_there = any(h.face_id == f and h.app_ref == r and h.status == "confirmed" for h in hints)
            if (who.qualifies and d < p.max_distance) or confirmed_there:
                who.suggestions.append((f, d, "face"))
                taken.add(f)
    # 5: unnamed groups that mostly sit in one person's pictures
    for g in groups:
        votes, hinted = defaultdict(int), 0
        for f in g:
            refs = refs_on.get(img_of.get(f, -1))
            if refs:
                hinted += 1
                for r in refs:
                    votes[r] += 1
        if not votes:
            continue
        r, n = max(sorted(votes.items()), key=lambda kv: kv[1])
        who = people[r]
        if not who.qualifies or n < p.group_min or n / hinted < p.group_share:
            continue
        members = [f for f in g if f in emb and f in ok and float(1.0 - emb[f] @ who.centroid) < p.max_distance]
        if not members:
            continue
        gc = unit(np.array([emb[f] for f in members]).mean(0))
        if float(1.0 - gc @ who.centroid) >= p.max_distance:
            continue
        for f in members:
            if f not in taken:
                who.suggestions.append((f, float(1.0 - emb[f] @ who.centroid), "group"))
                taken.add(f)
    return dict(sorted(people.items(), key=lambda kv: (-kv[1].pictures, kv[0])))
