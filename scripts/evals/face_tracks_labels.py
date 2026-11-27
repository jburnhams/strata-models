"""Label real clips for the face-tracks eval (`evals/specs/face-tracks-home.toml`) on the Faces screens, with as little work for the person
as possible.

1. propose: each clip is cut into shots (role shot-boundaries) and tracked by a generous setting of face-tracks (10 fps, `byte`, low thresholds:
   it splits more than it mixes, which is the cheap way round to correct). Its answer is saved (clip-NN.json):

       .venv/bin/python scripts/evals/face_tracks_labels.py propose OUT clip.mp4 other.mp4@60-150 ...

2. store: each track's stills go into the face store (app and tag `face-tracks-eval`, and `face-tracks-eval:SET/clip-NN`, SET being OUT's
   folder name, which prefixes every picture's ref so two proposal sets never share one), one picture a still holding that track's face,
   with the box and buffalo_l embedding face-tracks found. Each track's embedding is searched among the confirmed faces: when most of its
   nearest belong to one person, that person is suggested for its stills (their page's review), and the rest turn up in Unnamed groups.
   Clips named in `--skip` stay out, and the store itself never keeps a face too small or unsure to be worth a look (docs/database.md#faces). Storing again changes nothing already decided.

       .venv/bin/python scripts/evals/face_tracks_labels.py store OUT [--skip clip-03]

   `unstore OUT` takes them out again (DELETE /api/faces/images by their clip's tag), with every decision about them.

3. The person names them on the Faces screens like any face: confirmed for a person is that person; rejected with nobody is not a face.
   Anything else (undecided, only suggested, "not this person") is not labelled, and it is not held against anyone: an answered face where
   it was is ignored (MOT's "don't care" regions). So naming only the main people is enough.

4. build: each track takes the people its stills were confirmed as, in time order (a track whose stills are two people switches half way
   between them, so a tracker that mixed two people is caught); a track with only not-a-face stills is `none`, one with no decided still
   is ignored. Every clip an item, labelled at `--every` seconds (0.5) by interpolating each named track's boxes, the moments listed so
   that an answered face no label matches (and no ignored track covers) counts as false.

       .venv/bin/python scripts/evals/face_tracks_labels.py build OUT [--name face-tracks-home] [--every 0.5]

Items: input {path, clip, cuts}; reference {people: {person: [[time, x0, y0, x1, y1], ...]}, ignore: [[time, x0, y0, x1, y1], ...], times};
tags {source}. People are `p<identity id>` (a merge reads through to the person kept). Faces the generous setting missed are not labelled (a
candidate that finds one is charged a false face). Clips and labels are private: they stay in the data directory and the face store, never in
the repo."""

import argparse
import json
import os
import re
import sys
import urllib.parse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from stratamodels.evals import core as C  # noqa: E402
from stratamodels.evals import metrics as M  # noqa: E402

GENEROUS = dict(fps=10, tracker="byte", det_thresh=0.3, sim_min=0.25, min_track_s=0.3, stills=3, still_gap_s=0.5)
APP = "face-tracks-labels"
STORE = "face-tracks-eval"  # the face store's app and tag for the stills: kept apart from every app's own pictures, and found again by build
SUGGEST_K, SUGGEST_DISTANCE, SUGGEST_SHARE = (
    10,
    0.5,
    0.6,
)  # a track's nearest confirmed faces, how near (the store's propose uses 0.5), how many must agree
STORE_MODEL = "buffalo_l"  # the faces role's model, whose embeddings face-tracks' are (buffalo_l-tracks runs the same network): compared with every photo's


def parse_clip(arg):
    """`path` or `path@start-end` (seconds) -> (path, [start, end] or None)."""
    m = re.fullmatch(r"(.+)@([\d.]+)-([\d.]+)", arg)
    if m and not os.path.isfile(arg):
        return os.path.abspath(m[1]), [float(m[2]), float(m[3])]
    return os.path.abspath(arg), None


def name_at(label, t):
    """Who a track is at time t, from its label [(name, from_time)] (None when unlabelled or `none`)."""
    name = None
    for n, since in label:
        if t >= since - 1e-9:
            name = n
    return None if name in (None, "none", "") else name


def _keys(out_dir):
    return sorted(f[:-5] for f in os.listdir(out_dir) if re.fullmatch(r"clip-\d+\.json", f))


def _load(out_dir, key):
    with open(os.path.join(out_dir, f"{key}.json"), encoding="utf8") as f:
        return json.load(f)


def label_set(out_dir):
    """The proposal set's name in the face store (its folder's name): two sets both have a clip-01, and must not share its pictures."""
    return os.path.basename(os.path.abspath(out_dir))


def _ref(prefix, track, still):
    return f"{prefix}/t{int(track)}/{float(still['time']):.3f}"


def suggest(client, face_ids, embedding, k=SUGGEST_K, max_distance=SUGGEST_DISTANCE, share=SUGGEST_SHARE):
    """Suggest a known person for a track's undecided stills: the people confirmed on the `k` faces nearest the track's embedding, within
    `max_distance`; the commonest one is suggested when they hold `share` of those faces. The person confirms or rejects it on their page.
    Returns the person suggested, or None."""
    if not face_ids or not embedding:
        return None
    hits = client._call(
        "POST",
        "/api/faces/search",
        dict(app=STORE, embedding=embedding, k=k, named=True, max_distance=max_distance, model=STORE_MODEL),
    )["hits"]
    votes = Counter(h["face"]["identity_id"] for h in hits)
    if not votes:
        return None
    who, n = votes.most_common(1)[0]
    if n < max(2, share * len(hits)):
        return None
    client._call(
        "POST",
        "/api/faces/assign",
        dict(
            face_ids=face_ids,
            identity_id=who,
            app=STORE,
            status="suggested",
            source="face-tracks",
            confidence=round(n / len(hits), 3),
        ),
    )
    return who


def store(out_dir, client=None, skip=()):
    """Every worthwhile still of every clip's tracks into the face store, one picture each, and a known person suggested for each track's
    stills where the faces nearest it agree (`suggest`); returns how many stills."""
    if client is None:
        import strata_models_client as client
    n, suggested = 0, 0
    for key in _keys(out_dir):
        if key in skip:
            continue
        prefix = f"{label_set(out_dir)}/{key}"
        answer = _load(out_dir, key)["answer"]
        for tr in answer["tracks"]:
            undecided = []
            for st in tr.get("stills") or []:
                if not st.get("image") or not st.get("embedding"):
                    continue
                face = dict(
                    bbox=st["bbox"],
                    det_score=st.get("det_score"),
                    kps=st.get("kps"),
                    pose=st.get("pose"),
                    embedding=st["embedding"],
                    found_by="face-tracks",
                )
                img = client._call(
                    "POST",
                    "/api/faces/images",
                    dict(
                        app=STORE,
                        ref=_ref(prefix, tr["id"], st),
                        path=st["image"],
                        tags=[STORE, f"{STORE}:{prefix}"],
                        faces=[face],
                        model=STORE_MODEL,
                        width=answer["width"],
                        height=answer["height"],
                    ),
                )
                n += bool((img or {}).get("faces"))  # the store leaves out a face too small or unsure to keep
                undecided += [f["id"] for f in (img or {}).get("faces") or [] if f.get("status") is None]
            suggested += bool(suggest(client, undecided, tr.get("embedding")))
        print(f"{key}: stored")
    print(
        f"{n} faces in the face store under {STORE}, a known person suggested for {suggested} tracks: name them on the Faces screens"
    )
    return n


def unstore(out_dir, client=None):
    """Take this proposal set's stills out of the face store again (their faces and every decision about them; people stay)."""
    if client is None:
        import strata_models_client as client
    n = 0
    for key in _keys(out_dir):
        tag = urllib.parse.quote(f"{STORE}:{label_set(out_dir)}/{key}")
        n += client._call("DELETE", f"/api/faces/images?app={STORE}&tag={tag}")["deleted"]
    print(f"{n} stills taken out of the face store")
    return n


def decisions(out_dir, client=None):
    """ref -> 'p<identity>' (confirmed), 'none' (rejected as not a face) for every still of this proposal set a person decided about."""
    if client is None:
        import strata_models_client as client
    out, before = {}, None
    while True:
        page = client._call("GET", f"/api/faces/images?app={STORE}&limit=500" + (f"&before={before}" if before else ""))
        for img in page["items"]:
            if not img["ref"].startswith(label_set(out_dir) + "/"):
                continue
            for f in img["faces"][:1]:
                if f.get("status") == "confirmed" and f.get("identity_id") is not None:
                    out[img["ref"]] = f"p{f['identity_id']}"
                elif f.get("status") == "rejected" and f.get("identity_id") is None:
                    out[img["ref"]] = "none"
        before = page.get("next")
        if not before:
            return out


def track_labels(prefix, answer, decided):
    """track id -> [(person, from_time)] from its stills' decisions, in time order; a change of person starts half way between the stills."""
    out = {}
    for tr in answer["tracks"]:
        seen = sorted(
            (float(st["time"]), decided[r])
            for st in tr.get("stills") or []
            if (r := _ref(prefix, tr["id"], st)) in decided
        )
        names = [n for _, n in seen if n != "none"]
        if not seen:
            continue
        if not names:
            out[int(tr["id"])] = [("none", 0.0)]
            continue
        label, prev = [], None
        for t, who in [(t, n) for t, n in seen if n != "none"]:
            if not label:
                label.append((who, 0.0))
            elif who != label[-1][0]:
                label.append((who, round((prev + t) / 2, 3)))
            prev = t
        out[int(tr["id"])] = label
    return out


def propose(out_dir, clips, client=None):
    if client is None:
        import strata_models_client as client
    os.makedirs(out_dir, exist_ok=True)
    start = len(_keys(out_dir))
    for k, arg in enumerate(clips, start + 1):
        path, clip = parse_clip(arg)
        key = f"clip-{k:02d}"
        span = dict(clip=clip) if clip else {}
        shots = client.shots(path, app=APP, **span)
        cuts = [c["time"] for c in shots["cuts"]]
        answer = client.face_tracks(
            path, app=APP, cuts=cuts, out_dir=os.path.join(os.path.abspath(out_dir), key), **span, **GENEROUS
        )
        with open(os.path.join(out_dir, f"{key}.json"), "w", encoding="utf8") as f:
            json.dump(dict(path=path, clip=clip, cuts=cuts, answer=answer), f)
        print(f"{key}: {len(answer['tracks'])} tracks, {len(cuts)} cuts")
    print(f"next: store {out_dir}")


def item_for(key, saved, labels, every):
    """One dataset item from a proposal (saved: its JSON) and its tracks' labels (track id -> [(person, from_time)]; a track not in it is
    ignored)."""
    answer = saved["answer"]
    start = saved["clip"][0] if saved.get("clip") else 0.0
    end = start + float(answer["duration"])
    times = [
        round(start + every * k, 3) for k in range(int((end - start) / every) + 1) if start + every * k < end - 1e-9
    ]
    people, ignore = {}, []
    fps = float(answer["fps"])
    for tr in answer["tracks"]:
        label = labels.get(int(tr["id"])) or []
        for t in times:
            box = M.track_box_at(tr["boxes"], t, fps)
            if box is None:
                continue
            box = [round(v, 1) for v in box]
            if not label:
                ignore.append([t, *box])  # nobody named it: what is answered there is not held against anyone
            elif who := name_at(label, t):
                people.setdefault(who, []).append([t, *box])
    for obs in people.values():
        obs.sort()
    return dict(
        id=key,
        input=dict(path=saved["path"], **({"clip": saved["clip"]} if saved.get("clip") else {}), cuts=saved["cuts"]),
        reference=dict(people=people, ignore=sorted(ignore), times=times),
        tags=dict(source="video"),
    )


def build(out_dir, name="face-tracks-home", every=0.5, client=None):
    decided = decisions(out_dir, client)
    items, mixed = [], 0
    for key in _keys(out_dir):
        saved = _load(out_dir, key)
        labels = track_labels(f"{label_set(out_dir)}/{key}", saved["answer"], decided)
        mixed += sum(len(v) > 1 for v in labels.values())
        if labels:  # a clip whose decided faces are all `none` stays: its false faces count
            items.append(item_for(key, saved, labels, every))
    if not items:
        raise SystemExit("no clip has a face decided on the Faces screens yet")
    faces = sum(len(v) for it in items for v in it["reference"]["people"].values())
    people = len({p for it in items for p in it["reference"]["people"]})
    print(f"{mixed} tracks the generous setting mixed (stills of two people)")
    return C.write_dataset(
        name, items, description=f"{len(items)} real clips, {faces} faces of {people} people labelled every {every:g} s"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("out")
    p.add_argument("clips", nargs="+")
    s = sub.add_parser("store")
    s.add_argument("out")
    s.add_argument("--skip", action="append", default=[], help="a clip to leave out (clip-03), again for more")
    u = sub.add_parser("unstore", help="take the set's stills out of the face store again")
    u.add_argument("out")
    b = sub.add_parser("build")
    b.add_argument("out")
    b.add_argument("--name", default="face-tracks-home")
    b.add_argument("--every", type=float, default=0.5)
    a = ap.parse_args(argv)
    if a.cmd == "propose":
        propose(a.out, a.clips)
    elif a.cmd == "store":
        store(a.out, skip=set(a.skip))
    elif a.cmd == "unstore":
        unstore(a.out)
    else:
        print(build(a.out, a.name, a.every).to_json())


if __name__ == "__main__":
    main()
