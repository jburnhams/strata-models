"""Label real clips for the face-tracks eval (`evals/specs/face-tracks-home.toml`), with as little work for the person as possible.

1. propose: each clip is cut into shots (role shot-boundaries) and tracked by a generous setting of face-tracks (10 fps, `byte`, low thresholds:
   it splits more than it mixes, which is the cheap way round to correct). Its answer is saved (clip-NN.json), then `regroup` runs:

       .venv/bin/python scripts/evals/face_tracks_labels.py propose OUT clip.mp4 other.mp4@60-150 ...

2. regroup: a family film of two minutes gives hundreds of tracks (the first real clips, 2026-11-26: 12 to 354), too many to name one by one.
   So the tracks of each clip are grouped by their embeddings (average linkage, cosine GROUP_SIM; two tracks seen together never share a
   group), and the person names groups. The contact sheet (clip-NN.png) has a row a group, longest first: one face from each of its
   tracks, captioned with the track's id. labels.toml gets a line a group, listing its tracks. Run it alone to regroup without tracking again:

       .venv/bin/python scripts/evals/face_tracks_labels.py regroup OUT [--sim 0.4]

3. The person writes a name against each group worth naming (the same name in two groups is one person; `none` is not a face), and fixes a
   track that is in the wrong group under [clip-NN.tracks] (`17 = "ben"`, or `17 = "ana, ben@14.2"` for one that switches person at 14.2 s).
   A group or track left empty is not labelled, and it is not held against anyone: an answered face where it was is ignored, as are faces in
   tracks too short to show (MOT's "don't care" regions). So naming only the main people is enough.

4. build: the named tracks become the dataset: every clip an item, labelled at `--every` seconds (0.5) by interpolating each named track's
   boxes, the moments listed so that an answered face no label matches (and no blank track covers) counts as false.

       .venv/bin/python scripts/evals/face_tracks_labels.py build OUT [--name face-tracks-home] [--every 0.5]

Items: input {path, clip, cuts}; reference {people: {name: [[time, x0, y0, x1, y1], ...]}, ignore: [[time, x0, y0, x1, y1], ...], times};
tags {source}. Faces the generous setting missed are not labelled (a candidate that finds one is charged a false face): add them to the clip's
`extra` in labels.toml as `extra = [["ana", time, x0, y0, x1, y1], ...]`. Clips and labels are private: they stay in the data directory, never
in the repo."""

import argparse
import json
import os
import re
import sys
import tomllib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from stratamodels.evals import core as C  # noqa: E402
from stratamodels.evals import metrics as M  # noqa: E402

GENEROUS = dict(fps=10, tracker="byte", det_thresh=0.3, sim_min=0.25, min_track_s=0.3, stills=3, still_gap_s=0.5)
APP = "face-tracks-labels"
GROUP_SIM = 0.4  # ArcFace cosine above which two tracks are probably one person (same-person pairs are mostly over 0.4, others under 0.3)
PER_ROW = 10  # tracks shown in a group's row of the contact sheet


def parse_clip(arg):
    """`path` or `path@start-end` (seconds) -> (path, [start, end] or None)."""
    m = re.fullmatch(r"(.+)@([\d.]+)-([\d.]+)", arg)
    if m and not os.path.isfile(arg):
        return os.path.abspath(m[1]), [float(m[2]), float(m[3])]
    return os.path.abspath(arg), None


def parse_label(text):
    """A track's label -> [(name, from_time)]: `ana` is [("ana", 0)], `ana, ben@14.2` is ana then ben from 14.2 s; '' is []."""
    out = []
    for part in [p.strip() for p in (text or "").split(",") if p.strip()]:
        name, _, at = part.partition("@")
        try:
            out.append((name.strip(), float(at) if at else 0.0))
        except ValueError as e:
            raise SystemExit(f"bad label {text!r}: {part!r} is not name or name@seconds") from e
    if out and out[0][1] != 0.0:
        raise SystemExit(f"bad label {text!r}: the first name takes no time")
    return out


def name_at(label, t):
    """Who a labelled track is at time t (None when unlabelled or `none`)."""
    name = None
    for n, since in label:
        if t >= since - 1e-9:
            name = n
    return None if name in (None, "none", "") else name


def group_tracks(answer, sim=GROUP_SIM):
    """The clip's tracks in groups that are probably one person each: average-linkage clustering of the track embeddings, merging the most
    alike first while the mean similarity between two groups is at least `sim`; tracks seen together (`together`) never share a group. Groups
    as lists of track ids, the longest (total seconds) first; a track without an embedding is a group of its own."""
    tracks = [t for t in answer["tracks"] if t.get("embedding")]
    groups = [[t["id"]] for t in answer["tracks"] if not t.get("embedding")]
    if tracks:
        ids = [t["id"] for t in tracks]
        e = np.array([t["embedding"] for t in tracks], dtype=np.float32)
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
        s = e @ e.T
        n = len(ids)
        where = {tid: i for i, tid in enumerate(ids)}
        total = s.astype(np.float64)  # sum of similarities between two groups' tracks
        size = np.ones(n)
        apart = np.zeros((n, n), dtype=bool)
        for x, y in answer.get("together") or []:
            if x in where and y in where:
                apart[where[x], where[y]] = apart[where[y], where[x]] = True
        alive = np.ones(n, dtype=bool)
        members = {i: [i] for i in range(n)}
        while True:
            mean = total / np.outer(size, size)
            mean[apart | ~alive[:, None] | ~alive[None, :]] = -np.inf
            np.fill_diagonal(mean, -np.inf)
            a, b = np.unravel_index(int(np.argmax(mean)), mean.shape)
            if mean[a, b] < sim:
                break
            a, b = int(min(a, b)), int(max(a, b))  # b joins a: sums, sizes and what may not share a group add up
            total[a, :] += total[b, :]
            total[:, a] += total[:, b]
            size[a] += size[b]
            apart[a, :] |= apart[b, :]
            apart[:, a] |= apart[:, b]
            alive[b] = False
            members[a] += members.pop(b)
        groups += [[ids[i] for i in m] for m in members.values()]
    length = {t["id"]: t["end"] - t["start"] for t in answer["tracks"]}
    groups = [sorted(g, key=lambda i: -length[i]) for g in groups]
    return sorted(groups, key=lambda g: (-sum(length[i] for i in g), g[0]))


def _crop(answer, still, thumb):
    from PIL import Image

    with Image.open(still["image"]) as im:
        sx, sy = im.width / answer["width"], im.height / answer["height"]
        x0, y0, x1, y1 = still["bbox"]
        pad = 0.25 * max(x1 - x0, y1 - y0)
        crop = im.crop((int((x0 - pad) * sx), int((y0 - pad) * sy), int((x1 + pad) * sx), int((y1 + pad) * sy)))
        crop.thumbnail((thumb, thumb))
        return crop.copy()


def contact_sheet(answer, groups, path, thumb=80):
    """One row a group: its name and size, then the best face of each of its first PER_ROW tracks with the track's id under it. A PNG."""
    from PIL import Image, ImageDraw

    by_id = {t["id"]: t for t in answer["tracks"]}
    row_h = thumb + 22
    sheet = Image.new("RGB", (110 + PER_ROW * (thumb + 6), max(1, len(groups)) * row_h + 8), "white")
    draw = ImageDraw.Draw(sheet)
    for r, g in enumerate(groups):
        y = 8 + r * row_h
        secs = sum(by_id[i]["end"] - by_id[i]["start"] for i in g)
        draw.text((8, y + 4), f"g{r + 1}", fill="black")
        draw.text((8, y + 22), f"{len(g)} tracks", fill="gray")
        draw.text((8, y + 40), f"{secs:.0f} s", fill="gray")
        for k, tid in enumerate(g[:PER_ROW]):
            stills = [st for st in by_id[tid]["stills"] if st.get("image") and os.path.isfile(st["image"])]
            x = 110 + k * (thumb + 6)
            if stills:
                sheet.paste(_crop(answer, max(stills, key=lambda st: st.get("quality") or 0), thumb), (x, y))
            draw.text((x, y + thumb + 4), f"{tid}", fill="black")
    sheet.save(path)


def _keys(out_dir):
    return sorted(f[:-5] for f in os.listdir(out_dir) if re.fullmatch(r"clip-\d+\.json", f))


def has_names(path):
    """Whether a labels file holds any name (a group or track labelled, `none` included): a person's work, never to be overwritten."""
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        labels = tomllib.load(f)
    return any(
        str(v).strip()
        for clip in labels.values()
        if isinstance(clip, dict)
        for part in ("groups", "tracks")
        for v in (clip.get(part) or {}).values()
    )


def regroup(out_dir, sim=GROUP_SIM):
    """Groups, contact sheets and labels.toml from the saved answers. When labels.toml already holds a name, the new grouping goes to
    labels.new.toml and clip-NN.new.png instead, so the named file keeps the sheets its group numbers refer to."""
    keep = has_names(os.path.join(out_dir, "labels.toml"))
    suffix = ".new" if keep else ""
    lines = [
        "# Name each group worth naming: the same name = the same person; none = not a face. Leave the rest empty: they are not held against",
        '# anyone. Fix a track in the wrong group under [clip-NN.tracks]: 17 = "ben", or 17 = "ana, ben@14.2" (ben from 14.2 s).',
        "",
    ]
    for key in _keys(out_dir):
        with open(os.path.join(out_dir, f"{key}.json"), encoding="utf8") as f:
            saved = json.load(f)
        answer, clip = saved["answer"], saved.get("clip")
        groups = group_tracks(answer, sim)
        contact_sheet(answer, groups, os.path.join(out_dir, f"{key}{suffix}.png"))
        by_id = {t["id"]: t for t in answer["tracks"]}
        span = f" {clip[0]:g}-{clip[1]:g} s" if clip else ""
        lines += [
            f"[{key}]",
            f"# {os.path.basename(saved['path'])}{span}: {len(answer['tracks'])} tracks in {len(groups)} groups, sheet {key}{suffix}.png",
            f"sim = {sim:g}   # the grouping's similarity: build groups the tracks again with it, so leave it as it is",
            f"[{key}.groups]",
        ]
        for n, g in enumerate(groups, 1):
            secs = sum(by_id[i]["end"] - by_id[i]["start"] for i in g)
            lines.append(f'g{n} = ""   # {len(g)} tracks, {secs:.0f} s: {", ".join(str(i) for i in g)}')
        lines += [f"[{key}.tracks]", ""]
        print(f"{key}: {len(answer['tracks'])} tracks in {len(groups)} groups")
    labels = os.path.join(out_dir, f"labels{suffix}.toml")  # never overwrite a person's work
    with open(labels, "w", encoding="utf8") as f:
        f.write("\n".join(lines))
    print(f"fill in {labels}")


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
    regroup(out_dir)


def item_for(key, saved, labels, every):
    """One dataset item from a proposal (saved: its JSON) and its labels ({groups: {gN: label}, tracks: {id: label}, extra: [...]}). Groups
    are numbered as regroup numbered them (the grouping is recomputed with the same similarity, `sim` in the labels, default GROUP_SIM)."""
    answer = saved["answer"]
    named = {}
    group_labels = labels.get("groups") or {}
    if group_labels:
        for n, g in enumerate(group_tracks(answer, float(labels.get("sim", GROUP_SIM))), 1):
            for tid in g:
                named[int(tid)] = parse_label(group_labels.get(f"g{n}", ""))
    named.update({int(tid): parse_label(text) for tid, text in (labels.get("tracks") or {}).items()})
    start = saved["clip"][0] if saved.get("clip") else 0.0
    end = start + float(answer["duration"])
    times = [
        round(start + every * k, 3) for k in range(int((end - start) / every) + 1) if start + every * k < end - 1e-9
    ]
    people, ignore = {}, []
    fps = float(answer["fps"])
    for tr in answer["tracks"]:
        label = named.get(int(tr["id"])) or []
        for t in times:
            box = M.track_box_at(tr["boxes"], t, fps)
            if box is None:
                continue
            box = [round(v, 1) for v in box]
            if not label:
                ignore.append([t, *box])  # nobody named it: what is answered there is not held against anyone
            elif who := name_at(label, t):
                people.setdefault(who, []).append([t, *box])
    for name, t, *box in labels.get("extra") or []:
        t = min(times, key=lambda x: abs(x - float(t)))  # an added face counts at the nearest labelled moment
        people.setdefault(name, []).append([t, *box])
    for obs in people.values():
        obs.sort()
    return dict(
        id=key,
        input=dict(path=saved["path"], **({"clip": saved["clip"]} if saved.get("clip") else {}), cuts=saved["cuts"]),
        reference=dict(people=people, ignore=sorted(ignore), times=times),
        tags=dict(source=labels.get("source", "video")),
    )


def build(out_dir, name="face-tracks-home", every=0.5):
    with open(os.path.join(out_dir, "labels.toml"), "rb") as f:
        labels = tomllib.load(f)
    items = []
    for key in sorted(labels):
        with open(os.path.join(out_dir, f"{key}.json"), encoding="utf8") as f:
            saved = json.load(f)
        it = item_for(key, saved, labels[key], every)
        named = [v for part in ("groups", "tracks") for v in (labels[key].get(part) or {}).values() if str(v).strip()]
        if it["reference"]["people"] or named:  # a clip whose faces are all `none` stays: its false faces count
            items.append(it)
    if not items:
        raise SystemExit("no labelled clips")
    faces = sum(len(v) for it in items for v in it["reference"]["people"].values())
    people = len({p for it in items for p in it["reference"]["people"]})
    return C.write_dataset(
        name, items, description=f"{len(items)} real clips, {faces} faces of {people} people labelled every {every:g} s"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("out")
    p.add_argument("clips", nargs="+")
    g = sub.add_parser("regroup")
    g.add_argument("out")
    g.add_argument("--sim", type=float, default=GROUP_SIM)
    b = sub.add_parser("build")
    b.add_argument("out")
    b.add_argument("--name", default="face-tracks-home")
    b.add_argument("--every", type=float, default=0.5)
    a = ap.parse_args(argv)
    if a.cmd == "propose":
        propose(a.out, a.clips)
    elif a.cmd == "regroup":
        regroup(a.out, a.sim)
    else:
        print(build(a.out, a.name, a.every).to_json())


if __name__ == "__main__":
    main()
