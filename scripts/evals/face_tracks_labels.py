"""Label real clips for the face-tracks eval (`evals/specs/face-tracks-home.toml`), with as little work for the person as possible.

1. propose: each clip is cut into shots (role shot-boundaries) and tracked by a generous setting of face-tracks (10 fps, `byte`, low thresholds:
   it splits more than it mixes, which is the cheap way round to correct). For each clip it writes the answer (JSON), a contact sheet (one row
   a track: its id, span and stills) and a labels file to fill in:

       .venv/bin/python scripts/evals/face_tracks_labels.py propose OUT clip.mp4 other.mp4@60-150 ...

2. The person writes a name against each track in OUT/labels.toml. The same name in two tracks is the same person; `none` is not a face (or
   one not worth labelling); a track that switches person is `ana, ben@14.2` (ana until 14.2 s, then ben). Leave a track empty to skip it.

3. build: the labelled tracks become the dataset: every clip an item, labelled at `--every` seconds (0.5) by interpolating each named track's
   boxes, the moments listed so that an answered face no label matches counts as false.

       .venv/bin/python scripts/evals/face_tracks_labels.py build OUT [--name face-tracks-home] [--every 0.5]

Items: input {path, clip, cuts}; reference {people: {name: [[time, x0, y0, x1, y1], ...]}, times}; tags {source}. Faces the generous setting
missed are not labelled (a candidate that finds one is charged a false face): add them to the clip's `extra` in labels.toml as
`extra = [["ana", time, x0, y0, x1, y1], ...]`. Clips and labels are private: they stay in the data directory, never in the repo."""

import argparse
import json
import os
import re
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from stratamodels.evals import core as C  # noqa: E402
from stratamodels.evals import metrics as M  # noqa: E402

GENEROUS = dict(fps=10, tracker="byte", det_thresh=0.3, sim_min=0.25, min_track_s=0.3, stills=3, still_gap_s=0.5)
APP = "face-tracks-labels"


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


def contact_sheet(answer, path, thumb=96):
    """One row a track: its id and span, then crops of its stills. A PNG at `path`."""
    from PIL import Image, ImageDraw

    rows = answer["tracks"] or []
    width = 150 + 3 * (thumb + 8)
    sheet = Image.new("RGB", (width, max(1, len(rows)) * (thumb + 8) + 8), "white")
    draw = ImageDraw.Draw(sheet)
    for r, tr in enumerate(rows):
        y = 8 + r * (thumb + 8)
        draw.text((8, y + 4), f"track {tr['id']}", fill="black")
        draw.text((8, y + 22), f"{tr['start']:.1f}-{tr['end']:.1f} s", fill="black")
        draw.text((8, y + 40), f"{tr['faces']} faces, shot {tr['shot']}", fill="gray")
        for k, st in enumerate(tr["stills"][:3]):
            if not st.get("image") or not os.path.isfile(st["image"]):
                continue
            with Image.open(st["image"]) as im:
                sx, sy = im.width / answer["width"], im.height / answer["height"]
                x0, y0, x1, y1 = st["bbox"]
                pad = 0.25 * max(x1 - x0, y1 - y0)
                crop = im.crop((int((x0 - pad) * sx), int((y0 - pad) * sy), int((x1 + pad) * sx), int((y1 + pad) * sy)))
                crop.thumbnail((thumb, thumb))
                sheet.paste(crop, (150 + k * (thumb + 8), y))
    sheet.save(path)


def propose(out_dir, clips, client=None):
    if client is None:
        import strata_models_client as client
    os.makedirs(out_dir, exist_ok=True)
    lines = [
        "# Name each track: the same name = the same person; none = not a face; ana, ben@14.2 = ana, then ben from 14.2 s.",
        "",
    ]
    for k, arg in enumerate(clips, 1):
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
        contact_sheet(answer, os.path.join(out_dir, f"{key}.png"))
        lines += [
            f"[{key}]",
            f"# {os.path.basename(path)}{f' {clip[0]:g}-{clip[1]:g} s' if clip else ''}: {len(answer['tracks'])} tracks, sheet {key}.png",
            "[" + key + ".tracks]",
        ]
        lines += [
            f'{tr["id"]} = ""   # {tr["start"]:.1f}-{tr["end"]:.1f} s, {tr["faces"]} faces' for tr in answer["tracks"]
        ]
        lines.append("")
        print(f"{key}: {len(answer['tracks'])} tracks, {len(cuts)} cuts")
    labels = os.path.join(out_dir, "labels.toml")
    if os.path.exists(labels):
        labels = os.path.join(out_dir, "labels.new.toml")  # never overwrite a person's work
    with open(labels, "w", encoding="utf8") as f:
        f.write("\n".join(lines))
    print(f"fill in {labels}")


def item_for(key, saved, labels, every):
    """One dataset item from a proposal (saved: its JSON) and its labels ({tracks: {id: label}, extra: [...]})."""
    answer = saved["answer"]
    named = {int(tid): parse_label(text) for tid, text in (labels.get("tracks") or {}).items()}
    start = saved["clip"][0] if saved.get("clip") else 0.0
    end = start + float(answer["duration"])
    times = [
        round(start + every * k, 3) for k in range(int((end - start) / every) + 1) if start + every * k < end - 1e-9
    ]
    people = {}
    for tr in answer["tracks"]:
        label = named.get(int(tr["id"]))
        if not label:
            continue
        for t in times:
            who = name_at(label, t)
            box = M.track_box_at(tr["boxes"], t, float(answer["fps"])) if who else None
            if box is not None:
                people.setdefault(who, []).append([t, *[round(v, 1) for v in box]])
    for name, t, *box in labels.get("extra") or []:
        t = min(times, key=lambda x: abs(x - float(t)))  # an added face counts at the nearest labelled moment
        people.setdefault(name, []).append([t, *box])
    for obs in people.values():
        obs.sort()
    return dict(
        id=key,
        input=dict(path=saved["path"], **({"clip": saved["clip"]} if saved.get("clip") else {}), cuts=saved["cuts"]),
        reference=dict(people=people, times=times),
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
        if it["reference"]["people"] or labels[key].get("tracks"):
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
    b = sub.add_parser("build")
    b.add_argument("out")
    b.add_argument("--name", default="face-tracks-home")
    b.add_argument("--every", type=float, default=0.5)
    a = ap.parse_args(argv)
    if a.cmd == "propose":
        propose(a.out, a.clips)
    else:
        print(build(a.out, a.name, a.every).to_json())


if __name__ == "__main__":
    main()
