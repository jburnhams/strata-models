"""Build a video-enhance dataset from a folder of short clips and its manifest, `clips.toml`, which stays beside the clips (private: it
describes private footage):

    crop = [14, 0, 692, 568]           # the active picture in source pixels (no black side bars, no head-switching lines), for every clip
    field_order = "tff"                # forced: never trusted from the file
    # deinterlace is the candidate's to choose (the worker's default is bwdif); a clip may set its own ("none" for a progressive one)
    [[clips]]
    id = "05-face-reading-book"
    file = "05-face-reading-book.mkv"
    note = "boy reading aloud, medium close-up"
    tags = { content = "face", motion = "low" }
    regions = [{ id = "face", label = "Face", box = [200, 90, 260, 195] }]   # [x, y, w, h] in source pixels: compared up close when judging

Each item's input is the clip with its crop, field order and regions (what workers/video.py takes); its reference holds playable copies of the
source made the same way the candidates' are finished, but enlarged nearest-neighbour instead of enhanced, so the Judge and Outputs tabs can
show what was on the tape: `preview` (the whole frame) and `crops` (each region). They go in <data>/evals/media/<dataset>/.

    .venv/bin/python scripts/evals/video_clips.py /path/to/clips --name vhs-clips-8"""

import argparse
import os
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from stratamodels import settings  # noqa: E402
from stratamodels.evals import core as C  # noqa: E402

VIDEO_WORKER = os.path.join(HERE, "..", "..", "src", "stratamodels", "workers")


def worker():
    """workers/video.py (standalone: it imports only workers/protocol.py), for the same crop, region and finishing arithmetic."""
    if VIDEO_WORKER not in sys.path:
        sys.path.insert(0, VIDEO_WORKER)
    import video

    return video


def item(clip, folder, defaults, media_dir, V, make=True, run=None):
    """One dataset item from a manifest entry; with make, its source preview and region crops are written (skipped when present)."""
    path = os.path.join(folder, clip["file"])
    if not os.path.isfile(path):
        raise SystemExit(f"no clip {path}")
    inp = dict(path=os.path.abspath(path), field_order=clip.get("field_order", defaults.get("field_order", "tff")))
    if (
        "deinterlace" in clip
    ):  # only a clip that needs its own (a progressive one: "none"); otherwise the candidate's params choose (job params = candidate's, then the item's)
        inp["deinterlace"] = clip["deinterlace"]
    crop = clip.get("crop", defaults.get("crop"))
    if crop is not None:
        inp["crop"] = list(crop)
    regions = [
        dict(id=str(r["id"]), label=str(r.get("label") or r["id"]), box=list(r["box"]))
        for r in clip.get("regions") or []
    ]
    if regions:
        inp["regions"] = regions
    ref = dict(note=clip.get("note", ""))
    if make:
        os.makedirs(media_dir, exist_ok=True)
        p = {**V.DEFAULTS, **inp}
        kw = dict(run=run) if run else {}
        src = V.probe(path, **kw)
        stem = os.path.join(media_dir, clip["id"])
        preview = stem + ".source.mp4"
        if not os.path.isfile(preview):
            W, H = p["size"]
            vf = V.prep_filter({**p, "prep_size": [W, H]}, src).replace(
                f"scale={W}:{H}:flags=lanczos", f"scale={W}:{H}:flags=neighbor"
            )
            V.ffmpeg(
                [
                    "-i",
                    path,
                    "-vf",
                    vf + ",format=yuv420p",
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0?",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "12",
                    "-movflags",
                    "+faststart",
                    preview,
                ],
                **kw,
            )
        _, crops, _ = V.extras({**p, "stills": []}, preview, V.crop_box(p, src), stem + ".source", **kw)
        ref.update(preview=preview, crops=crops)
    return dict(id=clip["id"], input=inp, reference=ref, tags=dict(clip.get("tags") or {}))


def build(folder, name, make=True, run=None):
    with open(os.path.join(folder, "clips.toml"), "rb") as f:
        m = tomllib.load(f)
    V = worker()
    media_dir = os.path.join(settings.data_dir(), "evals", "media", name)
    items = [item(c, folder, m, media_dir, V, make=make, run=run) for c in m.get("clips") or []]
    if not items:
        raise SystemExit(f"{folder}/clips.toml lists no clips")
    return C.write_dataset(
        name,
        items,
        description=m.get("description")
        or f"{len(items)} video clips from {os.path.basename(os.path.abspath(folder))}",
    )


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("folder", help="the clips and their clips.toml")
    a.add_argument("--name", required=True, help="the dataset's name, e.g. vhs-clips-8")
    a.add_argument("--no-media", action="store_true", help="write the items only, not the source previews and crops")
    o = a.parse_args(argv)
    ds = build(o.folder, o.name, make=not o.no_media)
    print(f"{len(ds.items)} clips, version {ds.version}")


if __name__ == "__main__":
    main()
