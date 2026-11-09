"""Build the `speak` eval's dataset from strata-360 projects: the narration lines its script writer really wrote (every draft and script, so the
wording is the voice-over's: first person, race facts, numbers, place names), and a reference recording of the runner's own voice for the
cloning engines (the narration is written in the runner's voice, so "sounds like me" is one of the things to judge).

    .venv/bin/python scripts/evals/strata360_narration.py PROJECTS_DIR [--name narration-40] [--n 40] [--min-words 4] [--voice-s 15]

Items: input {text}; reference {text} (the same words: an intelligibility check transcribes the audio back and scores it against them);
tags {project, words (short <= 8, medium <= 16, long), numbers (the line has digits)}.

The reference voice: the runner's speech (segments strata-360 labelled `wearer`, English, with the highest similarity to the runner's voice
profile and enough level) cut from each clip's enhanced audio (`audio_clean.flac`), joined with short pauses until it is about --voice-s
seconds, into <data>/evals/voices/runner.wav, with its words in runner.txt (Qwen3-TTS clones more closely given the reference's words).
Race audio is wind and breathing: the reference is the best there is, not studio quality, and every report must say so."""

import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from stratamodels import settings
from stratamodels.evals import core as C


def narration(project):
    """Every distinct narration line of a project: `vo` items of the whole-race script drafts, the lines of the script plan and of the older scripts."""
    out = []
    for f in sorted(glob.glob(os.path.join(project, "script2", "*.json"))) + sorted(
        glob.glob(os.path.join(project, "scripts", "script-*.json"))
    ):
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        items = d.get("lines") or d.get("items") or (d.get("draft") or {}).get("items") or []
        for it in items:
            if isinstance(it, dict) and it.get("type") in (None, "vo") and isinstance(it.get("text"), str):
                out.append(re.sub(r"\s+", " ", it["text"]).strip())
    return out


def runner_voice(project, seconds, dest_dir, run=subprocess.run):
    """Cut and join the runner's clearest speech into dest_dir/runner.wav (24 kHz mono); returns (path, words) or None."""
    segs = []
    for f in glob.glob(os.path.join(project, "clips", "*", "speakers.json")):
        audio = os.path.join(os.path.dirname(f), "audio_clean.flac")
        if not os.path.isfile(audio):
            continue
        for s in json.load(open(f)).get("segments", []):
            if (
                s.get("label") == "wearer"
                and s.get("lang") == "en"
                and s["t1"] - s["t0"] >= 1.5
                and s.get("rms_db", -99) > -35
            ):
                segs.append((s.get("sim", 0), audio, s["t0"], s["t1"], s["text"].strip()))
    if not segs:
        return None
    segs.sort(key=lambda s: -s[0])
    pick, total = [], 0.0
    for s in segs:
        if total >= seconds:
            break
        pick.append(s)
        total += s[3] - s[2]
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, "runner.wav")
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, (_, audio, t0, t1, _) in enumerate(pick):
            p = os.path.join(tmp, f"{i}.wav")
            run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    f"{max(t0 - 0.1, 0):.2f}",
                    "-to",
                    f"{t1 + 0.15:.2f}",
                    "-i",
                    audio,
                    "-ac",
                    "1",
                    "-ar",
                    "24000",
                    "-af",
                    "apad=pad_dur=0.35",
                    p,
                ],
                check=True,
            )
            parts.append(p)
        lst = os.path.join(tmp, "list.txt")
        open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
        run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                lst,
                "-af",
                "loudnorm=I=-20",
                "-ar",
                "24000",
                out,
            ],
            check=True,
        )
    words = " ".join(s[4] for s in pick)
    open(os.path.join(dest_dir, "runner.txt"), "w").write(words + "\n")
    return out, words, total, len(pick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("projects")
    ap.add_argument("--name", default="narration-40")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--voice-s", type=float, default=15.0)
    ap.add_argument("--voice-project", default=None)
    a = ap.parse_args()
    projects = sorted(d for d in glob.glob(os.path.join(a.projects, "*", "")) if os.path.isdir(d))
    lines = {}
    for p in projects:
        for t in narration(p):
            if len(t.split()) >= a.min_words and re.search(r'[.!?…"”]$', t):
                lines.setdefault(
                    t, os.path.basename(p.rstrip("/"))
                )  # whole sentences: the plan splits a few lines mid-sentence
    if not lines:
        sys.exit("no narration lines found (script2/*.json or scripts/script-*.json)")
    keys = sorted(lines)
    random.Random(360).shuffle(keys)
    keys = keys[: a.n]

    def size(t):
        n = len(t.split())
        return "short" if n <= 8 else "medium" if n <= 16 else "long"

    items = [
        dict(
            id=f"n{i:03d}",
            input=dict(text=t),
            reference=dict(text=t),
            tags=dict(project=lines[t], words=size(t), numbers=bool(re.search(r"\d", t))),
        )
        for i, t in enumerate(keys)
    ]
    ds = C.write_dataset(
        a.name,
        items,
        description=f"{len(items)} narration lines written by strata-360's script writer ({', '.join(sorted(set(lines.values())))}), sampled with seed 360 from {len(lines)} distinct lines of {a.min_words}+ words",
    )
    print(
        f"{len(ds.items)} items, version {ds.version}: "
        + json.dumps({k: sum(1 for i in items if i["tags"]["words"] == k) for k in ("short", "medium", "long")})
    )

    def wearer(p):
        return sum(
            1
            for f in glob.glob(os.path.join(p, "clips", "*", "speakers.json"))
            for s in json.load(open(f)).get("segments", [])
            if s.get("label") == "wearer"
        )

    vp = a.voice_project or max(projects, key=wearer)
    v = vp and runner_voice(vp, a.voice_s, os.path.join(settings.data_dir(), "evals", "voices"))
    print(
        f"runner voice: {v[0]} ({v[2]:.1f} s from {v[3]} phrases): {v[1][:200]}"
        if v
        else "no runner voice: no wearer speech labelled in any project"
    )


if __name__ == "__main__":
    main()
