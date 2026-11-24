"""Build the `speak-consonants` eval's dataset: the 25 consonant sounds of Letters and Sounds (Phases 2 and 3), as stratamodels.workers.phonics
lists them, and (with --recordings) a person's own recordings of them as the baseline every model is measured against.

    .venv/bin/python scripts/evals/phonics_sounds.py [--name phonics-consonants-25] [--recordings DIR]

Items: input {text (the label, "/ʃ/ (sh)"), sound (the IPA)}; reference {sound}; tags {kind (hiss, buzz, nasal, liquid, stop, affricate,
brief), letters}.

--recordings DIR: one WAV per sound, named by its letters or its IPA (sh.wav or ʃ.wav; letters win: j.wav is the j of jam, never /j/, the y of yes; for the two th sounds th-thin.wav and th-then.wav, or
θ.wav and ð.wav; "c/k/ck" may be c.wav, k.wav or ck.wav). Each is copied, trimmed to its sound with 50 ms of silence either side, to
<data>/evals/recorded/phonics/<n>.wav, and listed in <data>/evals/recorded/phonics-person.jsonl, which model `phonics-person` (runtime
`recorded`, by sound) replays, so the person's clips stand in the eval, the blind judging and the metrics beside the models'. A sound with no
recording is reported and left out: its job then fails, which counts against the baseline rather than vanishing."""

import argparse
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from stratamodels import settings
from stratamodels.evals import core as C
from stratamodels.workers import pacing, phonics


def items():
    out = []
    for n, (letters, (ipa, kind)) in enumerate(phonics.CONSONANTS.items(), 1):
        slug = re.sub(r"[^a-z]+", "-", letters).strip("-")
        out.append(
            dict(
                id=f"c{n:02d}-{slug}",
                input=dict(text=f"/{ipa}/ ({letters})", sound=ipa),
                reference=dict(sound=ipa),
                tags=dict(kind=kind, letters=letters),
            )
        )
    return out


def names(letters, ipa):
    """The file names a recording of a sound may have (without .wav), its letters first. Its IPA counts only where no sound is spelled that
    way: the IPA of y is /j/, and j.wav is the j of jam."""
    spelled = {n for lt in phonics.CONSONANTS for n in lt.split("/")}
    own = letters.split("/")
    if letters.startswith("th "):
        own.append("th-" + letters[4:-1])
    return own + ([ipa] if ipa not in spelled else [])


def recordings(folder, dest, jsonl):
    """Copy a person's recordings (see the module doc); returns (recorded sounds, missing letters)."""
    import soundfile as sf

    have = {os.path.splitext(f)[0]: os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".wav")}
    os.makedirs(dest, exist_ok=True)
    rows, missing = [], []
    for letters, (ipa, _) in phonics.CONSONANTS.items():
        src = next((have[n] for n in names(letters, ipa) if n in have), None)
        if src is None:
            missing.append(letters)
            continue
        x, sr = sf.read(src, dtype="float32", always_2d=True)  # any WAV a phone or an app makes, mixed to mono
        x = x.mean(axis=1)
        y = pacing.tighten(x, sr, None, 0.05)
        path = os.path.join(dest, f"{len(rows) + len(missing):02d}.wav")
        sf.write(path, y, sr, subtype="PCM_16")
        rows.append(
            dict(
                key=ipa,
                output=dict(path=path, duration_s=round(len(y) / sr, 3), sample_rate=sr, seconds=0.0, engine="person"),
            )
        )
    with open(jsonl, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="phonics-consonants-25")
    ap.add_argument("--recordings", default=None)
    a = ap.parse_args()
    its = items()
    ds = C.write_dataset(
        a.name,
        its,
        description="The 25 consonant sounds of Letters and Sounds, Phases 2 and 3 (stratamodels.workers.phonics.CONSONANTS)",
    )
    print(f"{len(ds.items)} items, version {ds.version}")
    if a.recordings:
        rec = os.path.join(settings.data_dir(), "evals", "recorded")
        dest = os.path.join(rec, "phonics")
        shutil.rmtree(dest, ignore_errors=True)
        rows, missing = recordings(a.recordings, dest, os.path.join(rec, "phonics-person.jsonl"))
        print(f"{len(rows)} recordings copied to {dest}" + (f"; none for {', '.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
