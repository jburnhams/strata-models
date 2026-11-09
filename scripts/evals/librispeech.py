"""Build the eval dataset `librispeech-clean-200` from LibriSpeech test-clean (CC BY 4.0, https://www.openslr.org/12): 200 utterances, 5 from
each of 40 speakers, chosen at random with a fixed seed, so every transcription experiment uses the same items.

    curl -L https://www.openslr.org/resources/12/test-clean.tar.gz | tar xz -C ~/models/eval-data
    .venv/bin/python scripts/evals/librispeech.py [--root ~/models/eval-data/LibriSpeech/test-clean] [--per-speaker 5] [--name librispeech-clean-200]"""

import argparse, glob, os, random, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from stratamodels.evals import core as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/models/eval-data/LibriSpeech/test-clean"))
    ap.add_argument("--per-speaker", type=int, default=5)
    ap.add_argument("--name", default="librispeech-clean-200")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    by = {}
    for tr in sorted(glob.glob(os.path.join(a.root, "*", "*", "*.trans.txt"))):
        for line in open(tr, encoding="utf8"):
            uid, text = line.strip().split(" ", 1)
            spk = uid.split("-")[0]
            by.setdefault(spk, []).append(
                dict(
                    id=uid,
                    input=dict(path=os.path.join(os.path.dirname(tr), uid + ".flac")),
                    reference=dict(text=text.lower()),
                    tags=dict(speaker=spk),
                )
            )
    if not by:
        sys.exit(f"no LibriSpeech transcripts under {a.root}")
    items = []
    for spk in sorted(by):
        items += rng.sample(by[spk], min(a.per_speaker, len(by[spk])))
    ds = C.write_dataset(
        a.name,
        items,
        description=f"LibriSpeech test-clean (CC BY 4.0): {a.per_speaker} utterances from each of {len(by)} speakers, seed {a.seed}; read audiobook English",
    )
    print(f"{len(ds.items)} items, version {ds.version}")


if __name__ == "__main__":
    main()
