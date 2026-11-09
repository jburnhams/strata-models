"""Build an eval dataset of real race speech from a strata-360 project: every recognised phrase of every clip that has a second opinion, cut from the
clip's ORIGINAL audio (wind, breathing, footsteps, other runners: the audio the pipeline really gets), with a reference made from

  1. the clip's transcript as the pipeline wrote it (Whisper large-v3-turbo),
  2. with Gemini 3.5 Transcribe's corrections laid over it (`transcribe35.json`: Gemini transcribed the same speech on its own and every word
     where it heard something else is replaced; its insertions and deletions are not, so words Gemini added are missing from the reference),
  3. and the user's own corrections over that (`transcript_edits.json`, entries with "user").

So the reference is SILVER: where Whisper and Gemini agree it is their shared hearing, where they differ it is Gemini's (or the user's). It
favours large-v3-turbo a little (its words stand wherever Gemini did not object), which every report built on it must say.

    .venv/bin/python scripts/evals/strata360_speech.py "<project folder>" [--name legends-speech] [--pad 0.25]

Items: input {path: <clip>/audio_original.flac, clip: [t0 - pad, t1 + pad], language: the phrase's language as the pipeline chose it (from the
few the race allows)}; reference {text}; tags {clip, lang, corrected (user, gemini, user+gemini or none)}."""

import argparse, glob, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from stratamodels.evals import core as C


def words_of(tr):
    return [[w["w"] for w in (s.get("words") or [])] for s in tr["segments"]]


def reference(clip_dir, tr):
    """Per segment: (text, corrected by) with the user's corrections and Gemini's applied; None for a clip without Gemini's pass. Every user
    correction is applied and wins over Gemini's: a Gemini fix whose words overlap a user-corrected word is dropped. `by` says which kinds of
    correction a segment got ('user', 'gemini', 'user+gemini' or 'none')."""
    try:
        fx = json.load(open(os.path.join(clip_dir, "transcribe35.json")))
    except (OSError, ValueError):
        return None
    try:
        edits = json.load(open(os.path.join(clip_dir, "transcript_edits.json"))).get("edits", {})
    except (OSError, ValueError):
        edits = {}
    W = words_of(tr)
    users, gems = [set() for _ in W], [False] * len(W)
    for k, e in edits.items():  # user corrections first, on the original word positions
        if not e.get("user"):
            continue
        si, wi = (int(x) for x in k.split(":"))
        if si < len(W) and wi < len(W[si]):
            W[si][wi] = e["user"]["text"].strip()
            users[si].add(wi)
    for f in sorted(
        fx.get("fixes", []), key=lambda f: (f["seg"], -f["word"])
    ):  # then Gemini's, right to left so positions stay valid
        si, a, b = f["seg"], f["word"], f.get("through", f["word"])
        if si < len(W) and b < len(W[si]) and not any(a <= u <= b for u in users[si]):
            W[si][a : b + 1] = [f["to"].strip()]
            gems[si] = True
    by = ["+".join(x for x, on in (("user", bool(u)), ("gemini", g)) if on) or "none" for u, g in zip(users, gems)]
    return [
        (re.sub(r" (?=')", "", " ".join(w.strip() for w in ws if w.strip())), b) for ws, b in zip(W, by)
    ]  # Whisper splits "C'est" into "C" and "'est": joined back


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--name", default="legends-speech")
    ap.add_argument("--pad", type=float, default=0.25)
    a = ap.parse_args()
    items, clips = [], 0
    for d in sorted(glob.glob(os.path.join(a.project, "clips", "*", ""))):
        audio = os.path.join(d, "audio_original.flac")
        try:
            tr = json.load(open(os.path.join(d, "transcript.json")))
        except (OSError, ValueError):
            continue
        ref = reference(d, tr)
        if ref is None or not os.path.isfile(audio):
            continue
        clips += 1
        clip = os.path.basename(d.rstrip("/"))
        for si, (s, (text, by)) in enumerate(zip(tr["segments"], ref)):
            if not text or s.get("suspect"):
                continue
            items.append(
                dict(
                    id=f"{clip}:{si}",
                    input=dict(
                        path=audio,
                        clip=[round(max(s["t0"] - a.pad, 0), 2), round(s["t1"] + a.pad, 2)],
                        language=s["lang"],
                    ),
                    reference=dict(text=text),
                    tags=dict(clip=clip, lang=s["lang"], corrected=by),
                )
            )
    if not items:
        sys.exit("no clip with transcript.json, transcribe35.json and audio_original.flac")
    ds = C.write_dataset(
        a.name,
        items,
        description=f"{os.path.basename(a.project.rstrip('/'))}: {len(items)} phrases from {clips} clips, original race audio; SILVER reference (Whisper turbo + Gemini 3.5 Transcribe word fixes + user fixes)",
    )
    print(
        f"{len(ds.items)} items from {clips} clips, version {ds.version}; corrected: "
        + json.dumps(
            {k: sum(1 for i in items if i["tags"]["corrected"] == k) for k in ("gemini", "user", "user+gemini", "none")}
        )
    )


if __name__ == "__main__":
    main()
