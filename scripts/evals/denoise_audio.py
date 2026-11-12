"""Build the datasets of the denoise-audio evals from real recordings: strata-360 race footage (the wearer talking to the camera in wind, crowds,
footsteps) and any other media (a digitised tape's soundtrack: hiss, hum, head-switching buzz). The recordings are private, so the datasets
and the cut files stay in the data folder, never in git.

    .venv/bin/python scripts/evals/denoise_audio.py real --project P [--speech 16] [--media F --windows 12] [--files GLOB] [--name denoise-real]
    .venv/bin/python scripts/evals/denoise_audio.py mix --project P [--noise-files GLOB] --clean-root LIBRISPEECH [--n 32] [--name denoise-mix]

real   items to listen to (no reference: judged by people and model judges, scored by DNSMOS and SQUIM). From a strata-360 project (--project:
       clips/*/audio_original.flac, the camera's own sound, and speakers.json): --speech windows of speech (segments merged into 4-12 s,
       at most 2 a clip, the wearer's first); from a long recording (--media): --windows windows of --length s spread evenly over it (the
       first and last 2 % skipped); whole short files (--files): each from its start, at most --length s.
       Items: input {path: <data>/evals/audio/<name>/<id>.flac (48 kHz, the source's channels)}; tags {origin (--origin, or legends / the
       media file's stem), kind (speech | window | file), clip}.
mix    items with a reference (scored by SI-SDR, PESQ and STOI): clean read speech (LibriSpeech test-clean, CC BY 4.0) mixed into real noise
       at -5, 0, 5 and 10 dB. Noise: the project's stretches with nobody speaking (gaps of 4 s or more between speakers.json segments) and
       --noise-files (cut by hand, e.g. a tape's quiet stretches). Items: input {path: the mix}; reference {audio: the clean speech};
       tags {noise (its origin), snr}.
Both shuffle with a fixed --seed, so a rebuild makes the same dataset."""

import argparse
import glob
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from stratamodels import settings
from stratamodels.evals import core as C

SR = 48000
SNRS = (-5, 0, 5, 10)


def ffmpeg(args, run=subprocess.run, data=None):
    p = run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], input=data, capture_output=True)
    if p.returncode != 0:
        raise SystemExit(f"ffmpeg failed: {p.stderr.decode(errors='replace')[-300:]}")
    return p.stdout


def cut(src, t0, t1, dest, run=subprocess.run):
    """[t0, t1] of src as 48 kHz FLAC (its own channels)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    ffmpeg(["-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", src, "-vn", "-ar", str(SR), "-c:a", "flac", dest], run)
    return dest


def duration(path, run=subprocess.run):
    p = run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True,
        text=True,
    )
    try:
        return float(p.stdout.strip())
    except ValueError:
        raise SystemExit(f"ffprobe could not read the length of {path}")


def merge(segs, lo=4.0, hi=12.0, gap=1.0):
    """Speech segments [(t0, t1, label)] (sorted) joined while the gap between them is under `gap` s and the window under `hi` s; a short
    window is widened around its middle to `lo` s, a long one cut to its first `hi` s. The label is the first segment's."""
    out, cur = [], None
    for t0, t1, lab in segs:
        if cur and t0 - cur[1] < gap and t1 - cur[0] <= hi:
            cur[1] = t1
            continue
        if cur:
            out.append(cur)
        cur = [t0, t1, lab]
    if cur:
        out.append(cur)
    res = []
    for t0, t1, lab in out:
        if t1 - t0 < lo:
            mid = (t0 + t1) / 2
            t0, t1 = max(0.0, mid - lo / 2), max(0.0, mid - lo / 2) + lo
        t1 = min(t1, t0 + hi)  # one long segment: its first `hi` s
        res.append((round(t0, 3), round(t1, 3), lab))
    return res


def gaps(segs, length, lo=4.0, pad=0.5):
    """Stretches of at least `lo` s with no segment in them (pad s kept clear of speech on each side), as [(t0, t1)]."""
    out, t = [], 0.0
    for t0, t1, *_ in sorted(segs) + [(length, length)]:
        a, b = t + pad, t0 - pad
        if b - a >= lo:
            out.append((round(a, 3), round(b, 3)))
        t = max(t, t1)
    return out


def project_clips(project):
    """[(clip name, audio_original.flac, speakers.json segments [(t0, t1, label)])] of a strata-360 project."""
    out = []
    for f in sorted(glob.glob(os.path.join(project, "clips", "*", "speakers.json"))):
        d = os.path.dirname(f)
        audio = os.path.join(d, "audio_original.flac")
        if not os.path.isfile(audio):
            continue
        segs = sorted(
            (float(s["t0"]), float(s["t1"]), s.get("label") or "speech") for s in json.load(open(f)).get("segments", [])
        )
        out.append((os.path.basename(d), audio, segs))
    return out


def speech_windows(clips, n, rng, per_clip=2):
    """n speech windows, at most per_clip a clip, the wearer's first (they are what the film keeps), the rest by chance."""
    cands = []
    for name, audio, segs in clips:
        ws = merge(segs)
        rng.shuffle(ws)
        ws.sort(key=lambda w: w[2] != "wearer")
        cands += [(name, audio, w) for w in ws[:per_clip]]
    rng.shuffle(cands)
    cands.sort(key=lambda c: c[2][2] != "wearer")
    return cands[:n]


def even_windows(length, n, size):
    """n windows of `size` s spread evenly over [2 %, 98 %] of a recording `length` s long."""
    a, b = 0.02 * length, 0.98 * length - size
    if n <= 0 or b <= a:
        return [(0.0, min(size, length))] if n > 0 else []
    step = (b - a) / max(n - 1, 1)
    return [(round(a + i * step, 3), round(a + i * step + size, 3)) for i in range(n)]


def build_real(a, run=subprocess.run):
    rng = random.Random(a.seed)
    root = os.path.join(settings.data_dir(), "evals", "audio", a.name)
    items = []
    if a.project:
        for name, audio, (t0, t1, lab) in speech_windows(project_clips(a.project), a.speech, rng):
            iid = f"{a.origin or 'legends'}-{name}-{int(t0 * 10)}"
            items.append(
                dict(
                    id=iid,
                    input=dict(path=cut(audio, t0, t1, os.path.join(root, iid + ".flac"), run)),
                    tags=dict(origin=a.origin or "legends", kind="speech", speaker=lab, clip=name),
                )
            )
    if a.media:
        stem = a.origin or os.path.splitext(os.path.basename(a.media))[0].lower()
        for t0, t1 in even_windows(duration(a.media, run), a.windows, a.length):
            iid = f"{stem}-{int(t0)}"
            items.append(
                dict(
                    id=iid,
                    input=dict(path=cut(a.media, t0, t1, os.path.join(root, iid + ".flac"), run)),
                    tags=dict(origin=stem, kind="window"),
                )
            )
    for f in sorted(glob.glob(os.path.expanduser(a.files))) if a.files else []:
        stem = os.path.splitext(os.path.basename(f))[0]
        origin = a.origin or "files"
        iid = f"{origin}-{stem}"
        items.append(
            dict(
                id=iid,
                input=dict(path=cut(f, 0.0, min(a.length, duration(f, run)), os.path.join(root, iid + ".flac"), run)),
                tags=dict(origin=origin, kind="file", clip=stem),
            )
        )
    if not items:
        raise SystemExit("nothing to build: give --project, --media or --files")
    rng.shuffle(items)
    return items


# ---- mixes ----------------------------------------------------------------------------------------------------------------------------------
def decode(path, run=subprocess.run):
    import numpy as np

    return np.frombuffer(
        ffmpeg(["-i", path, "-vn", "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], run), np.float32
    ).copy()


def write(path, x, run=subprocess.run):
    import numpy as np

    os.makedirs(os.path.dirname(path), exist_ok=True)
    ffmpeg(
        ["-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", "-c:a", "flac", path],
        run,
        np.asarray(x, np.float32).tobytes(),
    )
    return path


def active_rms(x, frame=0.02):
    """RMS of the louder frames (above -40 dB of the loudest): the speech's level, not its pauses'."""
    import numpy as np

    n = max(int(frame * SR), 1)
    f = x[: len(x) // n * n].reshape(-1, n) if len(x) >= n else x.reshape(1, -1)
    e = np.sqrt(np.mean(f.astype(np.float64) ** 2, axis=1))
    if not len(e) or e.max() <= 0:
        return 0.0
    return float(np.sqrt(np.mean(e[e > e.max() * 0.01] ** 2)))


def mix(speech, noise, snr_db, peak=0.89):
    """speech + noise scaled so the speech's active level stands snr_db above the noise's RMS; the noise looped or cut to the speech's length;
    the pair scaled down together if the mix would clip. Returns (mix, the clean speech as it sits in the mix)."""
    import numpy as np

    noise = np.resize(noise, len(speech)) if len(noise) < len(speech) else noise[: len(speech)]
    ns = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
    g = active_rms(speech) / (ns * 10 ** (snr_db / 20)) if ns > 0 else 0.0
    m = speech + g * noise
    k = min(1.0, peak / max(float(np.abs(m).max()), 1e-9))
    return (m * k).astype(np.float32), (speech * k).astype(np.float32)


def build_mix(a, run=subprocess.run):
    rng = random.Random(a.seed)
    root = os.path.join(settings.data_dir(), "evals", "audio", a.name)
    noises = []  # (origin, file, t0, t1)
    if a.project:
        for _name, audio, segs in project_clips(a.project):
            for t0, t1 in gaps(segs, duration(audio, run)):
                noises.append(("legends", audio, t0, min(t1, t0 + 12)))
    for f in sorted(glob.glob(os.path.expanduser(a.noise_files))) if a.noise_files else []:
        noises.append((a.noise_origin, f, 0.0, None))
    if not noises:
        raise SystemExit("no noise: give --project (its stretches with nobody speaking) or --noise-files")
    clean = sorted(glob.glob(os.path.join(os.path.expanduser(a.clean_root), "*", "*", "*.flac")))
    if not clean:
        raise SystemExit(f"no LibriSpeech under {a.clean_root}")
    by_origin = {}
    for nz in noises:
        by_origin.setdefault(nz[0], []).append(nz)
    items = []
    for i in range(a.n):
        origin = sorted(by_origin)[i % len(by_origin)]  # the noise origins in turn, so each is as well represented
        o, f, t0, t1 = rng.choice(by_origin[origin])
        sp = rng.choice(clean)
        snr = SNRS[(i // len(by_origin)) % len(SNRS)]
        x = decode(sp, run)
        nz = decode(f, run)
        nz = nz[int(t0 * SR) : int(t1 * SR) if t1 else None]
        m, ref = mix(x, nz, snr)
        iid = f"mix-{i:03d}-{o}-{snr:+d}dB"
        items.append(
            dict(
                id=iid,
                input=dict(path=write(os.path.join(root, iid + ".flac"), m, run)),
                reference=dict(audio=write(os.path.join(root, iid + ".clean.flac"), ref, run)),
                tags=dict(noise=o, snr=snr, speech=os.path.basename(sp)[:-5]),
            )
        )
    return items


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["real", "mix"])
    ap.add_argument("--project", help="a strata-360 project folder (clips/*/audio_original.flac and speakers.json)")
    ap.add_argument(
        "--origin",
        default=None,
        help="the origin tag of the items (default: legends for a project, the media file's stem)",
    )
    ap.add_argument("--speech", type=int, default=16, help="real: speech windows from the project")
    ap.add_argument("--media", help="real: a long recording to take windows from")
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--length", type=float, default=10.0)
    ap.add_argument("--files", help="real: a glob of short files, each one item")
    ap.add_argument("--noise-files", help="mix: a glob of noise-only recordings")
    ap.add_argument("--noise-origin", default="files")
    ap.add_argument("--clean-root", default="~/models/eval-data/LibriSpeech/test-clean")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    a.name = a.name or f"denoise-{a.mode}"
    items = build_real(a) if a.mode == "real" else build_mix(a)
    desc = (
        "Real recordings to clean (race speech in wind and crowds, a tape's soundtrack): no reference, judged and scored by DNSMOS/SQUIM"
        if a.mode == "real"
        else f"LibriSpeech test-clean (CC BY 4.0) mixed into real noise at {', '.join(map(str, SNRS))} dB; reference = the clean speech"
    )
    ds = C.write_dataset(a.name, items, description=desc)
    print(f"{a.name}: {len(ds.items)} items, version {ds.version}")


if __name__ == "__main__":
    main()
