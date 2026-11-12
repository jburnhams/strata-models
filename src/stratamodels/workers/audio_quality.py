"""How good a recording of speech sounds, measured by models and formulas rather than people: the `audio-quality` role, run as a venv-worker in
its own venv (`scripts/setup_worker_venv.sh audio-quality`), on the CPU. An eval's [score] pass sends it every output file (evals.core).

Without a reference (real recordings: nothing clean to compare with), two learned estimators of what listeners would say:
  dnsmos   Microsoft's DNSMOS P.835 (DNS Challenge, sig_bak_ovr.onnx): mean opinion scores 1-5 for the speech itself (sig: distortion), the
           background (bak: how intrusive the noise is) and overall (ovrl). 9 s windows a second apart, averaged; 16 kHz.
  squim    torchaudio's SQUIM objective (Kumar et al. 2023): estimates of STOI (intelligibility, 0-1), PESQ (1-4.5) and SI-SDR (dB) as if a
           clean reference existed.
With a reference (`reference`: the clean speech a noisy test item was mixed from), the measured versions: si_sdr (dB, scale-invariant), pesq
(wide band, 16 kHz) and estoi (extended STOI). The two are aligned by cross-correlation first (a denoiser may shift by a few samples).

params: path (required), reference (optional), metrics (default all that apply: ["dnsmos", "squim", "intrusive"]).
Result: {dnsmos: {sig, bak, ovrl}, squim: {stoi, pesq, si_sdr}, intrusive: {si_sdr, pesq, estoi}, duration_s} (each only when asked and
possible). Weights: <models>/audio-quality/sig_bak_ovr.onnx (fetched by the setup script); SQUIM's come with torchaudio (cached in torch's
hub folder on first use)."""

import os
import subprocess

from protocol import ParamError, serve

SR = 16000
M = {}
RUN = [subprocess.run]
WINDOW = 9.01  # DNSMOS P.835's input length, seconds (dnsmos_local.py)
# DNSMOS P.835 calibration (dnsmos_local.py, not personalised): raw network outputs mapped to the P.835 scale.
POLY = dict(
    sig=(-0.08397278, 1.22083953, 0.0052439),
    bak=(-0.13166888, 1.60915514, -0.39604546),
    ovrl=(-0.06766283, 1.11546468, 0.04602535),
)


def decode(path, sr=SR):
    import numpy as np

    p = RUN[0](
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True,
    )
    if p.returncode != 0:
        raise ParamError(f"ffmpeg could not decode {path}: {p.stderr.decode(errors='replace').strip()[-300:]}")
    return np.frombuffer(p.stdout, np.float32).copy()


def load(model, models_dir):
    o = model.get("options") or {}
    M.clear()
    onnx = o.get("dnsmos") or os.path.join(models_dir, "audio-quality", "sig_bak_ovr.onnx")
    if not os.path.isfile(onnx):
        raise RuntimeError(f"missing {onnx}: run scripts/setup_worker_venv.sh audio-quality")
    import onnxruntime as ort

    M["dnsmos"] = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    import torchaudio

    M["squim"] = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().eval()
    return dict(device="cpu")


def dnsmos(x, sess):
    """{sig, bak, ovrl} of 16 kHz audio: windows of 9.01 s a second apart (a shorter clip is repeated to fill one), each mapped by POLY, averaged."""
    import numpy as np

    n = int(WINDOW * SR)
    if not len(x):
        raise ParamError("no audio to score")
    while len(x) < n:
        x = np.concatenate([x, x])
    hops = int(np.floor(len(x) / SR) - WINDOW) + 1
    name = sess.get_inputs()[0].name
    got = []
    for i in range(max(hops, 1)):
        seg = x[i * SR : i * SR + n]
        if len(seg) < n:
            continue
        sig, bak, ovrl = sess.run(None, {name: seg.astype(np.float32)[None]})[0][0]
        got.append([np.polyval(POLY[k], float(v)) for k, v in (("sig", sig), ("bak", bak), ("ovrl", ovrl))])
    m = np.mean(got, axis=0)
    return dict(sig=round(float(m[0]), 3), bak=round(float(m[1]), 3), ovrl=round(float(m[2]), 3))


def squim(x, model):
    import torch

    with torch.inference_mode():
        stoi, pesq, si_sdr = model(torch.from_numpy(x)[None])
    return dict(stoi=round(float(stoi[0]), 4), pesq=round(float(pesq[0]), 3), si_sdr=round(float(si_sdr[0]), 2))


def align(est, ref, max_lag=SR // 10):
    """est shifted to line up with ref (cross-correlation within +-max_lag samples), both cut to the same length."""
    import numpy as np

    n = min(len(est), len(ref))
    e, r = est[:n], ref[:n]
    if n > 4 * max_lag:
        c = np.correlate(
            e, r[max_lag:-max_lag], mode="valid"
        )  # c[k] peaks at k = max_lag + d when est lags ref by d samples
        d = int(np.argmax(np.abs(c))) - max_lag
        e = np.roll(e, -d) if d else e
    return e, r


def si_sdr(est, ref):
    import numpy as np

    est, ref = est - est.mean(), ref - ref.mean()
    a = float(np.dot(est, ref) / max(np.dot(ref, ref), 1e-12))
    t = a * ref
    return float(10 * np.log10(max(np.dot(t, t), 1e-12) / max(np.dot(est - t, est - t), 1e-12)))


def intrusive(x, ref):
    from pesq import pesq
    from pystoi import stoi

    e, r = align(x, ref)
    return dict(
        si_sdr=round(si_sdr(e, r), 2),
        pesq=round(float(pesq(SR, r, e, "wb")), 3),
        estoi=round(float(stoi(r, e, SR, extended=True)), 4),
    )


def run(p):
    path = p.get("path")
    if not isinstance(path, str) or not os.path.isfile(path):
        raise ParamError(f"no such audio file: {path!r}")
    ref = p.get("reference")
    if ref is not None and (not isinstance(ref, str) or not os.path.isfile(ref)):
        raise ParamError(f"no such reference file: {ref!r}")
    want = p.get("metrics") or ["dnsmos", "squim", "intrusive"]
    unknown = set(want) - {"dnsmos", "squim", "intrusive"}
    if unknown:
        raise ParamError(f"unknown metrics {sorted(unknown)}")
    x = decode(path)
    out = dict(duration_s=round(len(x) / SR, 3))
    if "dnsmos" in want:
        out["dnsmos"] = dnsmos(x, M["dnsmos"])
    if "squim" in want:
        out["squim"] = squim(x, M["squim"])
    if "intrusive" in want and ref:
        out["intrusive"] = intrusive(x, decode(ref))
    return out


if __name__ == "__main__":
    serve(load, run)
