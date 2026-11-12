"""The sound workers (stratamodels/workers/audio.py: denoise-audio; audio_quality.py: audio-quality) without ffmpeg, torch or any model: a fake
ffmpeg that decodes from a table of signals and runs filters as simple numpy stand-ins, and fake model libraries. The real thing: tests/gpu
(needs the venvs)."""

import os
import subprocess
import sys
import types

import numpy as np
import pytest
import soundfile as sf

WORKERS = os.path.join(os.path.dirname(__file__), "..", "..", "src", "stratamodels", "workers")
sys.path.insert(0, os.path.abspath(WORKERS))
import audio as A  # noqa: E402
import audio_quality as Q  # noqa: E402
from protocol import ParamError  # noqa: E402

SR = 48000


def sine(f=997.0, amp=1.0, s=2.0, sr=SR):
    t = np.arange(int(s * sr)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


class FakeFFmpeg:
    """decode: the signal registered for a path (else the file read with soundfile, at its own rate); -af: highpass, equalizer and speechnorm
    pass through, afftdn halves, alimiter clips at its limit. Every graph is recorded."""

    def __init__(self):
        self.sounds, self.graphs = {}, []

    def __call__(self, cmd, input=None, capture_output=True):
        a = cmd[cmd.index("-i") + 1]
        if "-af" in cmd:
            g = cmd[cmd.index("-af") + 1]
            self.graphs.append(g)
            x = np.frombuffer(input, np.float32).copy()
            if g.startswith("bad"):
                return subprocess.CompletedProcess(cmd, 1, b"", b"No such filter")
            if "afftdn" in g:
                x = x * 0.5
            if "alimiter" in g:
                x = np.clip(x, -0.89, 0.89)
            return subprocess.CompletedProcess(cmd, 0, x.astype(np.float32).tobytes(), b"")
        if a in self.sounds:
            x = self.sounds[a]
        elif os.path.isfile(a) and not a.endswith(".bin"):
            x, _ = sf.read(a, dtype="float32")
        else:
            return subprocess.CompletedProcess(cmd, 1, b"", b"Invalid data found")
        if "-ss" in cmd:
            t0, t1 = float(cmd[cmd.index("-ss") + 1]), float(cmd[cmd.index("-to") + 1])
            x = x[int(t0 * SR) : int(t1 * SR)]
        return subprocess.CompletedProcess(cmd, 0, np.asarray(x, np.float32).tobytes(), b"")


@pytest.fixture
def ff(monkeypatch, tmp_path):
    f = FakeFFmpeg()
    monkeypatch.setattr(A, "RUN", [f])
    monkeypatch.setattr(Q, "RUN", [f])
    src = tmp_path / "noisy.flac"
    src.write_bytes(b"x")
    f.sounds[str(src)] = sine(amp=0.1, s=3.0)
    return f, str(src), str(tmp_path / "out")


def load(engine, **o):
    return A.load(dict(key=f"k-{engine}", name=f"model-{engine}", options=dict(engine=engine, **o)), "/models")


# ---- loudness -------------------------------------------------------------------------------------------------------------------------
def test_loudness_is_bs1770_as_strata360_measured_it(ff):
    assert A.lufs(sine()) == pytest.approx(
        0.0, abs=0.15
    )  # 997 Hz full scale reads -3.01 LUFS on one channel; mono counts as both
    assert A.lufs(sine(amp=0.1)) == pytest.approx(-20.0, abs=0.15)
    assert A.lufs(np.zeros(SR, np.float32)) == -70.0 and A.lufs(np.zeros(0, np.float32)) == -70.0
    y = A.to_lufs(sine(amp=0.01), -14.0)
    assert A.lufs(y) == pytest.approx(-14.0, abs=0.4) and np.abs(y).max() <= 0.89
    assert np.all(A.to_lufs(np.zeros(SR, np.float32), -14.0) == 0)  # silence is not boosted
    assert A.fit(np.ones(5), 3).tolist() == [1, 1, 1] and A.fit(np.ones(2), 4).tolist() == [1, 1, 0, 0]


# ---- the chain -----------------------------------------------------------------------------------------------------------------------
def test_the_original_through_the_chain_is_only_levelled(ff):
    f, src, out = ff
    load("none")
    r = A.run(dict(path=src, highpass=0, lufs=-14, out_dir=out))
    y, sr = sf.read(r["path"], dtype="float32")
    assert (
        sr == SR
        and len(y) == 3 * SR
        and r["duration_s"] == 3.0
        and r["engine"] == "none"
        and r["model"] == "model-none"
    )
    assert (
        r["lufs_in"] == pytest.approx(-20, abs=0.2)
        and r["lufs_out"] == pytest.approx(-14, abs=0.4)
        and r["removed_db"] == 0
    )
    assert not any("highpass" in g for g in f.graphs) and not any("speechnorm" in g for g in f.graphs)
    r = A.run(dict(path=src, lufs=None, clip=[1.0, 2.0], out_dir=out))
    assert (
        r["duration_s"] == 1.0
        and r["lufs_out"] == pytest.approx(-20, abs=0.2)
        and f.graphs[-1].startswith("highpass=f=80")
    )


def test_strata360s_chain_caps_the_denoiser_and_finishes_the_speech(ff):
    f, src, out = ff
    load("afftdn", defaults=dict(highpass=80, atten_lim_db=20, finish="strata360", lufs=-14, nr=12))
    r = A.run(dict(path=src, out_dir=out))
    # afftdn halves the sound here; capped at 20 dB a tenth of what it took out comes back: 0.5 + 0.1 x 0.5 = 0.55 of the input
    assert r["removed_db"] == pytest.approx(-20 * np.log10(0.55), abs=0.05)
    assert (
        "afftdn=nr=12.0:nf=-40.0" in " ".join(f.graphs)
        and any(g == A.FINISH for g in f.graphs)
        and r["lufs_out"] == pytest.approx(-14, abs=0.4)
    )
    r = A.run(dict(path=src, atten_lim_db=None, out_dir=out))  # the job's own settings win over the model's defaults
    assert r["removed_db"] == pytest.approx(-20 * np.log10(0.5), abs=0.05)


@pytest.mark.parametrize(
    ("params", "msg"),
    [
        (dict(colour=1), "unknown params"),
        (dict(path="/no/such.wav"), "no such audio file"),
        (dict(clip=[2, 1]), "clip is"),
        (dict(clip="0-1"), "clip is"),
        (dict(finish="loud"), "finish is"),
        (dict(atten_lim_db=0), "positive number"),
        (dict(lufs=0), "lufs is between"),
        (dict(clip=[9, 10]), "no audio in"),
    ],
)
def test_bad_params_are_refused(ff, params, msg):
    _, src, out = ff
    load("none")
    with pytest.raises(ParamError, match=msg):
        A.run(dict(dict(path=src, out_dir=out), **params))


def test_ffmpeg_failures_name_the_step(ff, tmp_path):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"?")
    with pytest.raises(ParamError, match="decoding"):
        A.decode(str(bad))
    with pytest.raises(ParamError, match="ffmpeg -af bad"):
        A.ffilter(np.zeros(10, np.float32), "bad=1")


# ---- engines -----------------------------------------------------------------------------------------------------------------------------
class T:
    """A stand-in torch tensor."""

    def __init__(self, a):
        self.a = np.asarray(a, np.float32)

    def __getitem__(self, i):
        return T(self.a[i])

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def to(self, dev):
        self.dev = dev
        return self


@pytest.fixture
def fake_torch(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.from_numpy = T
    torch.Tensor = T  # scipy checks arrays against torch.Tensor when a torch module is loaded
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_noisereduce_and_unknown_engines(ff, monkeypatch):
    _, src, out = ff
    nr = types.ModuleType("noisereduce")
    seen = {}

    def reduce_noise(y, sr, stationary, prop_decrease):
        seen.update(sr=sr, stationary=stationary, prop=prop_decrease)
        return np.concatenate([y * 0.1, np.zeros(7, np.float32)])  # a few samples too many: cut back

    nr.reduce_noise = reduce_noise
    monkeypatch.setitem(sys.modules, "noisereduce", nr)
    assert load("noisereduce") == dict(device="cpu")
    r = A.run(dict(path=src, highpass=0, prop_decrease=0.8, stationary=False, out_dir=out))
    assert seen == dict(sr=SR, stationary=False, prop=0.8) and r["removed_db"] == pytest.approx(20, abs=0.05)
    with pytest.raises(RuntimeError, match="unknown engine"):
        load("magic")


def test_deepfilternet_caps_natively(ff, fake_torch, monkeypatch):
    _, src, out = ff
    df, enh = types.ModuleType("df"), types.ModuleType("df.enhance")
    calls = []
    enh.init_df = lambda log_level: (T([0]), "state", "suffix")

    def enhance(model, st, t, atten_lim_db=None):
        calls.append(atten_lim_db)
        return T(t.a * 0.25)

    enh.enhance = enhance
    monkeypatch.setitem(sys.modules, "df", df)
    monkeypatch.setitem(sys.modules, "df.enhance", enh)
    assert load("dfn3") == dict(device="cpu") and A.E["m"][0].dev == "cpu"
    r = A.run(dict(path=src, highpass=0, atten_lim_db=18, out_dir=out))
    assert calls == [18] and r["removed_db"] == pytest.approx(
        -20 * np.log10(0.25), abs=0.05
    )  # not mixed back again here


def test_clearvoice_runs_file_to_file(ff, fake_torch, monkeypatch):
    _, src, out = ff
    cv = types.ModuleType("clearvoice")

    class ClearVoice:
        def __init__(self, task, model_names):
            self.task, self.names = task, model_names

        def __call__(self, input_path, online_write):
            return sf.read(input_path, dtype="float32")[0] * 0.5

        def write(self, y, output_path):
            sf.write(output_path, y, SR, subtype="FLOAT")

    cv.ClearVoice = ClearVoice
    monkeypatch.setitem(sys.modules, "clearvoice", cv)
    load("clearvoice")
    assert A.E["m"].names == ["model-clearvoice"] and A.E["m"].task == "speech_enhancement"
    r = A.run(dict(path=src, highpass=0, out_dir=out))
    assert r["removed_db"] == pytest.approx(6.02, abs=0.05)


def test_resemble_denoises_or_enhances_and_comes_back_at_48k(ff, fake_torch, monkeypatch):
    _, src, out = ff
    pkg, enh, inf = (
        types.ModuleType(n)
        for n in ("resemble_enhance", "resemble_enhance.enhancer", "resemble_enhance.enhancer.inference")
    )
    got = {}
    inf.denoise = lambda t, sr, dev: (T(t.a * 0.5), SR)

    def enhance(t, sr, dev, nfe, solver, lambd, tau):
        got.update(nfe=nfe, lambd=lambd, tau=tau)
        return T(t.a[: len(t.a) // 2]), 24000  # another rate: written and decoded back at 48 kHz

    inf.enhance = enhance
    enh.inference = inf
    for n, m in (
        ("resemble_enhance", pkg),
        ("resemble_enhance.enhancer", enh),
        ("resemble_enhance.enhancer.inference", inf),
    ):
        monkeypatch.setitem(sys.modules, n, m)
    load("resemble")
    assert A.run(dict(path=src, highpass=0, out_dir=out))["removed_db"] == pytest.approx(6.02, abs=0.05)
    r = A.run(dict(path=src, highpass=0, mode="enhance", nfe=32, out_dir=out))
    assert got == dict(nfe=32, lambd=0.9, tau=0.5) and r["duration_s"] == 3.0
    with pytest.raises(ParamError, match="denoise or enhance"):
        A.run(dict(path=src, mode="loud", out_dir=out))


# ---- audio-quality ---------------------------------------------------------------------------------------------------------------------
class Sess:
    def __init__(self):
        self.n = 0

    def get_inputs(self):
        return [types.SimpleNamespace(name="input_1")]

    def run(self, _, feed):
        self.n += 1
        assert feed["input_1"].shape == (1, int(Q.WINDOW * Q.SR))
        return [np.array([[3.0, 4.0, 3.0]], np.float32)]


def test_dnsmos_windows_and_calibration():
    s = Sess()
    r = Q.dnsmos(
        np.zeros(Q.SR * 3, np.float32), s
    )  # short: doubled until it fills a window (6, then 12 s), as dnsmos_local.py does
    assert (
        s.n == 3
        and r["sig"] == pytest.approx(np.polyval(Q.POLY["sig"], 3.0), abs=1e-3)
        and r["bak"] == pytest.approx(np.polyval(Q.POLY["bak"], 4.0), abs=1e-3)
    )
    s = Sess()
    Q.dnsmos(np.zeros(int(Q.SR * 12.5), np.float32), s)
    assert s.n == 3  # 9.01 s windows, a second apart: starting at 0, 1 and 2 s
    with pytest.raises(ParamError, match="no audio"):
        Q.dnsmos(np.zeros(0, np.float32), Sess())


def test_intrusive_measures_align_first():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(Q.SR * 2).astype(np.float32)
    late = np.roll(ref, 37)
    e, r = Q.align(late, ref)
    assert np.allclose(e[100:-100], r[100:-100])
    assert Q.si_sdr(ref * 3, ref) > 60 and Q.si_sdr(
        ref + ref.std() * rng.standard_normal(len(ref)), ref
    ) == pytest.approx(0, abs=0.3)


def test_quality_run_reads_what_is_asked(ff, monkeypatch, tmp_path):
    f, src, _ = ff
    ref = tmp_path / "clean.flac"
    ref.write_bytes(b"x")
    f.sounds[str(ref)] = sine(amp=0.1, s=3.0)
    pesq, pystoi = types.ModuleType("pesq"), types.ModuleType("pystoi")
    pesq.pesq = lambda sr, r, e, mode: 4.2 if mode == "wb" and sr == Q.SR else 0
    pystoi.stoi = lambda r, e, sr, extended: 0.95 if extended else 0
    monkeypatch.setitem(sys.modules, "pesq", pesq)
    monkeypatch.setitem(sys.modules, "pystoi", pystoi)
    monkeypatch.setitem(Q.M, "dnsmos", Sess())
    monkeypatch.setattr(Q, "squim", lambda x, m: dict(stoi=0.9, pesq=3.0, si_sdr=15.0))
    monkeypatch.setitem(Q.M, "squim", None)
    r = Q.run(dict(path=src, reference=str(ref)))
    assert (
        set(r) == {"duration_s", "dnsmos", "squim", "intrusive"}
        and r["intrusive"]["pesq"] == 4.2
        and r["intrusive"]["estoi"] == 0.95
    )
    assert r["intrusive"]["si_sdr"] > 30  # the same signal, decoded at 16 kHz
    assert set(Q.run(dict(path=src, metrics=["dnsmos", "intrusive"]))) == {
        "duration_s",
        "dnsmos",
    }  # no reference: nothing to compare with
    for p, msg in [
        (dict(path="/no.wav"), "no such audio"),
        (dict(path=src, reference="/no.wav"), "no such reference"),
        (dict(path=src, metrics=["mos"]), "unknown metrics"),
    ]:
        with pytest.raises(ParamError, match=msg):
            Q.run(p)


def test_quality_load_needs_the_dnsmos_weights(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="setup_worker_venv.sh audio-quality"):
        Q.load(dict(options={}), str(tmp_path))
    w = tmp_path / "audio-quality" / "sig_bak_ovr.onnx"
    w.parent.mkdir()
    w.write_bytes(b"onnx")
    ort, ta = types.ModuleType("onnxruntime"), types.ModuleType("torchaudio")
    ort.InferenceSession = lambda path, providers: ("session", path, providers)
    ta.pipelines = types.SimpleNamespace(
        SQUIM_OBJECTIVE=types.SimpleNamespace(get_model=lambda: types.SimpleNamespace(eval=lambda: "squim"))
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "torchaudio", ta)
    assert (
        Q.load(dict(options={}), str(tmp_path)) == dict(device="cpu")
        and Q.M["dnsmos"][1] == str(w)
        and Q.M["squim"] == "squim"
    )


def test_squim_reads_the_three_estimates(fake_torch, monkeypatch):
    import contextlib

    fake_torch.inference_mode = contextlib.nullcontext
    model = lambda t: ([0.91], [3.2], [14.0])
    assert Q.squim(np.zeros(16, np.float32), model) == dict(stoi=0.91, pesq=3.2, si_sdr=14.0)
