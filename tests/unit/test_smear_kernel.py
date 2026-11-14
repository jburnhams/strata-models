"""scripts/evals/smear_kernel.py: a line's impulse response fitted from rows of dark marks on a light ground (synthetic, no video)."""

import importlib.util
import os
import subprocess

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "smear_kernel", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "evals", "smear_kernel.py")
)
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)


def rows_through(kernel, taps, n=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ideal = np.full(220, 200.0)
        for x in rng.choice(np.arange(70, 150, 12), 4, replace=False):
            ideal[x : x + rng.integers(3, 7)] = 30.0
        y = np.zeros_like(ideal)
        for t, v in zip(taps, kernel, strict=True):
            y += v * np.roll(ideal - 200, t)
        out.append(y + 200 + rng.normal(0, 1, ideal.size))
    return out


def test_fit_recovers_a_known_line_response():
    taps = [-1, 0, 1, 2, 3, 4]
    true = [0.2, 0.5, 0.3, 0.1, -0.06, -0.04]
    t, h, err = K.fit(rows_through(true, taps), left=4, right=20, smooth=10.0)
    got = dict(zip(t.tolist(), h, strict=True))
    assert err < 3 and all(abs(got[k] - v) < 0.03 for k, v in zip(taps, true, strict=True))
    kt, kh = K.kernel(t, h, reach=4)
    assert kt == list(range(-4, 5)) and abs(sum(kh) - 1) < 1e-3


def test_fit_needs_contrast_and_main_prints_toml(monkeypatch, capsys):
    with pytest.raises(SystemExit, match="enough contrast"):
        K.fit([np.full(100, 128.0)])
    frames = rows_through([0.2, 0.6, 0.2], [-1, 0, 1], n=30)
    monkeypatch.setattr(K, "frame", lambda path, t, crop: np.array(frames))
    K.main(["clip.mkv", "--box", "0", "0", "220", "30", "--times", "1"])
    out = capsys.readouterr().out
    assert out.startswith("# fitted on 30 rows") and "desmear = { taps = [-4," in out


def test_frame_reads_a_grey_frame_through_ffmpeg(monkeypatch):
    seen = []

    def run(cmd, capture_output=True, check=True):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, bytes(range(6)), b"")

    monkeypatch.setattr(K.subprocess, "run", run)
    f = K.frame("c.mkv", 1.5, (1, 2, 3, 2))
    assert (
        f.shape == (2, 3)
        and "crop=3:2:1:2" in seen[0][seen[0].index("-vf") + 1]
        and seen[0][seen[0].index("-ss") + 1] == "1.500"
    )
