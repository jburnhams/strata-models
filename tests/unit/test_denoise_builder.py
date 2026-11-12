"""The denoise-audio dataset builder (scripts/evals/denoise_audio.py) on a fake strata-360 project, a fake ffmpeg and ffprobe: windows of speech,
of a long recording and whole files; mixes of clean speech into noise at a known SNR."""

import importlib.util
import json
import os
import subprocess

import numpy as np
import pytest

from stratamodels.evals import core as C

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
spec = importlib.util.spec_from_file_location(
    "denoise_audio", os.path.join(ROOT, "scripts", "evals", "denoise_audio.py")
)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
SR = B.SR


def test_merge_gaps_and_even_windows():
    segs = [(1.0, 2.0, "wearer"), (2.5, 4.0, "other"), (10.0, 10.5, "other"), (11.0, 30.0, "wearer")]
    assert B.merge(segs) == [
        (0.5, 4.5, "wearer"),
        (8.25, 12.25, "other"),
        (11.0, 23.0, "wearer"),
    ]  # widened to 4 s, cut to 12
    assert B.merge([(0.2, 0.6, "w")]) == [(0.0, 4.0, "w")]  # widened around its middle, never before 0
    assert B.gaps(segs, 40.0) == [(4.5, 9.5), (30.5, 39.5)]  # nobody speaking for 4 s or more, 0.5 s clear of speech
    assert B.even_windows(100.0, 3, 10.0) == [(2.0, 12.0), (45.0, 55.0), (88.0, 98.0)]
    assert B.even_windows(5.0, 2, 10.0) == [(0.0, 5.0)] and B.even_windows(100.0, 0, 10.0) == []


def test_mix_puts_the_speech_at_the_asked_snr():
    rng = np.random.default_rng(0)
    speech = np.concatenate([np.zeros(SR // 2), 0.3 * np.sin(np.arange(SR) * 0.05), np.zeros(SR // 2)]).astype(
        np.float32
    )
    noise = rng.standard_normal(SR // 3).astype(np.float32)  # shorter: looped
    m, ref = B.mix(speech, noise, 5)
    n = m - ref
    assert len(m) == len(speech) and np.abs(m).max() <= 0.89 + 1e-6
    assert 20 * np.log10(B.active_rms(ref) / np.sqrt(np.mean(n.astype(np.float64) ** 2))) == pytest.approx(5, abs=0.1)
    assert B.active_rms(np.zeros(10, np.float32)) == 0.0 and B.mix(speech, np.zeros(10, np.float32), 0)[
        0
    ].tolist() == pytest.approx((speech * min(1.0, 0.89 / 0.3)).tolist(), abs=1e-6)


class Fake:
    """ffprobe answers a length; ffmpeg cuts write a marker file, decodes give a signal, encodes write the bytes they are given."""

    def __init__(self, length=60.0):
        self.length, self.cmds = length, []

    def __call__(self, cmd, input=None, capture_output=True, text=False):
        self.cmds.append(cmd)
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, "N/A\n" if "broken" in cmd[-1] else f"{self.length}\n", "")
        out = cmd[-1]
        if out == "-":  # decode
            return subprocess.CompletedProcess(cmd, 0, (0.1 * np.ones(SR * 2, np.float32)).tobytes(), b"")
        if "fail" in out:
            return subprocess.CompletedProcess(cmd, 1, b"", b"boom")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "wb").write(input or b"cut")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")


@pytest.fixture
def project(tmp_path):
    p = tmp_path / "Race"
    for name, segs in [
        ("0001", [(1, 3, "wearer"), (20, 25, "other")]),
        ("0002", [(5, 9, "wearer"), (12, 14, "wearer"), (40, 44, "other")]),
        ("0003", []),
    ]:
        d = p / "clips" / name
        d.mkdir(parents=True)
        (d / "speakers.json").write_text(
            json.dumps(dict(segments=[dict(t0=a, t1=b, label=l, text="hi") for a, b, l in segs]))
        )
        if name != "0003":
            (d / "audio_original.flac").write_bytes(b"x")
    return str(p)


def args(*a):
    import argparse

    ap = argparse.ArgumentParser()
    for k, v in dict(
        mode=None,
        project=None,
        origin=None,
        speech=16,
        media=None,
        windows=12,
        length=10.0,
        files=None,
        noise_files=None,
        noise_origin="files",
        clean_root=None,
        n=8,
        seed=1,
        name="t",
    ).items():
        ap.add_argument("--" + k.replace("_", "-"), dest=k, default=v, type=type(v) if v is not None else str)
    return ap.parse_args(list(a))


def test_real_items_from_a_project_a_long_recording_and_files(project, tmp_path, monkeypatch):
    monkeypatch.setenv("STRATAMODELS_DATA", str(tmp_path / "data"))
    clips = B.project_clips(project)
    assert [c[0] for c in clips] == ["0001", "0002"] and clips[1][2][0] == (5.0, 9.0, "wearer")
    import random

    w = B.speech_windows(clips, 3, random.Random(1))
    assert len(w) == 3 and [x[2][2] for x in w] == ["wearer"] * 3  # the wearer's first
    for n in ("a.mkv", "b.mkv"):
        (tmp_path / n).write_bytes(b"v")
    fake = Fake(100.0)
    items = B.build_real(
        args(
            "--project",
            project,
            "--speech",
            "4",
            "--media",
            str(tmp_path / "Tape.mp4"),
            "--windows",
            "3",
            "--files",
            str(tmp_path / "*.mkv"),
        ),
        run=fake,
    )
    kinds = sorted(i["tags"]["kind"] for i in items)
    assert kinds == ["file"] * 2 + ["speech"] * 4 + ["window"] * 3
    assert all(
        i["input"]["path"].startswith(str(tmp_path / "data" / "evals" / "audio" / "t"))
        and os.path.isfile(i["input"]["path"])
        for i in items
    )
    assert {i["tags"]["origin"] for i in items} == {"legends", "tape", "files"} and len(
        {i["id"] for i in items}
    ) == len(items)
    with pytest.raises(SystemExit, match="nothing to build"):
        B.build_real(args(), run=fake)
    with pytest.raises(SystemExit, match="could not read the length"):
        B.duration("broken.mp4", run=fake)
    with pytest.raises(SystemExit, match="ffmpeg failed"):
        B.cut("x", 0, 1, str(tmp_path / "fail.flac"), run=fake)


def test_mixes_take_noise_from_quiet_stretches_and_given_files(project, tmp_path, monkeypatch):
    monkeypatch.setenv("STRATAMODELS_DATA", str(tmp_path / "data"))
    ls = tmp_path / "ls" / "1" / "2"
    ls.mkdir(parents=True)
    (ls / "1-2-0001.flac").write_bytes(b"x")
    (tmp_path / "hiss.wav").write_bytes(b"x")
    a = args(
        "--project",
        project,
        "--noise-files",
        str(tmp_path / "hiss.wav"),
        "--noise-origin",
        "tape",
        "--clean-root",
        str(tmp_path / "ls"),
        "--n",
        "8",
    )
    items = B.build_mix(a, run=Fake(60.0))
    assert len(items) == 8 and {i["tags"]["noise"] for i in items} == {"legends", "tape"}
    assert sorted(i["tags"]["snr"] for i in items) == sorted(list(B.SNRS) * 2)
    assert all(os.path.isfile(i["input"]["path"]) and os.path.isfile(i["reference"]["audio"]) for i in items)
    with pytest.raises(SystemExit, match="no noise"):
        B.build_mix(args("--clean-root", str(tmp_path / "ls")), run=Fake())
    with pytest.raises(SystemExit, match="no LibriSpeech"):
        B.build_mix(
            args("--noise-files", str(tmp_path / "hiss.wav"), "--clean-root", str(tmp_path / "none")), run=Fake()
        )


def test_main_writes_the_dataset(project, tmp_path, monkeypatch):
    monkeypatch.setenv("STRATAMODELS_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(B, "build_real", lambda a: [dict(id="x", input=dict(path="/x.flac"), tags={})])
    B.main(["real", "--project", project])
    assert C.load_dataset("denoise-real").items[0]["id"] == "x"
