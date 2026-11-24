"""The shot-boundary worker (stratamodels/workers/shots.py) without torch, TransNet V2 or PySceneDetect: its own rules (settings, TransNet V2's
windows over a stream, runs, the one minimum-shot rule). The real detectors run in the shots venv (tests/gpu); ffmpeg's decode is in
tests/integration/test_shots_decode.py."""

import os
import sys

import pytest

WORKERS = os.path.join(os.path.dirname(__file__), "..", "..", "src", "stratamodels", "workers")
sys.path.insert(0, os.path.abspath(WORKERS))
import shots as S  # noqa: E402
from protocol import ParamError  # noqa: E402


def test_settings_are_the_engines_defaults_then_the_models_options_then_the_jobs_params():
    s = S.settings_of(
        "adaptive", {"threshold": 2.5, "worker": "shots", "engine": "adaptive"}, {"window_width": 3, "luma_only": True}
    )
    assert s == dict(
        threshold=2.5, min_shot_s=0.6, width=256, window_width=3, min_content_val=15.0, luma_only=True, deinterlace=True
    )
    assert S.settings_of("transnetv2", {}, {"threshold": 0.3, "deinterlace": False}) == dict(
        threshold=0.3, min_shot_s=0.6, deinterlace=False
    )
    assert S.settings_of("content", {}, {"width": 301})["width"] == 300  # even, as ffmpeg's scale needs


@pytest.mark.parametrize(
    ("engine", "params", "says"),
    [
        ("transnetv2", {"threshold": 1.5}, "chance from 0 to 1"),
        ("content", {"threshold": "high"}, "must be a number"),
        ("adaptive", {"min_shot_s": -1}, "must not be negative"),
        ("adaptive", {"luma_only": 1}, "true or false"),
    ],
)
def test_wrong_settings_are_the_jobs_fault(engine, params, says):
    with pytest.raises(ParamError, match=says):
        S.settings_of(engine, {}, params)


def reference_windows(n):
    """TransNet V2's own predict_frames padding (transnetv2-pytorch 1.0.5), on frame numbers."""
    frames = list(range(n))
    end = 25 + 50 - (n % 50 if n % 50 != 0 else 50)
    padded = [frames[0]] * 25 + frames + [frames[-1]] * end
    return [padded[p : p + 100] for p in range(0, len(padded) - 99, 50)]


@pytest.mark.parametrize("n", [1, 49, 50, 51, 100, 175, 1234])
def test_the_streamed_windows_are_transnets_own_and_their_middles_cover_every_frame_once(n):
    windows = list(S.transnet_windows(iter(range(n))))
    assert windows == reference_windows(n)
    middles = [f for w in windows for f in w[25:75]][:n]
    assert middles == list(range(n))


def test_no_frames_no_windows():
    assert list(S.transnet_windows(iter([]))) == []


def test_runs_over_a_threshold():
    assert S.runs_over([0.1, 0.9, 0.8, 0.2, 0.6, 0.1, 0.7], 0.5) == [(1, 2), (4, 4), (6, 6)]
    assert S.runs_over([], 0.5) == []


def test_of_cuts_too_close_the_strongest_wins_and_the_first_frame_is_never_a_cut():
    times = [i * 0.04 for i in range(100)]
    scores = [0.0] * 100
    scores[0], scores[10], scores[12], scores[60] = 1.0, 0.6, 0.9, 0.7
    assert S.pick_cuts(scores, times, [0, 10, 12, 60], 0.6) == [12, 60]
    assert S.pick_cuts(scores, times, [10, 12, 60], 0.0) == [10, 12, 60]


def test_black_stretches_and_times_from_ffmpegs_listing():
    text = "frame:0 pts:0 pts_time:0\nlavfi.scene_score=0\nframe:1 pts:1 pts_time:0.04\nlavfi.black_start=0.04\nframe:2 pts:2 pts_time:0.08\n"
    text += "lavfi.black_end=0.5\nframe:3 pts:3 pts_time:0.12\nlavfi.black_start=0.12\n"
    meta = S.parse_metadata(text)
    assert [m["time"] for m in meta] == [0.0, 0.04, 0.08, 0.12]
    assert S.black_stretches(meta, 0.16, 0.4) == [{"start": 0.04, "end": 0.5}, {"start": 0.12, "end": 0.16}]


def test_load_names_the_engines():
    with pytest.raises(ValueError, match="transnetv2, adaptive, content"):
        S.load({"key": "k", "options": {"engine": "magic"}}, "/models")


def test_a_bad_request_fails_the_job_not_the_worker(tmp_path):
    S.M.clear()
    S.M.update(key="k", engine="content", opts={})
    with pytest.raises(ParamError, match="no such video"):
        S.run({"path": str(tmp_path / "missing.mp4")})
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    with pytest.raises(ParamError, match=r"unknown params for content: \['window_width'\]"):
        S.run({"path": str(video), "window_width": 2})
    with pytest.raises(ParamError, match="clip must be"):
        S.run({"path": str(video), "clip": [3, 1]})


def meta_of(n, fps=25.0, black=None):
    meta = [{"time": i / fps} for i in range(n)]
    for k, i, t in black or []:
        meta[i][k] = t
    return meta


def test_transnets_runs_become_cuts_on_the_frame_after_their_peak_and_long_runs_are_transitions():
    scores = [0.0] * 100
    scores[24] = 0.95  # a hard cut: the old shot's last frame
    scores[59:66] = [0.55, 0.7, 0.8, 0.9, 0.8, 0.7, 0.55]  # a dissolve, peaking at 62
    scores[99] = 0.9  # peaks on the last frame looked at: the new shot would start after the end
    s = S.settings_of("transnetv2", {}, {})
    a = S.answer(
        "transnetv2", s, meta_of(100, black=[("black_start", 30, 1.2), ("black_end", 40, 1.6)]), scores, [], 10.0, 0.4
    )
    assert a["cuts"] == [{"time": 11.0, "frame": 25, "score": 0.95}, {"time": 12.52, "frame": 63, "score": 0.9}]
    assert a["transitions"] == [{"start": 12.36, "end": 12.64}]
    assert a["shots"] == [{"start": 10.0, "end": 11.0}, {"start": 11.0, "end": 12.52}, {"start": 12.52, "end": 14.0}]
    assert (
        a["black"] == [{"start": 11.2, "end": 11.6}]
        and a["fps"] == 25.0
        and a["frames"] == 100
        and a["duration"] == 4.0
    )


def test_pyscenedetects_cuts_are_its_own_frames_and_black_can_be_left_out():
    scores = [0.0] * 50
    scores[20], scores[22] = 40.0, 60.0
    s = S.settings_of("content", {}, {})
    a = S.answer("content", s, meta_of(40), scores, [20, 22, 45], 0.0, None)  # 45: past a stream cut short
    assert [c["frame"] for c in a["cuts"]] == [22] and a["black"] == [] and a["frames"] == 40 and a["transitions"] == []


class Arr(__import__("numpy").ndarray):
    """A numpy array that answers torch's .to(), .cpu() and .numpy()."""

    def to(self, _device):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return __import__("numpy").asarray(self)


@pytest.fixture
def fake_torch(monkeypatch):
    import contextlib
    import types

    import numpy as np

    torch = types.ModuleType("torch")
    torch.__version__ = "2.x"
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    torch.from_numpy = lambda a: np.asarray(a).view(Arr)
    torch.no_grad = contextlib.nullcontext

    class TransNetV2:
        def __init__(self, device):
            self.device = device

        def predict_raw(self, x):  # each frame's chance: its first pixel's red value / 100, shaped [batch, 100, 1]
            single = (x[:, :, 0, 0, 0:1].astype(np.float32) / 100).view(Arr)
            return single, single

    tn = types.ModuleType("transnetv2_pytorch")
    tn.TransNetV2 = TransNetV2
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transnetv2_pytorch", tn)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.0.5")


@pytest.mark.parametrize(("vram", "device", "want"), [(0.5, None, "cuda"), (0.5, "cpu", "cpu"), (0.0, None, "cpu")])
def test_transnet_goes_on_the_gpu_only_when_the_model_has_vram_and_does_not_say_cpu(fake_torch, vram, device, want):
    opts = {"engine": "transnetv2"} | ({"device": device} if device else {})
    info = S.load({"key": "transnetv2", "vram_gb": vram, "options": opts}, "/models")
    assert info == {"engine": "transnetv2-pytorch 1.0.5, torch 2.x", "device": want} and S.M["net"].device == want


def test_transnet_scores_every_frame_once_in_order_across_batches(fake_torch):
    import numpy as np

    S.load({"key": "transnetv2", "vram_gb": 0, "options": {"engine": "transnetv2"}}, "/models")
    n = 50 * S.BATCH + 37  # more windows than one batch, and a part window at the end
    frames = (np.full((27, 48, 3), i % 100, np.uint8) for i in range(n))
    assert S.transnet_scores(frames) == pytest.approx([(i % 100) / 100 for i in range(n)])


@pytest.fixture
def fake_scenedetect(monkeypatch):
    """PySceneDetect's detector interface as 0.7.3 has it: process_frame(timecode, frame) -> cuts, post_process, get_metrics, a
    StatsManager. These detectors call a frame a cut when its first pixel is over the threshold, one frame behind, and keep the last for
    post_process, as AdaptiveDetector's window and ContentDetector's flash filter hold frames back."""
    import types

    made = {}

    class FrameTimecode:
        def __init__(self, i, fps):
            self.frame_num = i

    class Stats:
        def __init__(self):
            self.m = {}

        def register_metrics(self, keys):
            self.keys = list(keys)

        def get_metrics(self, i, keys):
            return [self.m.get((i, k)) for k in keys]

    class Detector:
        def __init__(self, **kw):
            made.update(kw, kind=type(self).__name__)
            self.held = None

        def get_metrics(self):
            return (
                ["content_val", "adaptive_ratio (w=2)"]
                if type(self).__name__ == "AdaptiveDetector"
                else ["content_val"]
            )

        def process_frame(self, tc, frame):
            v = float(frame[0, 0, 0])
            self.stats_manager.m[(tc.frame_num, self.get_metrics()[-1])] = v
            out, self.held = self.held, (tc if v > made["threshold"] else None)
            return [out] if out else []

        def post_process(self, tc):
            return [self.held] if self.held else []

    class AdaptiveDetector(Detector):
        def __init__(self, adaptive_threshold, **kw):
            super().__init__(threshold=adaptive_threshold, **kw)

    class ContentDetector(Detector):
        pass

    mods = {
        "scenedetect": types.ModuleType("scenedetect"),
        "scenedetect.common": types.SimpleNamespace(FrameTimecode=FrameTimecode),
        "scenedetect.detectors": types.SimpleNamespace(
            AdaptiveDetector=AdaptiveDetector, ContentDetector=ContentDetector
        ),
        "scenedetect.stats_manager": types.SimpleNamespace(StatsManager=Stats),
    }
    mods["scenedetect"].__version__ = "0.7.3"
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    return made


@pytest.mark.parametrize("engine", ["adaptive", "content"])
def test_pyscenedetect_gets_every_frame_its_own_lengths_off_and_answers_its_score(fake_scenedetect, engine):
    import numpy as np

    assert S.load({"key": "k", "options": {"engine": engine}}, "/m") == {
        "engine": f"scenedetect 0.7.3 {engine}",
        "device": "cpu",
    }
    s = S.settings_of(engine, {}, {"threshold": 50})
    values = [0] * 10 + [90] + [0] * 8 + [80]
    scores, cuts = S.scenedetect_scores(engine, s, (np.full((4, 4, 3), v, np.uint8) for v in values))
    assert cuts == [10, 19] and scores == [float(v) for v in values]
    assert fake_scenedetect["min_scene_len"] == 0 and fake_scenedetect["threshold"] == 50
    if engine == "adaptive":
        assert fake_scenedetect["window_width"] == 2 and fake_scenedetect["min_content_val"] == 15.0


class Decoded:
    """What decode returns, without ffmpeg: a finished process, its stderr file, the frames and the listing it wrote."""

    def __init__(self, tmp, n, code=0, listing=True):
        import io

        self.stdout, self.code = io.BytesIO(b""), code
        self.listing = os.path.join(tmp, "frames.txt")
        if listing:
            with open(self.listing, "w") as f:
                f.write("".join(f"frame:{i} pts:{i} pts_time:{i / 25}\nlavfi.scene_score=0\n" for i in range(n)))

    def wait(self, _timeout=None):
        return self.code

    def poll(self):
        return self.code


@pytest.fixture
def no_ffmpeg(monkeypatch, tmp_path):
    """decode stood in for (the real one is in tests/integration/test_shots_decode.py), with TransNet V2 scoring frame 24 as a cut."""
    import io

    seen = {}

    def decode(path, s, engine, clip, tmp, black, black_min_s, black_pixel):
        seen.update(clip=clip, black=black, s=s)
        p = Decoded(tmp, 0 if "empty" in path else 50, code=1 if "broken" in path else 0)
        err = io.StringIO("broken: Invalid data\n" if "broken" in path else "")
        return p, err, iter(range(50)), p.listing

    monkeypatch.setattr(S, "decode", decode)
    monkeypatch.setattr(S, "transnet_scores", lambda frames: [0.9 if i == 24 else 0.0 for i in frames])
    S.M.clear()
    S.M.update(key="transnetv2", engine="transnetv2", opts={"threshold": 0.4}, version="stand-in", device="cuda")
    for name in ("v.mp4", "empty.mp4", "broken.mp4"):
        (tmp_path / name).write_bytes(b"")
    return seen


def test_run_answers_with_the_model_and_every_setting_it_ran_with(no_ffmpeg, tmp_path):
    r = S.run({"path": str(tmp_path / "v.mp4"), "clip": [2, 4], "black": False, "scores": True, "out_dir": "/gen"})
    assert no_ffmpeg["clip"] == (2.0, 4.0) and no_ffmpeg["black"] is False
    assert r["cuts"] == [{"time": 3.0, "frame": 25, "score": 0.9}] and r["scores"][24] == 0.9 and len(r["scores"]) == 50
    assert (r["model"], r["engine"], r["device"]) == ("transnetv2", "stand-in", "cuda")
    assert r["settings"] == {"threshold": 0.4, "min_shot_s": 0.6, "deinterlace": True} and r["seconds"] >= 0


@pytest.mark.parametrize(
    ("name", "params", "says"),
    [
        ("empty.mp4", {}, "no video frames"),
        ("broken.mp4", {}, "ffmpeg failed on .*Invalid data"),
        ("v.mp4", {"black_min_s": 0}, "black_min_s must be from"),
        ("v.mp4", {"black_pixel": "dark"}, "black_pixel must be a number"),
        ("v.mp4", {"clip": "all"}, r"clip must be \[start, end\] in seconds"),
    ],
)
def test_run_says_what_was_wrong_with_the_job(no_ffmpeg, tmp_path, name, params, says):
    with pytest.raises(ParamError, match=says):
        S.run({"path": str(tmp_path / name), **params})


@pytest.mark.parametrize(
    ("probe", "width", "want"),
    [
        (
            {"width": 1920, "height": 1080},
            272,
            154,
        ),  # 153: halves up to even, as scale=272:-2 would
        ({"width": 720, "height": 576}, 256, 204),
        (
            {"width": 1920, "height": 1080, "side_data_list": [{"rotation": -90}]},
            256,
            456,
        ),  # a phone video stored on its side
        ({"width": 1920, "height": 1080, "tags": {"rotate": "270"}}, 256, 456),
    ],
)
def test_the_analysis_height_keeps_the_shape_ffmpeg_shows(probe, width, want):
    import json
    import types

    def run(cmd, **kw):
        return types.SimpleNamespace(stdout=json.dumps({"streams": [probe]}), stderr="")

    assert S.analysis_height("v.mp4", width, run=run) == want


def test_a_file_ffprobe_cannot_read_is_the_jobs_fault():
    import types

    with pytest.raises(ParamError, match="ffprobe could not read v.mp4: bad"):
        S.analysis_height("v.mp4", 256, run=lambda cmd, **kw: types.SimpleNamespace(stdout="", stderr="bad"))


def test_a_probe_that_lists_the_stream_twice_reads_the_first():
    """omarchy, 2026-11-24: ffprobe's csv printed `640,360` twice with a blank line between on the spliced MP4s, and the first version read
    both as one number; the probe is JSON now and takes the first stream."""
    import json
    import types

    out = json.dumps({"streams": [{"width": 640, "height": 360}, {"width": 640, "height": 360}]}) + "\n\n"
    assert S.analysis_height("v.mp4", 256, run=lambda cmd, **kw: types.SimpleNamespace(stdout=out, stderr="")) == 144
