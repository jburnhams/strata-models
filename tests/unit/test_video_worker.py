"""The video-enhance venv-worker (stratamodels/workers/video.py) without ffmpeg, a model or a GPU: fake ffprobe/ffmpeg runs, a fake Popen for the
engines' own CLIs, a fake /proc. The real thing (spandrel on the card, the diffusion engines' CLIs): tests/gpu."""

import json
import os
import subprocess
import sys
import types

import pytest

WORKERS = os.path.join(os.path.dirname(__file__), "..", "..", "src", "stratamodels", "workers")
sys.path.insert(0, os.path.abspath(WORKERS))
import protocol  # noqa: E402
import video as V  # noqa: E402

SRC = dict(width=720, height=576, fps=25.0, frames=200, seconds=8.0, audio=True)


def P(**kw):
    return {**V.DEFAULTS, **kw}


class FakeRun:
    """ffprobe answers a 720 x 576 25 fps clip with audio (or the result's 1440 x 1080 50 fps); ffmpeg writes its last argument (the output)."""

    def __init__(self, fail=None, seconds=8.0):
        self.calls, self.fail, self.seconds = [], fail, seconds

    def __call__(self, cmd, capture_output=True, text=False, **kw):
        self.calls.append(cmd)
        if cmd[0] == "ffprobe":
            path = cmd[-1]
            if "missing" in path:
                return subprocess.CompletedProcess(cmd, 1, "", "No such file")
            out = path.endswith(".mp4") and "clip" not in os.path.basename(path)
            streams = [
                dict(
                    codec_type="video",
                    width=1440 if out else 720,
                    height=1080 if out else 576,
                    r_frame_rate="50/1" if out else "25/1",
                    nb_frames="400" if out else "N/A",
                )
            ]
            if "silent" not in path:
                streams.append(dict(codec_type="audio"))
            if "noimage" in path:
                streams = [dict(codec_type="audio")]
            return subprocess.CompletedProcess(
                cmd, 0, json.dumps(dict(streams=streams, format=dict(duration=str(self.seconds)))), ""
            )
        if self.fail and any(self.fail in str(a) for a in cmd):
            return subprocess.CompletedProcess(cmd, 1, b"", b"Error while filtering")
        with open(cmd[-1], "wb") as f:
            f.write(b"x")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "clip.mkv"
    p.write_bytes(b"x")
    return str(p)


def test_probe_reads_size_rate_frames_and_audio(clip):
    r = V.probe(clip, run=FakeRun())
    assert r == dict(
        width=720, height=576, fps=25.0, frames=200, seconds=8.0, audio=True
    )  # no nb_frames: seconds x fps


def test_probe_says_what_is_wrong(tmp_path):
    with pytest.raises(protocol.ParamError, match="cannot read"):
        V.probe(str(tmp_path / "missing.mkv"), run=FakeRun())
    with pytest.raises(protocol.ParamError, match="no video stream"):
        V.probe(str(tmp_path / "noimage.mkv"), run=FakeRun())


def test_prep_filter_deinterlaces_with_the_field_order_forced_crops_and_makes_square_pixels():
    f = V.prep_filter(P(crop=[14, 0, 692, 568]), SRC)
    assert f == "bwdif=mode=send_field:parity=tff:deint=all,crop=692:568:14:0,scale=768:576:flags=lanczos,setsar=1"
    assert V.prep_filter(P(deinterlace="yadif", field_order="bff"), SRC).startswith("yadif=mode=send_field:parity=bff")
    assert (
        V.prep_filter(P(deinterlace="none", crop=[0, 0, 720, 576]), SRC) == "scale=768:576:flags=lanczos,setsar=1"
    )  # the whole frame: no crop


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        (dict(field_order="auto"), "field_order"),
        (dict(deinterlace="qtgmc"), "deinterlace"),
        (dict(crop=[0, 0, 800, 576]), "not inside"),
        (dict(crop=[0, 0, 720]), r"\[x, y, w, h\]"),
        (dict(crop=[0, 0, 0, 576]), "not inside"),
        (dict(prep_size=[768]), r"\[width, height\]"),
    ],
)
def test_prep_filter_refuses_bad_params(kw, match):
    with pytest.raises(protocol.ParamError, match=match):
        V.prep_filter(P(**kw), SRC)


def test_region_box_maps_source_pixels_to_the_result_and_clips_to_the_crop():
    crop, size = (14, 0, 692, 568), (1440, 1080)
    b = V.region_box([200, 90, 260, 195], crop, size)
    assert b[0] % 2 == b[1] % 2 == b[2] % 2 == b[3] % 2 == 0
    assert b == [386, 170, 540, 370]
    assert V.region_box([0, 500, 100, 200], crop, size) == [
        0,
        950,
        178,
        128,
    ]  # clipped to the crop, kept inside the frame
    assert V.region_box([710, 0, 20, 20], crop, size) is None  # all outside the crop
    with pytest.raises(protocol.ParamError, match="box"):
        V.region_box([1, 2, 3], crop, size)


def test_finish_args_resize_encode_and_take_the_sources_audio():
    a = V.finish_args(["-i", "up.mp4"], P(), "out.mp4", 50.0, "clip.mkv")
    assert a[: a.index("-map")] == ["-i", "up.mp4", "-i", "clip.mkv"]
    assert a[a.index("-map", a.index("-map") + 1) + 1] == "1:a:0?"
    assert a[a.index("-vf") + 1] == "scale=1440:1080:flags=lanczos,setsar=1,format=yuv420p"
    assert a[a.index("-crf") + 1] == "14" and a[a.index("-r") + 1] == "50" and a[-1] == "out.mp4"
    own = V.finish_args(["-i", "clip.mkv"], P(size=[960, 720]), "o.mp4", 50.0, "clip.mkv", prefilter="hqdn3d")
    assert own.count("-i") == 1 and "0:a:0?" in own and own[own.index("-vf") + 1].startswith("hqdn3d,scale=960:720")
    silent = V.finish_args(["-i", "up.mp4"], P(), "o.mp4", 25.0)
    assert "-c:a" not in silent and silent.count("-map") == 1


def test_extras_crop_each_region_nearest_neighbour_and_keep_stills_inside_the_clip(tmp_path):
    run = FakeRun(seconds=6.0)
    res = str(tmp_path / "video-1.mp4")
    open(res, "wb").close()
    p = P(
        regions=[
            dict(id="face", label="Face", box=[200, 90, 260, 195]),
            dict(box=[710, 0, 10, 10]),
            dict(box=[0, 0, 100, 75]),
        ],
        crop_width=961,
    )
    info, crops, stills = V.extras(p, res, (14, 0, 692, 568), str(tmp_path / "video-1"), run=run)
    assert info["width"] == 1440 and [c["id"] for c in crops] == [
        "face",
        "region3",
    ]  # the second is outside the crop: no file
    face = run.calls[1]
    assert face[face.index("-vf") + 1] == "crop=540:370:386:170,scale=960:658:flags=neighbor,setsar=1,format=yuv420p"
    assert crops[0] == dict(id="face", label="Face", box=[200, 90, 260, 195], path=str(tmp_path / "video-1.face.mp4"))
    assert [os.path.basename(s) for s in stills] == [
        "video-1.still1s.png",
        "video-1.still3s.png",
        "video-1.still5s.png",
    ]  # 7 s is past the end
    with pytest.raises(protocol.ParamError, match="needs a box"):
        V.extras(P(regions=[dict(id="x")]), res, (0, 0, 720, 576), "s", run=run)


def test_ffmpeg_failure_names_ffmpegs_complaint(clip):
    with pytest.raises(RuntimeError, match="Error while filtering"):
        V.ffmpeg(["-i", clip, "out.mp4"], run=FakeRun(fail="out.mp4"))


def test_cli_flags():
    assert V.cli_flags(dict(blocks_to_swap=16, vae_decode_tiled=True, debug=False, x=None, **{"--seed": 1})) == [
        "--blocks_to_swap",
        "16",
        "--vae_decode_tiled",
        "--seed",
        "1",
    ]
    assert V.cli_flags(None) == []


def test_tiles_cover_the_frame_with_overlap():
    assert V._tiles(576, 0, 16) == [0] and V._tiles(200, 256, 16) == [0]
    t = V._tiles(576, 256, 16)
    assert t == [0, 240, 320] and t[-1] + 256 == 576


def test_descendants_walks_the_process_tree(tmp_path):
    for pid, ppid in [(10, 1), (11, 10), (12, 11), (13, 1)]:
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "stat").write_text(f"{pid} (py thon) S {ppid} 0 0")
    (tmp_path / "self").mkdir()
    (tmp_path / "14").mkdir()  # no stat: skipped
    assert V.descendants(10, proc=str(tmp_path)) == {10, 11, 12}
    assert V.descendants(5, proc=str(tmp_path / "none")) == {5}


def test_gpu_mb_sums_the_given_processes():
    out = "10, 2048\n11, 1024\n99, 4000\nbad line\n12, n/a\n"
    run = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, out, "")
    assert V.gpu_mb({10, 11, 12}, run=run) == 3072.0

    def boom(cmd, **kw):
        raise OSError("no nvidia-smi")

    assert V.gpu_mb({10}, run=boom) == 0.0


def test_peak_keeps_the_highest_reading():
    seen = iter([100.0, 900.0, 300.0] + [200.0] * 100)
    pk = V.Peak(1, every=0.001, sample=lambda: next(seen))
    assert pk.done() == 900.0


class FakeProc:
    def __init__(self, rc=0, on_wait=None):
        self.pid, self.rc, self.on_wait = 4242, rc, on_wait

    def wait(self, *a):
        if self.on_wait:
            self.on_wait()
        return self.rc


def test_run_cli_reports_peak_and_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(V, "gpu_mb", lambda pids, run=None: 512.0)
    log = open(tmp_path / "log", "w")
    assert V.run_cli(["py", "inference_cli.py"], str(tmp_path), log, popen=lambda *a, **k: FakeProc()) == 512.0
    with pytest.raises(RuntimeError, match="inference_cli.py exited 3"):
        V.run_cli(["py", "inference_cli.py"], str(tmp_path), log, popen=lambda *a, **k: FakeProc(3))
    log.close()


@pytest.fixture
def ffmpeg_on_path(monkeypatch):
    monkeypatch.setattr(V.shutil, "which", lambda name: f"/usr/bin/{name}")


def test_load_checks_ffmpeg_engine_and_files(monkeypatch, tmp_path, ffmpeg_on_path):
    assert V.load(dict(key="video-lanczos", options=dict(engine="ffmpeg")), str(tmp_path)) == dict(engine="ffmpeg")
    with pytest.raises(RuntimeError, match="unknown engine"):
        V.load(dict(options=dict(engine="magic")), str(tmp_path))
    with pytest.raises(RuntimeError, match="setup_worker_venv.sh seedvr2"):
        V.load(dict(options=dict(engine="seedvr2")), str(tmp_path))
    (tmp_path / "src" / "flashvsr-plus").mkdir(parents=True)
    assert V.load(dict(options=dict(engine="flashvsr")), str(tmp_path))["repo"] == str(
        tmp_path / "src" / "flashvsr-plus"
    )
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setitem(sys.modules, "spandrel", types.SimpleNamespace(ModelLoader=object))
    with pytest.raises(RuntimeError, match="missing .*RealESRGAN_x4plus.pth"):
        V.load(dict(name="RealESRGAN_x4plus.pth", options=dict(engine="spandrel")), str(tmp_path))
    monkeypatch.setattr(V.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg and ffprobe"):
        V.load(dict(options=dict(engine="ffmpeg")), str(tmp_path))


def test_load_spandrel_puts_the_model_on_the_device_in_half_when_it_can(monkeypatch, tmp_path, ffmpeg_on_path):
    (tmp_path / "spandrel").mkdir()
    (tmp_path / "spandrel" / "m.pth").write_bytes(b"x")

    class Net:
        def eval(self):
            return self

        def to(self, dev):
            self.dev = dev
            return self

        def half(self):
            self.halved = True
            return self

    desc = types.SimpleNamespace(
        model=Net(), scale=4, supports_half=True, architecture=types.SimpleNamespace(name="ESRGAN")
    )
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    )
    monkeypatch.setitem(
        sys.modules,
        "spandrel",
        types.SimpleNamespace(ModelLoader=lambda: types.SimpleNamespace(load_from_file=lambda p: desc)),
    )
    info = V.load(dict(name="x", options=dict(engine="spandrel", file="m.pth")), str(tmp_path))
    assert info == dict(engine="spandrel", scale=4, architecture="ESRGAN", half=True, device="cuda")
    assert desc.model.dev == "cuda" and desc.model.halved and V.E["scale"] == 4


def test_engine_cmd_builds_each_clis_command(tmp_path):
    V.E.clear()
    V.E.update(engine="seedvr2", repo="/r", models_dir="/m", options=dict(args=dict(blocks_to_swap=16, batch_size=5)))
    cmd, cwd = V.engine_cmd(P(), "prep.mp4", "up.mp4")
    assert (
        cmd[1:5] == [os.path.join("/r", "inference_cli.py"), "prep.mp4", "--output", "up.mp4"] and cwd == "/r"
    )  # the OS's separator
    assert (
        cmd[cmd.index("--resolution") + 1] == "1080"
        and cmd[cmd.index("--model_dir") + 1] == os.path.join("/m", "seedvr2")
        and cmd[cmd.index("--blocks_to_swap") + 1] == "16"
    )
    V.E.update(engine="flashvsr", options=dict(args=dict(mode="base", cf=True)))
    cmd, _ = V.engine_cmd(P(), "prep.mp4", "up.mp4")
    assert (
        cmd[1:3] == ["-m", "inference.run_stream"]
        and cmd[cmd.index("--width") + 1] == "1536"
        and cmd[cmd.index("--height") + 1] == "1152"
    )
    assert cmd[cmd.index("--mode") + 1] == "base" and "--cf" in cmd
    with pytest.raises(protocol.ParamError, match="multiples of 128"):
        V.engine_cmd(P(engine_size=[1440, 1080]), "prep.mp4", "up.mp4")


def test_run_ffmpeg_engine_end_to_end(monkeypatch, tmp_path, clip, ffmpeg_on_path):
    V.load(dict(key="video-lanczos-hqdn3d", options=dict(engine="ffmpeg", filters="hqdn3d=2:1.5:3:3")), str(tmp_path))
    run = FakeRun()
    r = V.run(
        dict(
            path=clip,
            crop=[14, 0, 692, 568],
            regions=[dict(id="face", box=[200, 90, 260, 195])],
            out_dir=str(tmp_path / "out"),
        ),
        run_=run,
    )
    main = run.calls[1]
    assert main[main.index("-vf") + 1].startswith(
        "bwdif=mode=send_field:parity=tff:deint=all,crop=692:568:14:0,scale=768:576:flags=lanczos,setsar=1,hqdn3d=2:1.5:3:3,scale=1440:1080"
    )
    assert (
        r["engine"] == "ffmpeg"
        and r["model"] == "video-lanczos-hqdn3d"
        and r["fps"] == 50.0
        and r["frames"] == 400
        and r["peak_vram_mb"] == 0.0
    )
    assert [c["id"] for c in r["crops"]] == ["face"] and len(r["stills"]) == 4 and r["path"].endswith(".mp4")


def test_run_refuses_a_missing_clip(tmp_path, ffmpeg_on_path):
    V.load(dict(options=dict(engine="ffmpeg")), str(tmp_path))
    with pytest.raises(protocol.ParamError, match="no such clip"):
        V.run(dict(path=str(tmp_path / "nope.mkv")))


def test_run_cli_engine_prepares_runs_the_cli_and_finishes(monkeypatch, tmp_path, clip, ffmpeg_on_path):
    (tmp_path / "src" / "seedvr2").mkdir(parents=True)
    V.load(dict(key="video-seedvr2-3b", options=dict(engine="seedvr2", args=dict(batch_size=5))), str(tmp_path))
    monkeypatch.setattr(V, "gpu_mb", lambda pids, run=None: 9000.0)
    seen = {}

    def popen(cmd, cwd=None, stdout=None, stderr=None):
        seen["cmd"] = cmd
        out = cmd[cmd.index("--output") + 1]
        return FakeProc(on_wait=lambda: open(out, "wb").write(b"up"))

    run = FakeRun()
    r = V.run(dict(path=clip, out_dir=str(tmp_path / "out")), run_=run, popen=popen)
    prep = run.calls[1]
    assert prep[prep.index("-qp") + 1] == "0" and seen["cmd"][2].endswith("prep.mp4")
    finish = run.calls[2]
    assert (
        finish[finish.index("-i") + 1].endswith("up.mp4") and r["peak_vram_mb"] == 9000.0 and r["engine"] == "seedvr2"
    )
    assert not [
        d for d in os.listdir(tmp_path / "out") if d.startswith("video-") and os.path.isdir(tmp_path / "out" / d)
    ]  # the work folder is gone

    def silent(cmd, cwd=None, stdout=None, stderr=None):
        return FakeProc()

    with pytest.raises(RuntimeError, match="wrote no video"):
        V.run(dict(path=clip, out_dir=str(tmp_path / "out")), run_=FakeRun(), popen=silent)


def test_run_spandrel_engine_pipes_frames_through_the_model(monkeypatch, tmp_path, clip):
    import numpy as np

    frame = np.zeros((576, 768, 3), np.uint8)

    class Pipe:
        def __init__(self, data=b""):
            self.data, self.written = data, []

        def read(self, n):
            out, self.data = self.data[:n], self.data[n:]
            return out

        def write(self, b):
            self.written.append(len(b))

        def close(self):
            pass

    procs = []

    def popen(cmd, stdout=None, stdin=None):
        p = types.SimpleNamespace(cmd=cmd, stdout=Pipe(frame.tobytes() * 2 + b"xx"), stdin=Pipe(), wait=lambda: 0)
        procs.append(p)
        return p

    resets = []
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            reset_peak_memory_stats=lambda: resets.append(1), max_memory_reserved=lambda: 3 * 2**30
        )
    )
    V.E.clear()
    V.E.update(
        engine="spandrel", key="video-realesrgan-x2plus", scale=2, device="cuda", torch=torch, options=dict(tile=0)
    )
    monkeypatch.setattr(
        V, "upscale_frame", lambda f, tile=0, mm=None: np.zeros((f.shape[0] * 2, f.shape[1] * 2, 3), np.uint8)
    )
    r = V.run(dict(path=clip, out_dir=str(tmp_path)), run_=FakeRun(), popen=popen)
    _dec, enc = procs
    assert (
        enc.cmd[enc.cmd.index("-s") + 1] == "1536x1152" and enc.stdin.written == [1536 * 1152 * 3] * 2
    )  # the half frame at the end is dropped
    assert r["peak_vram_mb"] == 3072.0 and resets == [1]
    procs.clear()
    monkeypatch.setattr(V, "upscale_frame", lambda f, tile=0, mm=None: f)

    def none(cmd, stdout=None, stdin=None):
        p = types.SimpleNamespace(cmd=cmd, stdout=Pipe(b""), stdin=Pipe(), wait=lambda: 0)
        return p

    with pytest.raises(RuntimeError, match="0 frames"):
        V.run(dict(path=clip, out_dir=str(tmp_path)), run_=FakeRun(), popen=none)


def test_prepare_file_writes_a_lossless_clip_without_audio(tmp_path, clip):
    run = FakeRun()
    path, peak = V.prepare_file(P(path=clip), SRC, str(tmp_path), run=run)
    assert (
        path == str(tmp_path / "prep.mp4")
        and peak == 0.0
        and "-an" in run.calls[0]
        and run.calls[0][run.calls[0].index("-qp") + 1] == "0"
    )


def test_spandrel_engine_stops_ffmpeg_when_the_model_fails(monkeypatch, tmp_path, clip):
    # A decoder blocked on its full pipe never exits on its own: a failed frame (out of VRAM) must kill both ffmpegs before waiting.
    import numpy as np

    frame = np.zeros((576, 768, 3), np.uint8)
    procs = []

    class Proc:
        def __init__(self, cmd):
            self.cmd, self.killed = cmd, False
            self.stdout = types.SimpleNamespace(read=lambda n: frame.tobytes()[:n])
            self.stdin = types.SimpleNamespace(write=lambda b: None, close=lambda: None)

        def kill(self):
            self.killed = True

        def wait(self):
            assert self.killed, "waited on an ffmpeg that was never stopped"
            return -9

    def popen(cmd, stdout=None, stdin=None):
        procs.append(Proc(cmd))
        return procs[-1]

    def oom(f, tile=0, mm=None):
        raise MemoryError("CUDA out of memory")

    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(reset_peak_memory_stats=lambda: None))
    V.E.clear()
    V.E.update(
        engine="spandrel", key="video-realesrgan-x2plus", scale=2, device="cuda", torch=torch, options=dict(tile=0)
    )
    monkeypatch.setattr(V, "upscale_frame", oom)
    with pytest.raises(MemoryError):
        V.run(dict(path=clip, out_dir=str(tmp_path)), run_=FakeRun(), popen=popen)
    assert len(procs) == 2 and all(p.killed for p in procs)


def test_cleanup_filters_run_on_the_tapes_own_pixels_before_the_resize():
    f = V.prep_filter(
        P(crop=[14, 0, 692, 568], cleanup=dict(chroma_shift=[-2, 0, -2, 0], filters="hqdn3d=0:4:0:6")), SRC
    )
    assert f == (
        "bwdif=mode=send_field:parity=tff:deint=all,crop=692:568:14:0,format=yuv444p,"
        "chromashift=cbh=-2:cbv=0:crh=-2:crv=0:edge=smear,hqdn3d=0:4:0:6,scale=768:576:flags=lanczos,setsar=1"
    )
    assert V.cleanup_filters(None) == [] and V.cleanup_filters(dict(model="m.pth")) == []


@pytest.mark.parametrize(
    ("c", "match"),
    [
        (dict(sharpen=1), "cleanup is a table"),
        ("hqdn3d", "cleanup is a table"),
        (dict(chroma_shift=[1, 2]), "chroma_shift"),
        (dict(chroma_shift=[1, 2, 3, 99]), "chroma_shift"),
        (dict(filters=3), "string of ffmpeg filters"),
    ],
)
def test_cleanup_refuses_bad_stages(c, match):
    with pytest.raises(protocol.ParamError, match=match):
        V.prep_filter(P(cleanup=c), SRC)


def fake_model(scale, device="cpu"):
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(reset_peak_memory_stats=lambda: None, max_memory_reserved=lambda: 2**30)
    )
    return dict(model=object(), scale=scale, device=device, half=False, torch=torch, architecture="Compact")


def test_cleanup_model_is_loaded_once_and_must_keep_the_size(monkeypatch, tmp_path):
    loads = []
    monkeypatch.setattr(
        V, "load_spandrel", lambda f, models_dir, half=True: loads.append(f) or fake_model(2 if "x2" in f else 1)
    )
    V.E.clear()
    V.E.update(models_dir=str(tmp_path))
    assert V.cleanup_model(P()) is None
    m = V.cleanup_model(P(cleanup=dict(model="1x_BleedOut.pth")))
    assert V.cleanup_model(P(cleanup=dict(model="1x_BleedOut.pth"))) is m and loads == ["1x_BleedOut.pth"]
    with pytest.raises(protocol.ParamError, match="keeps the size"):
        V.cleanup_model(P(cleanup=dict(model="x2.pth")))


class Pipe:
    def __init__(self, data=b""):
        self.data, self.written = data, []

    def read(self, n):
        out, self.data = self.data[:n], self.data[n:]
        return out

    def write(self, b):
        self.written.append(len(b))

    def close(self):
        pass


def frames_popen(procs, n=1):
    import numpy as np

    frame = np.zeros((576, 768, 3), np.uint8).tobytes()

    def popen(cmd, stdout=None, stdin=None, **kw):
        p = types.SimpleNamespace(cmd=cmd, stdout=Pipe(frame * n), stdin=Pipe(), wait=lambda: 0, pid=1)
        procs.append(p)
        return p

    return popen


def test_ffmpeg_engine_with_a_cleanup_model_pipes_frames_through_it(monkeypatch, tmp_path, clip, ffmpeg_on_path):
    V.load(dict(key="video-lanczos", options=dict(engine="ffmpeg", filters="hqdn3d=1")), str(tmp_path))
    cm = fake_model(1)
    seen = []
    monkeypatch.setattr(V, "cleanup_model", lambda p: cm)
    monkeypatch.setattr(V, "upscale_frame", lambda f, tile=0, mm=None: seen.append((mm is cm, tile)) or f)
    procs = []
    r = V.run(
        dict(path=clip, out_dir=str(tmp_path), cleanup=dict(model="1x.pth", tile=128)),
        run_=FakeRun(),
        popen=frames_popen(procs, 2),
    )
    _dec, enc = procs
    assert seen == [(True, 128)] * 2 and enc.cmd[enc.cmd.index("-s") + 1] == "768x576"
    assert enc.cmd[enc.cmd.index("-vf") + 1].startswith("hqdn3d=1,scale=1440:1080") and r["peak_vram_mb"] == 0.0


def test_spandrel_engine_chains_the_cleanup_model_before_the_upscaler(monkeypatch, tmp_path, clip):
    import numpy as np

    cm = fake_model(1, "cuda")
    V.E.clear()
    V.E.update(fake_model(2, "cuda"), engine="spandrel", key="x2", options=dict(tile=0), models_dir=str(tmp_path))
    order = []

    def up(f, tile=0, mm=None):
        order.append(mm["scale"])
        return np.zeros((f.shape[0] * mm["scale"], f.shape[1] * mm["scale"], 3), np.uint8)

    monkeypatch.setattr(V, "cleanup_model", lambda p: cm)
    monkeypatch.setattr(V, "upscale_frame", up)
    procs = []
    r = V.run(
        dict(path=clip, out_dir=str(tmp_path), cleanup=dict(model="1x.pth")), run_=FakeRun(), popen=frames_popen(procs)
    )
    assert order == [1, 2] and procs[1].cmd[procs[1].cmd.index("-s") + 1] == "1536x1152" and r["peak_vram_mb"] == 1024.0


def test_cli_engine_cleans_the_prepared_clip_with_the_model(monkeypatch, tmp_path, clip, ffmpeg_on_path):
    (tmp_path / "src" / "flashvsr-plus").mkdir(parents=True)
    V.load(dict(key="video-flashvsr-plus-turbo", options=dict(engine="flashvsr")), str(tmp_path))
    monkeypatch.setattr(V, "cleanup_model", lambda p: fake_model(1, "cuda"))
    monkeypatch.setattr(V, "upscale_frame", lambda f, tile=0, mm=None: f)
    monkeypatch.setattr(V, "gpu_mb", lambda pids, run=None: 500.0)
    procs = []
    base = frames_popen(procs)

    def popen(cmd, cwd=None, stdout=None, stderr=None, stdin=None):
        if cmd[0] == "ffmpeg":
            return base(cmd, stdout=stdout, stdin=stdin)
        out = cmd[cmd.index("--output") + 1]
        return types.SimpleNamespace(pid=2, wait=lambda: open(out, "wb").write(b"up") and 0)

    r = V.run(dict(path=clip, out_dir=str(tmp_path / "out"), cleanup=dict(model="1x.pth")), run_=FakeRun(), popen=popen)
    enc = procs[1].cmd
    assert (
        enc[-1].endswith("prep.mp4") and enc[enc.index("-qp") + 1] == "0" and r["peak_vram_mb"] == 1024.0
    )  # the cleanup's peak beats the CLI's
