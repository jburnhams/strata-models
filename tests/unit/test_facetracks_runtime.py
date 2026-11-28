"""The face-tracks runtime (stratamodels/runtimes/facetracks.py) without a model or ffmpeg: the tracker on synthetic faces, the quality score,
the choice of stills, and run() over fake frames and a fake detector. tests/integration/test_facetracks_runtime.py decodes real video."""

import itertools
import os

import numpy as np
import pytest

from stratamodels.runtimes import RunError, facetracks
from stratamodels.runtimes.facetracks import Face, Settings, Tracker


def unit(seed):
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


ANA, BEN = unit(1), unit(2)


def face(t, x, person=ANA, score=0.9, size=100, q=0.5, frame=0):
    return Face(time=t, bbox=[x, 50, x + size, 50 + size], det_score=score, embedding=person, quality=q, frame=frame)


def run_tracker(frames, tracker="iou-emb", cuts=(), fps=5, **kw):
    tr = Tracker(Settings(tracker=tracker, **kw), cuts, 1 / fps)
    for t, faces in frames:
        tr.step(t, faces)
    return tr, tr.finish()


def ids_of(tracks, person):
    return {tr.id for tr in tracks for f in tr.faces if f.embedding is person}


def test_iou_of_boxes():
    assert facetracks.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert facetracks.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert facetracks.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0
    assert facetracks.iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0


def test_frontal_falls_with_yaw_and_pitch():
    assert facetracks.frontal(None) == 1
    assert facetracks.frontal([0, 0, 0]) == 1
    assert facetracks.frontal([0, 35, 0]) == pytest.approx(0.75)
    assert facetracks.frontal([0, 80, 0]) == 0 and facetracks.frontal([60, 0, 0]) == 0


def test_quality_wants_a_big_sharp_frontal_sure_face():
    big = Face(time=0, bbox=[0, 0, 200, 200], det_score=0.9, sharpness=300, pose=[0, 0, 0])
    assert facetracks.quality(big) == pytest.approx(0.9)
    small = Face(time=0, bbox=[0, 0, 56, 56], det_score=0.9, sharpness=300, pose=[0, 0, 0])
    assert facetracks.quality(small) == pytest.approx(0.45)
    soft = Face(time=0, bbox=[0, 0, 200, 200], det_score=0.9, sharpness=75, pose=[0, 0, 0])
    assert facetracks.quality(soft) == pytest.approx(0.45)
    profile = Face(time=0, bbox=[0, 0, 200, 200], det_score=0.9, sharpness=300, pose=[0, 70, 0])
    assert facetracks.quality(profile) == 0


def test_sharpness_of_a_crop_is_higher_for_detail_than_for_a_flat_patch():
    img = np.zeros((200, 200, 3), np.uint8)
    img[::2, :100] = 255  # stripes on the left, flat on the right
    assert facetracks.sharpness(img, [0, 0, 100, 100]) > 1000
    assert facetracks.sharpness(img, [110, 0, 200, 100]) == 0
    assert facetracks.sharpness(img, [198, 198, 260, 260]) == 0  # too small once clipped to the picture


def test_one_face_moving_is_one_track():
    frames = [(k * 0.2, [face(k * 0.2, 10 + 8 * k)]) for k in range(10)]
    _, tracks = run_tracker(frames)
    assert len(tracks) == 1 and len(tracks[0].faces) == 10 and tracks[0].shot == 0


def test_two_people_side_by_side_are_two_tracks_seen_together():
    frames = [(k * 0.2, [face(k * 0.2, 10, ANA), face(k * 0.2, 400, BEN)]) for k in range(5)]
    tr, tracks = run_tracker(frames)
    assert len(tracks) == 2 and ids_of(tracks, ANA).isdisjoint(ids_of(tracks, BEN))
    assert tr.together == {(1, 2)}


def test_a_new_person_where_another_was_mixes_by_position_but_not_with_embeddings():
    # Ana leaves the frame and Ben steps into the same place: by position alone that is one track
    frames = [(k * 0.2, [face(k * 0.2, 100, ANA)]) for k in range(4)] + [
        (k * 0.2, [face(k * 0.2, 104, BEN)]) for k in range(4, 8)
    ]
    _, by_position = run_tracker(frames, tracker="iou")
    assert len(by_position) == 1
    _, by_both = run_tracker(frames)
    assert len(by_both) == 2


def test_a_cut_ends_every_track():
    frames = [(k * 0.2, [face(k * 0.2, 100)]) for k in range(10)]
    _, tracks = run_tracker(frames, cuts=[1.0])
    assert [(t.shot, len(t.faces)) for t in tracks] == [(0, 5), (1, 5)]


def test_a_face_gone_longer_than_max_gap_starts_a_new_track():
    frames = [(0.0, [face(0.0, 100)]), (0.2, [face(0.2, 100)]), (2.0, [face(2.0, 100)]), (2.2, [face(2.2, 100)])]
    _, tracks = run_tracker(frames, max_gap_s=1.0)
    assert len(tracks) == 2
    _, tracks = run_tracker(frames, max_gap_s=2.0)
    assert len(tracks) == 1


def test_a_strong_embedding_match_rejoins_a_face_that_jumped():
    frames = [(0.0, [face(0.0, 0)]), (0.2, [face(0.2, 0)]), (0.4, [face(0.4, 600)]), (0.6, [face(0.6, 600)])]
    _, emb = run_tracker(frames)
    assert len(emb) == 1
    _, pos = run_tracker(frames, tracker="iou")
    assert len(pos) == 2


def test_short_flicker_is_dropped():
    frames = [(0.0, [face(0.0, 100), face(0.0, 500, BEN)])] + [(k * 0.2, [face(k * 0.2, 100)]) for k in range(1, 5)]
    _, tracks = run_tracker(frames, min_track_s=0.4)
    assert len(tracks) == 1 and ids_of(tracks, BEN) == set()


def test_byte_lets_an_unsure_face_carry_a_track_but_never_start_one():
    frames = [
        (0.0, [face(0.0, 100)]),
        (
            0.2,
            [face(0.2, 102, score=0.35), face(0.2, 500, BEN, score=0.35)],
        ),  # unsure: Ana turning away, and a false face
        (0.4, [face(0.4, 104, score=0.35), face(0.4, 500, BEN, score=0.35)]),
        (0.6, [face(0.6, 106)]),
    ]
    _, byte = run_tracker(frames, tracker="byte")
    assert len(byte) == 1 and len(byte[0].faces) == 4
    _, plain = run_tracker(frames, tracker="iou-emb")  # without the second pass, unsure faces are ordinary ones
    assert len(plain) == 2


def test_an_unknown_tracker_is_refused():
    with pytest.raises(RunError, match="tracker must be"):
        Tracker(Settings(tracker="kalman"))


def test_a_track_without_embeddings_is_joined_by_position():
    frames = [(k * 0.2, [Face(time=k * 0.2, bbox=[100, 50, 200, 150], det_score=0.9)]) for k in range(3)]
    _, tracks = run_tracker(frames)
    assert len(tracks) == 1 and tracks[0].embedding() is None


def test_the_track_embedding_is_the_quality_weighted_mean():
    tr = facetracks.Track(id=1, shot=0)
    tr.add(face(0.0, 0, ANA, q=0.9))
    tr.add(face(0.2, 0, BEN, q=0.1))
    e = tr.embedding()
    assert e is not None and float(np.linalg.norm(e)) == pytest.approx(1)
    assert float(np.dot(e, ANA)) > float(np.dot(e, BEN))


def test_the_prediction_follows_the_motion_for_at_most_half_a_second():
    tr = facetracks.Track(id=1, shot=0)
    tr.add(face(0.0, 0))
    tr.add(face(0.2, 20))  # 100 px a second, smoothed to 50
    assert tr.predict(0.4)[0] == pytest.approx(30)
    assert tr.predict(5.0)[0] == pytest.approx(45)


def test_stills_are_the_best_and_spread_out():
    faces = [face(k * 0.2, 0, q=q) for k, q in enumerate([0.1, 0.9, 0.8, 0.3, 0.2, 0.7, 0.6])]
    got = facetracks.pick_stills(faces, 3, 0.5)
    assert [f.quality for f in got] == [0.9, 0.7]  # 0.8 and 0.3 are too close to 0.9, 0.6 to 0.7, 0.1 to 0.9...
    assert [f.quality for f in facetracks.pick_stills(faces, 2, 0.0)] == [0.9, 0.8]


def test_the_pool_keeps_the_best_face_of_each_span_and_lets_the_rest_go():
    written = []
    pool = facetracks.StillPool(2, 1, 0.5, lambda track, f, jpeg: written.append(jpeg) or f"/x/{f.frame}.jpg")
    tr = facetracks.Track(id=7, shot=0)
    # spans of 0.5 s: frames 0-2 are the first, 3-4 the second, 5 the third
    for k, q in enumerate([0.2, 0.5, 0.1, 0.9, 0.3, 0.05]):
        f = face(k * 0.2, 0, q=q, frame=k)
        tr.add(f)
        pool.offer(7, f, lambda k=k: f"jpeg{k}".encode())
    assert sorted(f.frame for f in pool.candidates(7)) == [1, 3]  # the third span's best is worse than both kept
    assert sorted(pool.frames) == [1, 3]
    pool.settle(tr, keep=True)
    assert written == [b"jpeg3"] and [(f.frame, p) for f, p in pool.stills[7]] == [(3, "/x/3.jpg")]
    assert pool.frames == {} and pool.wanted == {}


def test_a_better_span_pushes_out_the_worst():
    pool = facetracks.StillPool(2, 2, 1.0, None)
    for k, q in enumerate([0.3, 0.2, 0.8]):
        pool.offer(1, face(float(k), 0, q=q, frame=k), lambda: b"j")
    assert sorted(f.frame for f in pool.candidates(1)) == [0, 2] and sorted(pool.frames) == [0, 2]


def test_stills_without_a_gap_are_one_a_frame():
    pool = facetracks.StillPool(3, 2, 0.0, None)
    tr = facetracks.Track(id=1, shot=0)
    for k, q in enumerate([0.3, 0.2, 0.8]):
        f = face(k * 0.04, 0, q=q, frame=k)
        tr.add(f)
        pool.offer(1, f, lambda: b"j")
    pool.settle(tr, keep=True)
    assert [f.frame for f, _ in pool.stills[1]] == [0, 2]


def test_a_dropped_track_keeps_no_stills():
    pool = facetracks.StillPool(2, 1, 0.0, None)
    tr = facetracks.Track(id=1, shot=0)
    f = face(0, 0, frame=0)
    tr.add(f)
    pool.offer(1, f, lambda: b"j")
    pool.settle(tr, keep=False)
    assert pool.stills == {} and pool.frames == {}


# ---- run() over fake frames -------------------------------------------------------------------------------------------------------------
def fake_video(monkeypatch, frames, width=640, height=360):
    """frames: a list of lists of (person, x) faces, one list a decoded frame at 5 fps."""
    seen = {}

    def decode(exe, path, fps, size, clip):
        seen.update(fps=fps, size=size, clip=clip)
        start = clip[0] if clip else 0.0
        for k in range(len(frames)):
            img = np.zeros((size[1], size[0], 3), np.uint8)
            img[::2] = 200  # detail, so every face is sharp
            img[0, 0, 0] = k  # each frame its own picture
            yield start + k / fps, img

    def detect(pack, img, max_side, det_thresh, fields):
        k = int(img[0, 0, 0])
        seen.setdefault("thresh", []).append(det_thresh)
        return dict(
            faces=[
                dict(bbox=[x, 40, x + 120, 160], det_score=0.9, embedding=list(p), kps=[[x + 30, 80]], pose=[0, 0, 0])
                for p, x in frames[k]
            ]
        )

    monkeypatch.setattr(facetracks, "decode", decode)
    monkeypatch.setattr(
        facetracks, "probe", lambda exe, path: dict(width=width, height=height, duration=len(frames) / 5)
    )
    return dict(
        exe="ffmpeg", pack={}, detect=detect, key="buffalo_l-tracks", embedding_model="buffalo_l", options=dict(fps=5)
    ), seen


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "v.mp4"
    p.write_bytes(b"not read")
    return str(p)


def test_run_answers_tracks_with_stills_saved(monkeypatch, video, tmp_path):
    frames = [[(ANA, 10 + 4 * k), (BEN, 400)] for k in range(10)]
    handle, seen = fake_video(monkeypatch, frames)
    out = facetracks.run(handle, dict(path=video, out_dir=str(tmp_path / "out"), cuts=[1.0], still_gap_s=0.4))
    assert seen["size"] == (640, 360) and seen["fps"] == 5
    assert [(t["id"], t["shot"], t["start"], t["end"], t["faces"]) for t in out["tracks"]] == [
        (1, 0, 0.0, 1.0, 5),
        (2, 0, 0.0, 1.0, 5),
        (3, 1, 1.0, 2.0, 5),
        (4, 1, 1.0, 2.0, 5),
    ]
    assert out["together"] == [[1, 2], [3, 4]]
    assert out["shots"] == [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0}]
    assert (
        out["frames"] == 10
        and out["duration"] == 2.0
        and out["tracker"] == "iou-emb"
        and out["model"] == "buffalo_l-tracks"
        and out["embedding_model"] == "buffalo_l"
    )
    first = out["tracks"][0]
    assert len(first["boxes"]) == 5 and first["boxes"][0] == [0.0, 10.0, 40.0, 130.0, 160.0, 0.9]
    assert len(first["embedding"]) == 512 and float(np.dot(first["embedding"], ANA)) == pytest.approx(1, abs=1e-4)
    assert len(first["stills"]) == 3  # 5 faces in 1 s, 0.4 s apart at least: 0.0, 0.4, 0.8
    for s in first["stills"]:
        assert os.path.isfile(s["image"]) and s["image"].startswith(out["out_dir"])
        assert s["quality"] > 0 and len(s["embedding"]) == 512
    times = [s["time"] for s in first["stills"]]
    assert all(b - a >= 0.4 for a, b in itertools.pairwise(times))


def test_run_scales_big_video_and_gives_boxes_in_its_pixels(monkeypatch, video):
    handle, seen = fake_video(monkeypatch, [[(ANA, 100)]] * 3, width=3840, height=2160)
    out = facetracks.run(handle, dict(path=video, images=False, max_side=1920))
    assert seen["size"] == (1920, 1080)
    assert out["tracks"][0]["boxes"][0][1:5] == [200.0, 80.0, 440.0, 320.0]
    assert "image" not in out["tracks"][0]["stills"][0] and "out_dir" not in out


def test_run_with_a_clip_and_byte_looks_lower(monkeypatch, video, tmp_path):
    handle, seen = fake_video(monkeypatch, [[(ANA, 100)]] * 3)
    out = facetracks.run(handle, dict(path=video, clip=[10, 20], tracker="byte", out_dir=str(tmp_path)))
    assert seen["clip"] == (10.0, 20.0) and set(seen["thresh"]) == {facetracks.BYTE_LOW}
    assert out["tracks"][0]["start"] == 10.0 and out["shots"] == [{"start": 10.0, "end": 10.6}]
    facetracks.run(handle, dict(path=video, det_thresh=0.6, out_dir=str(tmp_path)))
    assert seen["thresh"][-1] == 0.6


def test_run_with_no_faces_has_no_tracks_and_writes_nothing(monkeypatch, video, tmp_path):
    handle, _ = fake_video(monkeypatch, [[]] * 3)
    out = facetracks.run(handle, dict(path=video, out_dir=str(tmp_path / "none")))
    assert out["tracks"] == [] and out["together"] == [] and not (tmp_path / "none").exists()


def test_run_default_out_dir_is_under_the_data_folder(monkeypatch, video, tmp_path):
    from stratamodels import settings

    monkeypatch.setattr(settings, "data_dir", lambda: str(tmp_path))
    handle, _ = fake_video(monkeypatch, [[(ANA, 100)]] * 3)
    first = facetracks.run(handle, dict(path=video))["out_dir"]
    second = facetracks.run(handle, dict(path=video))["out_dir"]
    assert (
        first.startswith(str(tmp_path / "generated"))
        and first.endswith("face-tracks-1")
        and second.endswith("face-tracks-2")
    )


def test_run_with_no_frames_fails(monkeypatch, video):
    handle, _ = fake_video(monkeypatch, [])
    with pytest.raises(RunError, match="no video frames"):
        facetracks.run(handle, dict(path=video))


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (dict(path="/no/such.mp4"), "no such video"),
        (dict(fps=0), "fps must be"),
        (dict(fps="x"), "fps must be a number"),
        (dict(tracker="kalman"), "tracker must be"),
        (dict(cuts="1,2"), "cuts must be"),
        (dict(clip=[5]), "clip must be"),
        (dict(clip=[5, 2]), "clip must be"),
        (dict(stills=9), "stills must be"),
    ],
)
def test_bad_params_are_refused(monkeypatch, video, params, message):
    handle, _ = fake_video(monkeypatch, [[(ANA, 100)]])
    with pytest.raises(RunError, match=message):
        facetracks.run(handle, {"path": video, **params})


def test_load_needs_ffmpeg(monkeypatch):
    from stratamodels.registry import Model
    from stratamodels.runtimes import scenes

    monkeypatch.setattr(scenes, "binary", lambda option=None: None)
    m = Model(
        key="buffalo_l-tracks", runtime="face-tracks", name="buffalo_l", roles=("face-tracks",), vram_gb=1, options={}
    )
    with pytest.raises(RunError, match="ffmpeg is not installed"):
        facetracks.load(m, "/models")


def test_load_and_unload_use_the_faces_pack(monkeypatch):
    from stratamodels.registry import Model
    from stratamodels.runtimes import faces, scenes

    monkeypatch.setattr(scenes, "binary", lambda option=None: "/usr/bin/ffmpeg")
    monkeypatch.setattr(faces, "load", lambda model, models_dir: {"fa": "pack"})
    freed = []
    monkeypatch.setattr(faces, "unload", lambda h: freed.append(dict(h)))
    m = Model(
        key="buffalo_l-tracks",
        runtime="face-tracks",
        name="buffalo_l",
        roles=("face-tracks",),
        vram_gb=1,
        options=dict(fps=10),
    )
    h = facetracks.load(m, "/models")
    assert (
        h["exe"] == "/usr/bin/ffmpeg"
        and h["pack"] == {"fa": "pack"}
        and h["options"] == dict(fps=10)
        and h["embedding_model"] == "buffalo_l"
        and h["detect"] is faces.detect
    )
    facetracks.unload(h)
    assert freed == [{"fa": "pack"}] and h == {}


def test_a_track_ends_at_its_cut_even_between_ticks(monkeypatch, video):
    handle, _ = fake_video(monkeypatch, [[(ANA, 100)]] * 10)
    out = facetracks.run(handle, dict(path=video, cuts=[0.95], images=False))
    assert [(t["start"], t["end"]) for t in out["tracks"]] == [(0.0, 0.95), (1.0, 2.0)]


def test_frame_times_are_read_from_showinfo():
    import collections
    import io
    import queue

    err = io.BytesIO(
        b"Input #0, mov,mp4...\n"
        b"[Parsed_showinfo_2 @ 0x1] config in time_base: 1/12800\n"
        b"[Parsed_showinfo_2 @ 0x1] n:   0 pts:      0 pts_time:0       duration: 512\n"
        b"[Parsed_showinfo_2 @ 0x1] n:   1 pts:  12800 pts_time:1.04    duration: 512\n"
        b"[vist#0:0/h264 @ 0x2] Error while decoding\n"
    )
    times, rest = queue.Queue(), collections.deque(maxlen=5)
    facetracks.read_times(err, times, rest)
    assert [times.get_nowait() for _ in range(3)] == [0.0, 1.04, None]
    assert list(rest) == ["Input #0, mov,mp4...", "[vist#0:0/h264 @ 0x2] Error while decoding"]
