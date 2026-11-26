"""The face-tracks runtime decoding real video with ffmpeg (installed in CI for the shot-boundaries tests), with a fake detector: the frames
arrive at the asked rate, in the file's time, at the size asked, and ffprobe reads the video's own size."""

import subprocess

import numpy as np
import pytest

from stratamodels.runtimes import RunError, facetracks, scenes


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("facetracks") / "moving-box.mp4"
    # a white square crossing a black frame, 4 s at 25 fps: the fake detector reports it as a face
    src = "color=black:s=640x360:r=25:d=4[b];color=white:s=120x120:r=25[w];[b][w]overlay=x='20+t*100':y=100:shortest=1"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            src,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return str(path)


def exe():
    found = scenes.binary()
    if not found:
        pytest.skip("ffmpeg is not installed")
    return found


def white_box(pack, img, max_side, det_thresh, fields):
    ys, xs = np.nonzero(img[..., 0] > 128)
    if not len(xs):
        return dict(faces=[])
    return dict(
        faces=[dict(bbox=[float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1], det_score=0.9)]
    )


def test_probe_reads_the_size_and_length(video):
    info = facetracks.probe(exe(), video)
    assert (info["width"], info["height"]) == (640, 360) and info["duration"] == pytest.approx(4, abs=0.1)


def test_probe_refuses_a_file_that_is_not_video(tmp_path):
    p = tmp_path / "x.mp4"
    p.write_text("not a video")
    with pytest.raises(RunError, match="no video stream"):
        facetracks.probe(exe(), str(p))


def test_decode_gives_frames_at_the_rate_and_size_asked(video):
    got = list(facetracks.decode(exe(), video, 5, (320, 180), None))
    assert len(got) == 20 and got[0][1].shape == (180, 320, 3)
    assert [t for t, _ in got[:3]] == pytest.approx([0, 0.2, 0.4])
    clipped = list(facetracks.decode(exe(), video, 2, (320, 180), (1.0, 3.0)))
    assert len(clipped) == 4 and clipped[0][0] == pytest.approx(1.0)


def test_one_moving_face_is_one_track_in_the_videos_pixels(video, tmp_path):
    handle = dict(exe=exe(), pack={}, detect=white_box, key="buffalo_l-tracks", options={})
    out = facetracks.run(handle, dict(path=video, fps=5, max_side=320, out_dir=str(tmp_path), still_gap_s=1.0))
    assert out["frames"] == 20 and (out["width"], out["height"]) == (640, 360)
    assert len(out["tracks"]) == 1
    tr = out["tracks"][0]
    assert tr["faces"] == 20 and tr["start"] == 0 and tr["end"] == 4.0
    t, x0, y0, x1, _, _ = tr["boxes"][5]  # at 1 s the box is at x 120 in the video's own pixels
    assert (
        t == 1.0
        and x0 == pytest.approx(120, abs=6)
        and y0 == pytest.approx(100, abs=4)
        and x1 - x0 == pytest.approx(120, abs=6)
    )
    assert len(tr["stills"]) == 3 and all(s["image"].endswith(".jpg") for s in tr["stills"])


def test_frames_keep_their_own_times_across_a_gap(tmp_path):
    """A recording that stops for 3 s (concat puts the second part at 1 + 3 s) (a variable-rate file, as phones write): the frames after the gap keep their real times."""
    path = tmp_path / "gap.mkv"
    graph = "color=black:s=320x180:r=25:d=1[a];color=gray:s=320x180:r=25:d=1,setpts=PTS+3/TB[b];[a][b]concat=n=2:v=1[v]"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "nullsrc",
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            str(path),
        ],
        check=True,
    )
    got = [t for t, _ in facetracks.decode(exe(), str(path), 5, (160, 90), None)]
    assert (
        got[:5] == pytest.approx([0, 0.2, 0.4, 0.6, 0.8])
        and got[5] == pytest.approx(4.0, abs=0.05)
        and got[-1] == pytest.approx(4.8, abs=0.05)
    )
