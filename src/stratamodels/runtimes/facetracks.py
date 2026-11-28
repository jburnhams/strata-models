"""Faces in a video, followed from frame to frame: role `face-tracks`. A track is one person's face through consecutive frames of one shot, so
an app names a person once per track instead of once per frame, and thirty near-identical frames do not swamp its groups. Two tracks on
screen together are two people, which clustering can rely on. Asked for 2026-11-26 (proposal: Claude Doc "Face tracks in video"); strata-360
(race footage) and strata-reels (home video) are the first users. Which videos to run, and when, is the apps' business.

How: ffmpeg decodes the video at `fps` frames a second (deinterlacing only the frames flagged interlaced, scaled to at most `max_side`), the
loaded buffalo_l pack (runtimes/faces.py: SCRFD boxes, landmarks, pose, ArcFace embedding) finds the faces of each, and a tracker joins each
face to a running track:
  - `iou`: by where the track's box is heading (constant velocity) and how much the new box overlaps it, the Hungarian assignment on 1 - IoU.
  - `iou-emb` (the default): the same, also requiring the face's embedding to agree with the track's, and allowing a face that agrees
    strongly (`sim_alone`) to rejoin its track without overlapping (a fast pan, a head turn). The embedding is what keeps two people apart
    when they cross.
  - `byte`: `iou-emb` on the faces the detector is sure of, then ByteTrack's second pass: the faces under `high` (found down to
    `det_thresh`) may extend a track by position alone, never start one. A face half turned away keeps its track instead of ending it.
A track ends at a cut (`cuts`, from the shot-boundaries role, seconds in the file) or after `max_gap_s` without its face. Tracks shorter than
`min_track_s` are dropped as flicker.

Every face gets a quality from its detection score, its size (the shorter side over 112 px, ArcFace's input), its sharpness (variance of the
Laplacian of the crop at 112 px, over SHARP_REF) and how frontal it is (yaw and pitch from the pose). A track's embedding is the
quality-weighted mean of its faces' (steadier than any one frame); its `stills` are the best `stills` faces at least `still_gap_s` apart, each
with its embedding and, unless `images` is false, its whole frame saved as a JPEG under `out_dir`.

The tracker, quality and still choice are evals' to settle (`evals/specs/face-tracks-*.toml`): fps, tracker and the quality score.

options (models.toml): the faces pack's (det_size, modules) and any param's default (served: fps 10, tracker byte, eval face-tracks-home
2026-11-27; when not given here: fps (5), tracker (iou-emb)), max_side (1600), stills (3).

params:
  path          the video
  clip          [start, end] seconds to look at (times in the answer are still the file's)
  cuts          shot cuts, seconds in the file: no track runs across one
  fps           frames a second to detect on (0.5 to 60)
  tracker       iou | iou-emb | byte
  det_thresh    the detector's threshold (default 0.5; `byte` looks down to 0.3 and treats faces under `high`, 0.5, as unsure)
  iou_min (0.2), sim_min (0.3), sim_alone (0.55), max_gap_s (1.0), min_track_s (0.4)
  stills (3, at most 5), still_gap_s (1.0), images (true), out_dir (default <data>/generated/<date>/face-tracks-<n>)
  max_side      frames are decoded at most this many pixels across (1600: the face store's medium copy)
Answer: {tracks: [{id, shot, start, end, faces, quality, embedding, boxes: [[time, x0, y0, x1, y1, det_score]], stills: [{time, bbox, kps,
pose, det_score, quality, sharpness, embedding, image?}]}], together: [[track, track]] (seen in the same frame), shots: [{start, end}], fps,
frames (decoded), width, height (the video's pixels: boxes are in them), duration, tracker, seconds, model, out_dir?}."""

import collections
import datetime
import importlib
import itertools
import json
import math
import os
import queue
import re
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import IO, Any

import numpy as np

from stratamodels.runtimes import RunError  # pyright: ignore[reportMissingTypeStubs]

TRACKERS = ("iou", "iou-emb", "byte")
FIELDS = ("bbox", "kps", "det_score", "embedding", "pose")
ARC_SIDE = 112.0  # ArcFace's input: a face this many pixels across is as good as it gets for recognition
SHARP_REF = 150.0  # Laplacian variance of a 112 px crop that counts as sharp (a crisp face is several hundred, a soft VHS one tens)
BYTE_LOW = 0.3  # how low the detector looks for `byte`'s second pass
NO = 1e6  # the cost of a pair that may not match
TIMEOUT_S = 6 * 3600


@dataclass
class Face:
    """One face found in one decoded frame, in the video's pixels."""

    time: float
    bbox: list[float]
    det_score: float
    embedding: np.ndarray | None = None
    kps: list[list[float]] | None = None
    pose: list[float] | None = None
    sharpness: float = 0.0
    quality: float = 0.0
    frame: int = 0  # which decoded frame: stills are cut from it


@dataclass
class Track:
    id: int
    shot: int
    faces: list[Face] = field(default_factory=list[Face])
    emb_sum: np.ndarray | None = None  # quality-weighted sum of its faces' embeddings
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # of each box edge, pixels a second

    @property
    def last(self) -> Face:
        return self.faces[-1]

    def embedding(self) -> np.ndarray | None:
        if self.emb_sum is None:
            return None
        n = math.sqrt(float((self.emb_sum * self.emb_sum).sum()))
        return self.emb_sum / n if n > 0 else None

    def predict(self, t: float) -> list[float]:
        """Where its box should be at time t, the last box moved on at its velocity (for at most half a second: a guess, not a flight)."""
        dt = min(t - self.last.time, 0.5)
        return [b + v * dt for b, v in zip(self.last.bbox, self.velocity, strict=True)]

    def add(self, f: Face) -> None:
        if self.faces:
            dt = f.time - self.last.time
            if dt > 0:
                new = [(a - b) / dt for a, b in zip(f.bbox, self.last.bbox, strict=True)]
                # smoothed: one jittery box does not fling the next prediction across the frame
                self.velocity = (
                    0.5 * self.velocity[0] + 0.5 * new[0],
                    0.5 * self.velocity[1] + 0.5 * new[1],
                    0.5 * self.velocity[2] + 0.5 * new[2],
                    0.5 * self.velocity[3] + 0.5 * new[3],
                )
        self.faces.append(f)
        if f.embedding is not None:
            w = max(f.quality, 1e-3)
            self.emb_sum = f.embedding * w if self.emb_sum is None else self.emb_sum + f.embedding * w


@dataclass
class Settings:
    tracker: str = "iou-emb"
    iou_min: float = 0.2
    sim_min: float = 0.3
    sim_alone: float = 0.55
    max_gap_s: float = 1.0
    min_track_s: float = 0.4
    high: float = 0.5  # `byte`: a face under this is unsure


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two [x0, y0, x1, y1] boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def frontal(pose: Sequence[float] | None) -> float:
    """1 for a face looking at the camera, falling to 0 at 70 degrees of yaw or 50 of pitch. insightface's pose is [pitch, yaw, roll]."""
    if not pose:
        return 1.0
    pitch, yaw = abs(float(pose[0])), abs(float(pose[1]))
    return max(0.0, 1 - (yaw / 70) ** 2) * max(0.0, 1 - (pitch / 50) ** 2)


def quality(f: Face) -> float:
    """How good a still of this face is for recognising its person: detection score x size x sharpness x frontal, each 0 to 1."""
    w, h = f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]
    size = min(1.0, max(0.0, min(w, h)) / ARC_SIDE)
    sharp = min(1.0, f.sharpness / SHARP_REF)
    return round(float(f.det_score) * size * sharp * frontal(f.pose), 6)


def sharpness(img: np.ndarray, bbox: Sequence[float]) -> float:
    """Variance of the Laplacian of the face's crop, scaled to 112 px across: the usual blur measure, comparable between big and small faces."""
    import cv2

    H, W = img.shape[:2]
    x0, y0 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x1, y1 = min(W, int(math.ceil(bbox[2]))), min(H, int(math.ceil(bbox[3])))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    crop = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (int(ARC_SIDE), int(ARC_SIDE)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


def _cost(tracks: list[Track], faces: list[Face], t: float, s: Settings, use_emb: bool) -> np.ndarray:
    """Cost of joining each face to each track; NO where they may not join."""
    c = np.full((len(tracks), len(faces)), NO)
    for i, tr in enumerate(tracks):
        box = tr.predict(t)
        emb = tr.embedding() if use_emb else None
        for j, f in enumerate(faces):
            ov = iou(box, f.bbox)
            if emb is None or f.embedding is None:
                if ov >= s.iou_min:
                    c[i, j] = 1 - ov
                continue
            sim = float((emb * f.embedding).sum())
            if (ov >= s.iou_min and sim >= s.sim_min) or sim >= s.sim_alone:
                c[i, j] = 1 - (0.5 * ov + 0.5 * max(sim, 0.0))
    return c


def _assign(cost: np.ndarray) -> list[tuple[int, int]]:
    """The cheapest one-to-one pairs of rows and columns, leaving out those that may not pair."""
    if not cost.size:
        return []
    solve: Any = importlib.import_module("scipy.optimize").linear_sum_assignment  # scipy has no type stubs
    rows, cols = solve(cost)
    pairs = [(int(r), int(c)) for r, c in zip(list(rows), list(cols), strict=True)]
    return [(r, c) for r, c in pairs if cost[r, c] < NO]


class Tracker:
    """Joins the faces of each decoded frame, in time order, into tracks."""

    def __init__(
        self,
        s: Settings,
        cuts: Sequence[float] = (),
        frame_s: float = 0.2,
        on_end: Callable[[Track, bool], None] | None = None,
    ):
        """frame_s: the time one decoded frame stands for (a track's length is its first to last face plus one). on_end(track, kept) is
        called as each track ends, kept being whether it is long enough to answer."""
        if s.tracker not in TRACKERS:
            raise RunError(f"tracker must be one of {', '.join(TRACKERS)}, not {s.tracker!r}")
        self.s = s
        self.cuts = sorted(float(c) for c in cuts)
        self.frame_s = frame_s
        self.on_end = on_end
        self.shot = 0
        self.active: list[Track] = []
        self.done: list[Track] = []
        self.together: set[tuple[int, int]] = set()
        self._ids = itertools.count(1)

    def long_enough(self, tr: Track) -> bool:
        return tr.last.time - tr.faces[0].time + self.frame_s >= self.s.min_track_s - 1e-6

    def _end(self, tracks: list[Track]) -> None:
        for tr in tracks:
            keep = self.long_enough(tr)
            if self.on_end:
                self.on_end(tr, keep)
            if keep:
                self.done.append(tr)
        self.active = [t for t in self.active if t not in tracks]

    def step(self, t: float, faces: list[Face]) -> None:
        s = self.s
        while self.shot < len(self.cuts) and t >= self.cuts[self.shot] - 1e-6:
            self.shot += 1
            self._end(list(self.active))
        self._end([tr for tr in self.active if t - tr.last.time > s.max_gap_s + 1e-6])
        use_emb = s.tracker != "iou"
        sure = [f for f in faces if s.tracker != "byte" or f.det_score >= s.high]
        unsure = [f for f in faces if s.tracker == "byte" and f.det_score < s.high]
        free = list(self.active)
        joined: list[Track] = []
        for tr_i, f_j in _assign(_cost(free, sure, t, s, use_emb)):
            free[tr_i].add(sure[f_j])
            joined.append(free[tr_i])
        new = [f for f in sure if not any(f is tr.last for tr in joined)]
        free = [tr for tr in free if tr not in joined]
        if unsure and free:  # ByteTrack's second pass: an unsure face carries a track on by position, never starts one
            for tr_i, f_j in _assign(_cost(free, unsure, t, s, False)):
                free[tr_i].add(unsure[f_j])
                joined.append(free[tr_i])
        for f in new:
            tr = Track(id=next(self._ids), shot=self.shot)
            tr.add(f)
            self.active.append(tr)
            joined.append(tr)
        ids = sorted(tr.id for tr in joined)
        self.together.update(itertools.combinations(ids, 2))

    def finish(self) -> list[Track]:
        """Ends the running tracks; every track long enough to keep, in the order they started."""
        self._end(list(self.active))
        return sorted(self.done, key=lambda tr: (tr.faces[0].time, tr.id))


def pick_stills(faces: Sequence[Face], n: int, gap_s: float) -> list[Face]:
    """The n best faces by quality, no two closer than gap_s seconds (a better one nearby wins), in time order."""
    kept: list[Face] = []
    for f in sorted(faces, key=lambda f: (-f.quality, f.time)):
        if len(kept) >= n:
            break
        if all(abs(f.time - k.time) >= gap_s - 1e-9 for k in kept):
            kept.append(f)
    return sorted(kept, key=lambda f: f.time)


class StillPool:
    """Keeps, for each running track, the frames its best faces were found in (encoded), so stills can be cut without holding the video. The
    track's time is cut into spans of `gap_s` and only the best face of each span is a candidate, so the candidates are spread through the
    track (its best faces are often a second of the same look); of those, the `size` best are kept. When the track ends, `settle` picks its
    stills, writes their frames out and lets the rest go, so memory holds a few frames per running track, not per track ever seen."""

    def __init__(self, size: int, n: int, gap_s: float, write: Callable[[Track, Face, bytes], str] | None):
        self.size, self.n, self.gap_s, self.write = size, n, gap_s, write
        self.best: dict[int, dict[int, Face]] = {}  # track id -> span -> its best face so far
        self.frames: dict[int, bytes] = {}
        self.wanted: dict[int, int] = {}  # frame -> how many kept faces need it
        self.stills: dict[
            int, list[tuple[Face, str | None]]
        ] = {}  # track id -> its stills and where their frames were written

    def offer(self, track_id: int, f: Face, encode: Callable[[], bytes]) -> None:
        spans = self.best.setdefault(track_id, {})
        span = int(f.time // self.gap_s) if self.gap_s > 0 else f.frame
        held = spans.get(span)
        if held is not None and f.quality <= held.quality:
            return
        if held is None and len(spans) >= self.size:
            worst = min(spans, key=lambda k: (spans[k].quality, -k))
            if f.quality <= spans[worst].quality:
                return
            self._release(spans.pop(worst).frame)
        if f.frame not in self.frames:
            self.frames[f.frame] = encode()
        self.wanted[f.frame] = self.wanted.get(f.frame, 0) + 1
        if held is not None:
            self._release(held.frame)
        spans[span] = f

    def _release(self, frame: int) -> None:
        self.wanted[frame] -= 1
        if not self.wanted[frame]:
            del self.wanted[frame], self.frames[frame]

    def candidates(self, track_id: int) -> list[Face]:
        return list(self.best.get(track_id, {}).values())

    def settle(self, track: Track, keep: bool) -> None:
        """The track has ended: pick its stills (when it is kept), write their frames, release the rest."""
        spans = self.best.pop(track.id, {})
        if keep:
            chosen = pick_stills(list(spans.values()) or track.faces, self.n, self.gap_s)
            self.stills[track.id] = [
                (f, self.write(track, f, self.frames[f.frame]) if self.write and f.frame in self.frames else None)
                for f in chosen
            ]
        for f in spans.values():
            self._release(f.frame)


def _number(params: dict[str, Any], key: str, default: float, low: float, high: float) -> float:
    value = params.get(key, default)
    try:
        n = float(value)
    except (TypeError, ValueError) as e:
        raise RunError(f"{key} must be a number: {value!r}") from e
    if not low <= n <= high:
        raise RunError(f"{key} must be from {low} to {high}: {n}")
    return n


def probe(exe: str, path: str) -> dict[str, Any]:
    """The video's width, height and duration, from ffprobe beside the ffmpeg in use."""
    folder, name = os.path.split(exe)
    probe_exe = os.path.join(folder, name.replace("ffmpeg", "ffprobe")) if folder else name.replace("ffmpeg", "ffprobe")
    cmd = [
        probe_exe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate:format=duration",
        "-of",
        "json",
        path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except OSError as e:
        raise RunError(f"cannot run {probe_exe}: {e}") from e
    try:
        info = json.loads(out.stdout or "{}")
        st = info["streams"][0]
        num, _, den = str(st.get("avg_frame_rate") or "0/1").partition("/")
        rate = float(num) / float(den or 1) if float(den or 1) else 0.0
        return dict(
            width=int(st["width"]),
            height=int(st["height"]),
            duration=float(info.get("format", {}).get("duration") or 0),
            rate=rate,
        )
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise RunError(f"{path}: no video stream ({(out.stderr or '').strip()[-300:]})") from e


SHOWINFO = re.compile(r"Parsed_showinfo.*?\sn:\s*\d+\s.*?pts_time:\s*(-?[\d.]+)")


def read_times(stream: IO[bytes], times: "queue.Queue[float | None]", rest: "collections.deque[str]") -> None:
    """ffmpeg's stderr: each picked frame's time (showinfo's pts_time) onto `times`, in order, then None at the end; the other lines into
    `rest` (the last few, for an error message)."""
    for raw in stream:
        line = raw.decode("utf8", "replace").rstrip()
        m = SHOWINFO.search(line)
        if m:
            times.put(float(m[1]))
        elif line and "Parsed_showinfo" not in line:
            rest.append(line)
    times.put(None)


def decode(
    exe: str, path: str, fps: float, size: tuple[int, int], clip: tuple[float, float] | None
) -> Iterator[tuple[float, np.ndarray]]:
    """(time in the file, BGR frame of `size`) at `fps`, from one ffmpeg process that is stopped when the caller stops reading."""
    w, h = size
    cmd = [
        exe,
        "-nostdin",
        "-hide_banner",
        "-nostats",
        "-v",
        "info",
    ]  # info: showinfo's lines, which carry each frame's own time
    offset = 0.0
    if clip is not None:
        offset = clip[0]
        cmd += ["-ss", f"{clip[0]:.6f}", "-t", f"{clip[1] - clip[0]:.6f}"]
    # yadif only on frames flagged interlaced (a tape's), so progressive video is left sharp. Then the first frame at or after each tick k/fps
    # (ffmpeg's fps filter keeps the last frame before the next half tick instead: two frames late at 5 fps on 25 fps video). Each frame's time
    # is its own, from showinfo: a variable-rate video or a gap in the recording skips ticks, and counting frames would shift every box after.
    pick = f"select='isnan(prev_selected_t)+gte(floor(t*{fps:g})-floor(prev_selected_t*{fps:g})\\,1)'"
    chain = f"yadif=deint=interlaced,{pick},showinfo,scale={w}:{h}:flags=area"
    cmd += [
        "-i",
        os.path.abspath(path),
        "-an",
        "-sn",
        "-dn",
        "-vf",
        chain,
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "-",
    ]
    size_b = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    times: queue.Queue[float | None] = queue.Queue()
    rest: collections.deque[str] = collections.deque(maxlen=20)
    reader = threading.Thread(target=read_times, args=(proc.stderr, times, rest), daemon=True)
    reader.start()
    try:
        if proc.stdout is None:  # pragma: no cover  (Popen with stdout=PIPE always has one)
            raise RunError("ffmpeg gave no output")
        while True:
            buf = proc.stdout.read(size_b)
            if len(buf) < size_b:
                break
            t = times.get(timeout=600)  # showinfo logs a frame before it is written out
            if t is None:  # pragma: no cover  (a frame with no showinfo line)
                raise RunError(f"ffmpeg gave a frame without its time on {path}")
            yield round(offset + t, 3), np.frombuffer(buf, np.uint8).reshape(h, w, 3)  # pyright: ignore[reportUnknownMemberType]
        proc.wait(timeout=TIMEOUT_S)
        reader.join(timeout=30)
        if proc.returncode:
            raise RunError(f"ffmpeg failed on {path}: {chr(10).join(rest)[-500:]}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def _faces() -> Any:
    """runtimes/faces.py, the buffalo_l pack's loader and detector (not typed strictly yet)."""
    return importlib.import_module("stratamodels.runtimes.faces")


def load(model: Any, models_dir: str) -> dict[str, Any]:
    from stratamodels.runtimes import scenes  # pyright: ignore[reportMissingTypeStubs]

    faces = _faces()
    opts: dict[str, Any] = dict(model.options)
    exe = scenes.binary(opts.get("binary"))
    if not exe:
        raise RunError(
            "ffmpeg is not installed (pacman -S ffmpeg; apt install ffmpeg; brew install ffmpeg; choco install ffmpeg)"
        )
    pack = faces.load(model, models_dir)
    # the embeddings are the pack's own (ArcFace in buffalo_l): the face store keys them by the pack's name, as the faces role stores them
    return dict(exe=exe, pack=pack, detect=faces.detect, key=model.key, embedding_model=model.name, options=opts)


def _out_dir(params: dict[str, Any]) -> str:
    if params.get("out_dir"):
        folder = str(params["out_dir"])
    else:
        settings: Any = importlib.import_module("stratamodels.settings")  # not typed strictly yet
        day = os.path.join(str(settings.data_dir()), "generated", datetime.date.today().isoformat())
        os.makedirs(day, exist_ok=True)
        n = 1
        while os.path.exists(os.path.join(day, f"face-tracks-{n}")):
            n += 1
        folder = os.path.join(day, f"face-tracks-{n}")
    os.makedirs(folder, exist_ok=True)
    return folder


def _encoder(img: np.ndarray) -> Callable[[], bytes]:
    """The frame as a JPEG, encoded the first time a track wants it and only once however many do."""
    done: list[bytes] = []

    def encode() -> bytes:
        if not done:
            import cv2

            done.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())
        return done[0]

    return encode


def run(handle: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    path = params.get("path")
    if not isinstance(path, str) or not os.path.isfile(path):
        raise RunError(f"no such video: {path!r}")
    opts: dict[str, Any] = handle.get("options", {})
    p = {**opts, **params}
    fps = _number(p, "fps", 5, 0.5, 60)
    tracker = str(p.get("tracker", "iou-emb"))
    s = Settings(
        tracker=tracker,
        iou_min=_number(p, "iou_min", 0.2, 0.0, 1.0),
        sim_min=_number(p, "sim_min", 0.3, -1.0, 1.0),
        sim_alone=_number(p, "sim_alone", 0.55, -1.0, 1.01),
        max_gap_s=_number(p, "max_gap_s", 1.0, 0.0, 60.0),
        min_track_s=_number(p, "min_track_s", 0.4, 0.0, 60.0),
        high=_number(p, "high", 0.5, 0.0, 1.0),
    )
    det_thresh = (
        _number(p, "det_thresh", 0.5, 0.05, 1.0)
        if p.get("det_thresh") is not None
        else (BYTE_LOW if tracker == "byte" else None)
    )
    n_stills = int(_number(p, "stills", 3, 1, 5))
    gap_s = _number(p, "still_gap_s", 1.0, 0.0, 3600.0)
    images = bool(p.get("images", True))
    max_side = int(_number(p, "max_side", 1600, 160, 7680))
    cuts: Any = p.get("cuts") or []
    if not isinstance(cuts, list) or not all(isinstance(c, (int, float)) for c in cuts):  # pyright: ignore[reportUnknownVariableType]
        raise RunError(f"cuts must be a list of seconds: {cuts!r}")
    clip: tuple[float, float] | None = None
    if p.get("clip") is not None:
        c = p["clip"]
        if not isinstance(c, (list, tuple)) or len(c) != 2:  # pyright: ignore[reportUnknownArgumentType]
            raise RunError(f"clip must be [start, end] in seconds: {c!r}")
        clip = (float(c[0]), float(c[1]))  # pyright: ignore[reportUnknownArgumentType]
        if not 0 <= clip[0] < clip[1]:
            raise RunError(f"clip must be [start, end] with 0 <= start < end: {c!r}")

    info = probe(handle["exe"], path)
    W, H = info["width"], info["height"]
    if info.get("rate"):
        fps = min(
            fps, float(info["rate"])
        )  # no more ticks than the video has frames: each tick must find a frame of its own
    scale = min(1.0, max_side / max(W, H))
    size = (max(2, round(W * scale / 2) * 2), max(2, round(H * scale / 2) * 2))
    sx, sy = W / size[0], H / size[1]  # decoded pixels back to the video's

    t0 = time.monotonic()
    frame_s = 1 / fps
    stem = os.path.splitext(os.path.basename(path))[0][:40]
    folder: list[str] = []  # made when the first still is written

    def write(track: Track, f: Face, jpeg: bytes) -> str:
        if not folder:
            folder.append(_out_dir(params))
        name = os.path.join(folder[0], f"{stem}-t{f.time:09.3f}-{track.id}.jpg")
        with open(name, "wb") as fh:
            fh.write(jpeg)
        return name

    pool = StillPool(4 * n_stills, n_stills, gap_s, write if images else None)
    tr = Tracker(s, [float(x) for x in cuts], frame_s, pool.settle)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
    frames = 0
    start_t: float | None = None
    last_t = 0.0
    for k, (t, img) in enumerate(decode(handle["exe"], path, fps, size, clip)):
        frames += 1
        start_t = t if start_t is None else start_t
        last_t = t
        found = handle["detect"](handle["pack"], img, None, det_thresh, FIELDS)["faces"]
        here: list[Face] = []
        for g in found:
            b = g["bbox"]
            f = Face(
                time=round(t, 3),
                bbox=[round(b[0] * sx, 2), round(b[1] * sy, 2), round(b[2] * sx, 2), round(b[3] * sy, 2)],
                det_score=float(g.get("det_score", 0.0)),
                embedding=np.asarray(g["embedding"], np.float32) if g.get("embedding") is not None else None,
                kps=[[round(x * sx, 2), round(y * sy, 2)] for x, y in g["kps"]] if g.get("kps") else None,
                pose=g.get("pose"),
                sharpness=round(sharpness(img, b), 2),
                frame=k,
            )
            f.quality = quality(f)
            here.append(f)
        tr.step(t, here)
        encode = _encoder(img)
        for track in tr.active:
            if track.last.frame == k:
                pool.offer(track.id, track.last, encode if images else lambda: b"")
    if not frames:
        raise RunError(f"{path}: no video frames (the clip is past its end?)")
    kept = tr.finish()
    renumber = {track.id: n for n, track in enumerate(kept, 1)}
    out_tracks: list[dict[str, Any]] = []
    for track in kept:
        emb = track.embedding()
        out_stills: list[dict[str, Any]] = []
        for f, image in pool.stills.get(track.id, []):
            d: dict[str, Any] = dict(
                time=f.time,
                bbox=f.bbox,
                kps=f.kps,
                pose=f.pose,
                det_score=round(f.det_score, 4),
                quality=f.quality,
                sharpness=f.sharpness,
            )
            d["embedding"] = [round(float(v), 6) for v in f.embedding] if f.embedding is not None else None
            if image:
                d["image"] = image
            out_stills.append(d)
        out_tracks.append(
            dict(
                id=renumber[track.id],
                shot=track.shot,
                start=track.faces[0].time,
                end=round(
                    min([track.last.time + frame_s, *[c for c in tr.cuts if c > track.faces[0].time]]), 3
                ),  # never past its shot's cut
                faces=len(track.faces),
                quality=max(f.quality for f in track.faces),
                embedding=[round(float(v), 6) for v in emb] if emb is not None else None,
                boxes=[[f.time, *f.bbox, round(f.det_score, 4)] for f in track.faces],
                stills=out_stills,
            )
        )
    together = sorted([renumber[a], renumber[b]] for a, b in tr.together if a in renumber and b in renumber)
    begin = start_t if start_t is not None else 0.0
    end = round(last_t + frame_s, 3)
    edges = [begin] + [c for c in tr.cuts if begin < c < end] + [end]
    result: dict[str, Any] = dict(
        tracks=out_tracks,
        together=together,
        shots=[{"start": round(a, 3), "end": round(b, 3)} for a, b in itertools.pairwise(edges)],
        fps=fps,
        frames=frames,
        width=W,
        height=H,
        duration=round(end - begin, 3),
        tracker=tracker,
        seconds=round(time.monotonic() - t0, 3),
        model=handle["key"],
        # the face store's model for these embeddings: searching or indexing a still under `model` would find nothing else
        embedding_model=handle.get("embedding_model") or handle["key"],
    )
    if folder:
        result["out_dir"] = folder[0]
    return result


def unload(handle: dict[str, Any]) -> None:
    if handle.get("pack"):
        _faces().unload(handle["pack"])
    handle.clear()
