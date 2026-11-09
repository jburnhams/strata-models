"""The real runtimes on the real GPU: run on the GPU machine with the GPU requirements installed. Not collected by CI (testpaths is tests/unit).

    STRATAMODELS_TEST_IMAGE=/path/photo-with-faces.jpg STRATAMODELS_TEST_AUDIO=/path/speech.flac pytest tests/gpu -s
Weights come from the real models folder (STRATAMODELS_MODELS or ~/models), so the isolated home of the other suites does not apply here."""

import os
import time

import numpy as np
import pytest

from stratamodels import gpu, runtimes
from stratamodels.registry import Registry

IMAGE, AUDIO = os.environ.get("STRATAMODELS_TEST_IMAGE"), os.environ.get("STRATAMODELS_TEST_AUDIO")
MODELS = os.path.expanduser(os.environ.get("STRATAMODELS_MODELS", "~/models"))


def measure(model, params):
    rt = runtimes.get(model.runtime)
    before = gpu.own_used_mb(gpu.query())
    t0 = time.perf_counter()
    h = rt.load(model, MODELS)
    load_s = time.perf_counter() - t0
    vram = gpu.own_used_mb(gpu.query()) - before
    t0 = time.perf_counter()
    out = rt.run(h, params)
    run_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    out2 = rt.run(h, params)
    warm_s = time.perf_counter() - t0
    rt.unload(h)
    print(f"\n{model.key}: load {load_s:.1f} s, VRAM {vram:.0f} MB, first run {run_s:.2f} s, warm run {warm_s:.2f} s")
    return out, out2


@pytest.mark.skipif(
    not IMAGE or runtimes.available()["insightface"], reason="needs STRATAMODELS_TEST_IMAGE and insightface"
)
def test_faces_on_the_gpu():
    out, _ = measure(
        Registry.load().resolve("faces"),
        dict(path=IMAGE, fields=["bbox", "det_score", "embedding", "landmark_3d_68", "pose"]),
    )
    f = out["faces"][0]
    assert (
        out["faces"]
        and len(f["embedding"]) == 512
        and abs(sum(v * v for v in f["embedding"]) - 1) < 1e-3
        and len(f["landmark_3d_68"]) == 68
    )
    small, _ = measure(Registry.load().resolve("faces"), dict(path=IMAGE, max_side=640))
    assert (
        abs(small["faces"][0]["bbox"][0] - f["bbox"][0]) < 0.05 * out["width"]
    )  # boxes come back in the original's pixels


@pytest.mark.skipif(
    not AUDIO or runtimes.available()["faster-whisper"], reason="needs STRATAMODELS_TEST_AUDIO and faster-whisper"
)
@pytest.mark.parametrize("role", ["transcribe", "transcribe-fast"])
def test_whisper_on_the_gpu(role):
    out, _ = measure(Registry.load().resolve(role), dict(path=AUDIO, language="en"))
    text = " ".join(s["text"] for s in out["segments"]).lower()
    assert "ask not what your country" in text and out["segments"][0]["words"]


@pytest.mark.skipif(
    not IMAGE or runtimes.available()["insightface"], reason="needs STRATAMODELS_TEST_IMAGE and insightface"
)
def test_views_give_boxes_back_in_the_original_picture(tmp_path):
    """A face lying on its side (the test picture turned anticlockwise) is found by the view that turns it back upright, and its box comes back
    in the turned picture's pixels; a crop of 70 % around the upright face gives the same box as the whole picture."""
    import cv2

    m = Registry.load().resolve("faces")
    rt = runtimes.get(m.runtime)
    h = rt.load(m, MODELS)
    try:
        whole = rt.run(h, dict(path=IMAGE, fields=["bbox"]))
        W, H = whole["width"], whole["height"]
        b = whole["faces"][0]["bbox"]
        lying = str(tmp_path / "lying.png")
        cv2.imwrite(lying, cv2.rotate(cv2.imread(IMAGE), cv2.ROTATE_90_COUNTERCLOCKWISE))
        want = [b[1], W - 1 - b[2], b[3], W - 1 - b[0]]  # the box, turned anticlockwise with the picture
        r = rt.run(
            h, dict(path=lying, fields=["bbox", "kps"], views=[dict(rotate=90), dict(crop=[0, 0, H, W], rotate=90)])
        )
        for v in r["views"]:
            got = min(v["faces"], key=lambda f: abs(f["bbox"][0] - want[0]) + abs(f["bbox"][1] - want[1]))["bbox"]
            assert all(abs(a - c) < 0.05 * W for a, c in zip(got, want, strict=False)), (v["view"], got, want)
        c = rt.run(
            h,
            dict(
                path=IMAGE,
                fields=["bbox"],
                views=[dict(crop=[max(b[0] - W * 0.3, 0), max(b[1] - H * 0.3, 0), W * 0.7, H * 0.7])],
            ),
        )
        got = c["views"][0]["faces"][0]["bbox"]
        assert all(abs(a - x) < 0.05 * W for a, x in zip(got, b, strict=False)), (got, b)
        with pytest.raises(runtimes.RunError, match="rotate"):
            rt.run(h, dict(path=IMAGE, views=[dict(rotate=45)]))
        with pytest.raises(runtimes.RunError, match="outside"):
            rt.run(h, dict(path=IMAGE, views=[dict(crop=[W + 5, 0, 10, 10])]))
        with pytest.raises(runtimes.RunError, match="one image"):
            rt.run(h, dict(paths=[IMAGE], views=[{}]))
        with pytest.raises(runtimes.RunError, match="crop must"):
            rt.run(h, dict(path=IMAGE, views=[dict(crop="x")]))
    finally:
        rt.unload(h)


@pytest.mark.skipif(
    not AUDIO or runtimes.available()["faster-whisper"], reason="needs STRATAMODELS_TEST_AUDIO and faster-whisper"
)
def test_whisper_detects_the_language_alone():
    m = Registry.load().resolve("transcribe-fast")
    rt = runtimes.get(m.runtime)
    h = rt.load(m, MODELS)
    try:
        r = rt.run(h, dict(path=AUDIO, detect_language_only=True))
        assert r["language"] == "en" and r["all_language_probs"]["en"] > 0.5 and "segments" not in r
        with pytest.raises(runtimes.RunError, match="unknown params"):
            rt.run(h, dict(path=AUDIO, colour="red"))
    finally:
        rt.unload(h)


@pytest.mark.skipif(
    bool(runtimes.available()["sd-cpp"]),
    reason="needs the sd-cli binary (stable-diffusion.cpp) and huggingface_hub; gated models need HF_TOKEN",
)
@pytest.mark.parametrize("model", ["flux3-klein", "qwen-image-2.1-turbo"])
def test_image_generation_on_the_gpu(model, tmp_path):
    m = Registry.load().resolve("image-generate", model)
    rt = runtimes.get(m.runtime)
    h = rt.load(m, MODELS)
    try:
        r = rt.run(
            h,
            dict(prompt="a red apple on a white table, studio photo", width=512, height=512, seed=1),
            out_dir=str(tmp_path),
        )
        print(f"\n{model}: {r['seconds']} s, peak VRAM {r['peak_vram_mb']} MB")
        assert os.path.getsize(r["path"]) > 10_000 and r["width"] == 512 and r["peak_vram_mb"] > 0
    finally:
        rt.unload(h)


@pytest.mark.skipif(
    not IMAGE or bool(runtimes.available()["open-clip"]), reason="needs STRATAMODELS_TEST_IMAGE, open_clip and torch"
)
def test_clip_embeddings_on_the_gpu(tmp_path):
    import cv2

    m = Registry.load().resolve("image-embed")
    rt = runtimes.get(m.runtime)
    h = rt.load(m, MODELS)
    try:
        r = rt.run(h, dict(paths=[IMAGE], texts=["a photo of a child", "a photo of a car"]))
        img, (child, car) = np.array(r["images"][0]), np.array(r["texts"])
        assert r["dim"] == 512 and abs(np.linalg.norm(img) - 1) < 1e-3 and img @ child > img @ car
        arr = tmp_path / "thumbs.npz"
        np.savez(arr, cam=np.stack([cv2.resize(cv2.imread(IMAGE), (256, 256))] * 3))
        o = rt.run(
            h, dict(array=dict(path=str(arr), key="cam", bgr=True), resize="square", out=str(tmp_path / "e.npz"))
        )
        e = np.load(o["out"])["images"]
        assert (
            o["images"] == 3 and e.shape == (3, 512) and float(e[0] @ img) > 0.8
        )  # the same picture squeezed to a square thumbnail against centre-cropped: close, not equal
    finally:
        rt.unload(h)
