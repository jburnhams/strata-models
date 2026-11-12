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
    not AUDIO or not os.path.isdir(os.path.join(MODELS, "venvs", "whisper")),
    reason="needs STRATAMODELS_TEST_AUDIO and the whisper worker venv (scripts/setup_worker_venv.sh whisper)",
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
            assert all(abs(a - c) < 0.05 * W for a, c in zip(got, want)), (v["view"], got, want)
        c = rt.run(
            h,
            dict(
                path=IMAGE,
                fields=["bbox"],
                views=[dict(crop=[max(b[0] - W * 0.3, 0), max(b[1] - H * 0.3, 0), W * 0.7, H * 0.7])],
            ),
        )
        got = c["views"][0]["faces"][0]["bbox"]
        assert all(abs(a - x) < 0.05 * W for a, x in zip(got, b)), (got, b)
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
    not AUDIO or not os.path.isdir(os.path.join(MODELS, "venvs", "whisper")),
    reason="needs STRATAMODELS_TEST_AUDIO and the whisper worker venv (scripts/setup_worker_venv.sh whisper)",
)
def test_whisper_detects_the_language_alone():
    m = Registry.load().resolve("transcribe-fast")
    rt = runtimes.get(m.runtime)
    h = rt.load(m, MODELS)
    try:
        r = rt.run(h, dict(path=AUDIO, detect_language_only=True))
        assert r["language"] == "en" and r["all_language_probs"]["en"] > 0.5 and "segments" not in r
        c = rt.run(
            h, dict(path=AUDIO, language="en", clip=[3.0, 8.0])
        )  # only that part is heard; times stay the file's
        assert (
            c["segments"]
            and c["segments"][0]["start"] >= 2.9
            and "country" in " ".join(x["text"] for x in c["segments"]).lower()
        )
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


SPEAKERS = ["kokoro-v1", "chatterbox", "chatterbox-turbo", "qwen3-tts-custom", "qwen3-tts-design", "qwen3-tts-clone"]
VOICE = os.environ.get("STRATAMODELS_TEST_VOICE")  # a reference recording of more than 5 s, for the cloning models


@pytest.mark.parametrize("key", SPEAKERS)
def test_speak_on_the_gpu(key):
    from stratamodels.runtimes import worker

    m = Registry.load().models[key]
    if not os.path.isfile(worker.python_of(m, MODELS)[1]):
        pytest.skip(f"no venv for {key}: scripts/setup_worker_venv.sh {m.options.get('venv')}")
    params = dict(text="Two hundred kilometres behind me, and the second night is coming.")
    if key in ("chatterbox-turbo", "qwen3-tts-clone"):
        if not VOICE:
            pytest.skip("needs STRATAMODELS_TEST_VOICE")
        params["voice"] = VOICE
    if key == "qwen3-tts-design":
        params["instruct"] = "A calm British man in his forties, narrating a documentary"
    out, _out2 = measure(m, params)
    assert os.path.isfile(out["path"]) and 2.0 < out["duration_s"] < 12.0 and out["sample_rate"] >= 16000


@pytest.mark.skipif(
    not IMAGE or bool(runtimes.available()["face-kind"]) or bool(runtimes.available()["insightface"]),
    reason="needs STRATAMODELS_TEST_IMAGE (a photo of a real person), insightface, open_clip and torch",
)
def test_face_kind_on_the_gpu():
    import dataclasses

    faces = Registry.load().resolve("faces")
    fr = runtimes.get(faces.runtime)
    fh = fr.load(faces, MODELS)
    try:
        boxes = [f["bbox"] for f in fr.run(fh, dict(path=IMAGE, fields=["bbox"]))["faces"]]
    finally:
        fr.unload(fh)
    assert boxes
    m = Registry.load().resolve("face-kind")
    out, _ = measure(m, dict(path=IMAGE, boxes=boxes[:1]))
    f = out["faces"][0]
    assert abs(sum(f["probs"].values()) - 1) < 1e-3 and "real" in f["probs"]
    if out["head"].startswith("trained"):
        assert f["real"] > 0.5 and f["kind"] == "real"  # the test picture is of a real person
    zs = dataclasses.replace(m, options={**m.options, "head": "no-such-head.npz"})
    rt = runtimes.get(zs.runtime)
    h = rt.load(zs, MODELS)
    try:
        z = rt.run(h, dict(path=IMAGE, boxes=boxes[:1]))
    finally:
        rt.unload(h)
    assert z["head"] == "zero-shot" and set(z["faces"][0]["probs"]) == {"real", "statue", "picture", "toy", "animal"}


@pytest.mark.skipif(
    bool(runtimes.available()["llama-server"]), reason="needs the llama-server binary (llama.cpp) and huggingface_hub"
)
@pytest.mark.parametrize("model", ["qwen3.5-9b-vl", "flux3-klein-pup"])
def test_text_role_expands_an_image_prompt_on_the_gpu(model):
    """Role `text`, text only (no picture, no projector needed): the image-prompt expansion a pipeline eval runs before the image model."""
    m = Registry.load().resolve("text", model)
    out, _ = measure(
        m,
        dict(
            prompt="Exactly five red apples in a row on a white shelf, studio photo",
            system_preset="image-prompt-bfl",
            max_tokens=1500,
            temperature=0.0,
        ),
    )
    print(out["text"][:300])
    assert len(out["text"].split()) > 40 and "five" in out["text"].lower() and out["finish"] == "stop"


def _judge(role, spec_q, item, right, wrong, key=None):
    """Both orders of one clear-cut pair: the real model must accept the verdict schema and pick the right answer whichever side it is on."""
    from stratamodels.evals import core as C
    from stratamodels.evals import judges as J

    spec = C.parse_spec(
        dict(
            name="g", question=spec_q, role="r", dataset="d", metrics=["failed"], candidates=[dict(name="a", model="m")]
        )
    )
    j = J.parse_judge(dict(name="g", kind=role.removeprefix("judge-")))
    m = Registry.load().models[key] if key else Registry.load().resolve(role)
    rt = runtimes.get(m.runtime)
    h = rt.load(m, MODELS)
    try:
        for x, y, want in ((right, wrong, "X"), (wrong, right, "Y")):
            system, text, images = J.prompt(spec, j, item, x, y)
            p = dict(
                temperature=0,
                max_tokens=1024,
                system=system,
                prompt=text,
                schema=J.SCHEMA,
                **({("audios" if j["kind"] == "audio" else "images"): images} if images else {}),
            )
            out = rt.run(h, p)
            said, why = J.verdict(out)
            print(f"\n{m.key} {role}: {said} ({out['seconds']} s): {why}")
            assert said == want
    finally:
        rt.unload(h)


@pytest.mark.skipif(
    bool(runtimes.available()["llama-server"]), reason="needs the llama-server binary (llama.cpp) and huggingface_hub"
)
def test_judge_text_on_the_gpu():
    """Role judge-text: a transcript that matches the reference against one that does not."""
    _judge(
        "judge-text",
        "Which transcript of the speech is right?",
        dict(id="t", input=dict(language="en"), reference=dict(text="the quick brown fox jumps over the lazy dog")),
        dict(segments=[dict(text="the quick brown fox jumps over the lazy dog")]),
        dict(segments=[dict(text="a quick round box dumps over the hazy fog")]),
    )


@pytest.mark.skipif(
    bool(runtimes.available()["llama-server"]), reason="needs the llama-server binary (llama.cpp) and huggingface_hub"
)
def test_judge_vision_on_the_gpu(tmp_path):
    """Role judge-vision: two descriptions of the item's picture, then two pictures made for one prompt (three pictures in one job)."""
    from PIL import Image, ImageDraw

    def pic(name, colour, shape):
        im = Image.new("RGB", (384, 384), "white")
        d = ImageDraw.Draw(im)
        (d.rectangle if shape == "square" else d.ellipse)((96, 96, 288, 288), fill=colour)
        im.save(tmp_path / name)
        return str(tmp_path / name)

    red = pic("red.png", "red", "square")
    _judge(
        "judge-vision",
        "Describe the picture",
        dict(id="p", input=dict(image=red)),
        dict(text="A red square on a white background."),
        dict(text="A blue circle on a black background."),
    )
    _judge(
        "judge-vision",
        "Draw the prompt: a red square on a white background",
        dict(id="g", input=dict(image=pic("ref.png", "red", "square"))),
        dict(path=red),
        dict(path=pic("blue.png", "blue", "circle")),
    )


# ---- recorded sound -------------------------------------------------------------------------------------------------------------------------
DENOISERS = [
    "deepfilternet3",
    "mossformer2-se-48k",
    "mossformergan-se-16k",
    "frcrn-se-16k",
    "resemble-enhance",
    "audio-passthrough",
    "afftdn",
    "noisereduce",
]


def _noisy(tmp_path, snr_db=0.0):
    """STRATAMODELS_TEST_AUDIO as 48 kHz mono with white noise at snr_db (WAV), and the clean version."""
    import subprocess

    import soundfile as sf

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", AUDIO, "-t", "10", "-f", "f32le", "-ac", "1", "-ar", "48000", "-"],
        capture_output=True,
        check=True,
    ).stdout
    x = np.frombuffer(raw, np.float32)
    n = (
        np.random.default_rng(0).standard_normal(len(x)).astype(np.float32)
        * float(np.sqrt(np.mean(x**2)))
        * 10 ** (-snr_db / 20)
    )
    clean, noisy = str(tmp_path / "clean.wav"), str(tmp_path / "noisy.wav")
    sf.write(clean, x, 48000)
    sf.write(noisy, np.clip(x + n, -1, 1), 48000)
    return clean, noisy


@pytest.mark.skipif(not AUDIO, reason="needs STRATAMODELS_TEST_AUDIO (speech)")
@pytest.mark.parametrize("key", DENOISERS)
def test_denoise_audio_on_the_gpu(key, tmp_path):
    from stratamodels.runtimes import worker

    m = Registry.load().models[key]
    if not os.path.isfile(worker.python_of(m, MODELS)[1]):
        pytest.skip(f"no venv for {key}: scripts/setup_worker_venv.sh {m.options.get('venv')}")
    _, noisy = _noisy(tmp_path)
    out, _ = measure(m, dict(path=noisy, highpass=80, lufs=-14))
    print(f"{key}: removed {out['removed_db']} dB, {out['seconds']} s, LUFS {out['lufs_in']} -> {out['lufs_out']}")
    assert os.path.isfile(out["path"]) and out["sample_rate"] == 48000 and abs(out["lufs_out"] + 14) < 1.0
    assert (
        key in ("audio-passthrough", "afftdn") or out["removed_db"] > 3
    )  # white noise at 0 dB: every learned denoiser takes a good part out (afftdn barely does: measured 0.1 dB, a weak classical baseline)


@pytest.mark.skipif(not AUDIO, reason="needs STRATAMODELS_TEST_AUDIO (speech)")
def test_audio_quality_scores_clean_above_noisy(tmp_path):
    """The intrusive measures are taken at 16 kHz: resampling drops two thirds of 48 kHz white noise, so a 0 dB mix reads about +4.77 dB."""
    from stratamodels.runtimes import worker

    m = Registry.load().resolve("audio-quality")
    if not os.path.isfile(worker.python_of(m, MODELS)[1]):
        pytest.skip("no venv: scripts/setup_worker_venv.sh audio-quality")
    clean, noisy = _noisy(tmp_path)
    rt = runtimes.get(m.runtime)
    h = rt.load(m, MODELS)
    try:
        c, n = rt.run(h, dict(path=clean, reference=clean)), rt.run(h, dict(path=noisy, reference=clean))
    finally:
        rt.unload(h)
    print(f"\nclean {c}\nnoisy {n}")
    assert (
        c["dnsmos"]["ovrl"] > n["dnsmos"]["ovrl"]
        and c["dnsmos"]["bak"] > n["dnsmos"]["bak"]
        and c["squim"]["stoi"] > n["squim"]["stoi"]
    )
    assert (
        c["intrusive"]["si_sdr"] > 30
        and abs(n["intrusive"]["si_sdr"] - 4.77) < 1.5
        and c["intrusive"]["pesq"] > n["intrusive"]["pesq"]
    )


@pytest.mark.skipif(
    not AUDIO or bool(runtimes.available()["llama-server"]),
    reason="needs STRATAMODELS_TEST_AUDIO and the llama-server binary",
)
@pytest.mark.parametrize("key", ["qwen3-omni-30b-a3b", "gemma-4-e4b"])
def test_judge_audio_on_the_gpu(key, tmp_path):
    """Role judge-audio: the clean recording against the same with loud noise, both orders, the noisy one heard first as the input."""
    clean, noisy = _noisy(tmp_path, -5.0)
    _judge(
        "judge-audio",
        "Which cleaned recording sounds better: clear speech, little noise?",
        dict(id="a", input=dict(path=noisy)),
        dict(path=clean),
        dict(path=noisy),
        key=key,
    )
