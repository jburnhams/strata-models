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
    not IMAGE or runtimes.available()["face-tracks"], reason="needs STRATAMODELS_TEST_IMAGE, insightface and ffmpeg"
)
def test_face_tracks_on_the_gpu(tmp_path):
    """The test photo as a 6 s video that pans across it with a cut at 3 s: every face it holds is one track a shot, seen with the others."""
    import subprocess

    video = tmp_path / "pan.mp4"
    pan = "scale=1920:-2,crop=1280:ih:'t*40':0,setsar=1"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-loop",
            "1",
            "-framerate",
            "25",
            "-t",
            "6",
            "-i",
            IMAGE,
            "-vf",
            pan,
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    out, _ = measure(
        Registry.load().resolve("face-tracks"), dict(path=str(video), cuts=[3.0], out_dir=str(tmp_path / "stills"))
    )
    assert out["tracks"] and {t["shot"] for t in out["tracks"]} <= {0, 1}
    for t in out["tracks"]:
        assert len(t["embedding"]) == 512 and t["stills"] and all(os.path.isfile(s["image"]) for s in t["stills"])
    per_shot = [[t for t in out["tracks"] if t["shot"] == k] for k in (0, 1)]
    print(f"tracks a shot: {[len(p) for p in per_shot]}, {out['frames']} frames in {out['seconds']} s")
    assert abs(len(per_shot[0]) - len(per_shot[1])) <= 1  # the same faces either side of the cut, each one track


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
    not AUDIO or not os.path.isdir(os.path.join(MODELS, "venvs", "speaker")),
    reason="needs STRATAMODELS_TEST_AUDIO and the speaker worker venv (scripts/setup_worker_venv.sh speaker)",
)
def test_speaker_embeddings_on_the_gpu():
    """Two stretches of the sample (one speaker) come out close together; a stretch too short to place answers null."""
    items = [dict(path=AUDIO, clip=[0.0, 4.0]), dict(path=AUDIO, clip=[5.0, 9.0]), dict(path=AUDIO, clip=[0.0, 0.2])]
    out, _ = measure(Registry.load().resolve("speaker-embedding"), dict(items=items))
    a, b, short = out["embeddings"]
    assert out["dim"] == 192 and short is None and abs(sum(v * v for v in a) - 1) < 1e-3
    print(f"same voice, two stretches: cosine {sum(x * y for x, y in zip(a, b)):.3f}")
    assert sum(x * y for x, y in zip(a, b)) > 0.5


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


def test_speak_ipa_on_kokoro():
    """A vowel sound given as IPA (`phonemes`): a short clip, not a spelled-out word. Listen to it: the numbers cannot say it is the ee of bee."""
    from stratamodels.runtimes import worker

    m = Registry.load().models["kokoro-v1"]
    if not os.path.isfile(worker.python_of(m, MODELS)[1]):
        pytest.skip("no venv for kokoro-v1: scripts/setup_worker_venv.sh kokoro")
    out, _out2 = measure(m, dict(text="/iː/", phonemes="iː", voice="bf_emma"))
    assert os.path.isfile(out["path"]) and 0.1 < out["duration_s"] < 2.0


def test_speak_pure_consonants_on_kokoro():
    """`sound`: /ʃ/ cut from /ʃɑː/ and held hisses with no vowel after it; said alone (carrier "") it does not. Measured in a cloud container
    on 2026-11-24 (bf_emma): 24 of the 25 consonants pass phonics.pure, against none said alone. Listen to the clips too."""
    from stratamodels.runtimes import worker

    m = Registry.load().models["kokoro-v1"]
    if not os.path.isfile(worker.python_of(m, MODELS)[1]):
        pytest.skip("no venv for kokoro-v1: scripts/setup_worker_venv.sh kokoro")
    cut, _ = measure(m, dict(text="/ʃ/", sound="ʃ", voice="bf_emma"))
    alone, _ = measure(m, dict(text="/ʃ/", sound="ʃ", carrier="", voice="bf_emma"))
    assert (
        cut["sound"]["kind"] == "hiss"
        and cut["high"] > 0.5
        and cut["vowel_leak"] < 0.1
        and abs(cut["duration_s"] - 0.6) < 0.02
    )
    assert alone["high"] < cut["high"]


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


@pytest.mark.skipif(
    bool(runtimes.available()["llama-server"]), reason="needs the llama-server binary (llama.cpp) and huggingface_hub"
)
def test_scene_describe_on_the_gpu(tmp_path):
    """Role scene-describe through the registry: three frames of one made-up scene and its context, answered in the preset's schema with what is seen kept apart from what is guessed."""
    from PIL import Image, ImageDraw

    frames = []
    for i in range(3):  # a cake with candles moving across a table, as frames of one shot would
        im = Image.new("RGB", (448, 336), (90, 60, 40))
        d = ImageDraw.Draw(im)
        x = 80 + 60 * i
        d.rectangle((x, 180, x + 140, 260), fill="white")
        for c in range(4):
            d.rectangle((x + 20 + 30 * c, 140, x + 30 + 30 * c, 180), fill="yellow")
        im.save(tmp_path / f"{i}.png")
        frames.append(str(tmp_path / f"{i}.png"))
    out, _ = measure(
        Registry.load().resolve("scene-describe"),
        dict(
            images=frames,
            preset="scene-describe-v2",
            context=dict(seconds=12.0, shots=1, speech="Happy birthday to you"),
            temperature=0.0,
        ),
    )
    a = out["json"]
    print(a)
    assert set(a) == {"summary", "observed", "inferred", "setting", "people", "children", "activity", "tags", "picture"}
    assert a["setting"] in ("indoor", "outdoor", "mixed", "unknown") and isinstance(a["people"], int)
    assert all(set(g) == {"guess", "because"} for g in a["inferred"])
    assert a["picture"]["quality"] in ("good", "fair", "poor")


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


VIDEO = os.environ.get(
    "STRATAMODELS_TEST_VIDEO"
)  # a short interlaced PAL clip (a few seconds of digitised VHS), e.g. one of vhs-clips-8
ENHANCERS = [
    "video-lanczos",
    "video-realesr-general-x4v3",
    "video-realesrgan-x2plus",
    "video-realesrgan-x4plus",
    "video-realplksr-nomos",
    "video-swinir-real-x4",
    "video-seedvr2-3b",
    "video-flashvsr-plus-turbo",
    "video-flashvsr-plus-base",
]


@pytest.mark.skipif(not VIDEO, reason="needs STRATAMODELS_TEST_VIDEO")
@pytest.mark.parametrize("key", ENHANCERS)
def test_video_enhance_on_the_gpu(key, tmp_path):
    """One clip through each engine: a 1440 x 1080 50p result with the source's length, a face region and stills. Slow for the diffusion engines."""
    from stratamodels.runtimes import worker

    m = Registry.load().models[key]
    if not os.path.isfile(worker.python_of(m, MODELS)[1]):
        pytest.skip(f"no venv for {key}: scripts/setup_worker_venv.sh {m.options.get('venv')}")
    rt = runtimes.get(m.runtime)
    h = rt.load(m, MODELS)
    try:
        out = rt.run(
            h, dict(path=VIDEO, crop=[14, 0, 692, 568], regions=[dict(id="face", box=[200, 90, 260, 195])], stills=[1])
        )
    finally:
        rt.unload(h)
    print(f"\n{key}: {out['enhance_s']:.1f} s ({out['fps_enhance']} frames/s), peak {out['peak_vram_mb']:.0f} MB")
    assert (out["width"], out["height"], out["fps"]) == (1440, 1080, 50.0) and os.path.isfile(out["path"])
    assert [c["id"] for c in out["crops"]] == ["face"] and len(out["stills"]) == 1
    assert out["peak_vram_mb"] < 11.5 * 1024  # fits the 12 GB card with room for the service


def test_page_build_draws_a_page_through_the_registered_role(tmp_path):
    # CPU lane (vram_gb 0), so it runs without measure()'s GPU readings: the registered role end to end, load, run twice, unload
    model = Registry.load().resolve("page-build")
    assert model.vram_gb == 0
    rt = runtimes.get(model.runtime)
    h = rt.load(model, MODELS)
    page = dict(
        width=1200,
        height=800,
        dpi=150,
        blocks=[
            dict(
                box=[100, 100, 1000, 60],
                role="heading",
                lines=[dict(box=[100, 100, 1000, 60], spans=[dict(text="PUNCH AND JUDY")])],
            )
        ],
    )
    out = rt.run(h, dict(page=page, out=str(tmp_path / "p.pdf")))
    again = rt.run(h, dict(page=page, out=str(tmp_path / "q.pdf")))
    rt.unload(h)
    assert out["text"].strip() == "PUNCH AND JUDY"
    assert open(out["pdf"], "rb").read() == open(again["pdf"], "rb").read()  # the same description draws the same bytes
    assert out["png"] and os.path.isfile(out["png"])


def test_font_match_fits_the_real_catalogue(tmp_path):
    """The real catalogue path: faces fetched from Google Fonts into the models folder, a block set in one of them, and the fit naming it.
    CPU only (vram 0), but here because it downloads."""
    from stratamodels.runtimes import fontfit as F

    model = Registry.load().resolve("font-match")
    h = runtimes.get(model.runtime).load(model, MODELS)
    face = F.Face("old-standard", F.fetch("old-standard", MODELS))
    canvas = np.zeros((220, 1400), dtype=bool)
    words, x = [], 40.0
    for w in ["The", "Punch", "Puppet", "has", "a", "big", "belly"]:
        m, left, base = face.word("roman", w, 60.0, 1.0, 0.0, 0.5)
        F._paste(canvas, m, left, base, x, 120)
        cols, rows = np.flatnonzero(m.any(0)), np.flatnonzero(m.any(1))
        bw = int(cols[-1] - cols[0] + 1)
        words.append(
            dict(text=w, box=[int(x), int(120 - (base - rows[0])), bw, int(rows[-1] - rows[0] + 1)], conf=95.0)
        )
        x += bw + 18
    from PIL import Image

    p = str(tmp_path / "block.png")
    Image.fromarray(np.where(canvas, 40, 225).astype(np.uint8)).save(p, dpi=(400, 400))
    line = dict(text=" ".join(w["text"] for w in words), box=[40, 60, int(x), 90], words=words)
    r = runtimes.get(model.runtime).run(
        h,
        dict(
            image=p,
            lines=[line],
            candidates=["old-standard", "libre-bodoni", "tinos", "rye"],
            out=str(tmp_path / "fit.png"),
        ),
    )
    print(f"\nfontfit: {r['face']} ink_f1 {r['ink_f1']} in {r['seconds']} s; ranked {[f['face'] for f in r['faces']]}")
    assert r["face"] == "old-standard"
    assert r["blocks"][0]["lines"][0]["words"][0]["size_px"] == pytest.approx(60.0, rel=0.08)


WRITE_SCHEMA = dict(
    type="object",
    properties=dict(
        prompts=dict(
            type="array",
            items=dict(
                type="object",
                properties=dict(text=dict(type="string"), style=dict(type="string")),
                required=["text", "style"],
                additionalProperties=False,
            ),
        )
    ),
    required=["prompts"],
    additionalProperties=False,
)


@pytest.mark.skipif(
    bool(runtimes.available()["claude-api"]), reason="needs the anthropic SDK and ANTHROPIC_API_KEY (keys.py)"
)
def test_write_role_follows_a_brief_and_a_schema():
    """Role `write` (a cloud model, no GPU): a recording-prompt brief of the kind strata-voice sends, answered in its JSON schema."""
    m = Registry.load().resolve("write")
    rt = runtimes.get(m.runtime)
    h = rt.load(m, MODELS)
    try:
        out = rt.run(
            h,
            dict(
                system="You write short sentences for a speaker to read aloud when recording a voice dataset, in British English.",
                prompt="Write 3 new prompts. Each is one sentence of 6 to 15 words, with no digits, and uses the sound /ʒ/ as in "
                "“measure”. Give each a style from: neutral, happy, surprised.",
                schema=WRITE_SCHEMA,
                max_tokens=1000,
            ),
        )
    finally:
        rt.unload(h)
    print(f"\n{m.key} write ({out['seconds']} s, {out['cost_usd']} USD): {out['json']}")
    prompts = out["json"]["prompts"]
    assert len(prompts) == 3 and all(p["style"] in {"neutral", "happy", "surprised"} for p in prompts)
    assert all(6 <= len(p["text"].split()) <= 15 and not any(c.isdigit() for c in p["text"]) for p in prompts)


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(MODELS, "venvs", "shots")),
    reason="needs the shots worker venv (scripts/setup_worker_venv.sh shots)",
)
@pytest.mark.parametrize("key", ["transnetv2", "transnetv2-cpu", "pyscenedetect-adaptive", "pyscenedetect-content"])
def test_shot_detectors_find_the_cuts_to_the_frame(key, tmp_path):
    """The rivals to ffmpeg's scene score on the four-shot video of tests/integration/test_scenes_runtime.py: cuts at 2, 3 and 5 s, black
    from 2 to 3 s, in the baseline's answer shape. TransNet V2 is on the GPU unless the model says the CPU."""
    import subprocess

    path = tmp_path / "four-shots.mp4"
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    for s in [
        "testsrc=s=320x240:r=25:d=2",
        "color=black:s=320x240:r=25:d=1",
        "mandelbrot=s=320x240:r=25",
        "smptebars=s=320x240:r=25:d=2",
    ]:
        cmd += ["-f", "lavfi", "-i", s]
    graph = "[2]trim=duration=2,setpts=PTS-STARTPTS[m];[0][1][m][3]concat=n=4:v=1[v]"
    subprocess.run(
        cmd + ["-filter_complex", graph, "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True
    )
    out, _ = measure(Registry.load().models[key], dict(path=str(path)))
    assert [c["time"] for c in out["cuts"]] == [2.0, 3.0, 5.0] and out["black"] == [{"start": 2.0, "end": 3.0}]
    assert out["model"] == key and out["device"] == ("cuda" if key == "transnetv2" else "cpu")


EMBED_MODELS = ["qwen3-embedding-0.6b", "qwen3-embedding-4b", "bge-m3", "embeddinggemma-300m", "nomic-embed-v1_5"]


@pytest.mark.skipif(
    bool(runtimes.available()["llama-server"]), reason="needs the llama-server binary (llama.cpp) and huggingface_hub"
)
@pytest.mark.parametrize("key", EMBED_MODELS)
def test_text_embed_on_the_gpu(key):
    """Role text-embed: vectors come back unit length, and a question ranks the passage that answers it above the ones that do not (the repeat is answered from the cache)."""
    model = Registry.load().models[key]
    assert "text-embed" in model.roles
    docs = [
        "Your parcel has left the depot and will arrive on Thursday between 9 and 1.",
        "Reminder: the quarterly gas meter reading is due before the end of the month.",
        "Booking confirmed: two nights at the lakeside cabin, check-in from 3 pm on Friday.",
    ]
    out, again = measure(model, dict(query="When can I check in to the cabin?", docs=docs))
    scores = out["scores"]
    assert len(scores) == 3 and max(range(3), key=scores.__getitem__) == 2 and out["dim"] > 0
    assert again["scores"] == scores  # deterministic, and the second run is the cached one
    vecs, _ = measure(model, dict(texts=docs))
    v = np.asarray(vecs["vectors"], dtype=np.float32)
    assert v.shape == (3, out["dim"]) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-3)
