"""Image generation with stable-diffusion.cpp (`sd-cli`, GGUF models): FLUX.3 Klein, Qwen-Image 2.1 Turbo, and whatever else it runs.

A model's files are named in models.toml as Hugging Face references, "repo:path/in/repo" (or absolute paths), under options:
  diffusion     the GGUF diffusion model (required)
  llm           the text encoder (Qwen3-VL for both models above)
  vae           the VAE
  steps, cfg_scale, sampler, args   defaults for this model, and extra sd-cli arguments (e.g. ["--clip-on-cpu", "--diffusion-fa"])
Files are fetched once into <models>/sd/ (gated repos need a Hugging Face token: HF_TOKEN, or `hf auth login`). The sd-cli binary is
STRATAMODELS_SD_BIN, else <models>/bin/sd/sd-cli.

Each job runs one sd-cli process (it loads the model itself; between jobs the card holds nothing for it). The scheduler still counts the
model's vram_gb while it is "loaded", so nothing else is put where the next image needs room.

params: prompt (required), negative_prompt, width, height (multiples of 16; default 1024x1024), steps, cfg_scale, sampler, seed (default
random, always returned), ref_images (paths: edits and references, FLUX.3 Klein takes up to 10). Result: {path, width, height, seed, steps,
seconds, peak_vram_mb, model}. Images go to <data>/generated/<YYYY-MM-DD>/."""

import datetime
import os
import random
import subprocess
import threading
import time
import uuid

from stratamodels import gpu, settings
from stratamodels.runtimes import RunError

FILES = ("diffusion", "llm", "vae")
FLAG = {"diffusion": "--diffusion-model", "llm": "--llm", "vae": "--vae"}


def binary(models_dir):
    return os.environ.get("STRATAMODELS_SD_BIN") or os.path.join(models_dir, "bin", "sd", "sd-cli")


def fetch(ref, models_dir):
    """A local path for "repo:path" (downloaded once into <models>/sd/<repo>/) or for an absolute path."""
    if os.path.isabs(ref):
        if not os.path.isfile(ref):
            raise RunError(f"no such model file: {ref}")
        return ref
    repo, _, path = ref.partition(":")
    if not path:
        raise RunError(f'a model file is "repo:path" or an absolute path, not {ref!r}')
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo, path, local_dir=os.path.join(models_dir, "sd", repo.replace("/", "--")))
    except Exception as e:
        raise RunError(
            f"could not fetch {ref}: {type(e).__name__}: {str(e)[:300]} (a gated repo needs its licence accepted on Hugging Face and a token: HF_TOKEN or `hf auth login`)"
        ) from e


def load(model, models_dir):
    o = model.options
    exe = binary(models_dir)
    if not os.path.isfile(exe):
        raise RunError(f"stable-diffusion.cpp is not installed: no {exe} (set STRATAMODELS_SD_BIN)")
    if not o.get("diffusion"):
        raise RunError(f"model {model.key} names no diffusion file in models.toml")
    files = {k: fetch(o[k], models_dir) for k in FILES if o.get(k)}
    return dict(
        key=model.key,
        exe=exe,
        files=files,
        defaults=dict(steps=o.get("steps", 20), cfg_scale=o.get("cfg_scale", 1.0), sampler=o.get("sampler", "euler")),
        args=list(o.get("args") or []),
    )


def _size(v, name):
    try:
        v = int(v)
    except (TypeError, ValueError) as exc:
        raise RunError(f"{name} must be a whole number of pixels") from exc
    if not 256 <= v <= 2048 or v % 16:
        raise RunError(f"{name} must be a multiple of 16 between 256 and 2048, not {v}")
    return v


def command(handle, params, out):
    """The sd-cli command line for a job (and the seed it will use)."""
    prompt = params.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise RunError("give a prompt")
    unknown = set(params) - {
        "prompt",
        "negative_prompt",
        "width",
        "height",
        "steps",
        "cfg_scale",
        "sampler",
        "seed",
        "ref_images",
    }
    if unknown:
        raise RunError(f"unknown params {sorted(unknown)}")
    d = handle["defaults"]
    seed = int(params["seed"]) if params.get("seed") is not None else random.randrange(2**31)
    cmd = [
        handle["exe"],
        "-M",
        "img_gen",
        "-p",
        prompt,
        "-W",
        str(_size(params.get("width", 1024), "width")),
        "-H",
        str(_size(params.get("height", 1024), "height")),
        "--steps",
        str(int(params.get("steps") or d["steps"])),
        "--cfg-scale",
        str(float(params.get("cfg_scale") or d["cfg_scale"])),
        "--sampling-method",
        str(params.get("sampler") or d["sampler"]),
        "-s",
        str(seed),
        "-o",
        out,
    ]
    for k, f in handle["files"].items():
        cmd += [FLAG[k], f]
    if params.get("negative_prompt"):
        cmd += ["-n", str(params["negative_prompt"])]
    refs = params.get("ref_images") or []
    if not isinstance(refs, list) or len(refs) > 10:
        raise RunError("ref_images is a list of at most 10 image paths")
    for r in refs:
        if not isinstance(r, str) or not os.path.isfile(r):
            raise RunError(f"no such reference image: {r!r}")
        cmd += ["-r", r]
    return cmd + handle["args"], seed


def run(handle, params, popen=subprocess.Popen, probe=gpu.query, out_dir=None):
    day = datetime.date.today().isoformat()
    folder = out_dir or os.path.join(settings.data_dir(), "generated", day)
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, f"{uuid.uuid4().hex[:12]}.png")
    cmd, seed = command(handle, params, out)
    env = dict(
        os.environ, LD_LIBRARY_PATH=os.path.dirname(handle["exe"]) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    )  # its ggml libraries sit beside it
    t0 = time.perf_counter()
    p = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    peak = [0.0]
    stop = threading.Event()

    def watch():  # peak VRAM of the sd-cli process, sampled while it runs
        while not stop.wait(0.5):
            st = probe() or {}
            for pr in st.get("processes", []):
                if pr["pid"] == p.pid:
                    peak[0] = max(peak[0], pr["used_mb"])

    threading.Thread(target=watch, daemon=True).start()
    try:
        log, _ = p.communicate(timeout=1800)
    except subprocess.TimeoutExpired as exc:
        p.kill()
        log, _ = p.communicate()
        raise RunError("sd-cli took more than 30 minutes; stopped") from exc
    finally:
        stop.set()
    if p.returncode != 0 or not os.path.isfile(out):
        raise RunError(f"sd-cli failed (exit {p.returncode}): {(log or '').strip()[-600:]}")
    w, h = int(cmd[cmd.index("-W") + 1]), int(cmd[cmd.index("-H") + 1])
    return dict(
        path=out,
        width=w,
        height=h,
        seed=seed,
        steps=int(cmd[cmd.index("--steps") + 1]),
        seconds=round(time.perf_counter() - t0, 2),
        peak_vram_mb=round(peak[0]),
        model=handle["key"],
    )


def unload(handle):
    handle.clear()
