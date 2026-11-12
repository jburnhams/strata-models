"""Calling a cloud model on a subscription plan through its vendor's own agent CLI, run headless: `codex exec` (ChatGPT plans) and `agy -p`
(Antigravity, Google AI plans; Gemini CLI no longer serves consumer plans since 2026-06-18). The plans have no API credits, so the official
binaries are the only sanctioned way in: never route their sign-in tokens through anything else (Google disables accounts for it). API keys
are removed from the CLI's environment so a call is always billed to the plan.

Used by scripts/evals/record_cloud.py (record a cloud model's outputs for the `recorded` runtime) and evals.judges (a cloud model as a judge).
Each call function takes (system, text, images, schema, model=, effort=, timeout=, run=) and returns {text, usage}; QuotaError means the
plan's allowance is spent for now (stop, rerun later), CallError that this one call failed."""

import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import time

QUOTA = re.compile(
    r"usage limit|rate limit|quota|resource[_ ]exhausted|too many requests|\b429\b|try again (later|in)", re.I
)
TEMPFAIL = 75


class QuotaError(RuntimeError):
    """The plan's allowance is spent for now: stop, rerun later."""


class CallError(RuntimeError):
    """This item failed (bad answer, CLI error): log it and go on."""


def prompt_of(system, text, names, schema, attached=False):
    """One prompt for an agent CLI: they take no separate system message. `names`: the pictures, attached to the message (codex -i) or saved
    in the agent's working directory (agy). The agent is told not to use tools beyond looking at the pictures."""
    parts = []
    if system:
        parts.append(system.strip())
    if names:
        many = len(names) > 1
        where = "attached" if attached else "in your working directory: " + ", ".join(names)
        parts.append(
            f"The picture{'s are' if many else ' is'} {where}. Look at {'them' if many else 'it'}; do not run commands or edit files."
        )
    parts.append(text.strip())
    if schema:
        parts.append("Answer with the JSON object only: no prose, no code fence.")
    return "\n\n".join(parts)


def parse_json(text):
    """The JSON object in an answer (tolerates a code fence or prose around it), or None."""
    if not text:
        return None
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except ValueError:
        a, b = t.find("{"), t.rfind("}")
        if a >= 0 and b > a:
            try:
                return json.loads(t[a : b + 1])
            except ValueError:
                return None
    return None


def clean_env():
    """The environment for the CLI without API keys, so a call is always billed to the plan, never to a paid API key by accident."""
    return {
        k: v
        for k, v in os.environ.items()
        if k not in ("OPENAI_API_KEY", "CODEX_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI")
    }


def _check(p, what):
    err = (p.stderr or "") + "\n" + (p.stdout or "")
    if QUOTA.search(err) and p.returncode != 0:
        raise QuotaError(f"{what}: {err.strip()[-400:]}")
    if p.returncode != 0:
        raise CallError(f"{what} exited {p.returncode}: {err.strip()[-400:]}")


def call_codex(system, text, images, schema, model=None, effort=None, timeout=600, run=subprocess.run):
    """codex exec attaches the pictures itself (-i) and enforces the schema (--output-schema); the last message is the answer."""
    prompt = prompt_of(system, text, [os.path.basename(i) for i in images], schema, attached=True)
    with tempfile.TemporaryDirectory(prefix="strata-cloud-") as d:
        last = os.path.join(d, "answer.txt")
        cmd = ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--ephemeral", "-o", last]
        if model:
            cmd += ["-m", model]
        if effort:
            cmd += ["-c", f"model_reasoning_effort={effort}"]
        if schema:
            sp = os.path.join(d, "schema.json")
            json.dump(schema, open(sp, "w"))
            cmd += ["--output-schema", sp]
        cmd += [f"--image={i}" for i in images]  # one value per flag: -i takes several, and would swallow the "-" below
        cmd += ["-"]  # the prompt on stdin: no argv length limit
        p = run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=d, env=clean_env())
        _check(p, "codex exec")
        text = open(last, encoding="utf8").read() if os.path.exists(last) else (p.stdout or "")
        return dict(text=text.strip(), usage=None)


def _agy(prompt, d, model=None, effort=None, timeout=600, run=subprocess.run, schema=None):
    """One headless agy turn in workspace `d`: its JSON envelope (conversation_id, response, structured_output, usage)."""
    cmd = [
        "agy",
        "-p",
        prompt,
        "--output-format",
        "json",
        "--print-timeout",
        f"{int(timeout)}s",
        "--disable-slash-commands",
    ]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
    if schema:
        sp = os.path.join(d, "schema.json")
        json.dump(schema, open(sp, "w"))
        cmd += ["--json-schema", sp]
    p = run(cmd, capture_output=True, text=True, timeout=timeout + 60, cwd=d, env=clean_env())
    env = parse_json(p.stdout) or {}
    status = str(env.get("status") or "")
    err = env.get("error")
    if p.returncode != 0 or (status and status != "SUCCESS"):
        msg = json.dumps(err) if err else (p.stderr or p.stdout or "")
        if QUOTA.search(msg) or QUOTA.search(p.stderr or ""):
            raise QuotaError(f"agy: {msg.strip()[-400:]}")
        raise CallError(f"agy exited {p.returncode} ({status or 'no status'}): {msg.strip()[-400:]}")
    return env


def call_agy(system, text, images, schema, model=None, effort=None, timeout=600, run=subprocess.run):
    """agy -p has no attachment flag: the pictures are copied into a fresh workspace (its working directory) and the prompt names them."""
    with tempfile.TemporaryDirectory(prefix="strata-cloud-") as d:
        env = _agy(prompt_of(system, text, stage_images(images, d), schema), d, model, effort, timeout, run, schema)
        so = env.get("structured_output")
        text = json.dumps(so) if so is not None and not isinstance(so, str) else (so or env.get("response") or "")
        return dict(text=str(text).strip(), usage=env.get("usage"))


CLIS = {"codex": call_codex, "agy": call_agy}


def stage_images(images, d):
    """Copy the pictures into the agent's workspace as img0.jpg, img1.png, ... (agy only reads its workspace)."""
    names = []
    for n, i in enumerate(images):
        name = f"img{n}{os.path.splitext(i)[1].lower() or '.jpg'}"
        shutil.copyfile(i, os.path.join(d, name))
        names.append(name)
    return names


# ---- pictures and speech (probed 2026-11-12: both CLIs make pictures; agy hears audio files; codex has no audio) --------------------------------
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")


def image_prompt(prompt, width, height):
    return (
        f"Generate one {width}x{height} image with your image generation tool for the description below, and save it as out.png in the "
        f"current directory. Make no other changes and do not ask questions.\n\nDescription: {prompt}"
    )


def image_codex(prompt, dest, width=1024, height=1024, model=None, effort=None, timeout=900, run=subprocess.run):
    """A picture from ChatGPT's image model (gpt-image, through Codex's own image skill): it saves out.png in its workspace (writable)."""
    with tempfile.TemporaryDirectory(prefix="strata-cloud-") as d:
        cmd = ["codex", "exec", "--skip-git-repo-check", "--sandbox", "workspace-write", "--ephemeral"]
        if model:
            cmd += ["-m", model]
        if effort:
            cmd += ["-c", f"model_reasoning_effort={effort}"]
        p = run(
            cmd + ["-"],
            input=image_prompt(prompt, width, height),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=d,
            env=clean_env(),
        )
        _check(p, "codex exec")
        made = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMAGE_EXT)]
        if not made:
            raise CallError(f"codex made no picture: {(p.stdout or '').strip()[-300:]}")
        return _keep(max(made, key=os.path.getmtime), dest)


def agy_brain():
    return os.environ.get("AGY_BRAIN") or os.path.expanduser("~/.gemini/antigravity-cli/brain")


def image_agy(
    prompt, dest, width=1024, height=1024, model=None, effort=None, timeout=900, run=subprocess.run, brain=None
):
    """A picture from Antigravity's generate_image tool. Headless mode denies the shell, so it cannot copy the file into the workspace: the
    picture is taken from the conversation's own folder (<brain>/<conversation_id>/), else from the workspace."""
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="strata-cloud-") as d:
        env = _agy(image_prompt(prompt, width, height), d, model, effort, timeout, run)
        made = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMAGE_EXT)]
        cid = env.get("conversation_id")
        if cid:
            made += [
                f
                for f in glob.glob(os.path.join(brain or agy_brain(), str(cid), "**", "*"), recursive=True)
                if f.lower().endswith(IMAGE_EXT)
            ]
        made = [f for f in made if os.path.getmtime(f) >= t0 - 1]
        if not made:
            raise CallError(f"agy made no picture: {str(env.get('response') or '')[-300:]}")
        return _keep(max(made, key=os.path.getmtime), dest)


def _keep(src, dest):
    """Copy a made picture to dest (keeping its own extension); its path and size."""
    dest = os.path.splitext(dest)[0] + os.path.splitext(src)[1].lower()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(src, dest)
    try:
        from PIL import Image

        with Image.open(dest) as im:
            w, h = im.size
    except Exception:
        w = h = None
    return dict(path=dest, width=w, height=h)


def cut_audio(path, clip, out, run=subprocess.run):
    """The file, or its clip [start, end] in seconds, as 16 kHz mono WAV (ffmpeg)."""
    cut = ["-ss", f"{float(clip[0]):.3f}", "-to", f"{float(clip[1]):.3f}"] if clip else []
    p = run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *cut, "-i", path, "-ac", "1", "-ar", "16000", out],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise CallError(f"ffmpeg could not cut {path}: {(p.stderr or '').strip()[-300:]}")
    return out


LANGS = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "es": "Spanish",
    "nl": "Dutch",
    "pt": "Portuguese",
}


def transcribe_agy(
    path, clip=None, language=None, model=None, effort=None, timeout=600, run=subprocess.run, cut=cut_audio
):
    """Speech to plain text by a Gemini model through agy: the audio goes into the workspace and the agent is told to open it with its file
    viewer (headless mode denies the shell, and an agent that tries one answers nothing). No word timings: they drift (probed), so this is
    text only; Whisper keeps alignment."""
    with tempfile.TemporaryDirectory(prefix="strata-cloud-") as d:
        cut(path, clip, os.path.join(d, "clip.wav"))
        lang = f" It is in {LANGS.get(language, language)}." if language else ""
        prompt = (
            "Open clip.wav in your working directory with your file viewing tool (do not run any shell commands) and transcribe the speech "
            f"in it verbatim.{lang} Output only the words that are spoken, nothing else."
        )
        env = _agy(prompt, d, model, effort, timeout, run)
        return dict(text=str(env.get("response") or "").strip(), usage=env.get("usage"))


# ---- remaining allowance: stop before a plan runs dry (Jonathan: stop at 10 % left) ---------------------------------------------------------
class UsageError(RuntimeError):
    """The CLI would not say how much allowance is left."""


def agy_group(model):
    """The Antigravity quota group a model draws from (agy -p /usage): Argon has its own on Pro, other Gemini models share one, and the
    third-party models (Claude, GPT-OSS) another."""
    m = (model or "").lower()
    if "argon" in m:
        return "Gemini 4 Argon"
    if m.startswith(("claude", "gpt")):
        return "Claude and GPT models"
    return "Gemini Models"  # the default model is a Gemini


def usage_agy(model=None, run=subprocess.run):
    """[(bucket, remaining 0-1, reset time)] of the model's group, from `agy -p /usage` (costs nothing: no model call)."""
    p = run(
        ["agy", "-p", "/usage", "--output-format", "json"], capture_output=True, text=True, timeout=120, env=clean_env()
    )
    try:
        groups = json.loads(p.stdout)["command"]["data"]["groups"]
    except (ValueError, KeyError, TypeError):
        raise UsageError(f"agy /usage gave no usage: {(p.stderr or p.stdout or '').strip()[-300:]}")
    want = agy_group(model)
    g = next((g for g in groups if g.get("name") == want), None)
    if g is None:
        raise UsageError(f"agy /usage has no group {want!r} (has {[g.get('name') for g in groups]})")
    return [
        (f"{want} {b.get('window')}", float(b["remaining_fraction"]), b.get("reset_time")) for b in g.get("buckets", [])
    ]


def usage_codex(model=None, popen=subprocess.Popen, timeout=60):
    """[(window, remaining 0-1, reset time)] of the ChatGPT plan's Codex limits, from the CLI's own app-server (`account/rateLimits/read`,
    JSON-RPC over stdio; no model call): primary is the 5-hour window, secondary the week."""
    p = popen(
        ["codex", "app-server", "--stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=clean_env(),
    )
    try:
        for msg in (
            dict(
                jsonrpc="2.0",
                id=1,
                method="initialize",
                params=dict(clientInfo=dict(name="strata-models-evals", version="0")),
            ),
            dict(jsonrpc="2.0", method="initialized"),
            dict(jsonrpc="2.0", id=2, method="account/rateLimits/read"),
        ):
            p.stdin.write(json.dumps(msg) + "\n")
        p.stdin.flush()
        t0 = time.monotonic()
        res = None
        while time.monotonic() - t0 < timeout:
            line = p.stdout.readline()
            if not line:
                break
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("id") == 2:
                if "error" in m:
                    raise UsageError(f"codex account/rateLimits/read: {m['error']}")
                res = m.get("result") or {}
                break
    finally:
        p.kill()
        p.wait()
    rl = (res or {}).get("rateLimits")
    if not rl:
        raise UsageError("codex app-server gave no rate limits")
    out = []
    for k, name in (("primary", "5h"), ("secondary", "weekly")):
        w = rl.get(k)
        if w and w.get("usedPercent") is not None:
            at = w.get("resetsAt")
            out.append(
                (
                    f"codex {name}",
                    1 - float(w["usedPercent"]) / 100,
                    datetime.datetime.fromtimestamp(at, datetime.UTC).isoformat() if at else None,
                )
            )
    return out


USAGE = {"codex": usage_codex, "agy": usage_agy}


def low_bucket(cli, model, min_left, usage=None):
    """The first bucket at or under `min_left` remaining as (name, remaining, reset), or None when there is room."""
    for b in (usage or USAGE[cli])(model):
        if b[1] <= min_left:
            return b
    return None
