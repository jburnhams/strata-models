"""Calling a cloud model on a subscription plan through its vendor's own agent CLI, run headless: `codex exec` (ChatGPT plans) and `agy -p`
(Antigravity, Google AI plans; Gemini CLI no longer serves consumer plans since 2026-06-18). The plans have no API credits, so the official
binaries are the only sanctioned way in: never route their sign-in tokens through anything else (Google disables accounts for it). API keys
are removed from the CLI's environment so a call is always billed to the plan.

Used by scripts/evals/record_cloud.py (record a cloud model's outputs for the `recorded` runtime) and evals.judges (a cloud model as a judge).
Each call function takes (system, text, images, schema, model=, effort=, timeout=, run=) and returns {text, usage}; QuotaError means the
plan's allowance is spent for now (stop, rerun later), CallError that this one call failed."""

import json
import os
import re
import shutil
import subprocess
import tempfile

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


def call_agy(system, text, images, schema, model=None, effort=None, timeout=600, run=subprocess.run):
    """agy -p has no attachment flag: the pictures are copied into a fresh workspace (its working directory) and the prompt names them."""
    with tempfile.TemporaryDirectory(prefix="strata-cloud-") as d:
        prompt = prompt_of(system, text, stage_images(images, d), schema)
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
