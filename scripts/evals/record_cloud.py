"""Record a cloud model's answers to an eval's items through a subscription CLI, so the model can stand in the eval as a `recorded` candidate.

Jonathan's Google AI Pro and ChatGPT Plus plans have no API credits: the only sanctioned way to call their models from a script is each vendor's
own agent CLI run headless (`codex exec` for OpenAI, `agy -p` for Antigravity; Gemini CLI no longer serves consumer plans since 2026-06-18).
Never route those CLIs' sign-in tokens through anything else (Google disables accounts for it). This script calls the official binaries only.

It takes a LOCAL candidate of the spec as the template, so the cloud model gets exactly the same job: the same preset (prompt and JSON schema),
the same context, the same pictures (`runtimes.llama.request` builds the prompt, as it does for llama-server). Each answer is appended to a JSON
lines file of {"key": <the item's `by` param>, "item": <id>, "output": {...}} that the `recorded` runtime replays (see models.toml).

    uv run python scripts/evals/record_cloud.py SPEC --like CANDIDATE --cli codex|agy [--model SLUG] [--effort low] \
        --out <data>/evals/recorded/<name>.jsonl [--by image] [--limit N] [--timeout 600]

Resumable: items already in --out are skipped, so rerun it after a quota runs out. A quota or rate-limit answer stops the run with exit code 75
(EX_TEMPFAIL) and writes nothing for that item; any other failure is recorded as {"error": ...} in the log on stderr and the item is left out,
so the eval counts it as missing against the cloud model (a recorded job with no record fails). Reads the spec and dataset, never writes the
database; uses no GPU and does not need the service running."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

QUOTA = re.compile(
    r"usage limit|rate limit|quota|resource[_ ]exhausted|too many requests|\b429\b|try again (later|in)", re.I
)
TEMPFAIL = 75


class QuotaError(RuntimeError):
    """The plan's allowance is spent for now: stop, rerun later."""


class CallError(RuntimeError):
    """This item failed (bad answer, CLI error): log it and go on."""


def job(params):
    """(system, prompt, images, schema) for a job's params, as llama-server would get them."""
    from stratamodels.runtimes import llama

    body, schema = llama.request(params)
    system = next((m["content"] for m in body["messages"] if m["role"] == "system"), None)
    user = body["messages"][-1]["content"]
    text = "".join(c["text"] for c in user if c.get("type") == "text")
    imgs = params.get("images") or ([params["image"]] if params.get("image") else [])
    return system, text, [str(i) for i in imgs], schema


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


def record_one(params, cli, model=None, effort=None, timeout=600, call=None, clock=time.perf_counter):
    system, text, images, schema = job(params)
    t0 = clock()
    r = (call or CLIS[cli])(system, text, images, schema, model=model, effort=effort, timeout=timeout)
    if not r["text"]:
        raise CallError(
            f"{cli} answered nothing"
        )  # a CLI hiccup, not the model's answer: left out, so a rerun tries the item again
    out = dict(text=r["text"], seconds=round(clock() - t0, 2), cli=cli, cli_model=model or "default")
    if schema:
        out["json"] = parse_json(r["text"])
    u = r.get("usage") or {}
    if isinstance(u, dict):
        out["tokens_in"] = u.get("input_tokens") or u.get("prompt_tokens")
        out["tokens_out"] = u.get("output_tokens") or u.get("completion_tokens")
    return out


def done_keys(path):
    if not os.path.exists(path):
        return set()
    return {str(json.loads(line)["key"]) for line in open(path, encoding="utf8") if line.strip()}


def main(argv=None, log=None):
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spec")
    ap.add_argument(
        "--like", required=True, help="the local candidate whose params (preset, context) the cloud model gets"
    )
    ap.add_argument("--cli", required=True, choices=sorted(CLIS))
    ap.add_argument(
        "--model", help="the CLI's model slug (codex: e.g. gpt-6.1-sol; agy: see `agy models`); default: the CLI's own"
    )
    ap.add_argument("--effort", help="reasoning effort (low, medium, high)")
    ap.add_argument(
        "--out", required=True, help="JSON lines file to append to (relative paths are under the data folder)"
    )
    ap.add_argument("--by", default="image", help="the job param the recorded runtime matches on (default image)")
    ap.add_argument("--limit", type=int, help="record at most this many new items")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per item")
    a = ap.parse_args(argv)

    from stratamodels import settings
    from stratamodels.evals import core as C

    spec = C.load_spec(a.spec)
    cand = spec.candidate(a.like)
    ds = C.load_dataset(spec.dataset)
    out = a.out if os.path.isabs(a.out) else os.path.join(settings.data_dir(), a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    have = done_keys(out)
    todo = [it for it in ds.items if str(C.job_params(spec, cand, it).get(a.by)) not in have]
    if a.limit is not None:
        todo = todo[: a.limit]
    log(
        f"{spec.name}: {len(ds.items)} items, {len(have)} recorded, {len(todo)} to do with {a.cli} {a.model or '(default model)'}"
    )
    failed = 0
    for n, it in enumerate(todo, 1):
        params = {k: v for k, v in C.job_params(spec, cand, it).items() if not k.startswith("__")}
        key = params.get(a.by)
        try:
            o = record_one(params, a.cli, a.model, a.effort, a.timeout)
        except QuotaError as e:
            log(f"quota: stopping after {n - 1} new items; rerun later to resume. {e}")
            return TEMPFAIL
        except (CallError, subprocess.TimeoutExpired) as e:
            failed += 1
            log(f"[{n}/{len(todo)}] {it['id']}: failed: {e}")
            continue
        with open(out, "a", encoding="utf8") as f:
            f.write(json.dumps(dict(key=key, item=it["id"], output=o), ensure_ascii=False) + "\n")
        ok = "json ok" if o.get("json") is not None else ("no json" if "json" in o else "text")
        log(f"[{n}/{len(todo)}] {it['id']}: {o['seconds']} s, {ok}")
    log(f"done: {len(todo) - failed} recorded, {failed} failed (left out: they count as missing)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
