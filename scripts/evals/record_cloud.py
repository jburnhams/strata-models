"""Record a cloud model's answers to an eval's items through a subscription CLI, so the model can stand in the eval as a `recorded` candidate.

Jonathan's Google AI Pro and ChatGPT Plus plans have no API credits: the only sanctioned way to call their models from a script is each vendor's
own agent CLI run headless (`codex exec` for OpenAI, `agy -p` for Antigravity; Gemini CLI no longer serves consumer plans since 2026-06-18).
Never route those CLIs' sign-in tokens through anything else (Google disables accounts for it). This script calls the official binaries only.

It takes a LOCAL candidate of the spec as the template, so the cloud model gets exactly the same job: the same preset (prompt and JSON schema),
the same context, the same pictures (`runtimes.llama.request` builds the prompt, as it does for llama-server). Each answer is appended to a JSON
lines file of {"key": <the item's `by` param>, "item": <id>, "output": {...}} that the `recorded` runtime replays (see models.toml).

    uv run python scripts/evals/record_cloud.py SPEC --like CANDIDATE --cli codex|agy [--model SLUG] [--effort low] \
        --out <data>/evals/recorded/<name>.jsonl [--by image] [--limit N] [--timeout 600] [--min-left 0.10]

The spec's role picks the job: describe-style roles (text and pictures in, JSON out) through either CLI; image-generate (both CLIs make
pictures, saved under <data>/generated/recorded/<out name>/, keyed by prompt); transcribe roles through agy only (Codex cannot hear),
keyed by path and clip, as plain text in one segment (agy's timings drift, so none are recorded).

Before every item it reads how much of the plan is left, through the CLIs themselves and at no cost (`agy -p /usage`; Codex's own app-server,
`account/rateLimits/read`), and stops with exit code 75 once a limit the model draws on has --min-left (default 10 %) or less.

Resumable: items already in --out are skipped, so rerun it after a quota runs out. A quota or rate-limit answer stops the run with exit code 75
(EX_TEMPFAIL) and writes nothing for that item; any other failure is recorded as {"error": ...} in the log on stderr and the item is left out,
so the eval counts it as missing against the cloud model (a recorded job with no record fails). Reads the spec and dataset, never writes the
database; uses no GPU and does not need the service running."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from stratamodels.evals.cloud import (  # the CLI calls live in the package: evals judges use them too
    CLIS,
    QUOTA,
    TEMPFAIL,
    USAGE,
    CallError,
    QuotaError,
    UsageError,
    agy_group,
    call_agy,
    call_codex,
    clean_env,
    image_agy,
    image_codex,
    low_bucket,
    parse_json,
    prompt_of,
    stage_images,
    transcribe_agy,
    usage_agy,
    usage_codex,
)

__all__ = [
    "CLIS",
    "QUOTA",
    "TEMPFAIL",
    "USAGE",
    "CallError",
    "QuotaError",
    "UsageError",
    "agy_group",
    "call_agy",
    "call_codex",
    "clean_env",
    "image_agy",
    "image_codex",
    "low_bucket",
    "parse_json",
    "prompt_of",
    "stage_images",
    "transcribe_agy",
    "usage_agy",
    "usage_codex",
]

IMAGERS = {"codex": image_codex, "agy": image_agy}
TRANSCRIBERS = {"agy": transcribe_agy}  # codex has no audio input (probed 2026-11-12)


def job(params):
    """(system, prompt, images, schema) for a job's params, as llama-server would get them."""
    from stratamodels.runtimes import llama

    body, schema = llama.request(params)
    system = next((m["content"] for m in body["messages"] if m["role"] == "system"), None)
    user = body["messages"][-1]["content"]
    text = "".join(c["text"] for c in user if c.get("type") == "text")
    imgs = params.get("images") or ([params["image"]] if params.get("image") else [])
    return system, text, [str(i) for i in imgs], schema


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


def key_of(params, by):
    """The record's key: one param as a string, or several (by = ["path", "clip"]) as a JSON list, as runtimes/recorded.py looks it up."""
    return json.dumps([params.get(b) for b in by]) if isinstance(by, list) else params.get(by)


def record_image(params, cli, dest, model=None, effort=None, timeout=900, make=None, clock=time.perf_counter):
    """One picture for an image-generate job: saved under dest (the data folder's generated/, so the web app and judges can show it)."""
    t0 = clock()
    r = (make or IMAGERS[cli])(
        params["prompt"],
        dest,
        int(params.get("width") or 1024),
        int(params.get("height") or 1024),
        model=model,
        effort=effort,
        timeout=timeout,
    )
    return dict(r, seconds=round(clock() - t0, 2), cli=cli, cli_model=model or "default")


def record_speech(params, cli, model=None, effort=None, timeout=600, hear=None, clock=time.perf_counter):
    """One transcript for a transcribe job, shaped like Whisper's (segments[].text, which the WER metric joins), without timings."""
    if hear is None and cli not in TRANSCRIBERS:
        raise SystemExit(f"{cli} cannot hear audio; use --cli agy")
    t0 = clock()
    r = (hear or TRANSCRIBERS[cli])(
        params["path"], params.get("clip"), params.get("language"), model=model, effort=effort, timeout=timeout
    )
    if not r["text"]:
        raise CallError(f"{cli} answered nothing")
    out = dict(
        text=r["text"],
        segments=[dict(text=r["text"])],
        seconds=round(clock() - t0, 2),
        cli=cli,
        cli_model=model or "default",
    )
    if params.get("language"):
        out["language"] = params["language"]
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
    ap.add_argument(
        "--by",
        help="the job param(s) the recorded runtime matches on, comma separated (default: image; prompt for image-generate; path,clip for transcribe)",
    )
    ap.add_argument("--limit", type=int, help="record at most this many new items")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per item")
    ap.add_argument(
        "--min-left",
        type=float,
        default=0.10,
        help="stop (exit 75) once any of the plan's limits this model draws on has this share or less left (default 0.10; 0 turns the check off)",
    )
    a = ap.parse_args(argv)

    from stratamodels import settings
    from stratamodels.evals import core as C

    spec = C.load_spec(a.spec)
    cand = spec.candidate(a.like)
    ds = C.load_dataset(spec.dataset)
    out = a.out if os.path.isabs(a.out) else os.path.join(settings.data_dir(), a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    role = spec.role
    if role.startswith("transcribe") and a.cli not in TRANSCRIBERS:
        log(f"{a.cli} cannot hear audio; transcribe through {', '.join(sorted(TRANSCRIBERS))}")
        return 2
    by = a.by or ("prompt" if role == "image-generate" else "path,clip" if role.startswith("transcribe") else "image")
    by = by.split(",") if "," in by else by
    have = done_keys(out)
    todo = [it for it in ds.items if str(key_of(C.job_params(spec, cand, it), by)) not in have]
    if a.limit is not None:
        todo = todo[: a.limit]
    log(
        f"{spec.name}: {len(ds.items)} items, {len(have)} recorded, {len(todo)} to do with {a.cli} {a.model or '(default model)'}"
    )
    failed = 0
    for n, it in enumerate(todo, 1):
        params = {k: v for k, v in C.job_params(spec, cand, it).items() if not k.startswith("__")}
        key = key_of(params, by)
        if a.min_left > 0:
            try:
                low = low_bucket(a.cli, a.model, a.min_left)
            except (UsageError, OSError, subprocess.TimeoutExpired) as e:
                log(f"cannot read the remaining allowance, stopping (--min-left 0 skips the check): {e}")
                return 1
            if low:
                log(
                    f"allowance: {low[0]} has {low[1]:.0%} left (at most {a.min_left:.0%}), resets {low[2]}; stopping after {n - 1} new items, rerun then to resume"
                )
                return TEMPFAIL
        try:
            if role == "image-generate":
                stem = os.path.splitext(os.path.basename(out))[0]
                dest = os.path.join(
                    settings.data_dir(),
                    "generated",
                    "recorded",
                    stem,
                    hashlib.sha256(str(key).encode()).hexdigest()[:16],
                )
                o = record_image(params, a.cli, dest, a.model, a.effort, a.timeout)
            elif role.startswith("transcribe"):
                o = record_speech(params, a.cli, a.model, a.effort, a.timeout)
            else:
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
        ok = (
            o["path"]
            if "path" in o
            else "json ok"
            if o.get("json") is not None
            else ("no json" if "json" in o else "text")
        )
        log(f"[{n}/{len(todo)}] {it['id']}: {o['seconds']} s, {ok}")
    log(f"done: {len(todo) - failed} recorded, {failed} failed (left out: they count as missing)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
