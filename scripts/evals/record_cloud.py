"""Record a cloud model's answers to an eval's items through a subscription CLI, so the model can stand in the eval as a `recorded` candidate.

Jonathan's Google AI Pro and ChatGPT Plus plans have no API credits: the only sanctioned way to call their models from a script is each vendor's
own agent CLI run headless (`codex exec` for OpenAI, `agy -p` for Antigravity; Gemini CLI no longer serves consumer plans since 2026-06-18).
Never route those CLIs' sign-in tokens through anything else (Google disables accounts for it). This script calls the official binaries only.

It takes a LOCAL candidate of the spec as the template, so the cloud model gets exactly the same job: the same preset (prompt and JSON schema),
the same context, the same pictures (`runtimes.llama.request` builds the prompt, as it does for llama-server). Each answer is appended to a JSON
lines file of {"key": <the item's `by` param>, "item": <id>, "output": {...}} that the `recorded` runtime replays (see models.toml).

    uv run python scripts/evals/record_cloud.py SPEC --like CANDIDATE --cli codex|agy [--model SLUG] [--effort low] \
        --out <data>/evals/recorded/<name>.jsonl [--by image] [--limit N] [--timeout 600]

Before every item it reads how much of the plan is left, through the CLIs themselves and at no cost (`agy -p /usage`; Codex's own app-server,
`account/rateLimits/read`), and stops with exit code 75 once a limit the model draws on has --min-left (default 10 %) or less.

Resumable: items already in --out are skipped, so rerun it after a quota runs out. A quota or rate-limit answer stops the run with exit code 75
(EX_TEMPFAIL) and writes nothing for that item; any other failure is recorded as {"error": ...} in the log on stderr and the item is left out,
so the eval counts it as missing against the cloud model (a recorded job with no record fails). Reads the spec and dataset, never writes the
database; uses no GPU and does not need the service running."""

import argparse
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
    CallError,
    QuotaError,
    call_agy,
    call_codex,
    clean_env,
    parse_json,
    prompt_of,
    stage_images,
)

__all__ = [
    "CLIS",
    "QUOTA",
    "TEMPFAIL",
    "CallError",
    "QuotaError",
    "call_agy",
    "call_codex",
    "clean_env",
    "parse_json",
    "prompt_of",
    "stage_images",
]


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
    import datetime

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
