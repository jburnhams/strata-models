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
