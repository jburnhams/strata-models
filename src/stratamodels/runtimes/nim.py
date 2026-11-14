"""NVIDIA NIM (runtime `nvidia-nim`): the hosted models of build.nvidia.com through their OpenAI-compatible API
(https://integrate.api.nvidia.com/v1), key NVIDIA_API_KEY (keys.py). Free and rate-limited (Jonathan, 2026-11-14): no budget, but every
call is still written to the ledger (spend.py) so the Cloud page shows what was sent. A 429 waits and tries again, up to three times.

options (models.toml):
  model     the NIM model id (nvidia/nemotron-3.5-super-vl-120b-a12b)
  base_url  another OpenAI-compatible endpoint (default the one above)
  extra     fields added to every request body (e.g. {chat_template_kwargs = {enable_thinking = false}})
Params and result: runtimes/cloud.py. Uses no GPU here."""

import time

from stratamodels import keys, spend
from stratamodels.runtimes import RunError, cloud

PROVIDER = "nim"
BASE = "https://integrate.api.nvidia.com/v1"
TRIES = 3


def load(model, models_dir):
    o = model.options
    if not o.get("model"):
        raise RunError(f"model {model.key} names no NIM model id (options.model)")
    try:
        keys.get(
            PROVIDER
        )  # checked now, read again per call: a key put in the file later is picked up without a restart
    except keys.MissingKey as e:
        raise RunError(str(e))
    return dict(
        key=model.key,
        model=o["model"],
        url=(o.get("base_url") or BASE).rstrip("/") + "/chat/completions",
        extra=dict(o.get("extra") or {}),
    )


def body(handle, job):
    content = [{"type": "image_url", "image_url": {"url": f"data:{m};base64,{b}"}} for _, m, b in job["images"]] + [
        {"type": "text", "text": job["text"]}
    ]
    msgs = ([{"role": "system", "content": job["system"]}] if job["system"] else []) + [
        {"role": "user", "content": content if job["images"] else job["text"]}
    ]
    b = dict(model=handle["model"], messages=msgs, max_tokens=job["max_tokens"], **handle["extra"])
    if job["schema"]:
        b["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": job["schema"], "strict": True},
        }
    return b


def run(handle, params, post=cloud.post, sleep=time.sleep):
    job = cloud.parse(params)
    b = body(handle, job)
    log = cloud.logged(job)
    ticket = spend.reserve(PROVIDER, 0.0)
    t0 = time.perf_counter()
    for attempt in range(TRIES):
        try:
            r = post(handle["url"], b, {"authorization": f"Bearer {keys.get(PROVIDER)}"})
            break
        except cloud.HttpError as e:
            if e.status == 429 and attempt + 1 < TRIES:
                sleep(min(e.retry_after or 5.0 * (attempt + 1), 60.0))
                continue
            spend.record(
                ticket,
                handle["model"],
                ok=False,
                error=str(e)[:2000],
                seconds=round(time.perf_counter() - t0, 2),
                model_key=handle["key"],
                request=log,
            )
            raise RunError(f"NVIDIA NIM {e.status}: {e.body[:300]}")
        except (RunError, keys.MissingKey) as e:
            spend.record(
                ticket,
                handle["model"],
                ok=False,
                error=str(e)[:2000],
                seconds=round(time.perf_counter() - t0, 2),
                model_key=handle["key"],
                request=log,
            )
            raise RunError(str(e))
    seconds = round(time.perf_counter() - t0, 2)
    ch = (r.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    text = msg.get("content") or ""
    u = r.get("usage") or {}
    tin, tout = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
    spend.record(
        ticket,
        r.get("model") or handle["model"],
        0.0,
        tokens_in=tin,
        tokens_out=tout,
        seconds=seconds,
        model_key=handle["key"],
        finish=ch.get("finish_reason"),
        request=log,
        response=text,
    )
    return dict(
        cloud.answer(job, text),
        tokens_in=tin,
        tokens_out=tout,
        cost_usd=0.0,
        seconds=seconds,
        model=handle["key"],
        provider=PROVIDER,
        finish=ch.get("finish_reason"),
    )


def unload(handle):
    handle.clear()
