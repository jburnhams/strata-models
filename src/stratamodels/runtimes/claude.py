"""The Claude API (runtime `claude-api`), through the official anthropic SDK, with the key from keys.py (ANTHROPIC_API_KEY) and every call
counted against the Claude budget (spend.py): the call's input is counted first (count_tokens, free), its worst case (that input plus
max_tokens of output) must fit what the budget has left, and the real cost is written to the ledger after.

options (models.toml):
  model   the Claude model id (claude-haiku-5-5)
  price   USD per million tokens (spend.py): a model without one is never loaded, so nothing runs unpriced
  effort  the default effort when a job gives none
Params and result: runtimes/cloud.py. A refusal (stop_reason "refusal") fails the job; what it cost is still counted. Uses no GPU."""

import time

from stratamodels import keys, spend
from stratamodels.runtimes import RunError, cloud

PROVIDER = "claude"


def load(model, models_dir, client=None):
    o = model.options
    if not o.get("model"):
        raise RunError(f"model {model.key} names no Claude model id (options.model)")
    if not o.get("price"):
        raise RunError(f"model {model.key} has no price in models.toml: its spend could not be counted")
    if client is None:
        import anthropic

        try:
            key = keys.get(PROVIDER)
        except keys.MissingKey as e:
            raise RunError(str(e))
        client = anthropic.Anthropic(api_key=key, max_retries=2, timeout=600.0)
    return dict(key=model.key, model=o["model"], price=o["price"], effort=o.get("effort"), client=client)


def _content(job):
    return [
        {"type": "image", "source": {"type": "base64", "media_type": m, "data": b}} for _, m, b in job["images"]
    ] + [{"type": "text", "text": job["text"]}]


def _usage(u):
    return dict(
        tokens_in=int(getattr(u, "input_tokens", 0) or 0),
        tokens_out=int(getattr(u, "output_tokens", 0) or 0),
        cache_read=int(getattr(u, "cache_read_input_tokens", 0) or 0),
        cache_write=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
    )


def run(handle, params):
    import anthropic

    job = cloud.parse(params)
    client = handle["client"]
    req = dict(model=handle["model"], messages=[{"role": "user", "content": _content(job)}])
    if job["system"]:
        req["system"] = job["system"]
    oc = {}
    if job["effort"] or handle["effort"]:
        oc["effort"] = job["effort"] or handle["effort"]
    if job["schema"]:
        oc["format"] = {"type": "json_schema", "schema": cloud.portable(job["schema"])}
    log = dict(cloud.logged(job), effort=oc.get("effort"))
    try:
        n_in = client.messages.count_tokens(**req).input_tokens
    except anthropic.APIError as e:
        raise RunError(keys.redact(f"Claude API refused to count the request: {e}"))
    ticket = spend.reserve(PROVIDER, spend.worst(handle["price"], n_in, job["max_tokens"]))
    t0 = time.perf_counter()
    try:
        r = client.messages.create(max_tokens=job["max_tokens"], **req, **({"output_config": oc} if oc else {}))
    except anthropic.APIStatusError as e:  # a refused request (400, 401, 429, 5xx) is not billed
        spend.record(
            ticket,
            handle["model"],
            ok=False,
            error=keys.redact(str(e))[:2000],
            seconds=round(time.perf_counter() - t0, 2),
            model_key=handle["key"],
            request=log,
        )
        raise RunError(keys.redact(f"Claude API {e.status_code}: {e.message}"))
    except anthropic.APIConnectionError as e:
        spend.record(
            ticket,
            handle["model"],
            ok=False,
            error=keys.redact(str(e))[:2000],
            seconds=round(time.perf_counter() - t0, 2),
            model_key=handle["key"],
            request=log,
        )
        raise RunError(keys.redact(f"could not reach the Claude API: {e}"))
    seconds = round(time.perf_counter() - t0, 2)
    u = _usage(r.usage)
    usd = spend.cost(handle["price"], **u)
    text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
    refused = r.stop_reason == "refusal"
    spend.record(
        ticket,
        r.model or handle["model"],
        usd,
        ok=not refused,
        error="refused" if refused else None,
        seconds=seconds,
        model_key=handle["key"],
        finish=r.stop_reason,
        request=log,
        response=text,
        **u,
    )
    if refused:
        cat = getattr(getattr(r, "stop_details", None), "category", None)
        raise RunError(f"Claude declined this request (refusal{', ' + cat if cat else ''})")
    return dict(
        cloud.answer(job, text),
        tokens_in=u["tokens_in"] + u["cache_read"] + u["cache_write"],
        tokens_out=u["tokens_out"],
        cost_usd=round(usd, 6),
        seconds=seconds,
        model=handle["key"],
        provider=PROVIDER,
        finish=r.stop_reason,
    )


def unload(handle):
    c = handle.get("client")
    if c is not None and hasattr(c, "close"):
        c.close()
    handle.clear()
