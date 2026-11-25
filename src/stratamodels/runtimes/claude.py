"""The Claude API (runtime `claude-api`), through the official anthropic SDK, with the key from keys.py (ANTHROPIC_API_KEY) and every call
counted against the Claude budget (spend.py): the call's input is counted first (count_tokens, free), its worst case (that input plus
max_tokens of output) must fit what the budget has left, and the real cost is written to the ledger after.

options (models.toml):
  model   the Claude model id (claude-haiku-5-5)
  price   USD per million tokens (spend.py): a model without one is never loaded, so nothing runs unpriced
  effort  the default effort when a job gives none
Params and result: runtimes/cloud.py. A refusal (stop_reason "refusal") fails the job; what it cost is still counted. Uses no GPU.
A job with `batch` in its params goes through the Message Batches API instead (batch_* below, driven by batches.py): half the price, an
answer within a day."""

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


def _request(handle, job):
    """(request, output_config, log) of one job's call: what messages.create, count_tokens and a batch's entry share."""
    req = dict(model=handle["model"], messages=[{"role": "user", "content": _content(job)}])
    if job["system"]:
        req["system"] = job["system"]
    oc = {}
    if job["effort"] or handle["effort"]:
        oc["effort"] = job["effort"] or handle["effort"]
    if job["schema"]:
        oc["format"] = {"type": "json_schema", "schema": cloud.portable(job["schema"])}
    return req, oc, dict(cloud.logged(job), effort=oc.get("effort"))


def _failed(ticket, handle, e, t0, log):
    spend.record(
        ticket,
        handle["model"],
        ok=False,
        error=keys.redact(str(e))[:2000],
        seconds=round(time.perf_counter() - t0, 2),
        model_key=handle["key"],
        request=log,
    )


def _finish(handle, job, log, r, ticket, seconds, batch=None):
    """Record a call's answer `r` (a Message) against its ticket and return the job's result; a refusal or an answer that is not one raises
    RunError, after the row is written. `batch`: the vendor's batch id it came in, priced at the batch discount."""
    u = _usage(r.usage)
    usd = spend.cost(handle["price"], **u, batch=batch is not None)
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
        batch=batch,
        **u,
    )
    if refused:
        cat = getattr(getattr(r, "stop_details", None), "category", None)
        raise RunError(f"Claude declined this request (refusal{', ' + cat if cat else ''})")
    return dict(
        cloud.answer(job, text, r.stop_reason),
        tokens_in=u["tokens_in"] + u["cache_read"] + u["cache_write"],
        tokens_out=u["tokens_out"],
        cost_usd=round(usd, 6),
        seconds=seconds,
        model=handle["key"],
        provider=PROVIDER,
        finish=r.stop_reason,
        **({"batch": batch} if batch is not None else {}),
    )


def run(handle, params):
    import anthropic

    job = cloud.parse(params)
    client = handle["client"]
    req, oc, log = _request(handle, job)
    try:
        n_in = client.messages.count_tokens(**req).input_tokens
    except anthropic.APIError as e:
        raise RunError(keys.redact(f"Claude API refused to count the request: {e}"))
    ticket = spend.reserve(PROVIDER, spend.worst(handle["price"], n_in, job["max_tokens"]))
    t0 = time.perf_counter()
    try:
        r = client.messages.create(max_tokens=job["max_tokens"], **req, **({"output_config": oc} if oc else {}))
    except anthropic.APIStatusError as e:  # a refused request (400, 401, 429, 5xx) is not billed
        _failed(ticket, handle, e, t0, log)
        raise RunError(keys.redact(f"Claude API {e.status_code}: {e.message}"))
    except anthropic.APIConnectionError as e:
        _failed(ticket, handle, e, t0, log)
        raise RunError(keys.redact(f"could not reach the Claude API: {e}"))
    return _finish(handle, job, log, r, ticket, round(time.perf_counter() - t0, 2))


# ---- the Message Batches API (batches.py drives these) -----------------------------------------------------------------------------------
# Half price, answered within 24 hours (most within one). The same request as run() sends, as one entry of a batch. Its worst case is bounded
# without count_tokens (which would send every picture twice): a token is at least one byte of text, and a picture is at most IMAGE_TOKENS
# (Claude scales a picture down to fit, and LONG_EDGE x LONG_EDGE / 750 pixels a token is under 9 000), plus OVERHEAD for the framing and the
# structured-output instructions. The real usage, priced at the batch rate, is what the ledger records.
IMAGE_TOKENS = 10_000
OVERHEAD = 2_000


def _bound(job, body):
    import json

    text = len((job["text"] or "").encode()) + len((job["system"] or "").encode())
    fmt = body.get("output_config", {}).get("format")
    schema = len(json.dumps(fmt)) * 2 if fmt else 0
    return text + schema + IMAGE_TOKENS * len(job["images"]) + OVERHEAD


def batch_request(handle, params):
    """(the entry's params, its worst case in USD at the batch price, the request as the ledger logs it: what batch_answer reads the answer
    by, kept with the batch) for one job. RunError for params that could not be sent."""
    job = cloud.parse(params)
    req, oc, log = _request(handle, job)
    body = dict(max_tokens=job["max_tokens"], **req, **({"output_config": oc} if oc else {}))
    return body, spend.worst(handle["price"], _bound(job, body), job["max_tokens"], batch=True), log


def batch_submit(handle, entries):
    """Send [(custom_id, params)] as one batch; returns the vendor's batch id. RunError when it is refused (nothing is billed)."""
    import anthropic

    try:
        b = handle["client"].messages.batches.create(requests=[{"custom_id": c, "params": p} for c, p in entries])
    except anthropic.APIStatusError as e:
        raise RunError(keys.redact(f"Claude API {e.status_code} refused the batch: {e.message}"))
    except anthropic.APIConnectionError as e:
        raise RunError(keys.redact(f"could not reach the Claude API: {e}"))
    return b.id


COUNTS = ("processing", "succeeded", "errored", "canceled", "expired")


def batch_poll(handle, vendor_id):
    """(processing status: in_progress | canceling | ended, {processing, succeeded, errored, canceled, expired}). Lets an APIError through:
    a poll that fails is tried again at the next one."""
    b = handle["client"].messages.batches.retrieve(vendor_id)
    rc = b.request_counts
    return b.processing_status, {k: int(getattr(rc, k, 0) or 0) for k in COUNTS}


def batch_results(handle, vendor_id):
    """(custom_id, outcome) of an ended batch, in any order; an outcome's type is succeeded (its message), errored, canceled or expired."""
    for r in handle["client"].messages.batches.results(vendor_id):
        yield r.custom_id, r.result


def batch_answer(handle, log, outcome, ticket, vendor_id, seconds):
    """A job's result from its outcome in the batch, recorded against its ticket; RunError for one that failed (recorded too, at no cost: the
    vendor bills only the entries that succeed). `log` is the job's request as batch_request logged it when the batch was sent (a preset
    edited since does not change how the answer is read), `handle` the model as it was then (its price)."""
    log = dict(log, batch=vendor_id)
    kind = getattr(outcome, "type", None)
    if kind != "succeeded":
        err = getattr(getattr(outcome, "error", None), "error", None)
        why = (
            f"{getattr(err, 'type', 'error')}: {getattr(err, 'message', '')}"
            if kind == "errored"
            else f"the batch entry {kind}"
        )
        spend.record(
            ticket,
            handle["model"],
            ok=False,
            error=keys.redact(why)[:2000],
            seconds=seconds,
            model_key=handle["key"],
            request=log,
            batch=vendor_id,
        )
        raise RunError(keys.redact(f"Claude batch {vendor_id}: {why}"))
    return _finish(handle, dict(schema=log.get("schema")), log, outcome.message, ticket, seconds, batch=vendor_id)


def unload(handle):
    c = handle.get("client")
    if c is not None and hasattr(c, "close"):
        c.close()
    handle.clear()
