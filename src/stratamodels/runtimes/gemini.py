"""Google AI Studio (runtime `gemini-api`): Gemini through the Generative Language API (generateContent) on GEMINI_FREE_API_KEY, the free
tier (rate-limited, no budget), or GEMINI_API_KEY, the paid key (a small monthly budget, spend.py), as the model's `keys` say.

The paid key is a deliberate last resort (Jonathan, 2026-11-15: after the free key and agy): a model uses it only when its `keys` name it.
Until then the free tier's 429 or 503 was a silent fall-over to the paid key inside the same call, and a burst of calls (an eval on the CPU
lane, four at once) ran into the free tier's per-minute limit and spent on the paid key almost every time (62 of 64 calls on 2026-11-15).
Now a free-tier 429 or 503 that names a short wait (RetryInfo's retryDelay, or Retry-After) is waited out and tried again on the free key,
a few times; after that, or for a daily quota, the call fails and says so. A model that does list both keys (keys = ["gemini-free",
"gemini"]) still moves on to the paid key, and that choice is in models.toml for everyone to see. The paid key's call is counted first
(countTokens, free) and refused if its worst case does not fit the budget. For Gemini work the subscription CLI (agy,
scripts/evals/record_cloud.py) comes before either key.

Note: Google may use what is sent on the free tier to improve its products (AI Studio terms): give private pictures a model with
keys = ["gemini"] only.

options (models.toml):
  model   the Gemini model id (gemini-3.8-flash)
  price   USD per million tokens on the paid key (spend.py; a list with `from` dates when Google has announced a change)
  keys    which keys to try, in order: "gemini-free" (the default, alone), "gemini" (the paid key)
Params and result: runtimes/cloud.py; effort sets thinkingLevel. Uses no GPU."""

import json
import re
import time

from stratamodels import keys, spend
from stratamodels.runtimes import RunError, cloud

BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
KEYS = ("gemini-free", "gemini")
DEFAULT_KEYS = ("gemini-free",)  # the paid key only where a model names it
FREE_WAITS = 3  # times a free-tier call is tried again after the wait it names
MAX_WAIT_S = 65.0  # a longer wait is a daily quota, not a per-minute one: fail instead


_day_spent_until = {}  # Gemini model id -> epoch seconds when its free daily quota comes back


def _per_day(e):
    """A 429 for the free tier's daily quota ("...requests per day...", quotaId ...PerDay...) rather than its per-minute one."""
    b = str(getattr(e, "body", "") or e).lower()
    return "per day" in b or "perday" in b


def _next_pacific_midnight(now):
    import datetime
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("America/Los_Angeles")  # AI Studio's daily quotas reset at midnight Pacific
    t = datetime.datetime.fromtimestamp(now, tz)
    return (datetime.datetime.combine(t.date() + datetime.timedelta(days=1), datetime.time(), tz)).timestamp()


def _mark_day_spent(model, now=None):
    _day_spent_until[model] = _next_pacific_midnight(time.time() if now is None else now)


def _day_spent(model, now=None):
    until = _day_spent_until.get(model)
    return until is not None and (time.time() if now is None else now) < until


def free_wait(e):
    """Seconds the free tier asks us to wait before trying again (RetryInfo's retryDelay, else Retry-After), or None."""
    try:
        for d in json.loads(e.body).get("error", {}).get("details", []):
            m = re.fullmatch(r"([\d.]+)s", str(d.get("retryDelay", "")))
            if m:
                return float(m.group(1))
    except (ValueError, AttributeError):
        pass
    return getattr(e, "retry_after", None)


def load(model, models_dir):
    o = model.options
    if not o.get("model"):
        raise RunError(f"model {model.key} names no Gemini model id (options.model)")
    order = list(o.get("keys") or DEFAULT_KEYS)
    if any(k not in KEYS for k in order):
        raise RunError(f"keys are among {list(KEYS)}")
    if "gemini" in order and not o.get("price"):
        raise RunError(f"model {model.key} has no price in models.toml: its spend on the paid key could not be counted")
    if not any(keys.have(k) for k in order):
        raise RunError(
            f"no Google AI Studio key: put {' or '.join(keys.NAMES[k] for k in order)} in {keys.path()} (chmod 600)"
        )
    return dict(key=model.key, model=o["model"], price=o.get("price"), keys=order)


def body(job):
    parts = [{"inline_data": {"mime_type": m, "data": b}} for _, m, b in job["images"]] + [{"text": job["text"]}]
    gc = dict(maxOutputTokens=job["max_tokens"])
    if job["effort"]:
        gc["thinkingConfig"] = {"thinkingLevel": job["effort"]}
    if job["schema"]:
        gc.update(responseMimeType="application/json", responseJsonSchema=cloud.portable(job["schema"]))
    b = dict(contents=[{"role": "user", "parts": parts}], generationConfig=gc)
    if job["system"]:
        b["systemInstruction"] = {"parts": [{"text": job["system"]}]}
    return b


def _usage(r):
    u = r.get("usageMetadata") or {}
    cached = int(u.get("cachedContentTokenCount") or 0)
    return dict(
        tokens_in=max(int(u.get("promptTokenCount") or 0) - cached, 0),
        tokens_out=int(u.get("candidatesTokenCount") or 0)
        + int(u.get("thoughtsTokenCount") or 0),  # thinking is billed as output
        cache_read=cached,
    )


def run(handle, params, post=cloud.post, sleep=time.sleep):
    job = cloud.parse(params)
    b = body(job)
    log = cloud.logged(job)
    url = BASE + handle["model"]
    errors = []
    waits = 0
    queue = [k for k in handle["keys"] if keys.have(k)]
    while queue:
        provider = queue.pop(0)
        if provider == "gemini-free" and _day_spent(handle["model"]):
            if "gemini" in queue:
                continue
            raise RunError(
                f"Google AI Studio free tier: {handle['model']}'s daily quota is spent (until midnight Pacific). The paid key is not "
                f"tried for {handle['key']}: use agy, or a model whose keys name the paid key"
            )
        hdr = {"x-goog-api-key": keys.get(provider)}
        worst = 0.0
        if spend.PROVIDERS[provider]["budgeted"]:
            try:
                n_in = int(
                    post(
                        url + ":countTokens",
                        {"generateContentRequest": dict(model="models/" + handle["model"], **b)},
                        hdr,
                    ).get("totalTokens")
                    or 0
                )
            except cloud.HttpError as e:
                raise RunError(f"Google AI Studio refused to count the request ({e.status}): {e.body[:300]}")
            worst = spend.worst(handle["price"], n_in, job["max_tokens"])
        ticket = spend.reserve(provider, worst)
        t0 = time.perf_counter()
        try:
            r = post(url + ":generateContent", b, hdr)
        except (cloud.HttpError, RunError) as e:
            status = getattr(e, "status", None)
            spend.record(
                ticket,
                handle["model"],
                ok=False,
                error=str(e)[:2000],
                seconds=round(time.perf_counter() - t0, 2),
                model_key=handle["key"],
                request=log,
            )
            errors.append(f"{provider}: {str(e)[:300]}")
            if provider == "gemini-free" and status in (429, 500, 503):
                wait = free_wait(e)
                daily = status == 429 and _per_day(e)
                if daily:
                    _mark_day_spent(handle["model"])  # its next calls skip the free key until Google's day ends
                elif waits < FREE_WAITS and wait is not None and wait <= MAX_WAIT_S:
                    waits += 1
                    sleep(wait + 1.0)  # a per-minute limit or a short overload: the free key again, not the paid one
                    queue.insert(0, provider)
                    continue
                if "gemini" in queue:
                    continue  # this model names the paid key after the free one (models.toml)
                raise RunError(
                    f"Google AI Studio free tier ({status}): {str(e)[:400]}. The paid key is not tried for {handle['key']}: "
                    "use agy, or a model whose keys name the paid key"
                )
            raise RunError(f"Google AI Studio ({provider}): {str(e)[:500]}")
        seconds = round(time.perf_counter() - t0, 2)
        u = _usage(r)
        price = handle["price"] if spend.PROVIDERS[provider]["budgeted"] else None
        usd = spend.cost(price, **u)
        cand = (r.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought"))
        finish = cand.get("finishReason") or (r.get("promptFeedback") or {}).get("blockReason")
        blocked = not cand or finish in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "RECITATION")
        spend.record(
            ticket,
            r.get("modelVersion") or handle["model"],
            usd,
            ok=not blocked,
            error="blocked" if blocked else None,
            seconds=seconds,
            model_key=handle["key"],
            finish=finish,
            request=log,
            response=text,
            **u,
        )
        if blocked:
            raise RunError(f"Gemini declined this request ({finish})")
        return dict(
            cloud.answer(job, text, finish),
            tokens_in=u["tokens_in"] + u["cache_read"],
            tokens_out=u["tokens_out"],
            cost_usd=round(usd, 6),
            seconds=seconds,
            model=handle["key"],
            provider=provider,
            finish=finish,
        )
    raise RunError("no Google AI Studio key could take this call: " + ("; ".join(errors) or "none is set"))


def unload(handle):
    handle.clear()
