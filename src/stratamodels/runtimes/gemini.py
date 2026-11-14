"""Google AI Studio (runtime `gemini-api`): Gemini through the Generative Language API (generateContent), with two keys tried in order
(Jonathan, 2026-11-14): GEMINI_FREE_API_KEY, the free tier (rate-limited, no budget), then GEMINI_API_KEY, the paid key (a small monthly
budget, spend.py). The free tier's 429 (its per-minute or per-day quota, or a model it does not serve: Argon has a free limit of 0) or
503 (overloaded, often on the free tier) moves the call on to the paid key; the paid key's call is counted first (countTokens, free) and refused if its worst case does not fit the budget.
For Gemini work the subscription CLI (agy, scripts/evals/record_cloud.py) comes before either key: this runtime is the fallback.

Note: Google may use what is sent on the free tier to improve its products (AI Studio terms): give private pictures a model with
keys = ["gemini"] only.

options (models.toml):
  model   the Gemini model id (gemini-3.8-flash)
  price   USD per million tokens on the paid key (spend.py; a list with `from` dates when Google has announced a change)
  keys    which keys to try, in order (default ["gemini-free", "gemini"])
Params and result: runtimes/cloud.py; effort sets thinkingLevel. Uses no GPU."""

import time

from stratamodels import keys, spend
from stratamodels.runtimes import RunError, cloud

BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
KEYS = ("gemini-free", "gemini")


def load(model, models_dir):
    o = model.options
    if not o.get("model"):
        raise RunError(f"model {model.key} names no Gemini model id (options.model)")
    order = list(o.get("keys") or KEYS)
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


def run(handle, params, post=cloud.post):
    job = cloud.parse(params)
    b = body(job)
    log = cloud.logged(job)
    url = BASE + handle["model"]
    errors = []
    for provider in handle["keys"]:
        if not keys.have(provider):
            continue
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
                continue  # the free tier's quota is spent, it does not serve this model, or it is overloaded: the paid key
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
            cloud.answer(job, text),
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
