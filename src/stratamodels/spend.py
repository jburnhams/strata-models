"""What the cloud API keys spend, and the budgets that stop them: every call is a row of `api_calls` (the ledger, kept for good), priced from
the model's `price` in models.toml at the time of the call.

Providers (one per key, see keys.py):
  claude        Claude API (paid): budget `claude_budget_usd`, rollover `claude_rollover`
  gemini        Google AI Studio, the paid key: budget `gemini_budget_usd`, rollover `gemini_rollover`; only after gemini-free
  gemini-free   Google AI Studio, the free tier: rate-limited, no budget
  nim           NVIDIA NIM (build.nvidia.com): free, rate-limited, no budget
None of the vendors shows a key's remaining credit to that key (checked 2026-11-14), so the ledger is the only count: a budget is what the
key may spend in the period, and the period runs from the last rollover (the rollover day, repeated monthly) to the next. With no rollover
date the budget is one pot that only goes down.

Before a call the runtime asks `reserve(provider, worst_usd)`: the call's worst case (its input, plus max_tokens of output at the output
price) must fit what is left once the calls still running have reserved theirs, else it is refused (OverBudget) and nothing is sent. After
it, `record(...)` writes the row with the real usage and frees the reservation. A failed call is recorded too (its cost is what the vendor
bills for it: nothing for a refused request).

A price (USD per million tokens) is a table under the model's options:
  price = { input = 0.10, output = 0.50, cache_read = 0.01, cache_write = 0.125, over = 100000, input_over = 0.50, output_over = 2.50 }
cache_read defaults to a tenth of input, cache_write to 1.25 x input; `over`: above that many prompt tokens the *_over prices apply. A list
of such tables with `from = "YYYY-MM-DD"` gives prices that change on a date (the latest one that has started applies)."""

import calendar
import contextlib
import contextvars
import datetime
import threading
import time

from stratamodels import settings
from stratamodels.runtimes import RunError

PROVIDERS = {
    "claude": dict(label="Claude API", budgeted=True),
    "gemini": dict(label="Google AI Studio (paid key)", budgeted=True),
    "gemini-free": dict(label="Google AI Studio (free tier)", budgeted=False),
    "nim": dict(label="NVIDIA NIM (free)", budgeted=False),
}
WARN_AT = (0.5, 0.8, 1.0)  # shares of a budget at which the web app warns

_store = None
clock = time.time  # tests set it
_lock = threading.Lock()  # guards _reserved
_budget = (
    threading.RLock()
)  # one boundary for check-then-reserve and reservation-to-ledger, so two calls can't both fit the same budget
_reserved = {}  # provider -> {ticket: usd} of calls under way
_ticket = [0]
_job = contextvars.ContextVar("stratamodels_spend_job", default=(None, None))


class OverBudget(RunError):
    """The call would take a key past its budget: it is not made."""


def bind(store):
    """Count against this store's ledger from now on (None: no ledger, and every budgeted call is refused)."""
    global _store
    _store = store


@contextlib.contextmanager
def job(app, job_id):
    """The scheduler runs a job inside this, so the rows its calls write name the app and the job."""
    tok = _job.set((app, job_id))
    try:
        yield
    finally:
        _job.reset(tok)


# ---- prices ------------------------------------------------------------------------------------------------------------------------------


def price_now(price, now=None):
    """The price table that applies at `now` (epoch seconds) from a model's `price` option (one table, or a list with `from` dates)."""
    if not price:
        return None
    if isinstance(price, dict):
        return price
    day = datetime.datetime.fromtimestamp(now if now is not None else clock(), datetime.UTC).date().isoformat()
    started = [p for p in price if str(p.get("from", "")) <= day]
    return max(started, key=lambda p: str(p.get("from", ""))) if started else None


def cost(price, tokens_in=0, tokens_out=0, cache_read=0, cache_write=0, now=None):
    """USD for a call's usage at a model's price (tokens_in: uncached input). 0 for a model with no price (a free provider)."""
    p = price_now(price, now)
    if not p:
        return 0.0
    prompt = tokens_in + cache_read + cache_write
    over = p.get("over") and prompt > float(p["over"])
    pin = float(p.get("input_over", p["input"]) if over else p["input"])
    pout = float(p.get("output_over", p["output"]) if over else p["output"])
    pread = float(p.get("cache_read", pin / 10))
    pwrite = float(p.get("cache_write", pin * 1.25))
    return (tokens_in * pin + tokens_out * pout + cache_read * pread + cache_write * pwrite) / 1e6


def worst(price, tokens_in, max_tokens, now=None):
    """The most a call can cost: all of its input uncached and every one of max_tokens produced."""
    return cost(price, tokens_in=tokens_in, tokens_out=max_tokens, now=now)


# ---- periods -----------------------------------------------------------------------------------------------------------------------------


def _shift(d, months, day):
    y, m = divmod(d.year * 12 + d.month - 1 + months, 12)
    return datetime.date(y, m + 1, min(day, calendar.monthrange(y, m + 1)[1]))


def period(rollover, now=None):
    """(start, next rollover) as UTC epoch seconds for a rollover date repeated monthly on its day; (0, None) without one."""
    if not rollover:
        return 0.0, None
    anchor = datetime.date.fromisoformat(rollover)
    today = datetime.datetime.fromtimestamp(now if now is not None else clock(), datetime.UTC).date()
    months = (today.year - anchor.year) * 12 + today.month - anchor.month
    start = _shift(anchor, months, anchor.day)
    if start > today:
        start = _shift(anchor, months - 1, anchor.day)
    nxt = _shift(start, 1, anchor.day)

    def ts(d):
        return datetime.datetime(d.year, d.month, d.day, tzinfo=datetime.UTC).timestamp()

    return ts(start), ts(nxt)


# ---- the ledger --------------------------------------------------------------------------------------------------------------------------


def state(provider, now=None):
    """One provider's budget as it stands: budget, spent and calls this period, reserved (calls under way), left, period, warn level."""
    now = now if now is not None else clock()
    info = PROVIDERS[provider]
    usd, rollover = settings.budget(provider) if info["budgeted"] else (None, None)
    start, nxt = period(rollover, now)
    spent, calls = _store.api_spent(provider, start) if _store is not None else (0.0, 0)
    with _lock:
        reserved = sum(_reserved.get(provider, {}).values())
    out = dict(
        provider=provider,
        label=info["label"],
        budgeted=info["budgeted"],
        unit="usd",
        budget=usd,
        rollover=rollover,
        period_start=start or None,
        next_rollover=nxt,
        spent=round(spent, 6),
        calls=calls,
        reserved=round(reserved, 6),
        left=None,
        used_share=None,
        warn=None,
    )
    if info["budgeted"]:
        left = max((usd or 0.0) - spent - reserved, 0.0)
        share = spent / usd if usd else None
        out.update(
            left=round(left, 6),
            used_share=None if share is None else round(share, 4),
            warn=1.0 if share is None else max([w for w in WARN_AT if share >= w], default=None),
        )
    return out


def summary(now=None):
    """Every provider's state (what GET /api/spend and the routing read)."""
    return [state(p, now) for p in PROVIDERS]


def reserve(provider, worst_usd, now=None):
    """Hold `worst_usd` of a provider's budget for a call about to be made; returns a ticket for `record`. Raises OverBudget if it does not fit."""
    with _budget:
        if PROVIDERS[provider]["budgeted"]:
            if _store is None:
                raise OverBudget(f"no spend ledger: {PROVIDERS[provider]['label']} calls are refused")
            st = state(provider, now)
            if not st["budget"]:
                raise OverBudget(
                    f"no budget is set for {st['label']}: set {provider}_budget_usd (and {provider}_rollover) first"
                )
            if worst_usd > st["left"]:
                raise OverBudget(
                    f"{st['label']}: this call could cost ${worst_usd:.4f} and ${st['left']:.4f} of the ${st['budget']:.2f} budget is left"
                    + (
                        f" until {datetime.datetime.fromtimestamp(st['next_rollover'], datetime.UTC).date()}"
                        if st["next_rollover"]
                        else ""
                    )
                )
        with _lock:
            _ticket[0] += 1
            t = (provider, _ticket[0])
            _reserved.setdefault(provider, {})[t] = float(worst_usd)
        return t


def release(ticket):
    with _lock:
        _reserved.get(ticket[0], {}).pop(ticket, None)


def record(ticket, model, cost_usd=0.0, now=None, **fields):
    """Write a call's row (fields: tokens_in, tokens_out, cache_read, cache_write, ok, error, seconds, model_key, finish, request, response)
    and free its reservation, the row first so the spend never drops out of the count in between. Returns the row."""
    app, job_id = _job.get()
    if fields.get("response") is not None:
        fields["response"] = str(fields["response"])[:100_000]
    with _budget:
        try:
            if _store is None:
                return None
            return _store.add_api_call(
                at=now if now is not None else clock(),
                provider=ticket[0],
                model=model,
                app=app,
                job_id=job_id,
                cost_usd=round(float(cost_usd), 8),
                **fields,
            )
        finally:
            release(ticket)


def per_task(since=0.0):
    """{(role, model key): {n, usd, tokens_in, tokens_out, failed}}: what one job of a role costs on a cloud model, from the ledger (mean
    over the jobs since `since`). What routing reads in place of a price-times-typical-tokens guess."""
    return _store.api_per_task(since) if _store is not None else {}


def series(since, bucket_s=86400.0, provider=None, model=None, app=None):
    """The ledger in time buckets since `since`, for the charts: [{t, provider, cost_usd, calls, failed, tokens}] with t the bucket start;
    only the calls of `provider`, `model` (the vendor's id) and `app` when given."""
    if _store is None:
        return []
    out = {}
    for at, provider_, model_, app_, usd, tokens, ok in _store.api_points(since):
        if (provider and provider_ != provider) or (model and model_ != model) or (app and app_ != app):
            continue
        t = since + ((at - since) // bucket_s) * bucket_s
        b = out.setdefault((t, provider_), dict(t=t, provider=provider_, cost_usd=0.0, calls=0, failed=0, tokens=0))
        b["cost_usd"] += usd
        b["calls"] += 1
        b["failed"] += 0 if ok else 1
        b["tokens"] += int(tokens or 0)
    return [dict(b, cost_usd=round(b["cost_usd"], 6)) for _, b in sorted(out.items())]
