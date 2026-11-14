import datetime

import pytest
from cloudfakes import ledger  # noqa: F401  (fixture)

from stratamodels import settings, spend


def ts(day, hour=12):
    return datetime.datetime.fromisoformat(f"{day}T{hour:02d}:00:00+00:00").timestamp()


def test_cost_prices_cache_tiers_and_dates():
    p = dict(input=1.0, output=5.0)
    assert spend.cost(p, tokens_in=1_000_000, tokens_out=200_000) == pytest.approx(2.0)
    assert spend.cost(p, cache_read=1_000_000, cache_write=1_000_000) == pytest.approx(0.1 + 1.25)
    assert spend.cost(
        dict(p, cache_read=0.5, cache_write=2.0), cache_read=1_000_000, cache_write=1_000_000
    ) == pytest.approx(2.5)
    tiered = dict(input=0.1, output=0.5, over=100_000, input_over=0.5, output_over=2.5)
    assert spend.cost(tiered, tokens_in=100_000, tokens_out=1_000_000) == pytest.approx(0.01 + 0.5)
    assert spend.cost(tiered, tokens_in=100_001, tokens_out=1_000_000) == pytest.approx(100_001 * 0.5 / 1e6 + 2.5)
    assert spend.cost(None, tokens_in=10**9) == 0.0 and spend.price_now(None) is None
    dated = [dict(input=1.0, output=2.0), dict({"from": "2027-01-01"}, input=3.0, output=4.0)]
    assert (
        spend.price_now(dated, ts("2026-12-31"))["input"] == 1.0
        and spend.price_now(dated, ts("2027-01-01"))["input"] == 3.0
    )
    assert spend.price_now([dict({"from": "2030-01-01"}, input=1, output=1)], ts("2026-01-01")) is None
    assert spend.worst(p, 1000, 2000) == pytest.approx((1000 * 1.0 + 2000 * 5.0) / 1e6)


@pytest.mark.parametrize(
    ("rollover", "now", "start", "nxt"),
    [
        (
            "2026-12-08",
            "2026-11-14",
            "2026-11-08",
            "2026-12-08",
        ),  # a date still to come: the period runs from a month before it
        ("2026-12-08", "2026-12-08", "2026-12-08", "2027-01-08"),
        ("2026-12-01", "2027-02-28", "2027-02-01", "2027-03-01"),
        ("2026-01-31", "2026-02-15", "2026-01-31", "2026-02-28"),  # the 31st falls on a month's last day
        ("2026-01-31", "2026-03-01", "2026-02-28", "2026-03-31"),
    ],
)
def test_period_repeats_monthly(rollover, now, start, nxt):
    assert spend.period(rollover, ts(now)) == (ts(start, 0), ts(nxt, 0))


def test_period_without_rollover_is_one_pot():
    assert spend.period(None) == (0.0, None)


def test_reserve_record_and_summary(ledger, monkeypatch):  # noqa: F811
    monkeypatch.setattr(spend, "clock", lambda: ts("2026-11-14"))
    with pytest.raises(spend.OverBudget, match="no budget is set for Claude API"):
        spend.reserve("claude", 0.01)
    settings.save(dict(claude_budget_usd=1.0, claude_rollover="2026-12-08"))
    t = spend.reserve("claude", 0.6)
    assert spend.state("claude")["reserved"] == 0.6 and spend.state("claude")["left"] == pytest.approx(0.4)
    with pytest.raises(
        spend.OverBudget, match=r"could cost \$0.5000 and \$0.4000 of the \$1.00 budget is left until 2026-12-08"
    ):
        spend.reserve("claude", 0.5)
    with spend.job("evals", 7):
        row = spend.record(
            t, "claude-haiku-5-5", 0.25, tokens_in=10, tokens_out=5, response="x" * 200_000, request=dict(prompt="hi")
        )
    assert (
        row["app"] == "evals" and row["job_id"] == 7 and row["provider"] == "claude" and len(row["response"]) == 100_000
    )
    st = spend.state("claude")
    assert (st["spent"], st["calls"], st["reserved"], st["left"], st["used_share"], st["warn"]) == (
        0.25,
        1,
        0.0,
        0.75,
        0.25,
        None,
    )
    assert st["period_start"] == ts("2026-11-08", 0) and st["next_rollover"] == ts("2026-12-08", 0)
    spend.record(spend.reserve("claude", 0.6), "claude-haiku-5-5", 0.6)
    assert spend.state("claude")["warn"] == 0.8
    monkeypatch.setattr(spend, "clock", lambda: ts("2026-12-09"))  # rolled over: the period starts again
    assert spend.state("claude")["spent"] == 0.0 and spend.state("claude")["left"] == 1.0
    free = {s["provider"]: s for s in spend.summary()}
    assert free["nim"]["budget"] is None and free["nim"]["left"] is None and free["nim"]["warn"] is None
    spend.record(spend.reserve("nim", 99.0), "nvidia/x", 0.0)  # a free key is never refused
    assert spend.state("nim")["calls"] == 1


def test_unset_budget_warns_full_and_no_ledger_refuses(ledger):  # noqa: F811
    assert spend.state("gemini")["warn"] == 1.0 and spend.state("gemini")["used_share"] is None
    spend.bind(None)
    with pytest.raises(spend.OverBudget, match="no spend ledger"):
        spend.reserve("gemini", 0.0)
    assert spend.record(spend.reserve("nim", 0.0), "m") is None and spend.series(0.0) == []
    assert spend.state("nim")["spent"] == 0.0


def test_series_buckets_by_provider(ledger, monkeypatch):  # noqa: F811
    day = 86400.0
    for at, prov, usd, ok in (
        (10.0, "claude", 0.1, True),
        (20.0, "claude", 0.2, False),
        (day + 5, "claude", 0.3, True),
        (30.0, "nim", 0.0, True),
    ):
        spend.record(
            spend.reserve("nim", 0.0) if prov == "nim" else ("claude", 0),
            "m",
            usd,
            now=at,
            ok=ok,
            tokens_in=1,
            tokens_out=2,
        )
    pts = spend.series(0.0, day)
    assert [(p["t"], p["provider"], p["calls"], p["failed"], p["tokens"]) for p in pts] == [
        (0.0, "claude", 2, 1, 6),
        (0.0, "nim", 1, 0, 3),
        (day, "claude", 1, 0, 3),
    ]
    assert pts[0]["cost_usd"] == pytest.approx(0.3)
    assert [p["provider"] for p in spend.series(0.0, day, provider="nim")] == ["nim"]
    assert spend.series(0.0, day, model="other") == [] and spend.series(0.0, day, app="evals") == []
    assert ledger.api_breakdown(0.0)[0] == dict(
        provider="claude",
        model="m",
        app=None,
        calls=3,
        failed=1,
        tokens_in=3,
        tokens_out=6,
        cost_usd=pytest.approx(0.6),
    )
    log = ledger.api_log(provider="claude", limit=2)
    assert [c["cost_usd"] for c in log] == [0.3, 0.2] and "request" not in log[0]
    assert [c["cost_usd"] for c in ledger.api_log(provider="claude", before=log[-1]["id"])] == [0.1]
    assert (
        len(ledger.api_log(ok=False)) == 1
        and len(ledger.api_log(since=day)) == 1
        and ledger.api_log(model="other") == []
    )
    assert ledger.api_call(log[0]["id"])["request"] is None and ledger.api_call(999) is None


def test_concurrent_reservations_cannot_overrun_the_budget(ledger, monkeypatch):  # noqa: F811
    """Two calls checking the same budget at once: only one $0.75 reservation fits in $1, however the threads interleave."""
    import threading

    monkeypatch.setattr(settings, "budget", lambda p: (1.0, None))
    real_state = spend.state
    gate = threading.Barrier(2, timeout=0.5)

    def slow_state(*a, **k):
        st = real_state(*a, **k)
        try:
            gate.wait()  # without the lock both threads read the budget here before either reserves
        except threading.BrokenBarrierError:
            pass
        return st

    monkeypatch.setattr(spend, "state", slow_state)
    got = []

    def go():
        try:
            got.append(spend.reserve("claude", 0.75))
        except spend.OverBudget:
            got.append(None)

    threads = [threading.Thread(target=go) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(x is None for x in got) == [False, True]
