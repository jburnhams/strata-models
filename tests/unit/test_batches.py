"""Jobs sent through the Claude Message Batches API (stratamodels.batches, the batch_* functions of runtimes/claude.py), the vendor faked."""

from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from cloudfakes import FakeAnthropic, claude_reply, ended, errored, ledger, succeeded  # noqa: F401  (ledger: fixture)

from stratamodels import settings, spend
from stratamodels.registry import Registry
from stratamodels.runtimes import claude, fake
from stratamodels.scheduler import Scheduler
from stratamodels.store import memory_store

PRICE = dict(input=1.0, output=5.0)  # USD per million tokens: the batch price is half


@pytest.fixture
def sch(clock, tmp_path, monkeypatch):
    """A scheduler over an in-memory store on the test's clock (the spend ledger and settings bound to it), whose role `caption` runs on a Claude API model (`haiku`), its client faked (`sch.client`), and role `echo` on a fake model."""
    reg = Registry.from_dict(
        dict(
            roles=dict(caption=dict(model="haiku"), echo=dict(model="small")),
            models=dict(
                haiku=dict(
                    runtime="claude-api",
                    roles=["caption"],
                    vram_gb=0,
                    options=dict(model="claude-haiku-5-5", price=PRICE),
                ),
                small=dict(runtime="fake", roles=["echo"], vram_gb=0),
            ),
        )
    )
    fake.LOADED.clear()
    st = memory_store(clock)
    settings.bind(st.engine)
    spend.bind(st)
    monkeypatch.setattr(spend, "_reserved", {})
    s = Scheduler(st, reg, str(tmp_path / "models"), probe=lambda: None, clock=clock)
    s.client = FakeAnthropic()
    real = claude.load
    monkeypatch.setattr(claude, "load", lambda m, d: real(m, d, client=s.client))  # every handle on the fake vendor
    settings.save(dict(claude_budget_usd=1.0))
    yield s
    fake.LOADED.clear()
    spend.bind(None)
    settings.bind(None)


def bulk(sch, role, items, app="app", batch=True, **kw):
    """submit_many, returning the job rows."""
    return [sch.store.get(i) for i in sch.submit_many(role, items, app, batch=batch, **kw)]


def caption(sch, n, **params):
    return bulk(
        sch, "caption", [dict(prompt=f"describe {i}", max_tokens=100, **params) for i in range(n)], "strata-stills"
    )


def test_batch_jobs_go_as_one_batch_at_the_next_step_holding_their_worst_case(sch, clock):
    jobs = caption(sch, 3)
    assert sch.step() is None  # never in the CPU lane
    (b,) = sch.batcher.step()  # at once: nothing is held to see whether more come
    (sent,) = sch.client.messages.batches.created
    assert [r["custom_id"] for r in sent] == [f"job-{j['id']}" for j in jobs]
    assert sent[0]["params"] == dict(
        model="claude-haiku-5-5",
        max_tokens=100,
        messages=[{"role": "user", "content": [{"type": "text", "text": "describe 0"}]}],
    )
    assert (b["vendor_id"], b["status"], b["vendor_status"], b["model_key"]) == (
        "msgbatch_1",
        "submitted",
        "in_progress",
        "haiku",
    )
    # worst case: (10 bytes of prompt + 2000 overhead) in + 100 out, at half of $1/$5 per M
    w = (2010 * 1.0 + 100 * 5.0) / 1e6 / 2
    assert b["jobs"] == {str(j["id"]): pytest.approx(w) for j in jobs}
    assert spend.state("claude")["reserved"] == pytest.approx(3 * w)
    rows = [sch.store.get(j["id"]) for j in jobs]
    assert {(r["status"], r["note"], r["settings_hash"]) for r in rows} == {
        ("running", "in batch msgbatch_1", sch.registry.models["haiku"].settings_hash)
    }
    assert sch.status()["batches"][0]["jobs"] == 3


def test_an_ended_batch_records_each_answer_at_half_price_and_finishes_its_jobs(sch, clock):
    a, b, c, d = caption(sch, 4)
    clock.t += 60
    sch.batcher.step()
    batches = sch.client.messages.batches
    clock.t += 30
    assert sch.batcher.step() == []  # not due a poll yet
    clock.t += 30
    (row,) = sch.batcher.step()
    assert (row["status"], row["counts"]["processing"]) == ("submitted", 4) and sch.store.get(a["id"])[
        "status"
    ] == "running"
    batches.status = "ended"
    batches.outcomes = {
        f"job-{a['id']}": succeeded(claude_reply("a dog", tokens_in=1000, tokens_out=20)),
        f"job-{b['id']}": errored(),
        f"job-{c['id']}": ended("expired"),
        "someone-else": succeeded(claude_reply()),
    }  # d has no entry at all
    clock.t += 60
    (row,) = sch.batcher.step()
    assert (row["status"], row["ended_at"]) == ("ended", clock.t)
    done = sch.store.get(a["id"])
    assert done["status"] == "done" and done["run_s"] == 120.0
    assert done["result"] == dict(
        text="a dog",
        tokens_in=1000,
        tokens_out=20,
        cost_usd=pytest.approx(0.00055),
        seconds=120.0,
        model="haiku",
        provider="claude",
        finish="end_turn",
        batch="msgbatch_1",
    )
    assert sch.store.get(b["id"])["error"] == "Claude batch msgbatch_1: invalid_request_error: bad image"
    assert sch.store.get(c["id"])["error"] == "Claude batch msgbatch_1: the batch entry expired"
    assert sch.store.get(d["id"])["error"] == "batch msgbatch_1 ended without an answer for this job"
    log = {r["job_id"]: r for r in sch.store.api_log(provider="claude")}
    assert (log[a["id"]]["cost_usd"], log[a["id"]]["ok"], log[a["id"]]["app"]) == (
        pytest.approx(0.00055),
        True,
        "strata-stills",
    )
    assert sch.store.api_call(log[a["id"]]["id"])["request"]["batch"] == "msgbatch_1"
    assert (log[b["id"]]["cost_usd"], log[b["id"]]["ok"]) == (0.0, False) and d["id"] not in log
    st = spend.state("claude")
    assert st["reserved"] == 0.0 and st["spent"] == pytest.approx(0.00055)
    assert (st["batch_spent"], st["batch_calls"], st["batch_saved"]) == (
        pytest.approx(0.00055),
        3,
        pytest.approx(0.00055),
    )
    assert [c["batch"] for c in sch.store.api_log(batch=True)] == [
        "msgbatch_1"
    ] * 3  # a's, b's and c's (failed, at no cost)
    assert sch.store.api_log(batch=False) == []
    (row,) = sch.store.api_breakdown(0.0)
    assert (row["calls"], row["batch_calls"], row["batch_cost_usd"]) == (3, 3, pytest.approx(0.00055))
    (pt,) = spend.series(0.0)
    assert (pt["batch_calls"], pt["batch_cost_usd"]) == (3, pytest.approx(0.00055))
    assert sch.status()["batches"] == []


def test_a_refusal_and_an_answer_that_is_not_json_fail_their_jobs_but_are_paid_for(sch, clock):
    a, b = caption(sch, 2, schema={"type": "object"})
    clock.t += 60
    sch.batcher.step()
    assert sch.client.messages.batches.created[0][0]["params"]["output_config"] == {
        "format": {"type": "json_schema", "schema": {"type": "object"}}
    }
    sch.client.messages.batches.status = "ended"
    sch.client.messages.batches.outcomes = {
        f"job-{a['id']}": succeeded(claude_reply("", stop="refusal", category="cyber")),
        f"job-{b['id']}": succeeded(claude_reply("not json")),
    }
    clock.t += 60
    sch.batcher.step()
    assert sch.store.get(a["id"])["error"] == "Claude declined this request (refusal, cyber)"
    assert sch.store.get(b["id"])["error"].startswith("no JSON answer")
    assert spend.state("claude")["calls"] == 2 and spend.state("claude")["reserved"] == 0.0


def test_jobs_past_what_the_budget_holds_fail_and_the_rest_go(sch, clock):
    w = (2010 * 1.0 + 100 * 5.0) / 1e6 / 2
    settings.save(dict(claude_budget_usd=2.5 * w))
    a, b, c = caption(sch, 3)
    clock.t += 60
    (row,) = sch.batcher.step()
    assert set(row["jobs"]) == {str(a["id"]), str(b["id"])}
    assert (
        sch.store.get(c["id"])["status"] == "failed"
        and "3 calls of this batch could cost" in sch.store.get(c["id"])["error"]
    )
    settings.save(dict(claude_budget_usd=0))
    (d,) = caption(sch, 1)
    clock.t += 60
    sch.batcher.step()
    assert "no budget is set" in sch.store.get(d["id"])["error"]


def test_a_refused_batch_fails_its_jobs_and_frees_the_budget(sch, clock):
    resp = httpx2.Response(400, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages/batches"))
    sch.client.messages.batches.error = anthropic.BadRequestError("too big", response=resp, body=None)
    (a,) = caption(sch, 1)
    clock.t += 60
    (row,) = sch.batcher.step()
    assert (row["status"], row["error"]) == ("failed", "Claude API 400 refused the batch: too big")
    assert sch.store.get(a["id"])["error"] == "Claude API 400 refused the batch: too big"
    assert spend.state("claude")["reserved"] == 0.0


def test_bad_params_fail_one_job_and_a_cancelled_job_is_not_sent(sch, clock):
    (bad,) = bulk(sch, "caption", [dict(images=["/no/such.png"], prompt="x")])
    a, b = caption(sch, 2)
    sch.cancel(b["id"])
    clock.t += 60
    (row,) = sch.batcher.step()
    assert list(row["jobs"]) == [str(a["id"])]
    assert sch.store.get(bad["id"])["error"] == "no such image: '/no/such.png'"


def test_a_batch_is_cut_at_max_jobs_and_max_bytes_and_the_rest_waits(sch, clock):
    sch.batcher.max_jobs, sch.batcher.poll_s = 2, 3600
    caption(sch, 3)
    (row,) = sch.batcher.step()
    assert len(row["jobs"]) == 2 and len(sch.store.queued()) == 1
    sch.batcher.max_jobs, sch.batcher.max_bytes = 10, 1
    caption(sch, 1)
    clock.t += 60
    (row,) = sch.batcher.step()  # every request is over max_bytes: one goes, the other waits
    assert len(row["jobs"]) == 1 and len(sch.store.queued()) == 1


def test_a_poll_or_a_read_that_fails_is_tried_again(sch, clock):
    a, b = caption(sch, 2)
    clock.t += 60
    sch.batcher.step()
    batches = sch.client.messages.batches
    batches.status = RuntimeError("network down")
    clock.t += 60
    (row,) = sch.batcher.step()
    assert (row["status"], row["error"]) == ("submitted", "RuntimeError: network down")
    batches.status, batches.broken = "ended", RuntimeError("cut off")
    batches.outcomes = {
        f"job-{a['id']}": succeeded(claude_reply("one")),
        f"job-{b['id']}": succeeded(claude_reply("two")),
    }
    clock.t += 60
    (row,) = sch.batcher.step()
    assert row["status"] == "submitted" and "cut off" in row["error"]
    assert (sch.store.get(a["id"])["status"], sch.store.get(b["id"])["status"]) == ("done", "running")
    batches.broken = None
    clock.t += 60
    (row,) = sch.batcher.step()
    assert row["status"] == "ended" and sch.store.get(b["id"])["result"]["text"] == "two"
    assert spend.state("claude")["calls"] == 2  # the first answer was not recorded twice


def test_a_restart_keeps_open_batches_and_holds_their_budget_again(sch, clock, monkeypatch):
    a, b = caption(sch, 2)
    clock.t += 60
    sch.batcher.step()
    (c,) = caption(sch, 1)
    unsent = sch.store.add_batch(
        provider="claude", model_key="haiku", submitted_at=clock.t, status="submitted", jobs={str(c["id"]): 0.01}
    )
    sch.store.claim(c["id"], status="running", started_at=clock.t)
    lost = sch.submit("echo", {}, "app")
    sch.store.claim(lost["id"], status="running", started_at=clock.t)
    monkeypatch.setattr(spend, "_reserved", {})  # a new process: nothing held
    sch.batcher.tickets.clear()
    assert sch.store.recover() == 1  # only the job outside a batch
    sch.batcher.resume()
    assert sch.store.get(a["id"])["status"] == "running" and sch.store.get(c["id"])["error"].startswith(
        "the service restarted"
    )
    assert sch.store.batches()[0]["id"] == unsent["id"] and sch.store.batches()[0]["status"] == "failed"
    w = (2010 * 1.0 + 100 * 5.0) / 1e6 / 2
    assert spend.state("claude")["reserved"] == pytest.approx(2 * w)
    sch.client.messages.batches.status = "ended"
    sch.client.messages.batches.outcomes = {
        f"job-{a['id']}": succeeded(claude_reply()),
        f"job-{b['id']}": succeeded(claude_reply()),
    }
    clock.t += 60
    sch.batcher.step()
    assert sch.store.get(b["id"])["status"] == "done" and spend.state("claude")["reserved"] == 0.0


def test_a_batch_is_collected_with_the_price_and_prompt_it_was_sent_with(sch, clock, tmp_path, monkeypatch):
    from stratamodels.runtimes import llama

    (tmp_path / "p.md").write_text("old prompt")
    monkeypatch.setattr(llama, "preset", lambda name: ((tmp_path / f"{name}.md").read_text(), {"type": "object"}))
    (a,) = bulk(sch, "caption", [dict(preset="p", max_tokens=100)])
    clock.t += 60
    sch.batcher.step()
    sch.registry.models["haiku"].options["price"] = dict(input=100.0, output=500.0)  # edited while the batch runs
    sch.registry.models.pop("haiku")  # or the model removed altogether
    (tmp_path / "p.md").unlink()  # and the preset gone
    sch.client.messages.batches.status = "ended"
    sch.client.messages.batches.outcomes = {
        f"job-{a['id']}": succeeded(claude_reply('{"ok": 1}', tokens_in=1000, tokens_out=20))
    }
    clock.t += 60
    (row,) = sch.batcher.step()
    done = sch.store.get(a["id"])
    assert row["status"] == "ended" and done["result"]["json"] == {"ok": 1}
    assert done["result"]["cost_usd"] == pytest.approx(0.00055)
    call = sch.store.api_call(sch.store.api_log()[0]["id"])
    assert call["request"]["prompt"] == "old prompt" and call["request"]["schema"] == {"type": "object"}


def test_a_picture_that_cannot_be_read_fails_only_its_job(sch, clock, tmp_path, monkeypatch):
    from stratamodels.runtimes import cloud

    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"x")
    monkeypatch.setattr(cloud, "encode", lambda p: (_ for _ in ()).throw(OSError("truncated")))
    (b,) = bulk(sch, "caption", [dict(prompt="x", image=str(bad))])
    (a,) = caption(sch, 1)
    clock.t += 60
    (row,) = sch.batcher.step()
    assert list(row["jobs"]) == [str(a["id"])] and sch.store.get(b["id"])["error"] == "OSError: truncated"


def test_snapshots_keep_each_distinct_request_once():
    from stratamodels.batches import requests, snapshot

    m = SimpleNamespace(runtime="claude-api", options=dict(model="x"))
    snap = snapshot(
        m, [(1, dict(prompt="p", images=["/a"])), (2, dict(prompt="p", images=["/b"])), (3, dict(prompt="q"))]
    )
    assert snap["requests"] == [dict(prompt="p"), dict(prompt="q")]
    assert requests(snap) == {
        1: dict(prompt="p", images=["/a"]),
        2: dict(prompt="p", images=["/b"]),
        3: dict(prompt="q", images=[]),
    }


def test_the_batcher_loads_its_own_client_and_a_model_that_cannot_load_fails_its_jobs(sch, clock, monkeypatch):
    sch.batcher.handles.clear()
    monkeypatch.setattr(claude, "load", lambda m, d: (_ for _ in ()).throw(claude.RunError("no key")))
    (a,) = caption(sch, 1)
    clock.t += 60
    assert sch.batcher.step() == []
    assert sch.store.get(a["id"])["error"] == "no key"
    unloaded = []
    monkeypatch.setattr(
        claude,
        "load",
        lambda m, d: dict(client=sch.client, key=m.key, model="claude-haiku-5-5", price=PRICE, effort=None),
    )
    monkeypatch.setattr(claude, "unload", unloaded.append)
    sch.batcher.handles["haiku"] = ("stale", "old handle")
    caption(sch, 1)
    clock.t += 60
    (row,) = sch.batcher.step()
    assert unloaded == ["old handle"] and row["vendor_id"] == "msgbatch_1"


def test_a_model_without_a_batch_api_runs_the_job_at_once(sch):
    (j,) = bulk(sch, "echo", [dict(x=1)])
    assert j["params"] == dict(x=1) and not sch.batcher.takes(j)  # not marked
    assert sch.step()["result"] == dict(model="small", echo=dict(x=1))


def test_batch_prices_and_reserving_many(ledger):  # noqa: F811
    assert spend.cost(PRICE, tokens_in=1_000_000, batch=True) == 0.5
    assert spend.cost(dict(PRICE, batch=0.6), tokens_out=1_000_000, batch=True) == pytest.approx(3.0)
    settings.save(dict(claude_budget_usd=1.0))
    tickets, refusal = spend.reserve_many("claude", [0.4, 0.4, 0.4, 0.1])
    assert len(tickets) == 2 and "3 calls of this batch could cost $1.2000" in str(refusal)
    assert spend.state("claude")["reserved"] == pytest.approx(0.8)
    assert spend.reserve_many("nim", [5.0, 5.0])[1] is None  # unbudgeted: everything goes
    spend.hold("claude", 5.0)  # no check
    assert spend.state("claude")["left"] == 0.0


def test_every_claude_job_is_batched_but_one_at_high_priority(sch, clock):
    """Jonathan, 2026-11-28: all models and evals use batch; only an individual high-priority call runs at once."""
    items = [
        dict(prompt=f"describe {i}", max_tokens=100, batch=False) for i in range(3)
    ]  # an item's own flag is not read
    for b in (None, False, True):  # nor the call's
        many = bulk(sch, "caption", items, batch=b)
        assert [j["params"] for j in many] == [
            dict(prompt=f"describe {i}", max_tokens=100, batch=True) for i in range(3)
        ]
    (one,) = bulk(sch, "caption", [dict(prompt="c")], batch=False)  # one, at normal priority: still batched
    low = sch.submit("caption", dict(prompt="x"), "app", priority="low")  # queued on its own, not high: batched
    assert sch.batcher.takes(one) and sch.batcher.takes(low)
    two_high = bulk(sch, "caption", [dict(prompt="a"), dict(prompt="b")], priority="high")
    assert all(sch.batcher.takes(j) for j in two_high)  # high, but more than one: batched
    (row,) = sch.batcher.step()
    assert len(row["jobs"]) == 3 * 3 + 1 + 1 + 2
    high = sch.submit("caption", dict(prompt="now", batch=True), "app", priority="high")
    (high_bulk,) = bulk(sch, "caption", [dict(prompt="now too")], priority="high")
    assert "batch" not in high["params"] and not sch.batcher.takes(high) and not sch.batcher.takes(high_bulk)
    sch.client.messages.reply = claude_reply("now")
    sch.client.messages.count = 50
    r = sch.step()["result"]
    assert r["text"] == "now" and "batch" not in r  # the CPU lane runs it as a call made at once


def test_a_bulk_submission_is_checked_before_anything_is_queued(sch):
    from stratamodels.registry import RegistryError
    from stratamodels.scheduler import QueueError

    for items, err in (("x", "a list of params"), ([1], "a list of params")):
        with pytest.raises(QueueError, match=err):
            sch.submit_many("caption", items, "app")
    with pytest.raises(QueueError, match="priority"):
        sch.submit_many("caption", [{}], "app", priority="now")
    with pytest.raises(RegistryError):
        sch.submit_many("nope", [{}], "app")
    assert sch.submit_many("caption", [], "app", batch=True) == []
    assert sch.store.queued() == []
