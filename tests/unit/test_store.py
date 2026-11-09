"""Job rows (stratamodels.store): listing with a cursor, usage per app, model stats, pruning old results."""

from stratamodels.store import memory_store


def test_list_filters_and_pages_newest_first(clock):
    s = memory_store(clock)
    for i in range(5):
        s.add(kind="run", role="faces" if i % 2 else "transcribe", model="m", app=f"app{i % 2}", priority="batch")
    assert [j["id"] for j in s.list(limit=2)] == [5, 4] and [j["id"] for j in s.list(before=4, limit=2)] == [3, 2]
    assert {j["role"] for j in s.list(role="faces")} == {"faces"} and all(
        j["app"] == "app0" for j in s.list(app="app0")
    )
    assert s.list(status="done") == [] and s.list(kind="lease") == [] and "result" not in s.list()[0]
    assert s.update(999, status="done") is None and s.get(999) is None


def test_usage_counts_run_time_and_lease_hold_time(clock):
    s = memory_store(clock)
    a = s.add(kind="run", role="faces", model="m", app="bh", priority="batch")
    s.update(a["id"], status="done", started_at=clock.t, finished_at=clock.t + 3, run_s=2.5)
    b = s.add(kind="run", role="faces", model="m", app="bh", priority="batch")
    s.update(b["id"], status="failed", started_at=clock.t, finished_at=clock.t + 1)
    lease = s.add(kind="lease", app="stills", priority="batch")
    s.update(lease["id"], status="running", started_at=clock.t)
    s.add(kind="run", role="faces", model="m", app="never-ran", priority="batch")
    clock.t += 30
    rows = {(r["app"], r["kind"]): r for r in s.usage(0)}
    assert rows[("bh", "run")] == dict(app="bh", kind="run", role="faces", jobs=2, failed=1, gpu_s=2.5)
    assert rows[("stills", "lease")]["gpu_s"] == 30.0 and ("never-ran", "run") not in rows
    assert s.model_stats() == {"m": dict(jobs=1, mean_run_s=2.5, last_at=clock.t - 27)}
    assert s.get(a["id"])["wait_s"] == 0


def test_prune_drops_old_results_and_recover_fails_running_jobs(clock):
    s = memory_store(clock)
    j = s.add(kind="run", role="r", model="m", app="a", priority="batch")
    s.update(j["id"], status="done", result=dict(x=1), finished_at=clock.t)
    clock.t += 8 * 86400
    assert s.prune() == 1 and s.get(j["id"])["result"] is None and s.get(j["id"])["status"] == "done"
    r = s.add(kind="run", role="r", model="m", app="a", priority="batch", status="running")
    assert s.recover() == 1 and s.get(r["id"])["status"] == "failed"
