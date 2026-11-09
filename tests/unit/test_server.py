"""The HTTP API (stratamodels.server.app) over the fake scheduler: jobs, waiting, leases, the registry, status, usage, settings."""

import threading
import time

import pytest
import uvicorn
from fastapi.testclient import TestClient

from stratamodels import gpu
from stratamodels.server import app as server_app
from stratamodels.server.app import create_app


@pytest.fixture
def client(scheduler):
    h = gpu.History(scheduler.probe, every_s=5)
    h.sample()
    return TestClient(create_app(scheduler, h))


def worker(scheduler, n=1):
    """Run `n` steps on a thread shortly after, as the real worker would, so a waiting request sees the job finish."""

    def go():
        for _ in range(n):
            time.sleep(0.05)
            scheduler.step()

    t = threading.Thread(target=go)
    t.start()
    return t


def test_landing_page_and_built_static_index(scheduler, tmp_path, monkeypatch):
    monkeypatch.setattr(server_app, "STATIC", str(tmp_path / "none"))
    c = TestClient(create_app(scheduler))
    assert "strata-models" in c.get("/").text and c.get("/api/health").json() == dict(
        ok=True, version="0.1.0", gpu=True
    )
    assert c.get("/api/gpu/history").json() == dict(every_s=None, samples=[])
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<h1>built</h1>", encoding="utf8")
    (static / "assets" / "a.js").write_text("1", encoding="utf8")
    monkeypatch.setattr(server_app, "STATIC", str(static))
    c2 = TestClient(create_app(scheduler))
    assert c2.get("/").text == "<h1>built</h1>" and c2.get("/assets/a.js").text == "1"


def test_submit_poll_and_list_jobs(client, scheduler):
    j = client.post("/api/jobs", json=dict(role="echo", params=dict(x=1), app="bh")).json()
    assert j["status"] == "queued" and client.get(f"/api/jobs/{j['id']}").json()["status"] == "queued"
    t = worker(scheduler)
    done = client.get(f"/api/jobs/{j['id']}", params=dict(wait=5)).json()
    t.join()
    assert done["status"] == "done" and done["result"]["echo"] == dict(x=1)
    for _ in range(3):
        client.post("/api/jobs", json=dict(role="echo", app="other"))
    page = client.get("/api/jobs", params=dict(limit=2)).json()
    assert [x["id"] for x in page["items"]] == [4, 3] and page["next"] == 3
    assert [x["id"] for x in client.get("/api/jobs", params=dict(before=3, limit=2)).json()["items"]] == [2, 1]
    assert [x["app"] for x in client.get("/api/jobs", params=dict(app="bh")).json()["items"]] == ["bh"] and client.get(
        "/api/jobs", params=dict(app="bh")
    ).json()["next"] is None
    assert client.delete("/api/jobs/4").json()["status"] == "cancelled"
    assert client.get("/api/jobs/99").status_code == 404
    assert client.post("/api/jobs", json=dict(role="nope", app="x")).status_code == 400
    assert (
        client.post("/api/jobs", json=dict(role="echo", app="x", priority="now"))
        .json()["detail"]
        .startswith("priority")
    )


def test_run_waits_for_the_result_or_says_why_not(client, scheduler):
    t = worker(scheduler)
    r = client.post("/api/run/echo", json=dict(params=dict(y=2), app="s360"))
    t.join()
    assert r.status_code == 200 and r.json()["result"]["echo"] == dict(y=2) and r.json()["priority"] == "interactive"
    t = worker(scheduler)
    r = client.post("/api/run/echo", json=dict(params=dict(fail="no such audio file"), app="s360"))
    t.join()
    assert r.status_code == 422 and r.json()["detail"] == "no such audio file" and r.json()["job"]["status"] == "failed"
    r = client.post("/api/run/echo", json=dict(app="s360", timeout=0.1))
    assert r.status_code == 504 and r.json()["job"]["status"] == "queued"


def test_leases_over_http(client, scheduler):
    lease = client.post("/api/leases", json=dict(app="strata-stills", vram_gb=4, note="arena")).json()
    assert lease["status"] == "queued" and lease["vram_gb"] == 4
    t = worker(scheduler)
    got = client.get(f"/api/leases/{lease['id']}", params=dict(wait=5)).json()
    t.join()
    assert got["status"] == "running"
    assert client.post(f"/api/leases/{lease['id']}/heartbeat").json()["heartbeat_at"] is not None
    assert client.delete(f"/api/leases/{lease['id']}").json()["status"] == "done"
    assert client.post(f"/api/leases/{lease['id']}/heartbeat").status_code == 400
    j = client.post("/api/jobs", json=dict(role="echo", app="x")).json()
    assert client.get(f"/api/leases/{j['id']}").status_code == 400


def test_status_registry_models_and_usage(client, scheduler):
    client.post("/api/jobs", json=dict(role="echo", app="bh"))
    scheduler.step()
    st = client.get("/api/status").json()
    assert (
        st["gpu"]["name"] == "Fake RTX" and st["resident"][0]["model"] == "small" and st["budget"]["reserve_mb"] == 1024
    )
    reg = client.get("/api/registry").json()
    m = {x["key"]: x for x in reg["models"]}
    assert (
        m["small"]["resident"]
        and m["small"]["stats"]["jobs"] == 1
        and m["small"]["missing"] == []
        and m["planned"]["served"] is False
        and m["big"]["stats"] is None
    )
    assert (
        client.post("/api/models/big/load").status_code == 202
        and client.post("/api/models/planned/load").status_code == 400
    )
    scheduler.step()
    assert "big" in scheduler.resident
    assert client.get("/api/gpu/history").json()["every_s"] == 5
    u = client.get("/api/usage", params=dict(hours=1)).json()
    assert u["hours"] == 1 and u["rows"][0]["app"] == "bh"


def test_settings_roundtrip_and_main(client, scheduler, tmp_path, monkeypatch):
    assert client.put("/api/settings", json=dict(models_dir="relative")).status_code == 400
    assert client.put("/api/settings", json=dict(vram_reserve_gb=99)).status_code == 400
    r = client.put("/api/settings", json=dict(models_dir=str(tmp_path / "w"), vram_reserve_gb=2)).json()
    assert (
        r["models_dir"] == str(tmp_path / "w")
        and r["vram_reserve_gb"] == 2
        and scheduler.reserve_mb == 2048
        and scheduler.models_dir == str(tmp_path / "w")
    )
    assert client.get("/api/settings").json() == r
    runs = []
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port, log_level: runs.append((host, port)))
    monkeypatch.setattr(gpu, "query", lambda *a, **k: None)
    server_app.main(["--port", "8366", "--memory"])
    assert runs == [("127.0.0.1", 8366)]


def test_generated_pictures_gallery_files_and_uploads(client, scheduler, tmp_path):
    import os

    from stratamodels import settings

    gen = os.path.join(settings.data_dir(), "generated", "2026-11-09")
    os.makedirs(gen)
    pic = os.path.join(gen, "a.png")
    open(pic, "wb").write(b"png")
    j = scheduler.store.add(
        kind="run",
        role="image-generate",
        model="flux3-klein",
        app="web",
        priority="interactive",
        params=dict(prompt="a fox"),
    )
    scheduler.store.update(
        j["id"],
        status="done",
        result=dict(path=pic, width=1024, height=1024, seed=1, steps=16, seconds=9.5, peak_vram_mb=6100),
        finished_at=1.0,
    )
    k = scheduler.store.add(
        kind="run",
        role="image-generate",
        model="flux3-klein",
        app="web",
        priority="interactive",
        params=dict(prompt="broken"),
    )
    scheduler.store.update(k["id"], status="done", result={})
    g = client.get("/api/images").json()
    assert [x["prompt"] for x in g["items"]] == ["a fox"] and g["items"][0]["seconds"] == 9.5 and g["next"] is None
    assert client.get("/api/images", params=dict(limit=1)).json()["next"] == k["id"]
    assert client.get("/api/images/file", params=dict(path=pic)).content == b"png"
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"no")
    assert (
        client.get("/api/images/file", params=dict(path=str(secret))).status_code == 404
        and client.get("/api/images/file", params=dict(path=os.path.join(gen, "..", "..", "..", "x"))).status_code
        == 404
    )
    up = client.post("/api/images/upload", params=dict(name="ref.jpg"), content=b"jpegdata").json()["path"]
    assert up.endswith(".jpg") and client.get("/api/images/file", params=dict(path=up)).content == b"jpegdata"
    assert (
        client.post("/api/images/upload", params=dict(name="x.exe"), content=b"x").status_code == 400
        and client.post("/api/images/upload", content=b"").status_code == 400
    )


def test_registry_preferences_and_files_through_the_api(client, scheduler):
    from stratamodels import registry as reg

    reg.save_model(scheduler.store.engine, "small", roles=["echo"], runtime="fake", vram_gb=2)
    reg.save_model(scheduler.store.engine, "big", roles=["echo"], runtime="fake", vram_gb=6)
    reg.set_role(scheduler.store.engine, "echo", "small")
    scheduler.reload_registry()
    assert (
        client.put("/api/roles/echo", json=dict(model="big")).json()["roles"][0]["model"] == "big"
        and scheduler.registry.resolve("echo").key == "big"
    )
    assert client.put("/api/roles/echo", json=dict(model="nope")).status_code == 400
    m = client.put("/api/models/tiny", json=dict(runtime="fake", vram_gb=0.5, roles=["echo"])).json()["models"]
    assert any(x["key"] == "tiny" and x["served"] for x in m)
    assert (
        client.put("/api/preferences/bh/echo", json=dict(model="tiny", params=dict(k=1))).json()["items"][0]["model"]
        == "tiny"
    )
    assert client.put("/api/preferences/bh/echo", json={}).status_code == 400
    j = client.post("/api/jobs", json=dict(role="echo", app="bh", params=dict(path="/p/a.wav"))).json()
    assert j["model"] == "tiny" and j["params"] == dict(k=1, path="/p/a.wav")
    assert (
        client.get("/api/files", params=dict(job=j["id"])).json()["items"][0]["path"] == "/p/a.wav"
        and client.get("/api/files", params=dict(path="/p/a.wav")).json()["items"][0]["job_id"] == j["id"]
    )
    assert client.get("/api/files").status_code == 400
    assert client.delete("/api/preferences/bh/echo").json()["items"] == []
    assert client.post("/api/registry/seed", params=dict(follow="nope")).status_code == 400
    out = client.post("/api/registry/seed", params=dict(follow="role:echo")).json()
    assert "role faces" in out["added"] and scheduler.registry.resolve("faces").key == "buffalo_l"
