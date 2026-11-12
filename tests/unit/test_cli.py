import importlib
import json

import pytest

from stratamodels import cli, gpu


class Resp:
    def __init__(self, d):
        self.d = d

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def read(self):
        return json.dumps(self.d).encode()


def test_doctor_reports_libraries_runtimes_and_the_gpu(capsys, monkeypatch):
    monkeypatch.setattr(gpu, "query", lambda *a, **k: dict(name="RTX 3060", total_mb=12288, used_mb=1024))
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "fastapi" in out and "runtime fake" in out and "GPU RTX 3060, 12.0 GB" in out
    real = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda n: (_ for _ in ()).throw(ImportError(n)) if n in ("uvicorn", "insightface") else real(n),
    )
    monkeypatch.setattr(gpu, "query", lambda *a, **k: None)
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "MISSING  uvicorn" in out and "needs insightface" in out and "no NVIDIA GPU" in out


def test_status_jobs_and_registry_read_the_service(capsys, monkeypatch):
    import urllib.request

    status = dict(
        gpu=dict(name="RTX", used_mb=2048, total_mb=12288, util_pct=5),
        resident=[dict(model="buffalo_l", vram_mb=1024, jobs=3)],
        current=dict(id=7, role="faces", app="bh"),
        queued=[dict(id=8, kind="lease", role=None, note="arena", app="stills", priority="batch")],
    )
    jobs = dict(items=[dict(id=7, status="done", kind="run", role="faces", note=None, app="bh", run_s=0.2)], next=None)
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: Resp(status if "status" in url else jobs))
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert (
        "RTX: 2.0 of 12.0 GB" in out
        and "loaded   buffalo_l" in out
        and "running  #7 faces for bh" in out
        and "queued   #8 lease arena for stills" in out
    )
    assert cli.main(["status", "--json"]) == 0 and json.loads(capsys.readouterr().out)["current"]["id"] == 7
    status["running"] = [
        dict(id=7, role="faces", app="bh", model="buffalo_l", lane="gpu"),
        dict(id=9, role="text", app="x", model="cloud", lane="cpu"),
    ]
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "running  #7 faces for bh (buffalo_l, gpu lane)" in out and "running  #9 text for x (cloud, cpu lane)" in out
    del status["running"]
    assert cli.main(["jobs"]) == 0 and "#7" in capsys.readouterr().out
    assert cli.main(["jobs", "--json"]) == 0 and json.loads(capsys.readouterr().out)["items"][0]["id"] == 7
    status["gpu"] = None
    assert cli.main(["status"]) == 0 and "no GPU" in capsys.readouterr().out
    assert cli.main(["registry"]) == 0
    out = capsys.readouterr().out
    assert "faces" in out and "buffalo_l" in out and "not served yet: strata-360" in out
    assert cli.main(["registry", "--json"]) == 0 and "roles" in json.loads(capsys.readouterr().out)


def test_set_and_serve(capsys, monkeypatch):
    assert cli.main(["set", "vram_reserve_gb"]) == 0 and "vram_reserve_gb = 1.0" in capsys.readouterr().out
    assert cli.main(["set", "vram_reserve_gb", "2"]) == 0 and "= 2.0" in capsys.readouterr().out
    assert cli.main(["set", "vram_reserve_gb", "default"]) == 0 and "= 1.0" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="unknown setting"):
        cli.main(["set", "bad", "x"])
    with pytest.raises(SystemExit, match="number"):
        cli.main(["set", "vram_reserve_gb", "x"])
    from stratamodels.server import app as server_app

    called = []
    monkeypatch.setattr(server_app, "main", lambda argv: called.append(argv))
    assert cli.main(["serve", "--port", "9000", "--memory"]) == 0 and called == [
        ["--host", "127.0.0.1", "--port", "9000", "--memory"]
    ]


def test_eval_commands(capsys, monkeypatch, tmp_path, setup):
    monkeypatch.setenv("STRATAMODELS_DATABASE_URL", "sqlite:///" + str(tmp_path / "db.sqlite"))
    assert cli.main(["eval", "list"]) == 0
    out = capsys.readouterr().out
    assert "fake-asr" in out and "dataset speech" in out
    with pytest.raises(SystemExit, match="name the experiment"):
        cli.main(["eval", "report"])
    with pytest.raises(SystemExit, match="no experiment"):
        cli.main(["eval", "report", "nope"])
    import sys
    import types

    monkeypatch.setitem(
        sys.modules, "strata_models_client", types.SimpleNamespace(available=lambda: False, url=lambda: "http://x")
    )
    with pytest.raises(SystemExit, match="not running"):
        cli.main(["eval", "run", "fake-asr"])
    assert cli.main(["eval", "report", "fake-asr", "--out", str(tmp_path / "r.md")]) == 0 and (
        tmp_path / "r.md"
    ).read_text().startswith("# fake-asr")
    capsys.readouterr()
    assert cli.main(["eval", "report", "fake-asr"]) == 0 and "## Answer" in capsys.readouterr().out
    assert cli.main(["eval", "report", "fake-asr", "--json"]) == 0 and '"decision"' in capsys.readouterr().out
    assert (
        cli.main(["eval", "forget", "fake-asr", "--candidate", "same"]) == 0
        and "0 outputs forgotten" in capsys.readouterr().out
    )
    calls = []

    class SM:
        available = staticmethod(lambda: True)

        def submit(self, *a):
            calls.append(a)
            return dict(id=len(calls))

        def wait_job(self, i):
            return dict(id=i, status="done", result=dict(segments=[dict(text="x")]))

    sm = SM()
    monkeypatch.setitem(sys.modules, "strata_models_client", sm)
    assert (
        cli.main(["eval", "run", "fake-asr", "--limit", "2"]) == 0
        and '"ran": 6' in capsys.readouterr().out
        and len(calls) == 6
    )


def test_role_prefer_and_seed_change_the_database(capsys, monkeypatch, tmp_path):
    import urllib.request

    calls = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: calls.append(req.full_url) or Resp({}))
    assert (
        cli.main(["seed"]) == 0
        and "model buffalo_l" in capsys.readouterr().out
        and calls[-1].endswith("/api/registry/seed")
    )
    assert (
        cli.main(["role", "transcribe", "whisper-small"]) == 0
        and "transcribe -> whisper-small" in capsys.readouterr().out
    )
    with pytest.raises(SystemExit, match="not in the registry"):
        cli.main(["role", "transcribe", "nope"])  # the foreign key refuses it on a real database
    assert (
        cli.main(["prefer", "strata-360", "transcribe", "whisper-large-v3", "--params", '{"language": "en"}']) == 0
        and '"language": "en"' in capsys.readouterr().out
    )
    with pytest.raises(SystemExit, match="JSON"):
        cli.main(["prefer", "a", "transcribe", "--params", "{"])
    with pytest.raises(SystemExit, match="no role"):
        cli.main(["prefer", "a", "nope", "x"])
    assert (
        cli.main(["prefer", "strata-360", "transcribe", "--clear"]) == 0 and "no preference" in capsys.readouterr().out
    )
    assert cli.main(["seed", "--follow", "role:transcribe"]) == 0 and "role transcribe" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no model"):
        cli.main(["seed", "--follow", "model:nope"])
    assert cli.main(["registry"]) == 0 and "whisper-large-v3-turbo" in capsys.readouterr().out
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: (_ for _ in ()).throw(OSError("down")))
    assert cli.main(["role", "transcribe", "whisper-small"]) == 0 and "next start" in capsys.readouterr().err


def test_the_cli_never_migrates_a_database_at_another_revision(capsys, monkeypatch, tmp_path):
    import sqlalchemy as sa

    from stratamodels.store import StaleDatabase, Store

    Store.connect()
    e = sa.create_engine("sqlite:///" + str(tmp_path / "x.sqlite"))
    url = "sqlite:///" + str(tmp_path / "old.sqlite")
    monkeypatch.setenv("STRATAMODELS_DATABASE_URL", url)
    from omarchy_db import upgrade

    from stratamodels.db.models import Base

    upgrade(sa.create_engine(url), "migrations", Base.metadata, "e1efec22b916")
    with pytest.raises(StaleDatabase, match="e1efec22b916"):
        Store.connect(migrate="new")
    with pytest.raises(SystemExit, match="cannot be reached"):
        cli.main(["role", "faces", "buffalo_l"])
    assert cli.main(["eval", "list"]) == 0 and "reading the files" in capsys.readouterr().err
    e.dispose()
