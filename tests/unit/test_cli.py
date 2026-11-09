import importlib, json

import pytest

from stratamodels import cli, gpu


class Resp:
    def __init__(self, d): self.d = d
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def read(self): return json.dumps(self.d).encode()


def test_doctor_reports_libraries_runtimes_and_the_gpu(capsys, monkeypatch):
    monkeypatch.setattr(gpu, 'query', lambda *a, **k: dict(name='RTX 3060', total_mb=12288, used_mb=1024))
    assert cli.main(['doctor']) == 0
    out = capsys.readouterr().out; assert 'fastapi' in out and 'runtime fake' in out and 'GPU RTX 3060, 12.0 GB' in out
    real = importlib.import_module
    monkeypatch.setattr(importlib, 'import_module', lambda n: (_ for _ in ()).throw(ImportError(n)) if n in ('uvicorn', 'insightface') else real(n))
    monkeypatch.setattr(gpu, 'query', lambda *a, **k: None)
    assert cli.main(['doctor']) == 1
    out = capsys.readouterr().out; assert 'MISSING  uvicorn' in out and 'needs insightface' in out and 'no NVIDIA GPU' in out


def test_status_jobs_and_registry_read_the_service(capsys, monkeypatch):
    import urllib.request
    status = dict(gpu=dict(name='RTX', used_mb=2048, total_mb=12288, util_pct=5), resident=[dict(model='buffalo_l', vram_mb=1024, jobs=3)], current=dict(id=7, role='faces', app='bh'), queued=[dict(id=8, kind='lease', role=None, note='arena', app='stills', priority='batch')])
    jobs = dict(items=[dict(id=7, status='done', kind='run', role='faces', note=None, app='bh', run_s=0.2)], next=None)
    monkeypatch.setattr(urllib.request, 'urlopen', lambda url, timeout: Resp(status if 'status' in url else jobs))
    assert cli.main(['status']) == 0; out = capsys.readouterr().out
    assert 'RTX: 2.0 of 12.0 GB' in out and 'loaded   buffalo_l' in out and 'running  #7 faces for bh' in out and 'queued   #8 lease arena for stills' in out
    assert cli.main(['status', '--json']) == 0 and json.loads(capsys.readouterr().out)['current']['id'] == 7
    assert cli.main(['jobs']) == 0 and '#7' in capsys.readouterr().out
    assert cli.main(['jobs', '--json']) == 0 and json.loads(capsys.readouterr().out)['items'][0]['id'] == 7
    status['gpu'] = None; assert cli.main(['status']) == 0 and 'no GPU' in capsys.readouterr().out
    assert cli.main(['registry']) == 0; out = capsys.readouterr().out
    assert 'faces' in out and 'buffalo_l' in out and 'not served yet: strata-360' in out
    assert cli.main(['registry', '--json']) == 0 and 'roles' in json.loads(capsys.readouterr().out)


def test_set_and_serve(capsys, monkeypatch):
    assert cli.main(['set', 'vram_reserve_gb']) == 0 and 'vram_reserve_gb = 1.0' in capsys.readouterr().out
    assert cli.main(['set', 'vram_reserve_gb', '2']) == 0 and '= 2.0' in capsys.readouterr().out
    assert cli.main(['set', 'vram_reserve_gb', 'default']) == 0 and '= 1.0' in capsys.readouterr().out
    with pytest.raises(SystemExit, match='unknown setting'): cli.main(['set', 'bad', 'x'])
    with pytest.raises(SystemExit, match='number'): cli.main(['set', 'vram_reserve_gb', 'x'])
    from stratamodels.server import app as server_app
    called = []; monkeypatch.setattr(server_app, 'main', lambda argv: called.append(argv))
    assert cli.main(['serve', '--port', '9000', '--memory']) == 0 and called == [['--host', '127.0.0.1', '--port', '9000', '--memory']]
