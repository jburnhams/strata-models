"""venv-worker (stratamodels.runtimes.worker): a model in a Python of its own, spoken to in JSON lines. A real child process stands in for the
worker (this test's own Python running a tiny script), so the protocol, failures and shutdown are exercised without any model."""
import json, os, sys, textwrap

import pytest

from stratamodels import gpu
from stratamodels.registry import Registry
from stratamodels.runtimes import RunError, worker

SCRIPT = textwrap.dedent('''
    import json, sys
    for line in sys.stdin:
        m = json.loads(line)
        if m['op'] == 'quit': break
        if m['op'] == 'load': print(json.dumps(dict(ok=m['model']['options'].get('engine') != 'broken', result=dict(engine=m['model']['options'].get('engine')), error='cannot load')), flush=True); continue
        p = m['params']
        if p.get('die'): sys.exit(3)
        if p.get('sleep'): import time; time.sleep(p['sleep'])
        if p.get('junk'): print('not json', flush=True); continue
        if p.get('bad'): print(json.dumps(dict(ok=False, params=True, error='bad text')), flush=True); continue
        if p.get('crash'): print(json.dumps(dict(ok=False, error='CUDA out of memory')), flush=True); continue
        print(json.dumps(dict(ok=True, result=dict(echo=p))), flush=True)
''')


@pytest.fixture
def env(tmp_path, monkeypatch):
    wd = tmp_path / 'workers'; wd.mkdir(); (wd / 'echo.py').write_text(SCRIPT); monkeypatch.setattr(worker, 'WORKERS', str(wd))
    venv = tmp_path / 'models' / 'venvs' / 'echo' / 'bin'; venv.mkdir(parents=True); os.symlink(sys.executable, venv / 'python')
    return str(tmp_path / 'models')


def model(**options): return Registry.from_dict(dict(roles=dict(speak=dict(model='m')), models=dict(m=dict(runtime='venv-worker', roles=['speak'], options=options)))).models['m']


def test_loads_runs_and_unloads(env):
    h = worker.load(model(worker='echo', engine='kokoro'), env); assert h['info'] == dict(engine='kokoro')
    r = worker.run(h, dict(text='hi')); assert r['echo']['text'] == 'hi' and r['echo']['out_dir'].endswith(os.path.join('generated', r['echo']['out_dir'][-10:]))
    with pytest.raises(RunError, match='bad text'): worker.run(h, dict(bad=1))                    # the job's fault: the model stays loaded
    assert worker.run(h, dict(text='again'))['echo']['text'] == 'again'
    with pytest.raises(RuntimeError, match='out of memory'): worker.run(h, dict(crash=1))
    with pytest.raises(RuntimeError, match='not JSON'): worker.run(h, dict(junk=1))
    p = h['worker'].p; worker.unload(h); assert p.poll() is not None and 'worker' not in h; worker.unload(h)
    assert os.path.isfile(os.path.join(env, 'venvs', 'echo.log'))


def test_a_worker_that_dies_or_hangs_is_reported(env):
    h = worker.load(model(worker='echo'), env)
    with pytest.raises(RuntimeError, match='has stopped'): worker.run(h, dict(die=1))
    with pytest.raises(RuntimeError, match='has stopped'): worker.run(h, dict(text='x'))
    worker.unload(h)
    h = worker.load(model(worker='echo'), env); w = h['worker']
    with pytest.raises(RuntimeError, match='more than 0.2 s'): w.ask(dict(op='run', params=dict(sleep=5)), timeout=0.2)
    assert w.p.poll() is not None


def test_refuses_what_cannot_load(env, tmp_path):
    with pytest.raises(RunError, match='names no worker'): worker.load(model(), env)
    with pytest.raises(RunError, match='no worker script'): worker.load(model(worker='nope'), env)
    with pytest.raises(RunError, match='not set up'): worker.load(model(worker='echo', venv='other'), env)
    with pytest.raises(RunError, match='could not load: cannot load'): worker.load(model(worker='echo', engine='broken'), env)
    assert worker.python_of(model(worker='echo', venv=str(tmp_path / 'v')), env)[0] == str(tmp_path / 'v')


def test_a_child_process_on_the_card_is_ours():
    class Out:
        def __init__(self, out, code=0): self.stdout, self.returncode = out, code
    def smi(cmd, **k): return Out('RTX, 12288, 3000, 10, 50, 40, 170') if '--query-gpu' in cmd[1] else Out('10, 100\n11, 200\n12, 300')
    s = gpu.query(smi, own_pid=10, parent=lambda pid: 10 if int(pid) == 11 else 1)
    assert [p['ours'] for p in s['processes']] == [True, True, False] and gpu.own_used_mb(s) == 300
    assert gpu.parent_of(os.getpid()) == os.getppid() and gpu.parent_of(-5) is None
