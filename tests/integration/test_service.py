"""The real service process, driven through the client package as an app would: jobs, failures, leases, cancellation, history."""
import threading, time, urllib.request, json, warnings

import pytest
import strata_models_client as sm


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as r: return json.load(r)


def test_run_submit_wait_and_failures(service):
    assert sm.available()
    assert sm.run('echo', dict(a=1), app='it') == dict(model='echo-small', echo=dict(a=1))
    j = sm.submit('echo', dict(seconds=0.2), app='it')
    with pytest.raises(TimeoutError): sm.wait(j['id'], timeout=0.01)
    assert sm.wait(j['id'], timeout=10)['echo'] == dict(seconds=0.2)
    with pytest.raises(sm.JobFailed, match='not a real file'): sm.run('echo', dict(fail='not a real file'), app='it')
    with pytest.raises(sm.JobFailed, match='no role'): sm.run('nope', {}, app='it')
    with pytest.raises(sm.JobFailed, match="does not serve"): sm.faces('/x.jpg', app='it', model='echo-small')
    with pytest.raises(sm.JobFailed, match="does not serve"): sm.transcribe('/a.wav', app='it', model='buffalo_l', clip=(0, 1))
    with pytest.raises(sm.JobFailed, match="does not serve"): sm.speak('hello', app='it', model='buffalo_l', voice=__file__)
    assert sm.role_model('echo') == 'echo-small' and sm.role_model('nope') is None
    hist = get(service, '/api/jobs?app=it')['items']
    assert [x['status'] for x in hist[:2]] == ['failed', 'done']
    assert get(service, '/api/status')['resident'][0]['model'] == 'echo-small'


def test_a_sized_lease_lets_jobs_run_beside_it(service):
    with sm.gpu_lease('it-stills', vram_gb=1, ttl_s=30, note='shared') as lease:
        assert lease.job['status'] == 'running' and sm.run('echo', dict(z=1), app='it', timeout=20) == dict(model='echo-small', echo=dict(z=1))


def test_a_whole_gpu_lease_holds_it_and_jobs_wait_for_it(service):
    order = []
    with sm.gpu_lease('it-stills', vram_gb=None, ttl_s=6, note='integration') as lease:
        assert lease.job['status'] == 'running'
        t = threading.Thread(target=lambda: order.append(('job', sm.run('echo', {}, app='it')))); t.start()
        time.sleep(3)                                                    # longer than ttl/3: the heartbeat thread must keep it
        assert get(service, f'/api/leases/{lease.id}')['status'] == 'running'
        order.append(('lease-end', None))
    t.join(10)
    assert [o[0] for o in order] == ['lease-end', 'job']
    assert get(service, f'/api/leases/{lease.id}')['status'] == 'done'


def test_a_second_lease_waits_and_can_time_out(service):
    first = sm.acquire('it-a', ttl_s=30)
    try:
        with pytest.raises(TimeoutError): sm.acquire('it-b', ttl_s=30, timeout=0.5)
        cancelled = get(service, '/api/jobs?kind=lease&app=it-b')['items'][0]
        assert cancelled['status'] == 'cancelled'
    finally: first.release()


def test_without_the_service_the_client_says_so(monkeypatch):
    monkeypatch.setenv('STRATA_MODELS_URL', 'http://127.0.0.1:9')
    assert not sm.available(timeout=0.5)
    with pytest.raises(sm.Unavailable): sm.run('echo', {}, app='x')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        with sm.gpu_lease('x') as lease: assert lease is None
    assert 'not running' in str(w[0].message)
    with pytest.raises(sm.Unavailable):
        with sm.gpu_lease('x', required=True): pass


def test_the_judge_plays_only_the_phrase(service, tmp_path):
    import os, shutil, subprocess
    if not shutil.which('ffmpeg'): pytest.skip('needs ffmpeg')
    wav = tmp_path / 'long.wav'; subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'sine=f=440:d=10', str(wav)], check=True)
    data = os.environ['STRATAMODELS_DATA']; os.makedirs(os.path.join(data, 'evals', 'datasets'), exist_ok=True); os.makedirs(os.path.join(data, 'evals', 'specs'), exist_ok=True)
    open(os.path.join(data, 'evals', 'datasets', 'phr.jsonl'), 'w').write(json.dumps(dict(id='p', input=dict(path=str(wav), clip=[2.0, 3.0]), reference=dict(text='x'))) + '\n')
    open(os.path.join(data, 'evals', 'specs', 'phr.toml'), 'w').write('name = "phr"\nquestion = "q"\nrole = "echo"\ndataset = "phr"\nmetrics = ["failed"]\n[[candidates]]\nname = "a"\nmodel = "echo-small"\n')
    with urllib.request.urlopen(service + '/api/evals/phr/media/p?cut=1', timeout=30) as r: body = r.read(); ctype = r.headers['content-type']
    assert ctype == 'audio/wav' and len(body) < wav.stat().st_size / 5
