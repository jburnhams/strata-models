"""The Evals API (stratamodels.evals.api): listing, running in the background, the analysis, the write-up, blind judging, the media file."""
import time

import pytest
from fastapi.testclient import TestClient

from stratamodels.server.app import create_app


@pytest.fixture
def client(scheduler, setup):
    stop = scheduler.start(); yield TestClient(create_app(scheduler)); stop.set()


def wait_run(client, name):
    for _ in range(1200):                                                               # up to a minute: a loaded machine runs the fake jobs slowly
        st = client.get(f'/api/evals/{name}/status').json()
        if st['state'] != 'running': return st
        time.sleep(0.05)
    raise AssertionError('the run did not finish')


def test_list_run_analyse_and_report(client):
    d = client.get('/api/evals').json(); e = d['experiments'][0]
    assert e['name'] == 'fake-asr' and e['outputs'] == 0 and e['total'] == 120 and d['datasets'][0]['n'] == 40
    assert client.get('/api/evals/fake-asr/status').json()['state'] == 'idle'
    assert client.post('/api/evals/fake-asr/run', params=dict(limit=10)).json()['state'] == 'running'
    st = wait_run(client, 'fake-asr'); assert st['state'] == 'done' and st['result']['ran'] == 30
    assert client.post('/api/evals/fake-asr/run').status_code == 202 and wait_run(client, 'fake-asr')['result']['ran'] == 90
    a = client.get('/api/evals/fake-asr').json(); assert a['decision']['recommendation'] == 'better' and a['dataset']['compared'] == 40
    r = client.get('/api/evals/fake-asr/report'); assert r.status_code == 200 and r.text.startswith('# fake-asr') and '## Method' in r.text
    assert client.get('/api/evals/nope').status_code == 404 and client.post('/api/evals/nope/run').status_code == 404


def test_blind_judging(client, scheduler):
    assert client.get('/api/evals/fake-asr/pair').json() == dict(pair=None)
    client.post('/api/evals/fake-asr/run'); wait_run(client, 'fake-asr')
    p = client.get('/api/evals/fake-asr/pair').json()['pair']
    assert p['x'] and p['y'] and 'path' not in p['input'] and not p['has_media'] and 'better' not in p['token']
    j = client.post('/api/evals/fake-asr/judgments', json=dict(token=p['token'], winner='x')).json(); assert j['winner'] == j['x']
    for _ in range(5):
        p = client.get('/api/evals/fake-asr/pair').json()['pair']; client.post('/api/evals/fake-asr/judgments', json=dict(token=p['token'], winner='tie'))
    assert client.get('/api/evals/fake-asr').json()['judgments']['n'] == 6
    assert client.post('/api/evals/fake-asr/judgments', json=dict(token=p['token'], winner='z')).status_code == 400
    assert client.post('/api/evals/fake-asr/judgments', json=dict(token=p['token'][:-3] + 'abc', winner='x')).status_code == 400
    assert client.post('/api/evals/other/judgments', json=dict(token=p['token'], winner='x')).status_code == 400


def test_media_only_for_files_a_dataset_names(client, tmp_path, setup):
    from stratamodels.evals import core as C
    wav = tmp_path / 'a.wav'; wav.write_bytes(b'RIFF0000WAVE')
    C.write_dataset('speech', [dict(id='a', input=dict(path=str(wav), say='hello there'), reference=dict(text='hello there')), dict(id='b', input=dict(say='x'), reference=dict(text='x'))])
    r = client.get('/api/evals/fake-asr/media/a'); assert r.status_code == 200 and r.content == b'RIFF0000WAVE'
    assert client.get('/api/evals/fake-asr/media/b').status_code == 404 and client.get('/api/evals/fake-asr/media/zz').status_code == 404


def test_broken_specs_and_datasets_are_reported(client, setup):
    import os
    from stratamodels.evals import core as C
    folder = os.path.join(os.environ['STRATAMODELS_DATA'], 'evals', 'specs')
    open(os.path.join(folder, 'bad.toml'), 'w').write('name = ')
    open(os.path.join(folder, 'nodata.toml'), 'w').write('name = "nodata"\nquestion = "q"\nrole = "echo"\ndataset = "missing"\nmetrics = ["wer"]\n[[candidates]]\nname = "a"\nmodel = "small"\n')
    ex = {e['name']: e for e in client.get('/api/evals').json()['experiments']}
    assert 'error' in ex['bad'] and 'no dataset' in ex['nodata']['error'] and client.get('/api/evals/bad').status_code == 400
    assert client.get('/api/evals/nodata').status_code == 400
    one = os.path.join(folder, 'one.toml'); open(one, 'w').write('name = "one"\nquestion = "q"\nrole = "echo"\ndataset = "speech"\nmetrics = ["wer"]\n[[candidates]]\nname = "a"\nmodel = "small"\n')
    assert client.get('/api/evals/one/pair').status_code == 400
    def boom(*a, **k): raise RuntimeError('db away')
    C_run = C.run; C.run = boom
    try: client.post('/api/evals/one/run'); st = wait_run(client, 'one'); assert st['state'] == 'failed' and 'db away' in st['error']
    finally: C.run = C_run


def test_no_observations_come_back_as_null_not_nan(client):
    from stratamodels.evals.api import finite
    assert finite(dict(a=float('nan'), b=[1.0, float('inf')], c=(2, 'x'))) == dict(a=None, b=[1.0, None], c=[2, 'x'])
    a = client.get('/api/evals/fake-asr').json()                                                             # nothing run yet: every value is null, and it is valid JSON
    assert a['dataset']['compared'] == 0


def test_judgments_count_only_while_their_outputs_are_current(client, scheduler):
    from stratamodels.evals import core as C
    client.post('/api/evals/fake-asr/run'); wait_run(client, 'fake-asr')
    for _ in range(4):
        p = client.get('/api/evals/fake-asr/pair').json()['pair']; client.post('/api/evals/fake-asr/judgments', json=dict(token=p['token'], winner='x'))
    assert client.get('/api/evals/fake-asr').json()['judgments']['n'] == 4
    s = C.load_spec('fake-asr'); C.forget(s, scheduler.store.engine)                  # outputs regenerated: new keys? same keys (same params), so still counted
    assert len(C.judgments(scheduler.store.engine, 'fake-asr')) == 4
    keys = {(c.name, it['id']): 'changed' for c in s.candidates for it in C.load_dataset('speech').items}
    assert C.judgments(scheduler.store.engine, 'fake-asr', keys) == []


def test_pairs_with_the_same_output_are_not_offered(client, scheduler):
    from stratamodels.evals.api import same
    assert same(dict(segments=[dict(text=" Anders' face grew red.")]), dict(segments=[dict(text="anders' face grew red")]))
    assert not same(dict(segments=[dict(text='a b')]), dict(segments=[dict(text='a c')]))
    assert same(dict(path='/a.png', seconds=1.0), dict(path='/a.png', seconds=2.0)) and not same(dict(path='/a.png'), dict(path='/b.png'))
    client.post('/api/evals/fake-asr/run'); wait_run(client, 'fake-asr')
    seen = set()
    for _ in range(30):
        p = client.get('/api/evals/fake-asr/pair').json()['pair']
        assert not same(p['x'], p['y']); seen.add(p['item_id'])
        client.post('/api/evals/fake-asr/judgments', json=dict(token=p['token'], winner='tie'))
    assert len(seen) > 20                                                               # unjudged pairs come first, so it moves through the items


def test_media_of_a_phrase_is_cut_to_its_clip(client, tmp_path, setup, fake_run):
    from stratamodels.evals import core as C
    wav = tmp_path / 'long.wav'; wav.write_bytes(b'RIFF' + b'0' * 1000)
    C.write_dataset('speech', [dict(id='p', input=dict(path=str(wav), clip=[2.0, 3.0], say='x'), reference=dict(text='x'))])
    fake_run.returns['ffmpeg'] = b'RIFFcut'
    r = client.get('/api/evals/fake-asr/media/p'); assert r.headers['content-type'] == 'audio/wav' and r.content == b'RIFFcut'
    cmd = fake_run.calls[-1]; assert cmd[cmd.index('-ss') + 1] == '2.0' and cmd[cmd.index('-to') + 1] == '3.0'
    fake_run.code = 1; assert client.get('/api/evals/fake-asr/media/p').content.startswith(b'RIFF0')          # ffmpeg failed: the whole file

def test_the_most_different_pairs_come_first(client, scheduler, setup):
    from stratamodels.evals import core as C
    client.post('/api/evals/fake-asr/run'); wait_run(client, 'fake-asr')
    first = client.get('/api/evals/fake-asr/pair').json()
    assert 0 < first['pair']['similarity'] < 1 and first['remaining'] > 0
    sims = []
    for _ in range(10):
        p = client.get('/api/evals/fake-asr/pair').json(); sims.append(p['pair']['similarity'])
        client.post('/api/evals/fake-asr/judgments', json=dict(token=p['pair']['token'], winner='tie'))
    assert sims == sorted(sims) or max(sims) - min(sims) < 0.05                     # least alike first (ties within 0.02 are picked at random)
    import pytest
    with pytest.raises(C.EvalError, match='unknown similarity'): C.parse_spec(dict(name='x', question='q', role='r', dataset='d', metrics=['wer'], similarity='video', candidates=[dict(name='a', model='m')]))


def test_judging_skips_only_judgments_of_current_outputs_and_rescores_new_outputs(client, scheduler):
    from stratamodels.evals import core as C
    from stratamodels.db.models import EvalJudgment
    client.post('/api/evals/fake-asr/run'); wait_run(client, 'fake-asr')
    first = client.get('/api/evals/fake-asr/pair').json(); n0 = first['remaining']
    with C.session(scheduler.store.engine) as ses:                                    # a stale judgment (old outputs) of every pair: must not count as judged
        for it in C.load_dataset('speech').items: ses.add(EvalJudgment(experiment='fake-asr', item_id=it['id'], a='base', b='better', winner='tie', rater='x', at=1.0, a_key='old', b_key='old'))
        ses.commit()
    assert client.get('/api/evals/fake-asr/pair').json()['remaining'] == n0
    s = C.load_spec('fake-asr'); C.forget(s, scheduler.store.engine, 'same')
    client.post('/api/evals/fake-asr/run'); wait_run(client, 'fake-asr')
    assert client.get('/api/evals/fake-asr/pair').json()['pair'] is not None
