"""Experiments end to end on the fake runtime (stratamodels.evals.core and report): datasets, specs, cached runs, analysis, the decision, the write-up."""
import json, os

import pytest

from stratamodels.evals import core as C, report as R
from stratamodels.runtimes import fake



@pytest.fixture
def engine(scheduler): return scheduler.store.engine


def test_datasets_are_versioned_and_checked(setup, tmp_path):
    ds = C.load_dataset('speech'); assert ds.version == setup.version and len(ds.items) == 40 and ds.description == 'forty fake utterances for tests'
    assert ds.to_json()['tags'] == ['speaker'] and 'items' in ds.to_json(items=True)
    assert [d['name'] for d in C.list_datasets()] == ['speech']
    bad = tmp_path / 'bad'; bad.mkdir()
    for name, text, msg in (('dup', '{"id": 1, "input": {}}\n{"id": 1, "input": {}}', 'twice'), ('noinput', '{"id": 1}', 'needs'), ('json', '{x', 'line 1'), ('empty', '# nothing\n', 'no items')):
        (bad / f'{name}.jsonl').write_text(text)
        with pytest.raises(C.EvalError, match=msg): C.load_dataset(name, str(bad))
    assert {d['name']: 'error' in d for d in C.list_datasets(str(bad))} == dict(dup=True, empty=True, json=True, noinput=True)
    with pytest.raises(C.EvalError, match='no dataset'): C.load_dataset('nope')


def test_specs_are_parsed_and_refused_when_wrong(setup):
    s = C.load_spec('fake-asr'); assert s.baseline == 'base' and s.guards == dict(latency_s=1000.0) and s.judge and s.candidate('better').model == 'big'
    with pytest.raises(C.EvalError, match='no candidate'): s.candidate('x')
    with pytest.raises(C.EvalError, match='no experiment'): C.load_spec('nope')
    with pytest.raises(C.EvalError, match='unknown metrics'): C.parse_spec(dict(name='x', question='q', role='r', dataset='d', metrics=['bleu'], candidates=[dict(name='a', model='m')]))
    with pytest.raises(C.EvalError, match='share a name'): C.parse_spec(dict(name='x', question='q', role='r', dataset='d', metrics=['wer'], candidates=[dict(name='a', model='m'), dict(name='a', model='m')]))
    with pytest.raises(C.EvalError, match='KeyError'): C.parse_spec(dict(name='x'))
    p = C.parse_spec(dict(name='x', question='q', role='r', dataset='d', metrics=['cer'], decision=dict(primary='wer', guards=dict(failed=1)), candidates=[dict(name='a', model='m')]))
    assert p.metrics == ['wer', 'cer', 'failed'] and p.baseline == 'a'
    open(os.path.join(os.environ['STRATAMODELS_DATA'], 'evals', 'specs', 'broken.toml'), 'w').write('name = ')
    assert isinstance(C.list_specs()['broken'], C.EvalError)
    with pytest.raises(C.EvalError): C.load_spec('broken')


def test_run_caches_outputs_and_analysis_decides(setup, scheduler, engine):
    stop = scheduler.start()
    try:
        s = C.load_spec('fake-asr'); log = []
        r = C.run(s, engine, C.SchedulerExecutor(scheduler), log=log.append); assert r == dict(ran=120, cached=0, failed=0) and 'to run' in log[0]
        assert C.run(s, engine, C.SchedulerExecutor(scheduler), log=log.append) == dict(ran=0, cached=120, failed=0)
        assert any(k.startswith('small~') for k in scheduler.registry.models)                 # the candidate with options ran on a variant of its own
        res = C.analyse(s, engine)
        assert res['dataset']['compared'] == 40 and res['candidates']['base']['metrics']['wer']['value'] > 0.2 > res['candidates']['better']['metrics']['wer']['value']
        assert res['comparisons']['better']['wer']['improves'] is True and res['comparisons']['same']['wer']['improves'] is None and 'p_holm' in res['comparisons']['same']['wer']
        d = res['decision']; assert d['recommendation'] == 'better' and d['change'] and d['per_candidate']['better']['verdict'] == 'adopt' and d['per_candidate']['same']['verdict'] == 'unclear'
        assert res['comparisons']['better']['latency_s']['n'] == 40
        ds, outs, _ = C.outputs_for(s, engine); text = R.render(res, R.worst(s, outs, ds.items, 'wer'), date='2026-11-09')
        for h in ('## Answer', '## Setup', '## Results', '## Compared with the baseline', '## Cost', '## Failures and worst items', '## Method', '## Reproduce', '**Change to `better`.**'): assert h in text
        assert 'u0' in text and '"drop": 12' in text
        assert C.forget(s, engine, 'same') == 40 and C.run(s, engine, C.SchedulerExecutor(scheduler), candidates=['same'], limit=5, log=log.append)['ran'] == 5
        partial = C.analyse(s, engine); assert partial['dataset']['compared'] == 5
    finally: stop.set()


def test_failures_guards_and_keeping_the_baseline(setup, scheduler, engine, monkeypatch):
    s = C.parse_spec(dict(name='g', question='q', role='echo', dataset='speech', metrics=['wer', 'failed'], decision=dict(primary='wer', baseline='a', min_effect=0.5, guards=dict(failed=1.0)),
                          candidates=[dict(name='a', model='small', params=dict(drop=4)), dict(name='b', model='big', params=dict(drop=12)), dict(name='c', model='big', params=dict(fail='broken'))]))
    stop = scheduler.start()
    try: assert C.run(s, engine, C.SchedulerExecutor(scheduler), log=lambda m: None)['failed'] == 40
    finally: stop.set()
    res = C.analyse(s, engine); d = res['decision']['per_candidate']
    assert d['b']['verdict'] == 'reject' and 'under the' in d['b']['why'] and d['c']['verdict'] == 'reject' and not res['decision']['change']
    text = R.render(res, R.worst(s, C.outputs_for(s, engine)[1], C.load_dataset('speech').items, 'wer')); assert '**Keep `a`.**' in text and 'failed: broken' in text
    s2 = C.parse_spec(dict(name='g2', question='q', role='echo', dataset='speech', metrics=['wer'], decision=dict(primary='wer', baseline='b'), candidates=[dict(name='b', model='big', params=dict(drop=12)), dict(name='a', model='small', params=dict(drop=4))]))
    assert C.analyse(s2, engine)['decision']['per_candidate']['a']['verdict'] == 'reject'
    s3 = C.parse_spec(dict(name='g3', question='q', role='echo', dataset='speech', metrics=['latency_s'], decision=dict(primary='wer', baseline='a', guards=dict(latency_s=0.0001)), candidates=[dict(name='a', model='small', params=dict(drop=4)), dict(name='b', model='big', params=dict(drop=12))]))
    assert C.analyse(s3, engine)['decision']['per_candidate']['b']['verdict'] in ('reject', 'adopt')


def test_executors_pass_through(scheduler):
    calls = []
    class SM:
        def submit(self, role, params, app, priority, model, options): calls.append((role, app, priority, model, options)); return dict(id=7)
        def wait_job(self, i): return dict(id=i, status='done')
    ex = C.ClientExecutor(SM()); assert ex.submit('r', {}, 'm', {}) == 7 and ex.wait(7)['status'] == 'done' and calls == [('r', 'evals', 'batch', 'm', None)]
    j = C.SchedulerExecutor(scheduler).submit('echo', dict(x=1), 'small', None); scheduler.step()
    assert C.SchedulerExecutor(scheduler).wait(j)['status'] == 'done'
