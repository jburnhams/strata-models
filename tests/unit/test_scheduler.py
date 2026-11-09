"""The queue and the GPU budget (stratamodels.scheduler): order, residency, eviction, leases, failures, waiting."""
import threading, time

import pytest

from stratamodels.registry import RegistryError
from stratamodels.runtimes import fake
from stratamodels.scheduler import QueueError


def test_a_job_loads_its_model_once_and_later_jobs_reuse_it(scheduler):
    a = scheduler.submit('echo', dict(x=1), 'bright-horizons')
    assert a['status'] == 'queued' and a['model'] == 'small' and a['kind'] == 'run'
    done = scheduler.step()
    assert done['status'] == 'done' and done['result'] == dict(model='small', echo=dict(x=1)) and done['load_s'] >= 0
    assert list(scheduler.resident) == ['small'] and scheduler.resident['small'].vram_mb == 2048
    scheduler.submit('echo', dict(x=2), 'strata-360'); scheduler.step()
    assert len(fake.LOADED) == 1 and scheduler.resident['small'].jobs == 2
    assert scheduler.step() is None                                                 # nothing left


def test_unknown_roles_models_and_priorities_are_refused(scheduler):
    with pytest.raises(RegistryError, match='no role'): scheduler.submit('nope', {}, 'x')
    with pytest.raises(RegistryError, match='does not serve'): scheduler.submit('echo', {}, 'x', model='other')
    with pytest.raises(RegistryError, match='not served here yet'): scheduler.submit('later', {}, 'x')
    with pytest.raises(QueueError, match='priority'): scheduler.submit('echo', {}, 'x', priority='urgent')
    with pytest.raises(QueueError, match='priority'): scheduler.lease('x', priority='now')
    with pytest.raises(QueueError, match='vram_gb'): scheduler.lease('x', vram_gb=-1)


def test_interactive_goes_first_then_jobs_for_the_loaded_model(scheduler, clock):
    scheduler.submit('other', {}, 'a'); scheduler.step()                            # `other` is now loaded
    b1 = scheduler.submit('echo', {}, 'batch-1')
    b2 = scheduler.submit('other', {}, 'batch-2')
    i = scheduler.submit('echo', {}, 'web', priority='interactive')
    assert scheduler.step()['id'] == i['id']                                         # interactive first, whatever it needs
    assert scheduler.step()['id'] == b1['id']                                        # `small` is loaded now, so its batch job goes before `other`'s
    assert scheduler.step()['id'] == b2['id']


def test_a_job_that_waited_too_long_is_not_skipped_for_the_loaded_model(scheduler, clock):
    scheduler.submit('echo', {}, 'a'); scheduler.step()
    old = scheduler.submit('other', {}, 'old'); clock.t += 500
    scheduler.submit('echo', {}, 'new')
    assert scheduler.step()['id'] == old['id']


def test_models_are_evicted_least_recently_used_to_make_room(scheduler, clock):
    scheduler.submit('echo', {}, 'a'); scheduler.step(); clock.t += 1
    scheduler.submit('other', {}, 'a'); scheduler.step(); clock.t += 1
    assert list(scheduler.resident) == ['small', 'other']                             # 4 GB of 12, 0.5 GB other, 1 GB reserve
    scheduler.submit('echo', {}, 'a'); scheduler.step()                              # touches `small`: now `other` is the oldest
    scheduler.submit('echo', {}, 'a', model='big'); scheduler.step()                 # 6 GB: 12 - 0.5 - 1 - 4 = 6.5 free, fits without eviction
    assert set(scheduler.resident) == {'small', 'other', 'big'}
    scheduler.probe.other_mb = 3000                                                  # someone else took 2.5 GB more
    scheduler._unload('big', 'test'); scheduler.submit('echo', {}, 'a', model='big'); scheduler.step()
    assert 'other' not in scheduler.resident and 'big' in scheduler.resident and any('unloaded other' in e['text'] for e in scheduler.events)


def test_without_a_gpu_nothing_is_evicted_and_status_has_no_budget(scheduler):
    scheduler.probe.present = False
    for m in ('small', 'big'): scheduler.submit('echo', {}, 'a', model=m); scheduler.step()
    assert set(scheduler.resident) == {'small', 'big'}
    st = scheduler.status()
    assert st['gpu'] is None and st['budget'] is None and len(st['resident']) == 2


def test_a_lease_frees_the_gpu_and_holds_it_until_released(scheduler, clock):
    scheduler.submit('echo', {}, 'a'); scheduler.step()
    lease = scheduler.lease('strata-stills', note='arena build')
    assert lease['status'] == 'queued' and lease['ttl_s'] == 120
    queued_job = scheduler.submit('echo', {}, 'b')
    granted = scheduler.step()
    assert granted['id'] == lease['id'] and granted['status'] == 'running' and scheduler.resident == {}        # no figure: the whole GPU
    assert scheduler.step() is None and scheduler.store.get(queued_job['id'])['status'] == 'queued'          # nothing runs beside a lease
    clock.t += 100; scheduler.heartbeat(lease['id']); clock.t += 100
    assert scheduler.step() is None                                                  # the heartbeat kept it alive
    assert scheduler.release(lease['id'])['status'] == 'done'
    assert scheduler.step()['id'] == queued_job['id']
    with pytest.raises(QueueError, match='is done'): scheduler.heartbeat(lease['id'])


def test_a_lease_with_a_figure_only_frees_that_much(scheduler):
    scheduler.submit('echo', {}, 'a'); scheduler.step()
    scheduler.lease('strata-stills', vram_gb=4); scheduler.step()
    assert list(scheduler.resident) == ['small']                                     # 12 - 0.5 - 1 - 2 = 8.5 GB free already


def test_a_silent_lease_expires_and_a_queued_one_can_be_withdrawn(scheduler, clock):
    l1 = scheduler.lease('a', ttl_s=10); l2 = scheduler.lease('b')
    scheduler.step(); clock.t += 11
    assert scheduler.step()['id'] == l2['id']                                         # l1 expired, l2 granted
    assert scheduler.store.get(l1['id'])['status'] == 'expired'
    l3 = scheduler.lease('c')
    assert scheduler.release(l3['id'])['status'] == 'cancelled'
    assert scheduler.cancel(l2['id'])['status'] == 'done'                             # cancel on a lease gives it back
    with pytest.raises(QueueError, match='no lease'): scheduler.heartbeat(999)
    j = scheduler.submit('echo', {}, 'a')
    with pytest.raises(QueueError, match='no lease'): scheduler.release(j['id'])


def test_cancel_withdraws_queued_jobs_only(scheduler):
    a = scheduler.submit('echo', {}, 'a'); b = scheduler.submit('echo', {}, 'a')
    assert scheduler.cancel(a['id'])['status'] == 'cancelled'
    scheduler.step(); assert scheduler.cancel(b['id'])['status'] == 'done'
    with pytest.raises(QueueError, match='no job'): scheduler.cancel(999)


def test_run_errors_fail_the_job_and_crashes_unload_the_model(scheduler, monkeypatch):
    j = scheduler.submit('echo', dict(fail='no such image: x.jpg'), 'a'); r = scheduler.step()
    assert r['status'] == 'failed' and r['error'] == 'no such image: x.jpg' and 'small' in scheduler.resident
    monkeypatch.setattr(fake, 'run', lambda h, p: (_ for _ in ()).throw(RuntimeError('CUDA out of memory')))
    scheduler.submit('echo', {}, 'a'); r = scheduler.step()
    assert r['status'] == 'failed' and r['error'] == 'RuntimeError: CUDA out of memory' and 'small' not in scheduler.resident
    monkeypatch.setattr(fake, 'load', lambda m, d: (_ for _ in ()).throw(ImportError('insightface')))
    scheduler.submit('echo', {}, 'a'); assert scheduler.step()['error'] == 'ImportError: insightface'
    assert j['id'] < r['id']


def test_a_job_for_a_model_dropped_from_the_registry_fails(scheduler):
    j = scheduler.submit('echo', {}, 'a'); del scheduler.registry.models['small']
    assert 'no longer served' in scheduler.step()['error'] and j


def test_load_and_unload_requests_run_on_the_worker(scheduler, monkeypatch):
    scheduler.request('load', 'big'); assert scheduler.resident == {}
    scheduler.step(); assert list(scheduler.resident) == ['big']
    scheduler.request('unload', 'big'); scheduler.step(); assert scheduler.resident == {}
    with pytest.raises(QueueError, match='no model'): scheduler.request('load', 'nope')
    with pytest.raises(QueueError, match='not served'): scheduler.request('load', 'planned')
    with pytest.raises(QueueError, match='no action'): scheduler.request('reload', 'big')
    monkeypatch.setattr(fake, 'load', lambda m, d: (_ for _ in ()).throw(OSError('disk')))
    scheduler.request('load', 'small'); scheduler.step()
    assert any('could not load small' in e['text'] for e in scheduler.events)
    monkeypatch.setattr(fake, 'unload', lambda h: (_ for _ in ()).throw(OSError('busy')))
    scheduler.resident.clear(); scheduler.submit('echo', {}, 'a', model='big')
    monkeypatch.undo(); scheduler.step(); monkeypatch.setattr(fake, 'unload', lambda h: (_ for _ in ()).throw(OSError('busy')))
    scheduler._unload('big', 'test'); assert 'big' not in scheduler.resident


def test_status_reports_budget_queue_and_events(scheduler):
    scheduler.submit('echo', {}, 'a'); scheduler.step(); scheduler.lease('x', note='render')
    st = scheduler.status()
    b = st['budget']
    assert b['total_mb'] == 12288 and b['own_mb'] == 2048 and b['other_mb'] == 500 and b['reserve_mb'] == 1024 and b['free_for_models_mb'] == 12288 - 2548 - 1024
    assert st['resident'][0]['model'] == 'small' and st['queued'][0]['note'] == 'render' and st['current'] is None and st['lease'] is None
    scheduler.step(); assert scheduler.status()['lease']['note'] == 'render'
    assert st['events'][0]['text'].startswith('loaded small')


def test_wait_returns_when_done_granted_or_timed_out(scheduler):
    j = scheduler.submit('echo', {}, 'a')
    assert scheduler.wait(j['id'], 0)['status'] == 'queued'
    t = threading.Thread(target=lambda: (time.sleep(0.05), scheduler.step())); t.start()
    assert scheduler.wait(j['id'], 5)['status'] == 'done'; t.join()
    lease = scheduler.lease('a'); scheduler.step()
    assert scheduler.wait(lease['id'], 5)['status'] == 'running'
    with pytest.raises(QueueError): scheduler.wait(999)


def test_the_worker_thread_runs_jobs_and_recovers_after_a_restart(scheduler):
    stale = scheduler.store.add(kind='run', role='echo', model='small', app='a', priority='batch', status='running')
    stop = scheduler.start()
    try:
        assert scheduler.store.get(stale['id'])['status'] == 'failed' and 'service stopped' in scheduler.events[0]['text']
        j = scheduler.submit('echo', dict(seconds=0), 'a')
        assert scheduler.wait(j['id'], 10)['status'] == 'done'
    finally: stop.set()


def test_the_worker_survives_an_error_in_a_step(scheduler, monkeypatch):
    calls = []
    def boom(): calls.append(1); raise RuntimeError('db away')
    monkeypatch.setattr(scheduler, 'step', boom)
    stop = scheduler.start(); deadline = time.time() + 5
    while not any('worker error' in e['text'] for e in scheduler.events) and time.time() < deadline: time.sleep(0.01)
    stop.set(); assert calls and any('db away' in e['text'] for e in scheduler.events)


def test_load_requests_wait_while_a_lease_is_held(scheduler):
    lease = scheduler.lease('stills'); scheduler.step()
    scheduler.request('load', 'small'); scheduler.step()
    assert scheduler.resident == {} and len(scheduler.control) == 1
    scheduler.release(lease['id']); scheduler.step()
    assert list(scheduler.resident) == ['small']


def test_a_lease_withdrawn_while_room_is_made_stays_withdrawn(scheduler, monkeypatch):
    scheduler.submit('echo', {}, 'a'); scheduler.step()
    lease = scheduler.lease('stills')
    real = scheduler._make_room
    def withdraw_meanwhile(need, keep=None): real(need, keep); scheduler.release(lease['id'])
    monkeypatch.setattr(scheduler, '_make_room', withdraw_meanwhile)
    assert scheduler.step() is None and scheduler.store.get(lease['id'])['status'] == 'cancelled'
    j = scheduler.submit('echo', {}, 'a'); real_claim = scheduler.store.claim
    monkeypatch.setattr(scheduler.store, 'claim', lambda i, **kw: (scheduler.cancel(i), real_claim(i, **kw))[1])
    assert scheduler.step() is None and scheduler.store.get(j['id'])['status'] == 'cancelled'


def test_a_sized_lease_waits_for_room_and_one_that_never_fits_fails(scheduler):
    scheduler.probe.other_mb = 9000                                                  # 12 GB card, 9 GB elsewhere, 1 GB reserve: 2.3 GB free
    lease = scheduler.lease('stills', vram_gb=4); job = scheduler.submit('echo', {}, 'a')
    assert scheduler.step() is None and scheduler.step() is None and scheduler.store.get(lease['id'])['status'] == 'queued'
    assert sum('waits for 4.0 GB' in e['text'] for e in scheduler.events) == 1 and scheduler.store.get(job['id'])['status'] == 'queued'
    scheduler.probe.other_mb = 500
    assert scheduler.step()['status'] == 'running'
    scheduler.release(lease['id'])
    huge = scheduler.lease('stills', vram_gb=20); scheduler.step()                    # the echo job goes first (submitted before)
    r = scheduler.step(); assert r['id'] == huge['id'] and r['status'] == 'failed' and 'never fit' in r['error']


def test_a_sized_lease_shares_the_card_with_jobs_that_fit(scheduler):
    lease = scheduler.lease('strata-stills', vram_gb=6, pid=777, note='arena build'); assert lease['params'] == dict(pid=777)
    assert scheduler.step()['status'] == 'running'                                  # granted: 12 - 0.5 - 1 = 10.5 GB free
    st = scheduler.status(); assert st['budget']['reserved_mb'] == 6 * 1024 and st['leases'][0]['id'] == lease['id'] and st['lease']['id'] == lease['id']
    small = scheduler.submit('echo', {}, 'bh')                                       # 2 GB: fits in the 4.5 GB left
    assert scheduler.step()['id'] == small['id'] and 'small' in scheduler.resident
    big = scheduler.submit('echo', {}, 'x', model='big')                             # 6 GB: does not fit beside the 6 GB reservation
    other = scheduler.submit('other', {}, 'y')                                       # 2 GB: does
    assert scheduler.step()['id'] == other['id'] and scheduler.store.get(big['id'])['status'] == 'queued'
    assert scheduler.step() is None                                                  # only the big one is left, and it waits
    scheduler.probe.extra = [(777, 6 * 1024)]                                        # the app now uses its 6 GB: nothing outstanding, but the card is just as full
    assert scheduler.status()['budget']['reserved_mb'] == 0 and scheduler.step() is None
    scheduler.release(lease['id']); scheduler.probe.extra = []
    assert scheduler.step()['id'] == big['id']


def test_an_exclusive_lease_waits_for_sized_ones_and_blocks_everything(scheduler):
    a = scheduler.lease('a', vram_gb=2); b = scheduler.lease('b', vram_gb=2)
    assert scheduler.step()['id'] == a['id'] and scheduler.step()['id'] == b['id']     # two sized leases held together
    whole = scheduler.lease('c'); job = scheduler.submit('echo', {}, 'x')
    assert scheduler.step() is None and scheduler.step() is None and any('waits for the whole GPU' in e['text'] for e in scheduler.events)
    assert scheduler.store.get(job['id'])['status'] == 'queued'                      # nothing passes the waiting lease
    scheduler.release(a['id']); scheduler.release(b['id'])
    assert scheduler.step()['id'] == whole['id'] and scheduler.step() is None and scheduler.store.get(job['id'])['status'] == 'queued'
    scheduler.release(whole['id']); assert scheduler.step()['id'] == job['id']


def test_fits_without_a_gpu_or_for_an_unknown_model(scheduler):
    scheduler.lease('a', vram_gb=2); scheduler.step()
    j = scheduler.submit('echo', {}, 'x'); j['model'] = 'gone'; assert scheduler._fits(j)
    scheduler.probe.present = False; assert scheduler._fits(dict(j, model='big'))


def test_a_queued_variant_job_survives_a_restart(scheduler):
    j = scheduler.submit('echo', dict(x=1), 'evals', model='small', options=dict(tweak=2))
    assert j['params'] == dict(x=1, __options__=dict(tweak=2)) and '~' in j['model']
    del scheduler.registry.models[j['model']]                                        # what a fresh registry looks like after a restart
    done = scheduler.step()
    assert done['status'] == 'done' and done['result']['echo'] == dict(x=1) and j['model'] in scheduler.registry.models
    k = scheduler.submit('echo', {}, 'evals', model='small', options=dict(tweak=3)); del scheduler.registry.models[k['model']]
    del scheduler.registry.models['small']; assert 'no longer served' in scheduler.step()['error']


def test_a_resident_model_whose_runtime_or_options_were_edited_is_reloaded(scheduler):
    import dataclasses
    scheduler.submit('echo', {}, 'a'); scheduler.step(); old = scheduler.resident['small'].handle
    reg = scheduler.registry; m = reg.models['small']
    reg.models['small'] = dataclasses.replace(m, vram_gb=m.vram_gb + 0.5)               # a figure the handle does not depend on: kept
    scheduler.submit('echo', {}, 'a'); scheduler.step(); assert scheduler.resident['small'].handle is old
    reg.models['small'] = dataclasses.replace(m, options=dict(ctx=4096))                # the handle was loaded with other settings: reloaded
    scheduler.submit('echo', {}, 'a'); scheduler.step()
    assert scheduler.resident['small'].handle is not old and any('were edited' in e['text'] for e in scheduler.events)
