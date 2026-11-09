"""nvidia-smi parsing and the sample history (stratamodels.gpu)."""
import os, subprocess

from stratamodels import gpu


class Out:
    def __init__(self, stdout, code=0): self.stdout, self.returncode = stdout, code


def smi(gpu_line='NVIDIA GeForce RTX 3060, 12288, 1841, 7, 57, 53.12, 170.00', apps='1036378, 1118\n%d, 364\n' % os.getpid(), apps_code=0):
    def run(cmd, **kw): return Out(gpu_line) if '--query-gpu' in cmd[1] else Out(apps, apps_code)
    return run


def test_query_parses_the_card_and_its_processes():
    s = gpu.query(smi())
    assert s['name'] == 'NVIDIA GeForce RTX 3060' and s['total_mb'] == 12288 and s['free_mb'] == 12288 - 1841 and s['util_pct'] == 7 and s['power_limit_w'] == 170
    own = [p for p in s['processes'] if p['ours']]
    assert len(s['processes']) == 2 and own[0]['used_mb'] == 364 and 'python' in own[0]['command'] and gpu.own_used_mb(s) == 364
    assert gpu.query(smi(apps='garbage\n')).get('processes') == [] and gpu.query(smi(apps_code=9))['processes'] == []
    assert gpu.query(smi(gpu_line='RTX, 12288, [N/A], [N/A], 40, [N/A], [N/A]'))['util_pct'] is None


def test_query_is_none_without_an_nvidia_gpu():
    def missing(cmd, **kw): raise FileNotFoundError('nvidia-smi')
    assert gpu.query(missing) is None and gpu.query(lambda c, **k: Out('', 9)) is None and gpu.query(lambda c, **k: Out('a, b')) is None
    def timeout(cmd, **kw): raise subprocess.TimeoutExpired(cmd, 10)
    assert gpu.query(timeout) is None and gpu.own_used_mb(None) == 0


def test_command_of_reads_proc_or_gives_empty():
    assert 'python' in gpu.command_of(os.getpid()) and gpu.command_of(2 ** 31) == '' and gpu.command_of('x') == ''


def test_history_keeps_the_last_samples_and_its_thread_stops(clock):
    s = gpu.query(smi()); h = gpu.History(lambda: s, every_s=5, keep_s=10, clock=clock)
    for _ in range(3): h.sample(); clock.t += 5
    assert len(h.to_json()) == 2 and h.to_json()[-1]['used_mb'] == 1841 and h.latest is s
    h2 = gpu.History(lambda: None); assert h2.sample() is None and h2.to_json() == []
    calls = []
    h3 = gpu.History(lambda: calls.append(1) or (_ for _ in ()).throw(OSError()), every_s=0.01); h3.start()
    import time; deadline = time.time() + 5
    while len(calls) < 2 and time.time() < deadline: time.sleep(0.01)
    h3.stop(); assert len(calls) >= 2
