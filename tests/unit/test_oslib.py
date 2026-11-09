"""OS helpers (app.oslib): file lock, process liveness, termination, priority, CLI command."""
import os, subprocess, sys, time, types
from app import oslib


def test_file_lock_excludes_other_processes(tmp_path):
    p = tmp_path / 'lock'
    code = ("import sys, time; from app import oslib\n"
            "with open(sys.argv[1], 'a+') as f, oslib.file_lock(f):\n    print('locked', flush=True); time.sleep(0.5)\n")
    child = subprocess.Popen([sys.executable, '-c', code, str(p)], stdout=subprocess.PIPE, text=True, env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))
    assert child.stdout.readline().strip() == 'locked'
    t0 = time.time()
    with open(p, 'a+') as f, oslib.file_lock(f): waited = time.time() - t0
    child.wait()
    assert waited > 0.2, waited


def test_file_lock_is_released_after_the_block(tmp_path):
    with open(tmp_path / 'l', 'a+') as f:
        for _ in range(3):
            with oslib.file_lock(f): pass


def test_file_lock_windows_branch(tmp_path, monkeypatch):
    calls = []
    fake_msvcrt = types.SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0)
    def locking(fd, mode, n):
        calls.append(mode)
        if len(calls) == 1: raise OSError('busy')
    fake_msvcrt.locking = locking
    monkeypatch.setattr(oslib, 'WIN', True)
    monkeypatch.setitem(sys.modules, 'msvcrt', fake_msvcrt)
    monkeypatch.setattr(time, 'sleep', lambda _: None)
    with open(tmp_path / 'wlock', 'a+') as f, oslib.file_lock(f):
        pass
    assert calls == [2, 2, 0]

    flock_calls = []
    fake_fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_UN=8, flock=lambda f, op: flock_calls.append(op))
    monkeypatch.setattr(oslib, 'WIN', False)
    monkeypatch.setitem(sys.modules, 'fcntl', fake_fcntl)
    with open(tmp_path / 'plock', 'a+') as f, oslib.file_lock(f):
        pass
    assert flock_calls == [1, 8]


def test_pid_alive(monkeypatch):
    assert oslib.pid_alive(os.getpid()) and oslib.pid_alive(str(os.getpid()))
    assert not oslib.pid_alive(None) and not oslib.pid_alive('x') and not oslib.pid_alive(0)
    p = subprocess.Popen([sys.executable, '-c', 'pass']); p.wait()
    assert not oslib.pid_alive(p.pid)

    monkeypatch.setattr(oslib, 'WIN', False)
    monkeypatch.setattr(os, 'kill', lambda pid, sig: None)
    assert oslib.pid_alive(1234)
    monkeypatch.setattr(os, 'kill', lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()))
    assert not oslib.pid_alive(1234)
    monkeypatch.setattr(os, 'kill', lambda pid, sig: (_ for _ in ()).throw(PermissionError()))
    assert oslib.pid_alive(1234)
    monkeypatch.setattr(os, 'kill', lambda pid, sig: (_ for _ in ()).throw(OSError('other')))
    assert not oslib.pid_alive(1234)


def test_pid_alive_windows_branch(monkeypatch):
    import ctypes
    monkeypatch.setattr(oslib, 'WIN', True)
    closed = []
    class FakeKernel32:
        def OpenProcess(self, access, inherit, pid): return 0 if pid == 99 else 42
        def GetExitCodeProcess(self, h, ptr):
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ulong))[0] = 259
            return 1
        def CloseHandle(self, h): closed.append(h)
    monkeypatch.setattr(ctypes, 'windll', types.SimpleNamespace(kernel32=FakeKernel32()), raising=False)
    assert not oslib.pid_alive(99)
    assert oslib.pid_alive(100) and closed == [42]


def test_kill_tree_ends_the_process(monkeypatch):
    p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    assert oslib.pid_alive(p.pid)
    oslib.kill_tree(p.pid)
    p.wait(timeout=10)
    assert not oslib.pid_alive(p.pid)

    # Child process recursion and OSError fallback when pgrep is absent
    monkeypatch.setattr(oslib, 'WIN', False)
    killed = []
    monkeypatch.setattr(os, 'kill', lambda pid, sig: killed.append(pid))
    calls = [types.SimpleNamespace(stdout='222\n'), types.SimpleNamespace(stdout='')]
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: calls.pop(0) if calls else (_ for _ in ()).throw(OSError('no pgrep')))
    oslib.kill_tree(111)
    oslib.kill_tree(333)
    assert killed == [222, 111, 333]

    # Windows branch
    win_cmds = []
    monkeypatch.setattr(oslib, 'WIN', True)
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kw: win_cmds.append(cmd))
    oslib.kill_tree(123)
    assert win_cmds == [['taskkill', '/T', '/F', '/PID', '123']]


def test_lower_priority_and_venv_python(monkeypatch):
    monkeypatch.setattr(oslib, 'WIN', False)
    nices = []
    monkeypatch.setattr(os, 'nice', lambda n: nices.append(n), raising=False)
    oslib.lower_priority(10)
    assert nices == [10]
    assert oslib.venv_python('/tmp/v').endswith('bin/python') or oslib.venv_python('/tmp/v').endswith('bin\\python')

    import ctypes
    prio = []
    class FakeK32:
        def GetCurrentProcess(self): return 1
        def SetPriorityClass(self, h, cls): prio.append(cls)
    monkeypatch.setattr(oslib, 'WIN', True)
    monkeypatch.setattr(ctypes, 'windll', types.SimpleNamespace(kernel32=FakeK32()), raising=False)
    oslib.lower_priority(19)
    oslib.lower_priority(5)
    assert prio == [0x40, 0x4000]
    assert oslib.venv_python('C:\\v').endswith('Scripts/python.exe') or oslib.venv_python('C:\\v').endswith('Scripts\\python.exe')


def test_cli_command_runs_the_cli():
    out = subprocess.run([*oslib.cli_command(), '--help'], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and 'hello' in out.stdout, out.stderr[-300:]
