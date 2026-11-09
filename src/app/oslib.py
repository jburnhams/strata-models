"""OS differences in one place: file locks, process liveness and termination, priority, and how to start the CLI.

POSIX (macOS, Linux) uses fcntl, os.kill and os.nice; Windows uses msvcrt, the Win32 API through ctypes and taskkill. Other modules should call these
instead of importing fcntl or calling os.kill(pid, 0) (on Windows that would terminate the process)."""
import contextlib, os, subprocess, sys, time

WIN = sys.platform == 'win32'
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


@contextlib.contextmanager
def file_lock(f):
    """Exclusive lock on an open file, held for the duration of the block (other processes block until it is released)."""
    if WIN:
        import msvcrt
        f.seek(0)
        while True:
            try: msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1); break
            except OSError: time.sleep(0.02)
        try: yield
        finally:
            f.seek(0)
            with contextlib.suppress(OSError): msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(f, fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)


def pid_alive(pid):
    """Is that process still running?"""
    try: pid = int(pid)
    except (TypeError, ValueError): return False
    if pid <= 0: return False
    if WIN:
        import ctypes
        k = ctypes.windll.kernel32; h = k.OpenProcess(0x1000, False, pid)             # PROCESS_QUERY_LIMITED_INFORMATION
        if not h: return False
        try:
            code = ctypes.c_ulong(); ok = k.GetExitCodeProcess(h, ctypes.byref(code)); return bool(ok) and code.value == 259     # STILL_ACTIVE
        finally: k.CloseHandle(h)
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False
    except PermissionError: return True
    except OSError: return False


def kill_tree(pid, sig=15):
    """Terminate a worker and every process it started; errors are ignored."""
    if WIN:
        with contextlib.suppress(OSError): subprocess.run(['taskkill', '/T', '/F', '/PID', str(int(pid))], capture_output=True)
        return
    try: kids = subprocess.run(['pgrep', '-P', str(pid)], stdout=subprocess.PIPE, text=True).stdout.split()
    except OSError: kids = []
    for k in kids: kill_tree(int(k), sig)
    with contextlib.suppress(OSError): os.kill(int(pid), sig)


def lower_priority(nice=19):
    """Run this process at lower CPU priority (nice value on POSIX, below-normal/idle class on Windows). Children inherit it."""
    if WIN:
        import ctypes
        with contextlib.suppress(Exception): ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40 if nice >= 15 else 0x4000)    # IDLE / BELOW_NORMAL
        return
    with contextlib.suppress(OSError, AttributeError): os.nice(nice)


def venv_python(venv):
    """The interpreter of a virtual environment: bin/python on POSIX, Scripts\\python.exe on Windows."""
    return os.path.join(venv, 'Scripts', 'python.exe') if WIN else os.path.join(venv, 'bin', 'python')


def cli_command():
    """Command prefix that starts the app CLI with the interpreter running this process (works on POSIX and Windows, with or without a .venv)."""
    os.environ['PYTHONPATH'] = os.path.join(ROOT_DIR, 'src') + os.pathsep + os.environ.get('PYTHONPATH', ''); os.environ.setdefault('PYTHONWARNINGS', 'ignore')
    return [sys.executable, '-m', 'app']
