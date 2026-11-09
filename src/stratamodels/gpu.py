"""What the GPU is doing, from `nvidia-smi`: the card's memory, load, temperature and power, and every compute process on it with its command line.

No NVIDIA GPU (or no nvidia-smi) gives None, and the service then runs everything on the CPU with no VRAM budget. A small ring buffer keeps
the last hour of samples for the Overview chart."""
import collections, os, subprocess, threading, time

GPU_FIELDS = ('name', 'memory.total', 'memory.used', 'utilization.gpu', 'temperature.gpu', 'power.draw', 'power.limit')


def _num(s):
    try: return float(s)
    except (TypeError, ValueError): return None


def command_of(pid):
    """A short, readable command line for a process (`python scripts/build_arena.py`), or '' when it cannot be read."""
    try:
        with open(f'/proc/{int(pid)}/cmdline', 'rb') as f: parts = [p.decode('utf8', 'replace') for p in f.read().split(b'\0') if p]
    except (OSError, ValueError): return ''
    if not parts: return ''
    parts[0] = os.path.basename(parts[0]); return ' '.join(parts)[:300]


def parent_of(pid):
    """The parent pid of a process, or None (on Linux, from /proc; elsewhere unknown)."""
    try:
        with open(f'/proc/{int(pid)}/stat') as f: return int(f.read().rsplit(')', 1)[1].split()[1])
    except (OSError, ValueError, IndexError): return None


def query(run=subprocess.run, own_pid=None, parent=parent_of):
    """The card's state now, or None without an NVIDIA GPU. A process is `ours` when it is this service or one of its children (model workers)."""
    try:
        g = run(['nvidia-smi', '--query-gpu=' + ','.join(GPU_FIELDS), '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        p = run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError): return None
    if g.returncode != 0 or not g.stdout.strip(): return None
    v = [s.strip() for s in g.stdout.strip().splitlines()[0].split(',')]
    if len(v) < len(GPU_FIELDS): return None
    own = os.getpid() if own_pid is None else own_pid; procs = []
    for line in (p.stdout or '').strip().splitlines() if p.returncode == 0 else []:
        a = [s.strip() for s in line.split(',')]
        if len(a) >= 2 and a[0].isdigit(): procs.append(dict(pid=int(a[0]), used_mb=_num(a[1]) or 0.0, command=command_of(a[0]), ours=int(a[0]) == own or parent(a[0]) == own))      # a model's worker process (venv-worker) is ours too
    total, used = _num(v[1]) or 0.0, _num(v[2]) or 0.0
    return dict(name=v[0], total_mb=total, used_mb=used, free_mb=max(total - used, 0.0), util_pct=_num(v[3]), temp_c=_num(v[4]), power_w=_num(v[5]), power_limit_w=_num(v[6]), processes=procs)


def own_used_mb(state):
    """MB this service's process holds on the card (0 without a GPU)."""
    return sum(p['used_mb'] for p in (state or {}).get('processes', []) if p['ours'])


class History:
    """The last `keep_s` seconds of samples (time, util %, memory used MB), taken every `every_s` by a daemon thread."""
    def __init__(self, probe, every_s=5.0, keep_s=3600, clock=time.time):
        self.probe, self.every_s, self.clock = probe, every_s, clock
        self.samples = collections.deque(maxlen=max(int(keep_s / every_s), 1)); self.latest = None; self._stop = threading.Event()

    def sample(self):
        s = self.probe(); self.latest = s
        if s: self.samples.append(dict(at=self.clock(), util_pct=s['util_pct'], used_mb=s['used_mb'], own_mb=own_used_mb(s)))
        return s

    def start(self):
        def loop():
            while not self._stop.is_set():
                try: self.sample()
                except Exception: pass                    # a failed sample must not kill the sampler
                self._stop.wait(self.every_s)
        threading.Thread(target=loop, name='gpu-history', daemon=True).start()

    def stop(self): self._stop.set()

    def to_json(self): return list(self.samples)
