"""Client for the strata-models service: run a model by role, or hold the GPU for your own kernels. Standard library only.

    import strata_models_client as sm
    sm.faces('/x.jpg', app='bright-horizons')                 # -> {'width', 'height', 'faces': [{'bbox', 'kps', 'det_score', 'embedding', 'pose'}]}
    sm.transcribe('/a.wav', app='strata-360', language='en')  # -> {'language', 'duration', 'segments': [{'start', 'end', 'text', 'words': [...]}]}
    with sm.gpu_lease('strata-stills', vram_gb=4): ...

The service URL is STRATA_MODELS_URL (default http://127.0.0.1:8366). Inputs are file paths: the service runs on the same machine."""
import contextlib, json, os, threading, time, urllib.error, urllib.request, warnings

__version__ = '0.1.0'
POLL_S = 30                 # one long-poll request waits this long at most


class Unavailable(RuntimeError):
    """The service is not running (or not reachable)."""


class JobFailed(RuntimeError):
    """The service ran the job and it failed; `.job` is the job row."""
    def __init__(self, msg, job=None): super().__init__(msg); self.job = job


def url():
    return os.environ.get('STRATA_MODELS_URL', 'http://127.0.0.1:8366').rstrip('/')


def _call(method, path, body=None, timeout=POLL_S + 15):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url() + path, data=data, method=method, headers={'Content-Type': 'application/json'} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)
    except urllib.error.HTTPError as e:
        try: d = json.load(e)
        except Exception: d = {}
        if e.code in (400, 404, 422): raise JobFailed(str(d.get('detail') or e), d.get('job'))
        raise Unavailable(f'strata-models answered {e.code}: {d.get("detail") or e.reason}')
    except (urllib.error.URLError, OSError) as e: raise Unavailable(f'strata-models is not reachable at {url()}: {e}')


def available(timeout=2.0):
    """Is the service running?"""
    try: return bool(_call('GET', '/api/health', timeout=timeout).get('ok'))
    except (Unavailable, JobFailed): return False


def submit(role, params, app, priority='batch', model=None, options=None):
    """Queue a job and return its row at once (poll with `wait`). `options` changes the model's load options (a variant loaded apart: evals)."""
    return _call('POST', '/api/jobs', dict(role=role, params=params or {}, app=app, priority=priority, model=model, options=options))


def wait_job(job_id):
    """The job's row once it has finished (done, failed or cancelled), with result, error and timings; never raises for a failed job."""
    while True:
        j = _call('GET', f'/api/jobs/{int(job_id)}?wait={POLL_S}')
        if j['status'] not in ('queued', 'running'): return j


def wait(job_id, timeout=None):
    """The job's result once it is done; raises JobFailed if it failed or was cancelled, TimeoutError after `timeout` seconds."""
    end = None if timeout is None else time.monotonic() + timeout
    while True:
        step = POLL_S if end is None else max(min(POLL_S, end - time.monotonic()), 0)
        j = _call('GET', f'/api/jobs/{int(job_id)}?wait={step:.1f}')
        if j['status'] == 'done': return j['result']
        if j['status'] in ('failed', 'cancelled'): raise JobFailed(j.get('error') or f'job {job_id} was {j["status"]}', j)
        if end is not None and time.monotonic() >= end: raise TimeoutError(f'job {job_id} is still {j["status"]}')


def run(role, params, app, priority='batch', model=None, timeout=None):
    """Run a model job and return its result (queued behind the GPU's other work; `interactive` goes ahead of `batch`)."""
    return wait(submit(role, params, app, priority, model)['id'], timeout)


def faces(path, app, priority='batch', model=None, **params):
    """Faces in one image: params max_side, det_thresh, fields. Several images: path=None and `paths=[...]`, giving {'images': [...]}."""
    if path is not None: params['path'] = os.path.abspath(path)
    if 'paths' in params: params['paths'] = [os.path.abspath(p) for p in params['paths']]
    return run('faces', params, app, priority=priority, model=model)


def transcribe(path, app, role='transcribe', priority='batch', model=None, **params):
    """Whisper on an audio or video file: params language, task, word_timestamps, vad_filter, beam_size, initial_prompt, clip=(start, end)."""
    params['path'] = os.path.abspath(path)
    if 'clip' in params: params['clip'] = list(params['clip'])
    return run(role, params, app, priority=priority, model=model)


def role_model(role):
    """The model that serves `role` now (from the service's registry), or None for an unknown role. Apps key their caches on it, so output made
    by an earlier model is made again when an eval changes the role's model."""
    return next((r['model'] for r in _call('GET', '/api/registry').get('roles', []) if r['key'] == role), None)


def speak(text, app, priority='batch', model=None, **params):
    """Speech from text (role `speak`): params voice (a preset name, or a reference recording's path for the cloning models), speed (1.0 = the
    model's own pace), instruct, language, seed. Without model and voice, the voice and settings the role's eval chose. Returns {path (a WAV on
    this machine, the service's copy: copy it if it must last), duration_s, sample_rate, seconds, engine, voice}."""
    if params.get('voice') and os.path.exists(str(params['voice'])): params['voice'] = os.path.abspath(params['voice'])
    return run('speak', dict(params, text=text), app, priority=priority, model=model)


class Lease:
    """A held lease: `.id`, `.job` (the row when granted). Heartbeats run on a daemon thread until `release()`."""
    def __init__(self, job, ttl_s):
        self.id, self.job, self._stop = job['id'], job, threading.Event()
        self._t = threading.Thread(target=self._beat, args=(max(ttl_s / 3, 1.0),), daemon=True); self._t.start()

    def _beat(self, every):
        while not self._stop.wait(every):
            with contextlib.suppress(Exception): _call('POST', f'/api/leases/{self.id}/heartbeat', timeout=10)

    def release(self):
        self._stop.set()
        with contextlib.suppress(Exception): _call('DELETE', f'/api/leases/{self.id}', timeout=10)


def acquire(app, vram_gb=None, ttl_s=120, priority='batch', note=None, timeout=None):
    """Ask for the GPU and wait until it is granted; returns a Lease (call `.release()`), TimeoutError after `timeout` (the request is withdrawn)."""
    j = _call('POST', '/api/leases', dict(app=app, vram_gb=vram_gb, ttl_s=ttl_s, priority=priority, note=note, pid=os.getpid()))
    end = None if timeout is None else time.monotonic() + timeout; last_beat = time.monotonic()
    while j['status'] == 'queued':
        if end is not None and time.monotonic() >= end:
            with contextlib.suppress(Exception): _call('DELETE', f'/api/leases/{j["id"]}', timeout=10)
            raise TimeoutError(f'the GPU was not free within {timeout} s')
        if time.monotonic() - last_beat > ttl_s / 3: _call('POST', f'/api/leases/{j["id"]}/heartbeat', timeout=10); last_beat = time.monotonic()
        step = min(POLL_S, ttl_s / 3) if end is None else max(min(POLL_S, ttl_s / 3, end - time.monotonic()), 0)
        j = _call('GET', f'/api/leases/{j["id"]}?wait={step:.1f}')
    if j['status'] != 'running': raise JobFailed(f'lease {j["id"]} was {j["status"]}', j)
    return Lease(j, ttl_s)


@contextlib.contextmanager
def gpu_lease(app, vram_gb=None, ttl_s=120, priority='batch', note=None, timeout=None, required=False):
    """`with gpu_lease('strata-stills', vram_gb=4):` 4 GB of the GPU are this app's for the block, and other work goes on in the rest (give the
    most this block will use); with vram_gb=None the whole GPU, and nothing else runs. Without the service: yields None and runs the block anyway,
    with a warning (or raises Unavailable when `required`)."""
    try: lease = acquire(app, vram_gb, ttl_s, priority, note, timeout)
    except Unavailable:
        if required: raise
        warnings.warn('strata-models is not running: using the GPU without a lease', RuntimeWarning, stacklevel=3); lease = None
    try: yield lease
    finally:
        if lease is not None: lease.release()
