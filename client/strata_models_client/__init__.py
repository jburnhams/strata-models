"""Client for the strata-models service: run a model by role, or hold the GPU for your own kernels. Standard library only.

    import strata_models_client as sm
    sm.faces('/x.jpg', app='bright-horizons')                 # -> {'width', 'height', 'faces': [{'bbox', 'kps', 'det_score', 'embedding', 'pose'}]}
    sm.transcribe('/a.wav', app='strata-360', language='en')  # -> {'language', 'duration', 'segments': [{'start', 'end', 'text', 'words': [...]}]}
    with sm.gpu_lease('strata-stills', vram_gb=4): ...

The service URL is STRATA_MODELS_URL (default http://127.0.0.1:8366). Inputs are file paths: the service runs on the same machine."""

import contextlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
import warnings

__version__ = "0.1.0"
POLL_S = 30  # one long-poll request waits this long at most


class Unavailable(RuntimeError):
    """The service is not running (or not reachable)."""


class JobFailed(RuntimeError):
    """The service ran the job and it failed; `.job` is the job row."""

    def __init__(self, msg, job=None):
        super().__init__(msg)
        self.job = job


def url():
    return os.environ.get("STRATA_MODELS_URL", "http://127.0.0.1:8366").rstrip("/")


def _call(method, path, body=None, timeout=POLL_S + 15):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url() + path, data=data, method=method, headers={"Content-Type": "application/json"} if data else {}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            d = json.load(e)
        except Exception:
            d = {}
        if e.code in (400, 404, 422):
            raise JobFailed(str(d.get("detail") or e), d.get("job"))
        raise Unavailable(f"strata-models answered {e.code}: {d.get('detail') or e.reason}")
    except (urllib.error.URLError, OSError) as e:
        raise Unavailable(f"strata-models is not reachable at {url()}: {e}")


def available(timeout=2.0):
    """Is the service running?"""
    try:
        return bool(_call("GET", "/api/health", timeout=timeout).get("ok"))
    except (Unavailable, JobFailed):
        return False


def submit(role, params, app, priority="normal", model=None, options=None):
    """Queue a job and return its row at once (poll with `wait`). `options` changes the model's load options (a variant loaded apart: evals)."""
    return _call(
        "POST",
        "/api/jobs",
        dict(role=role, params=params or {}, app=app, priority=priority, model=model, options=options),
    )


def submit_many(role, items, app, priority="normal", model=None, options=None, batch=None):
    """Queue one job per params dict in `items` in a single call; returns their ids, in order (wait on each with `wait`). On a Claude model
    enough of them go through the batch API (half the price, answered within 24 hours) automatically; `batch=True` always, `batch=False`
    never. Nothing is queued when the role or model is refused."""
    return _call(
        "POST",
        "/api/jobs/bulk",
        dict(role=role, items=list(items), app=app, priority=priority, model=model, options=options, batch=batch),
    )["ids"]


def wait_job(job_id):
    """The job's row once it has finished (done, failed or cancelled), with result, error and timings; never raises for a failed job."""
    while True:
        j = _call("GET", f"/api/jobs/{int(job_id)}?wait={POLL_S}")
        if j["status"] not in ("queued", "running"):
            return j


def wait(job_id, timeout=None):
    """The job's result once it is done; raises JobFailed if it failed or was cancelled, TimeoutError after `timeout` seconds."""
    end = None if timeout is None else time.monotonic() + timeout
    while True:
        step = POLL_S if end is None else max(min(POLL_S, end - time.monotonic()), 0)
        j = _call("GET", f"/api/jobs/{int(job_id)}?wait={step:.1f}")
        if j["status"] == "done":
            return j["result"]
        if j["status"] in ("failed", "cancelled"):
            raise JobFailed(j.get("error") or f"job {job_id} was {j['status']}", j)
        if end is not None and time.monotonic() >= end:
            raise TimeoutError(f"job {job_id} is still {j['status']}")


def run(role, params, app, priority="normal", model=None, timeout=None):
    """Run a model job and return its result (queued behind the GPU's other work; priority `high` goes ahead of `normal` (the default), which goes ahead of `low`)."""
    return wait(submit(role, params, app, priority, model)["id"], timeout)


def run_with_provenance(role, params, app, priority="normal", model=None, timeout=None):
    """`run`, plus what the result was made with: (result, {model, options, settings_hash, job_id}) (see `provenance`)."""
    job_id = submit(role, params, app, priority, model)["id"]
    result = wait(job_id, timeout)
    return result, provenance(_call("GET", f"/api/jobs/{int(job_id)}"))


def provenance(job):
    """What a run job's output was made with, from its row: {model, options, settings_hash, job_id}. `settings_hash` changes when the model's
    runtime, file or load options do, even when its key stays the same (None on rows from before the service recorded it)."""
    return dict(
        model=job.get("model"), options=job.get("options"), settings_hash=job.get("settings_hash"), job_id=job["id"]
    )


def faces(path, app, priority="normal", model=None, **params):
    """Faces in one image: params max_side, det_thresh, fields. Several images: path=None and `paths=[...]`, giving {'images': [...]}."""
    if path is not None:
        params["path"] = os.path.abspath(path)
    if "paths" in params:
        params["paths"] = [os.path.abspath(p) for p in params["paths"]]
    return run("faces", params, app, priority=priority, model=model)


def transcribe(path, app, role="transcribe", priority="normal", model=None, **params):
    """Whisper on an audio or video file: params language, task, word_timestamps, vad_filter, beam_size, initial_prompt, clip=(start, end)."""
    params["path"] = os.path.abspath(path)
    if "clip" in params:
        params["clip"] = list(params["clip"])
    return run(role, params, app, priority=priority, model=model)


def ocr(path, app, priority="normal", model=None, **params):
    """Printed text from a page picture (role `ocr`): params lang ('eng', 'eng+fra'), psm (Tesseract's page segmentation), dpi, clean (steps that clean the page first, such as "flatten,sharpen": see runtimes/pageclean.py; boxes are then in the cleaned picture's pixels), pdf (a .pdf path
    to write as well: the picture with its text searchable). Returns {width, height, text, lines: [{text, box: [x, y, w, h], conf, block,
    para, words: [{text, box, conf}]}], blocks: [{block, box, kind}], unread: [{box, conf}], engine, lang, seconds}; conf is 0-100. `blocks`
    is every block the reader found (kind "text", or "picture": nothing read there), `unread` every word box a detector found but read as
    nothing; a detect-then-read model (docTR) has no `para` and may add `regions` [{box, kind, conf}] from a layout detector."""
    params["path"] = os.path.abspath(path)
    if params.get("pdf"):
        params["pdf"] = os.path.abspath(params["pdf"])
    return run("ocr", params, app, priority=priority, model=model)


def shots(path, app, priority="normal", model=None, **params):
    """Where a video cuts from one shot to the next (role `shot-boundaries`): params clip ([start, end] seconds), threshold, min_shot_s, width,
    deinterlace, black (find black stretches too, default True), scores (every frame's score as well). Returns {cuts: [{time, frame, score}],
    shots: [{start, end}], black: [{start, end}], fps, frames, duration, ...}; times are seconds in the file."""
    params["path"] = os.path.abspath(path)
    return run("shot-boundaries", params, app, priority=priority, model=model)


def summarise(
    inputs,
    app,
    *,
    purpose="",
    audience="",
    target_size="one short paragraph (3-4 sentences)",
    title="",
    priority="normal",
    model=None,
    timeout=None,
    **params,
):
    """Distill a set of input texts (`inputs`) for a `purpose`, `audience`, and `target_size` using role `summarise`."""
    context = dict(title=title, purpose=purpose, audience=audience, target_size=target_size, inputs=list(inputs))
    params.setdefault("max_tokens", 1500)
    params.setdefault("effort", "low")
    params.setdefault("preset", "summarise-v3-xml")
    return run("summarise", dict(params, context=context), app, priority=priority, model=model, timeout=timeout)


def role_model(role):
    """The model that serves `role` now (from the service's registry), or None for an unknown role. Apps key their caches on it, so output made
    by an earlier model is made again when an eval changes the role's model."""
    return next((r["model"] for r in _call("GET", "/api/registry").get("roles", []) if r["key"] == role), None)


def role_settings(role):
    """`role_model` with its settings hash: {model, settings_hash}, or None for an unknown role. A cache keyed on both is also made again when the
    model's options change under the same key."""
    r = next((r for r in _call("GET", "/api/registry").get("roles", []) if r["key"] == role), None)
    return None if r is None else dict(model=r["model"], settings_hash=r.get("settings_hash"))


def speak(text, app, priority="normal", model=None, **params):
    """Speech from text (role `speak`): params voice (a preset name, or a reference recording's path for the cloning models), speed (1.0 = the
    model's own pace), instruct, language, seed, phonemes (Kokoro: IPA said as written, for a sound no word spells alone; `text` is then
    its label), sound (Kokoro: one phonics sound in IPA, said pure with no vowel added, "mmm" not "muh": a consonant is cut from a
    carrier syllable and held for hold_s, a stop keeps its burst; text is then its label). Without model and voice, the voice and settings the role's eval chose. Returns {path (a WAV on
    this machine, the service's copy: copy it if it must last), duration_s, sample_rate, seconds, engine, voice}."""
    if params.get("voice") and os.path.exists(str(params["voice"])):
        params["voice"] = os.path.abspath(params["voice"])
    return run("speak", dict(params, text=text), app, priority=priority, model=model)


class Lease:
    """A held lease: `.id`, `.job` (the row when granted). Heartbeats run on a daemon thread until `release()`."""

    def __init__(self, job, ttl_s):
        self.id, self.job, self._stop = job["id"], job, threading.Event()
        self._t = threading.Thread(target=self._beat, args=(max(ttl_s / 3, 1.0),), daemon=True)
        self._t.start()

    def _beat(self, every):
        while not self._stop.wait(every):
            with contextlib.suppress(Exception):
                _call("POST", f"/api/leases/{self.id}/heartbeat", timeout=10)

    def release(self):
        self._stop.set()
        with contextlib.suppress(Exception):
            _call("DELETE", f"/api/leases/{self.id}", timeout=10)


def acquire(app, vram_gb=None, ttl_s=120, priority="normal", note=None, timeout=None):
    """Ask for the GPU and wait until it is granted; returns a Lease (call `.release()`), TimeoutError after `timeout` (the request is withdrawn)."""
    j = _call(
        "POST",
        "/api/leases",
        dict(app=app, vram_gb=vram_gb, ttl_s=ttl_s, priority=priority, note=note, pid=os.getpid()),
    )
    end = None if timeout is None else time.monotonic() + timeout
    last_beat = time.monotonic()
    while j["status"] == "queued":
        if end is not None and time.monotonic() >= end:
            with contextlib.suppress(Exception):
                _call("DELETE", f"/api/leases/{j['id']}", timeout=10)
            raise TimeoutError(f"the GPU was not free within {timeout} s")
        if time.monotonic() - last_beat > ttl_s / 3:
            _call("POST", f"/api/leases/{j['id']}/heartbeat", timeout=10)
            last_beat = time.monotonic()
        step = min(POLL_S, ttl_s / 3) if end is None else max(min(POLL_S, ttl_s / 3, end - time.monotonic()), 0)
        j = _call("GET", f"/api/leases/{j['id']}?wait={step:.1f}")
    if j["status"] != "running":
        raise JobFailed(f"lease {j['id']} was {j['status']}", j)
    return Lease(j, ttl_s)


@contextlib.contextmanager
def gpu_lease(app, vram_gb=None, ttl_s=120, priority="normal", note=None, timeout=None, required=False):
    """`with gpu_lease('strata-stills', vram_gb=4):` 4 GB of the GPU are this app's for the block, and other work goes on in the rest (give the
    most this block will use); with vram_gb=None the whole GPU, and nothing else runs. Without the service: yields None and runs the block anyway,
    with a warning (or raises Unavailable when `required`)."""
    try:
        lease = acquire(app, vram_gb, ttl_s, priority, note, timeout)
    except Unavailable:
        if required:
            raise
        warnings.warn("strata-models is not running: using the GPU without a lease", RuntimeWarning, stacklevel=3)
        lease = None
    try:
        yield lease
    finally:
        if lease is not None:
            lease.release()
