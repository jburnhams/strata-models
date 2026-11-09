"""The GPU's one owner: a single worker that takes jobs from the queue in the database, keeps models loaded within a VRAM budget, and lends the
whole GPU to an app that holds a lease.

Order: interactive before batch; within a priority, a job for a model that is already loaded goes first (batches of one kind run together instead
of reloading weights), but never ahead of a lease, and not when the oldest job has waited longer than `max_skip_s`, which then goes next whatever it needs. A lease is exclusive:
the worker frees the VRAM the app asked for (all of it when it gave no figure), marks the lease granted, and runs nothing else until the app
releases it or stops sending heartbeats for `ttl_s`. That is what keeps two GPU users from corrupting each other's work.

Loading, running and unloading happen only on the worker thread; the API threads queue requests and wait on a condition variable."""
import collections, threading, time

from stratamodels import gpu, runtimes
from stratamodels.store import FINISHED

PRIORITY = {'interactive': 0, 'batch': 1}
MB = 1024.0


class QueueError(ValueError):
    """A request the queue refuses (unknown job, a lease that is not one, a bad priority)."""


class Resident:
    def __init__(self, model, handle, vram_mb, load_s, clock):
        self.model, self.handle, self.vram_mb, self.load_s = model, handle, vram_mb, load_s
        self.loaded_at = self.last_used = clock(); self.jobs = 0

    def to_json(self): return dict(model=self.model.key, runtime=self.model.runtime, roles=list(self.model.roles), vram_mb=round(self.vram_mb), load_s=round(self.load_s, 2), loaded_at=self.loaded_at, last_used=self.last_used, jobs=self.jobs)


class Scheduler:
    def __init__(self, store, registry, models_dir, probe=lambda: None, reserve_gb=1.0, clock=time.time, get_runtime=runtimes.get, max_skip_s=120.0):
        self.store, self.registry, self.models_dir, self.probe = store, registry, models_dir, probe
        self.reserve_mb, self.clock, self.get_runtime, self.max_skip_s = reserve_gb * MB, clock, get_runtime, max_skip_s
        self.resident = collections.OrderedDict()            # model key -> Resident, least recently used first
        self.cv = threading.Condition(); self.control = collections.deque(); self.current = None; self.events = collections.deque(maxlen=200); self._waiting = None

    # ---- requests (any thread) -------------------------------------------------------------------------------------------------------
    def _priority(self, p):
        if p not in PRIORITY: raise QueueError(f'priority must be interactive or batch, not {p!r}')
        return p

    def _notify(self):
        with self.cv: self.cv.notify_all()

    def submit(self, role, params, app, priority='batch', model=None, options=None):
        """Queue a model job; returns its row. `options` changes the model's load options (a variant, see Registry.variant). Raises RegistryError
        for an unknown role or a model that does not serve it."""
        m = self.registry.resolve(role, model)
        if options: m = self.registry.variant(m.key, options)
        j = self.store.add(kind='run', role=role, model=m.key, app=str(app or 'unknown')[:64], priority=self._priority(priority), params=dict(params or {})); self._notify(); return j

    def lease(self, app, vram_gb=None, ttl_s=120.0, priority='batch', note=None):
        """Queue a request for the whole GPU; returns its row (status queued, then running once granted)."""
        if vram_gb is not None and not 0 <= float(vram_gb) <= 1024: raise QueueError('vram_gb must be a number of GB')
        ttl = min(max(float(ttl_s or 120.0), 5.0), 3600.0)
        j = self.store.add(kind='lease', app=str(app or 'unknown')[:64], priority=self._priority(priority), vram_gb=None if vram_gb is None else float(vram_gb), ttl_s=ttl, note=(note or None) and str(note)[:300], heartbeat_at=self.clock()); self._notify(); return j

    def _lease_row(self, job_id):
        j = self.store.get(job_id, result=False)
        if j is None or j['kind'] != 'lease': raise QueueError(f'no lease {job_id}')
        return j

    def heartbeat(self, job_id):
        j = self._lease_row(job_id)
        if j['status'] in FINISHED: raise QueueError(f'lease {job_id} is {j["status"]}')
        return self.store.update(job_id, heartbeat_at=self.clock())

    def release(self, job_id):
        """Give a lease back (or withdraw one still queued)."""
        j = self._lease_row(job_id)
        if j['status'] == 'running': j = self.store.update(job_id, status='done', finished_at=self.clock())
        elif j['status'] == 'queued': j = self.store.update(job_id, status='cancelled', finished_at=self.clock())
        self._notify(); return j

    def cancel(self, job_id):
        """Withdraw a queued job (a running one finishes: a GPU kernel cannot be stopped half way)."""
        j = self.store.get(job_id, result=False)
        if j is None: raise QueueError(f'no job {job_id}')
        if j['kind'] == 'lease': return self.release(job_id)
        if j['status'] == 'queued': j = self.store.update(job_id, status='cancelled', finished_at=self.clock()); self._notify()
        return j

    def wait(self, job_id, timeout=0.0):
        """The job once it has finished (a lease: once granted or finished), or as it stands after `timeout` seconds."""
        end = time.monotonic() + max(float(timeout or 0), 0.0)
        with self.cv:
            while True:
                j = self.store.get(job_id)
                if j is None: raise QueueError(f'no job {job_id}')
                if j['status'] in FINISHED or (j['kind'] == 'lease' and j['status'] == 'running'): return j
                left = end - time.monotonic()
                if left <= 0: return j
                self.cv.wait(min(left, 1.0))

    def request(self, action, key):
        """Load or unload a model on the worker thread (from the Models page); returns at once, the change shows on the next status read."""
        if action not in ('load', 'unload'): raise QueueError(f'no action {action!r}')
        m = self.registry.models.get(key)
        if m is None: raise QueueError(f'no model {key!r}')
        if not m.served: raise QueueError(f'model {key!r} ({m.runtime}) is not served here yet')
        self.control.append((action, key)); self._notify()

    # ---- state ------------------------------------------------------------------------------------------------------------------------
    def _event(self, text): self.events.append(dict(at=self.clock(), text=text))

    def status(self, state=None):
        state = self.probe() if state is None else state
        with self.cv: res = [r.to_json() for r in self.resident.values()]; cur = self.current
        q = self.store.queued(); budget = None; lease = next((j for j in self.store.running() if j['kind'] == 'lease'), None)
        if state:
            own = gpu.own_used_mb(state); budget = dict(total_mb=state['total_mb'], used_mb=state['used_mb'], own_mb=own, other_mb=max(state['used_mb'] - own, 0.0), reserve_mb=self.reserve_mb, free_for_models_mb=max(self.free_mb(state), 0.0))
        return dict(gpu=state, budget=budget, resident=res, current=cur, lease=lease, queued=q, events=list(self.events)[-30:])

    def free_mb(self, state):
        """MB the service may still fill: the card's free memory less the reserve kept for the desktop and other processes."""
        return state['total_mb'] - state['used_mb'] - self.reserve_mb

    # ---- the worker -------------------------------------------------------------------------------------------------------------------
    def _unload(self, key, why):
        r = self.resident.pop(key, None)
        if r is None: return
        try: self.get_runtime(r.model.runtime).unload(r.handle)
        except Exception: pass                                  # freeing must not fail the job that needed the room
        self._event(f'unloaded {key} ({why})')

    def _make_room(self, need_mb, keep=None):
        """Unload least recently used models until `need_mb` fits (None: unload every model). Without a GPU there is no budget to keep."""
        if need_mb is None:
            for k in list(self.resident): self._unload(k, 'lease wants the whole GPU')
            return
        state = self.probe()
        if not state: return
        free = self.free_mb(state)
        for k in list(self.resident):
            if free >= need_mb: break
            if k == keep: continue
            free += self.resident[k].vram_mb; self._unload(k, f'room for {need_mb / MB:.1f} GB')

    def _ensure(self, model):
        """The loaded handle for `model`, loading it (after making room) when needed; returns (Resident, seconds spent loading)."""
        r = self.resident.get(model.key)
        if r is not None: self.resident.move_to_end(model.key); return r, 0.0
        self._make_room(model.vram_gb * MB)
        before = gpu.own_used_mb(self.probe()); t0 = time.perf_counter()
        handle = self.get_runtime(model.runtime).load(model, self.models_dir); load_s = time.perf_counter() - t0
        measured = gpu.own_used_mb(self.probe()) - before
        r = Resident(model, handle, measured if measured > 0 else model.vram_gb * MB, load_s, self.clock)
        with self.cv: self.resident[model.key] = r
        self._event(f'loaded {model.key} in {load_s:.1f} s ({r.vram_mb / MB:.1f} GB)'); return r, load_s

    def _do_control(self):
        while self.control:
            action, key = self.control.popleft(); m = self.registry.models[key]
            try:
                if action == 'load': self._ensure(m)
                else: self._unload(key, 'asked from the Models page')
            except Exception as e: self._event(f'could not {action} {key}: {e}')

    def _next(self, queued):
        if not queued: return None
        top = min(PRIORITY.get(j['priority'], 1) for j in queued)
        cands = [j for j in queued if PRIORITY.get(j['priority'], 1) == top]
        if self.clock() - cands[0]['submitted_at'] > self.max_skip_s: return cands[0]
        ahead = []                                                  # a job for a loaded model may jump the queue, but never past a lease: an app is waiting on it
        for j in cands:
            if j['kind'] == 'lease': break
            ahead.append(j)
        return next((j for j in ahead if j['model'] in self.resident), cands[0])

    def _held_lease(self):
        """The lease that holds the GPU now, after expiring one whose app went quiet."""
        for j in self.store.running():
            if j['kind'] != 'lease': continue
            if self.clock() - (j['heartbeat_at'] or j['started_at'] or 0) > (j['ttl_s'] or 120.0):
                self.store.update(j['id'], status='expired', finished_at=self.clock(), error='no heartbeat within its ttl'); self._event(f'lease {j["id"]} ({j["app"]}) expired'); self._notify(); continue
            return j
        return None

    def _grant(self, j):
        """Free what the lease asked for and grant it. A sized lease waits (stays queued, and nothing passes it) while other programs hold so much
        that the room is not there even with every model freed; one that could never fit on this card fails at once."""
        need = None if j['vram_gb'] is None else j['vram_gb'] * MB
        self._make_room(need)
        state = self.probe() if need is not None else None
        if state and need > state['total_mb'] - self.reserve_mb:
            j = self.store.update(j['id'], status='failed', finished_at=self.clock(), error=f'{j["vram_gb"]:g} GB will never fit: the card has {state["total_mb"] / MB:.1f} GB and {self.reserve_mb / MB:.1f} GB is kept free')
            self._notify(); return j
        if state and self.free_mb(state) < need:
            if self._waiting != j['id']: self._waiting = j['id']; self._event(f'lease {j["id"]} ({j["app"]}) waits for {need / MB:.1f} GB: other programs hold the card')
            return None
        claimed = self.store.claim(j['id'], status='running', started_at=self.clock(), heartbeat_at=self.clock())
        if claimed is None: return None                                                             # withdrawn while the room was being made
        self._event(f'lease {j["id"]} granted to {j["app"]}' + (f' ({j["note"]})' if j['note'] else '')); self._notify(); return claimed

    def _run(self, j):
        j = self.store.claim(j['id'], status='running', started_at=self.clock())
        if j is None: return None                                                                   # withdrawn after the queue was read
        with self.cv: self.current = j
        model = self.registry.models.get(j['model']); load_s = 0.0
        try:
            if model is None or not model.served: raise runtimes.RunError(f'model {j["model"]!r} is no longer served')
            r, load_s = self._ensure(model); t0 = time.perf_counter()
            try: result = self.get_runtime(model.runtime).run(r.handle, j['params'] or {})
            except runtimes.RunError: raise
            except Exception: self._unload(model.key, 'it failed while running'); raise           # a model that threw (out of memory, a CUDA error) is reloaded fresh next time
            finally: r.last_used = self.clock(); r.jobs += 1
            j = self.store.update(j['id'], status='done', result=result, finished_at=self.clock(), load_s=round(load_s, 3), run_s=round(time.perf_counter() - t0, 3))
        except Exception as e:
            j = self.store.update(j['id'], status='failed', error=str(e) if isinstance(e, runtimes.RunError) else f'{type(e).__name__}: {e}', finished_at=self.clock(), load_s=round(load_s, 3))
        with self.cv: self.current = None
        self._notify(); return j

    def step(self):
        """Do one thing, unless a lease holds the GPU: pending load/unload requests, then the next job. Returns the job it handled, or None."""
        if self._held_lease(): return None                          # load requests wait too: the lease holder has the card to itself
        self._do_control()
        j = self._next(self.store.queued())
        if j is None: return None
        return self._grant(j) if j['kind'] == 'lease' else self._run(j)

    def start(self):
        """Run the worker on a daemon thread; returns the event that stops it."""
        stop = threading.Event(); n = self.store.recover()
        if n: self._event(f'{n} job(s) were running when the service stopped; marked failed')

        def loop():
            while not stop.is_set():
                try: did = self.step()
                except Exception as e: did = None; self._event(f'worker error: {type(e).__name__}: {e}')
                if did is None:
                    with self.cv: self.cv.wait(1.0)
        threading.Thread(target=loop, name='gpu-worker', daemon=True).start(); return stop
