"""The GPU's one owner: a dispatcher that takes jobs from the queue in the database, keeps models loaded within a VRAM budget, runs jobs on
threads of their own (several at once), and lends the GPU, or part of it, to an app that holds a lease.

Two lanes. A job whose model needs no VRAM (vram_gb = 0: recorded outputs, a CPU model, a cloud CLI) runs in the CPU lane, up to `cpu_jobs` at
once, and never waits for the GPU or a lease: a 30 s cloud call does not hold up the card. Every other job, and every lease, is in the GPU
lane: up to `gpu_jobs` run at once, as long as their models fit in the budget together. One job per model at a time (a worker process or a
llama-server answers one request at a time anyway); a model that is running or loading is never evicted.

Memory. A model's vram_gb is its peak; what it measured when it loaded is what it holds idle. While a job runs, the gap between the two (a
runtime that allocates per job, like sd-cpp, or a KV cache that grows) is promised to it, and so is a whole estimate while a model loads, so a
second job is only started into room nobody running is about to take. A GPU job that needs room that running jobs hold waits, and nothing
in the GPU lane passes it (so a big job is not starved by a stream of small ones); with nothing else running it starts whatever the card
shows, as one job at a time always did. A load is measured from the card only when no other GPU job was busy meanwhile; otherwise the
estimate stands.

Order: interactive before batch; within a priority, a job for a model that is already loaded (and idle) goes first (batches of one kind run
together instead of reloading weights), but never ahead of a lease, and not when the oldest job has waited longer than `max_skip_s`, which
then goes next whatever it needs.

Leases: an app that runs its own CUDA code asks for a lease. With a size (vram_gb) the lease RESERVES that much: the dispatcher frees room for
it (idle models only: it waits for running jobs to finish if it must), and keeps running other jobs in what is left (models that fit; a job
whose model does not fit waits), and other sized leases can be held at the same time. The reservation counts until the app's own process
(the pid it sent) is seen using that memory, so a model is never loaded into room the app is about to take. Without a size the lease is
EXCLUSIVE: it waits for running GPU jobs, every GPU model is freed and nothing else on the GPU runs until it is given back (the CPU lane
carries on). Either way a lease ends when released or when the app stops sending heartbeats for `ttl_s`.

Whole card: a model job or a sized lease that needs more than `whole_card_gb` (or more than the budget could ever give: the card less the
reserve) does not share the card. It waits until no GPU job runs and no lease is held, every idle GPU model is freed, the reserve is ignored,
and nothing else on the GPU starts until it ends (the CPU lane carries on). It is a way to try a model that is too big to budget for: the
card's other programs may still leave it short, and then it fails like any job that runs out of memory. Only a lease bigger than the card
itself is refused. Not sharing memory is what keeps
two GPU users from corrupting each other's work; sharing the card's time just makes both a little slower.

Deciding happens on one dispatcher thread (`step` decides one thing and, called directly, also runs it, which is how the tests drive it);
loading, running and unloading a model happen on its job's thread, and the API threads queue requests and wait on a condition variable."""

import collections
import threading
import time

from stratamodels import gpu, runtimes
from stratamodels.store import FINISHED

PRIORITY = {"interactive": 0, "batch": 1}
OPTIONS = "__options__"  # a job's params key holding the load options of a model variant (never passed to the runtime)
MB = 1024.0


class QueueError(ValueError):
    """A request the queue refuses (unknown job, a lease that is not one, a bad priority)."""


class Resident:
    def __init__(self, model, handle, vram_mb, load_s, clock, held_mb=None):
        self.model, self.handle, self.vram_mb, self.load_s = model, handle, vram_mb, load_s
        self.held_mb = held_mb  # what loading it really took on the card (0 for sd-cpp, which allocates per job); None: not known, its load overlapped other GPU work
        self.loaded_at = self.last_used = clock()
        self.jobs = 0

    @property
    def growth_mb(self):
        """What a job on it may still take beyond what it holds idle: its estimate less what it measured (the whole estimate when unknown)."""
        return max(self.model.vram_gb * MB - (self.held_mb or 0.0), 0.0)

    @property
    def frees_mb(self):
        """What unloading it gives back."""
        return self.vram_mb if self.held_mb is None else self.held_mb

    def to_json(self):
        return dict(
            model=self.model.key,
            runtime=self.model.runtime,
            roles=list(self.model.roles),
            vram_mb=round(self.vram_mb),
            load_s=round(self.load_s, 2),
            loaded_at=self.loaded_at,
            last_used=self.last_used,
            jobs=self.jobs,
        )


class Scheduler:
    def __init__(
        self,
        store,
        registry,
        models_dir,
        probe=lambda: None,
        reserve_gb=1.0,
        clock=time.time,
        get_runtime=runtimes.get,
        max_skip_s=120.0,
        gpu_jobs=2,
        cpu_jobs=4,
        whole_card_gb=10.0,
    ):
        self.store, self.registry, self.models_dir, self.probe = store, registry, models_dir, probe
        self.reserve_mb, self.clock, self.get_runtime, self.max_skip_s = reserve_gb * MB, clock, get_runtime, max_skip_s
        self.gpu_jobs, self.cpu_jobs = max(int(gpu_jobs), 1), max(int(cpu_jobs), 1)
        self.whole_card_mb = whole_card_gb * MB
        self.alone = set()  # keys of the models running or loading with the whole card to themselves
        self.resident = collections.OrderedDict()  # model key -> Resident, least recently used first
        self.cv = threading.Condition()
        self.control = collections.deque()
        self.events = collections.deque(maxlen=200)
        self._waiting = None
        self.held = []
        self.busy = {}  # model key -> what holds it (a job id, or "load"): never evicted, one job per model at a time
        self.loading = {}  # model key -> MB promised to a load in progress
        self.running = {}  # job id -> (row, lane) of every job running now
        self._epoch = 0  # moves whenever GPU work starts or ends, so a load can tell whether its measurement overlapped other work
        self._dirty = False  # something changed since the dispatcher last looked
        self._snap = None  # (epoch, card state) read once per dispatch: nvidia-smi is slow, and nothing moved while the epoch stands

    # ---- requests (any thread) -------------------------------------------------------------------------------------------------------
    def _priority(self, p):
        if p not in PRIORITY:
            raise QueueError(f"priority must be interactive or batch, not {p!r}")
        return p

    def _notify(self):
        with self.cv:
            self._dirty = True
            self.cv.notify_all()

    def submit(self, role, params, app, priority="batch", model=None, options=None):
        """Queue a model job; returns its row. `options` changes the model's load options (a variant, see Registry.variant). Raises RegistryError
        for an unknown role or a model that does not serve it."""
        app = str(app or "unknown")[:64]
        pref = self.registry.preference(app, role)
        m = self.registry.resolve(role, model, app)
        params = {
            **((pref.params or {}) if pref else {}),
            **(params or {}),
        }  # the app's default params (preferences), its job's own over them
        if options:
            m = self.registry.variant(m.key, options)
            params[OPTIONS] = dict(options)  # kept with the job, so a restart can rebuild the variant
        j = self.store.add(
            kind="run", role=role, model=m.key, app=app, priority=self._priority(priority), params=params
        )
        self._notify()
        return j

    def lease(self, app, vram_gb=None, ttl_s=120.0, priority="batch", note=None, pid=None):
        """Queue a request for `vram_gb` of the GPU (None: all of it, exclusively); returns its row (status queued, then running once granted).
        `pid` is the app's process, whose memory on the card then counts against its reservation."""
        if vram_gb is not None and not 0 <= float(vram_gb) <= 1024:
            raise QueueError("vram_gb must be a number of GB")
        ttl = min(max(float(ttl_s or 120.0), 5.0), 3600.0)
        j = self.store.add(
            kind="lease",
            app=str(app or "unknown")[:64],
            priority=self._priority(priority),
            vram_gb=None if vram_gb is None else float(vram_gb),
            ttl_s=ttl,
            note=(note or None) and str(note)[:300],
            heartbeat_at=self.clock(),
            params=dict(pid=int(pid)) if pid else None,
        )
        self._notify()
        return j

    def _lease_row(self, job_id):
        j = self.store.get(job_id, result=False)
        if j is None or j["kind"] != "lease":
            raise QueueError(f"no lease {job_id}")
        return j

    def heartbeat(self, job_id):
        j = self._lease_row(job_id)
        if j["status"] in FINISHED:
            raise QueueError(f"lease {job_id} is {j['status']}")
        return self.store.update(job_id, heartbeat_at=self.clock())

    def release(self, job_id):
        """Give a lease back (or withdraw one still queued)."""
        j = self._lease_row(job_id)
        if j["status"] == "running":
            j = self.store.update(job_id, status="done", finished_at=self.clock())
        elif j["status"] == "queued":
            j = self.store.update(job_id, status="cancelled", finished_at=self.clock())
        self._notify()
        return j

    def cancel(self, job_id):
        """Withdraw a queued job (a running one finishes: a GPU kernel cannot be stopped half way)."""
        j = self.store.get(job_id, result=False)
        if j is None:
            raise QueueError(f"no job {job_id}")
        if j["kind"] == "lease":
            return self.release(job_id)
        if j["status"] == "queued":
            j = self.store.update(job_id, status="cancelled", finished_at=self.clock())
            self._notify()
        return j

    def wait(self, job_id, timeout=0.0):
        """The job once it has finished (a lease: once granted or finished), or as it stands after `timeout` seconds."""
        end = time.monotonic() + max(float(timeout or 0), 0.0)
        with self.cv:
            while True:
                j = self.store.get(job_id)
                if j is None:
                    raise QueueError(f"no job {job_id}")
                if j["status"] in FINISHED or (j["kind"] == "lease" and j["status"] == "running"):
                    return j
                left = end - time.monotonic()
                if left <= 0:
                    return j
                self.cv.wait(min(left, 1.0))

    def reload_registry(self, registry=None):
        """Take up a changed registry (an edit from the web app or the CLI): new jobs resolve against it, loaded models stay loaded."""
        from stratamodels.registry import Registry

        self.registry = registry if registry is not None else Registry.from_db(self.store.engine, keep=self.registry)
        self._notify()
        return self.registry

    def request(self, action, key):
        """Load or unload a model (from the Models page); returns at once, the change shows on the next status read."""
        if action not in ("load", "unload"):
            raise QueueError(f"no action {action!r}")
        m = self.registry.models.get(key)
        if m is None:
            raise QueueError(f"no model {key!r}")
        if not m.served:
            raise QueueError(f"model {key!r} ({m.runtime}) is not served here yet")
        self.control.append((action, key))
        self._notify()

    # ---- state ------------------------------------------------------------------------------------------------------------------------
    def _event(self, text):
        self.events.append(dict(at=self.clock(), text=text))

    @property
    def current(self):
        """The job that has been running longest (None when nothing runs); `status()["running"]` lists them all."""
        with self.cv:
            return next(iter(self.running.values()), (None,))[0]

    def status(self, state=None):
        state = self.probe() if state is None else state
        with self.cv:
            res = [dict(r.to_json(), busy=k in self.busy) for k, r in self.resident.items()]
            running = [dict(j, lane=lane) for j, lane in self.running.values()]
        q = self.store.queued()
        budget = None
        leases = [j for j in self.store.running() if j["kind"] == "lease"]
        if state:
            own = gpu.own_used_mb(state)
            budget = dict(
                total_mb=state["total_mb"],
                used_mb=state["used_mb"],
                own_mb=own,
                other_mb=max(state["used_mb"] - own, 0.0),
                reserve_mb=self.reserve_mb,
                reserved_mb=self.outstanding_mb(state, leases),
                pending_mb=self.pending_mb(),
                free_for_models_mb=max(self.free_mb(state, leases), 0.0),
            )
        exclusive = next((j for j in leases if j["vram_gb"] is None), None)
        return dict(
            gpu=state,
            budget=budget,
            resident=res,
            current=running[0] if running else None,
            running=running,
            slots=dict(gpu=self.gpu_jobs, cpu=self.cpu_jobs),
            whole_card_mb=self.whole_card_mb,
            lease=exclusive or (leases[0] if leases else None),
            leases=leases,
            queued=q,
            events=list(self.events)[-30:],
        )

    @staticmethod
    def outstanding_mb(state, leases):
        """MB promised to sized leases that their processes have not taken yet (a reservation counts until the app's pid is seen using it)."""
        used = {p["pid"]: p["used_mb"] for p in (state or {}).get("processes", [])}
        return sum(
            max(l["vram_gb"] * MB - used.get((l.get("params") or {}).get("pid"), 0.0), 0.0)
            for l in leases
            if l["vram_gb"] is not None
        )

    def pending_mb(self):
        """MB the service's own work is about to take that the card may not show yet: a loading model's whole estimate, and for a model running
        a job, its estimate beyond what it measured when it loaded (the room a job's own allocations grow into)."""
        with self.cv:
            p = sum(self.loading.values())
            for k in self.busy:
                r = self.resident.get(k)
                if r is not None and k not in self.loading:
                    p += r.growth_mb
        return p

    def free_mb(self, state, leases=None):
        """MB the service may still fill: the card's free memory less the reserve kept for the desktop, what sized leases have reserved, and
        what running and loading jobs are about to take."""
        leases = self.held if leases is None else leases
        return (
            state["total_mb"]
            - state["used_mb"]
            - self.reserve_mb
            - self.outstanding_mb(state, leases)
            - self.pending_mb()
        )

    # ---- models (loads and runs on a job's thread; room is made by the dispatcher) ---------------------------------------------------
    def _state(self):
        """The card's state for the dispatcher's decisions: read again once GPU work has started, ended, loaded or been freed since."""
        snap = self._snap
        if snap is not None and snap[0] == self._epoch:
            return snap[1]
        epoch, state = self._epoch, self.probe()
        self._snap = (epoch, state)
        return state

    @staticmethod
    def _on_gpu(model):
        return model is not None and model.vram_gb > 0

    def _gpu_busy(self, but=None):
        """Keys of the GPU models running or loading now (other than `but`)."""
        with self.cv:
            return [k for k in self.busy if k != but and self._on_gpu(self._model_of_key(k))]

    def _model_of_key(self, key):
        r = self.resident.get(key)
        return r.model if r is not None else self.registry.models.get(key)

    def _unload(self, key, why):
        with self.cv:
            r = self.resident.pop(key, None)
            if r is not None and self._on_gpu(r.model):
                self._epoch += 1
        if r is None:
            return
        try:
            self.get_runtime(r.model.runtime).unload(r.handle)
        except Exception:
            pass  # freeing must not fail the job that needed the room
        self._event(f"unloaded {key} ({why})")

    def _idle(self):
        """Resident models no job is using, least recently used first: the only ones that may be evicted."""
        with self.cv:
            return [(k, r) for k, r in self.resident.items() if k not in self.busy]

    def _whole(self, need_mb):
        """Whether `need_mb` (a model's estimate or a lease's size) takes the whole card: more than `whole_card_gb`, or more than the budget
        could ever give."""
        if need_mb > self.whole_card_mb:
            return True
        state = self._state()
        return bool(state) and need_mb > state["total_mb"] - self.reserve_mb

    def _whole_model(self, model):
        return self._on_gpu(model) and self._whole(model.vram_gb * MB)

    def _card_taken(self):
        """Why a whole-card job or lease must wait now (GPU jobs running or loading, leases held), or None when the card is free."""
        if self.held:
            return f"{len(self.held)} lease(s) hold part of it"
        if self._gpu_busy():
            return "the jobs running on it finish first"
        return None

    def _make_room(self, need_mb, keep=None, why="lease wants the whole GPU"):
        """Unload idle models, least recently used first, until `need_mb` fits (None: every idle GPU model). Without a GPU there is no budget
        to keep. Models running a job are never unloaded."""
        if need_mb is None:
            for k, r in self._idle():
                if k != keep and (self._on_gpu(r.model) or r.vram_mb > 0):
                    self._unload(k, why)
            return
        state = self._state()
        if not state:
            return
        free = self.free_mb(state)
        for k, r in self._idle():
            if free >= need_mb:
                break
            if k == keep:
                continue
            free += r.frees_mb
            self._unload(k, f"room for {need_mb / MB:.1f} GB")

    def _need_mb(self, model):
        """MB a job for `model` will take beyond what the card shows now: the whole estimate to load it, or its growth if it is loaded."""
        with self.cv:
            r = self.resident.get(model.key)
        if not self._same(r, model):
            return model.vram_gb * MB
        return max(model.vram_gb * MB - (r.held_mb or 0.0), 0.0)

    @staticmethod
    def _same(r, model):
        """Whether resident `r` is `model` as it stands (not loaded with a runtime, file or options since edited)."""
        return r is not None and (r.model.runtime, r.model.name, r.model.options) == (
            model.runtime,
            model.name,
            model.options,
        )

    def _room_now(self, model):
        """Whether `model` fits now beside the work running, counting idle models as room (they can be evicted)."""
        state = self._state()
        if not state:
            return True
        idle = sum(r.frees_mb for k, r in self._idle() if k != model.key)
        return self.free_mb(state) + idle >= self._need_mb(model)

    def _ensure(self, model):
        """The loaded handle for `model`, loading it when needed (the dispatcher has made room); returns (Resident, seconds spent loading)."""
        with self.cv:
            r = self.resident.get(model.key)
        if r is not None and (r.model.runtime, r.model.name, r.model.options) != (
            model.runtime,
            model.name,
            model.options,
        ):
            self._unload(model.key, "its runtime, file or options were edited")
            r = None  # the old handle belongs to the old runtime and settings
        if r is not None:
            with self.cv:
                self.resident.move_to_end(model.key)
            return r, 0.0
        with self.cv:
            alone, epoch = not self._gpu_busy(but=model.key), self._epoch
        before = gpu.own_used_mb(self.probe())
        t0 = time.perf_counter()
        handle = self.get_runtime(model.runtime).load(model, self.models_dir)
        load_s = time.perf_counter() - t0
        measured = gpu.own_used_mb(self.probe()) - before
        with self.cv:
            alone = alone and epoch == self._epoch and not self._gpu_busy(but=model.key)
            held = max(measured, 0.0) if alone else None
            r = Resident(model, handle, held if held else model.vram_gb * MB, load_s, self.clock, held)
            self.resident[model.key] = r
            self.loading.pop(model.key, None)
            self._epoch += 1
        self._event(f"loaded {model.key} in {load_s:.1f} s ({r.vram_mb / MB:.1f} GB)")
        return r, load_s

    def _model_of(self, j):
        """The model a queued job runs on (a variant from before a restart is rebuilt from its options), or None."""
        model = self.registry.models.get(j["model"])
        options = (j["params"] or {}).get(OPTIONS)
        if model is None and options and "~" in (j["model"] or ""):
            try:
                model = self.registry.variant(j["model"].split("~")[0], options)
            except Exception:
                model = None
        return model

    def _lane(self, j):
        return "gpu" if j["kind"] == "lease" or self._on_gpu(self._model_of(j)) else "cpu"

    # ---- dispatching (one thread) -----------------------------------------------------------------------------------------------------
    def _do_control(self):
        """Pending load/unload requests. An unload of a model in use waits for its job; a load is returned as a ticket to run like a job."""
        later, ticket = [], None
        while self.control:
            action, key = self.control.popleft()
            m = self.registry.models.get(key)
            if m is None:
                continue
            with self.cv:
                in_use = key in self.busy
            if action == "unload":
                if in_use:
                    later.append((action, key))
                else:
                    self._unload(key, "asked from the Models page")
                continue
            if in_use or ticket is not None:
                later.append((action, key))
                continue
            with self.cv:
                if key in self.resident:
                    continue
            if self._whole_model(m):
                if self._card_taken():
                    later.append((action, key))  # waits for the whole card like a job
                    continue
                self._make_room(None, keep=key, why=f"{key} takes the whole GPU")
            elif self._on_gpu(m) and self._gpu_busy() and not self._room_now(m):
                later.append((action, key))  # waits for room like a job
                continue
            else:
                self._make_room(m.vram_gb * MB)
            ticket = self._take(None, m, "gpu" if self._on_gpu(m) else "cpu")
        self.control.extend(later)
        return ticket

    def _next(self, queued):
        if not queued:
            return None
        top = min(PRIORITY.get(j["priority"], 1) for j in queued)
        cands = [j for j in queued if PRIORITY.get(j["priority"], 1) == top]
        if self.clock() - cands[0]["submitted_at"] > self.max_skip_s:
            return cands[0]
        ahead = []  # a job for a loaded model may jump the queue, but never past a lease: an app is waiting on it
        for j in cands:
            if j["kind"] == "lease":
                break
            ahead.append(j)
        with self.cv:
            idle = [j for j in ahead if j["model"] in self.resident]
        return idle[0] if idle else cands[0]

    def _held_leases(self):
        """The leases held now, after expiring those whose app went quiet."""
        held = []
        for j in self.store.running():
            if j["kind"] != "lease":
                continue
            if self.clock() - (j["heartbeat_at"] or j["started_at"] or 0) > (j["ttl_s"] or 120.0):
                self.store.update(
                    j["id"], status="expired", finished_at=self.clock(), error="no heartbeat within its ttl"
                )
                self._event(f"lease {j['id']} ({j['app']}) expired")
                self._notify()
                continue
            held.append(j)
        self.held = held
        return held

    def _fits(self, j):
        """Whether a model job can run beside the sized leases held now: its model is loaded, or would fit once every other model is freed."""
        if not self.held:
            return True
        m = self.registry.models.get(j["model"])
        if m is None or m.key in self.resident:
            return True
        state = self._state()
        if not state:
            return True
        return self.free_mb(state) + sum(r.frees_mb for r in list(self.resident.values())) >= m.vram_gb * MB

    def _wait_note(self, j, text):
        if self._waiting != j["id"]:
            self._waiting = j["id"]
            self._event(text)

    def _grant(self, j):
        """Free what the lease asked for and grant it. A sized lease waits (stays queued, and nothing on the GPU passes it) while running jobs,
        other programs and leases hold so much that the room is not there even with every idle model freed. An exclusive lease waits until the sized ones held now are given back and the GPU jobs running now finish, and so
        does a sized one too big to share the card (see Whole card), which is then granted whatever the reserve."""
        need = None if j["vram_gb"] is None else j["vram_gb"] * MB
        if need is not None and self._whole(need):
            return self._grant_whole(j, need)
        if need is None and self.held:
            self._wait_note(
                j, f"lease {j['id']} ({j['app']}) waits for the whole GPU: {len(self.held)} lease(s) hold part of it"
            )
            return None
        if need is None and self._gpu_busy():
            self._wait_note(
                j, f"lease {j['id']} ({j['app']}) waits for the whole GPU: the jobs running on it finish first"
            )
            return None
        self._make_room(need)
        state = self._state() if need is not None else None
        if state and self.free_mb(state) < need:
            who = (
                "running jobs and other programs hold the card" if self._gpu_busy() else "other programs hold the card"
            )
            self._wait_note(j, f"lease {j['id']} ({j['app']}) waits for {need / MB:.1f} GB: {who}")
            return None
        claimed = self.store.claim(j["id"], status="running", started_at=self.clock(), heartbeat_at=self.clock())
        if claimed is None:
            return None  # withdrawn while the room was being made
        self._event(f"lease {j['id']} granted to {j['app']}" + (f" ({j['note']})" if j["note"] else ""))
        self._notify()
        return claimed

    def _grant_whole(self, j, need):
        state = self._state()
        if state and need > state["total_mb"]:
            j = self.store.update(
                j["id"],
                status="failed",
                finished_at=self.clock(),
                error=f"{j['vram_gb']:g} GB will never fit: the card has {state['total_mb'] / MB:.1f} GB",
            )
            self._notify()
            return j
        why = self._card_taken()
        if why:
            self._wait_note(j, f"lease {j['id']} ({j['app']}) waits for the whole GPU ({need / MB:.1f} GB): {why}")
            return None
        self._make_room(None, why=f"lease {j['id']} takes the whole GPU")
        claimed = self.store.claim(j["id"], status="running", started_at=self.clock(), heartbeat_at=self.clock())
        if claimed is None:
            return None
        self._event(
            f"lease {j['id']} granted to {j['app']} with the whole GPU ({need / MB:.1f} GB, nothing else on it, at its own risk)"
            + (f" ({j['note']})" if j["note"] else "")
        )
        self._notify()
        return claimed

    def _lane_jobs(self, lane):
        with self.cv:
            return sum(1 for _, ln in self.running.values() if ln == lane)

    def _pick_gpu(self, queued, exclusive):
        """The GPU lane's next lease or job that may start now, or None (an exclusive lease is held, the slots are full, or the head waits)."""
        if exclusive:
            return None
        with self.cv:
            busy = set(self.busy)
        cands = [
            q for q in queued if q["kind"] == "lease" or q["model"] not in busy
        ]  # a busy model's job waits for it, others may pass
        j = self._next(cands)
        if j is None or j["kind"] == "lease":
            return j
        m = self._model_of(j)
        if m is not None and self._whole_model(m):
            why = self._card_taken()
            if why:  # nothing on the GPU passes it, or it would never find the card empty
                self._wait_note(j, f"job {j['id']} ({j['role']}) waits for the whole GPU ({m.vram_gb:g} GB): {why}")
                return None
            return j
        if self._lane_jobs("gpu") >= self.gpu_jobs:
            return None
        if not self._fits(j):  # does not fit beside the leases: something smaller may
            j = next(
                (q for q in cands if q["kind"] == "run" and q["priority"] == j["priority"] and self._fits(q)), None
            )
            if j is None:
                return None
        m = self._model_of(j)
        if m is not None and self._whole_model(m):
            return None  # a smaller job stood in for one that cannot run beside the leases; a whole-card one waits for them
        if m is not None and self._gpu_busy() and not self._room_now(m):
            self._wait_note(j, f"job {j['id']} ({j['role']}) waits for {m.vram_gb:g} GB: the jobs running now hold it")
            return None  # nothing on the GPU passes it, so a big job is not starved by small ones
        return j

    def _pick_cpu(self, queued):
        if self._lane_jobs("cpu") >= self.cpu_jobs:
            return None
        with self.cv:
            busy = set(self.busy)
        cands = [q for q in queued if q["model"] not in busy]
        return min(cands, key=lambda q: (PRIORITY.get(q["priority"], 1), q["id"]), default=None)

    def _take(self, j, model, lane):
        """Mark `model` busy for job `j` (None: a load asked from the Models page) and return the ticket its thread runs."""
        with self.cv:
            key = model.key if model is not None else j["model"]
            self.busy[key] = j["id"] if j else "load"
            if model is not None and not self._same(self.resident.get(key), model) and self._on_gpu(model):
                self.loading[key] = model.vram_gb * MB
            if model is not None and lane == "gpu" and self._whole_model(model):
                self.alone.add(key)
            if j is not None:
                self.running[j["id"]] = (j, lane)
            if lane == "gpu":
                self._epoch += 1
        return dict(job=j, model=model, key=key, lane=lane)

    def dispatch(self):
        """Decide one thing: grant a lease (returns its row), or claim a job or a load (returns a ticket for `execute`), or None."""
        self._snap = None
        exclusive = any(l["vram_gb"] is None or self._whole(l["vram_gb"] * MB) for l in self._held_leases())
        with self.cv:
            exclusive = exclusive or bool(self.alone)  # a whole-card job or load runs with nothing beside it
        if not exclusive:  # load requests wait too: the holder has the card to itself
            t = self._do_control()
            if t is not None:
                return t
        queued = self.store.queued()
        gpu_q, cpu_q = [], []
        for q in queued:
            (gpu_q if self._lane(q) == "gpu" else cpu_q).append(q)
        g = self._pick_gpu(gpu_q, exclusive)
        c = self._pick_cpu(cpu_q)
        order = sorted(
            (x for x in (g, c) if x is not None),
            key=lambda q: (PRIORITY.get(q["priority"], 1), q["submitted_at"], q["id"]),
        )
        for j in order:
            if j["kind"] == "lease":
                r = self._grant(j)
                if r is not None:
                    return r
                continue  # it waits; nothing on the GPU passes it, the CPU lane may
            model, lane = self._model_of(j), ("gpu" if j is g else "cpu")
            if lane == "gpu" and model is not None and self._whole_model(model):
                self._make_room(None, keep=model.key, why=f"{model.key} takes the whole GPU")
                self._event(
                    f"job {j['id']} ({j['role']}) runs {model.key} on the whole GPU ({model.vram_gb:g} GB, nothing else on it, at its own risk)"
                )
            elif lane == "gpu" and model is not None:
                self._make_room(self._need_mb(model), keep=model.key)
            claimed = self.store.claim(j["id"], status="running", started_at=self.clock())
            if claimed is None:
                continue  # withdrawn after the queue was read
            if model is None or not model.served:
                return dict(job=claimed, model=None, key=None, lane=lane)
            return self._take(claimed, model, lane)
        return None

    def execute(self, t):
        """Run a ticket from `dispatch` (on its own thread, or inline from `step`); returns the job's final row (None for a load)."""
        try:
            if t["job"] is None:
                try:
                    self._ensure(t["model"])
                except Exception as e:
                    self._event(f"could not load {t['key']}: {e}")
                return None
            return self._run(t["job"], t["model"])
        finally:
            with self.cv:
                if t["key"] is not None:
                    self.busy.pop(t["key"], None)
                    self.loading.pop(t["key"], None)
                    self.alone.discard(t["key"])
                if t["job"] is not None:
                    self.running.pop(t["job"]["id"], None)
                if t["lane"] == "gpu":
                    self._epoch += 1
            self._notify()

    def _run(self, j, model):
        params = dict(j["params"] or {})
        params.pop(OPTIONS, None)
        load_s = 0.0
        try:
            if model is None or not model.served:
                raise runtimes.RunError(f"model {j['model']!r} is no longer served")
            r, load_s = self._ensure(model)
            t0 = time.perf_counter()
            try:
                result = self.get_runtime(model.runtime).run(r.handle, params)
            except runtimes.RunError:
                raise
            except Exception:
                self._unload(model.key, "it failed while running")
                raise  # a model that threw (out of memory, a CUDA error) is reloaded fresh next time
            finally:
                r.last_used = self.clock()
                r.jobs += 1
            return self.store.update(
                j["id"],
                status="done",
                result=result,
                finished_at=self.clock(),
                load_s=round(load_s, 3),
                run_s=round(time.perf_counter() - t0, 3),
            )
        except Exception as e:
            return self.store.update(
                j["id"],
                status="failed",
                error=str(e) if isinstance(e, runtimes.RunError) else f"{type(e).__name__}: {e}",
                finished_at=self.clock(),
                load_s=round(load_s, 3),
            )

    def step(self):
        """Decide one thing and do it here, start to finish (what the tests drive): a load/unload request, then the next job that can run
        beside the sized leases held. Returns the job or lease it handled, or None."""
        t = self.dispatch()
        if t is None or "job" not in t:
            return t  # nothing, or a lease's row
        return self.execute(t)

    def start(self):
        """Run the dispatcher on a daemon thread, each job on a thread of its own; returns the event that stops it (running jobs finish)."""
        stop = threading.Event()
        n = self.store.recover()
        if n:
            self._event(f"{n} job(s) were running when the service stopped; marked failed")

        def loop():
            while not stop.is_set():
                with self.cv:
                    self._dirty = False
                try:
                    t = self.dispatch()
                except Exception as e:
                    t = None
                    self._event(f"worker error: {type(e).__name__}: {e}")
                if t is not None and "job" in t:
                    name = f"job-{t['job']['id']}" if t["job"] else f"load-{t['key']}"
                    threading.Thread(target=self.execute, args=(t,), name=name, daemon=True).start()
                elif t is None:
                    with self.cv:
                        if not self._dirty:
                            self.cv.wait(1.0)

        threading.Thread(target=loop, name="gpu-dispatcher", daemon=True).start()
        return stop
