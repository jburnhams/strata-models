"""The service: `strata-models serve` starts the GPU worker, the GPU sampler and this FastAPI app, which serves the API and the built React app.

    strata-models serve [--host 127.0.0.1] [--port 8366] [--memory]

API (JSON), for apps (through the client in client/) and for the web app:
  GET  /api/health                       version, and whether a GPU was found
  GET  /api/status                       the GPU now, the VRAM budget, loaded models, the running job, the queue, recent events
  GET  /api/gpu/history                  samples of the last hour (util %, memory used)
  GET  /api/registry                     roles, models (with load stats and which runtimes are installed) and apps' preferences
  PUT  /api/roles/{key}                  {model?, description?} choose a role's model (the row then stops following models.toml)
  PUT  /api/models/{key}                 {runtime?, name?, vram_gb?, licence?, source?, used_by?, options?, roles?} add or change a model
  POST /api/registry/seed                import models.toml again (new entries; changed ones not edited here); ?follow=model:KEY|role:KEY first
  GET  /api/preferences                  every app's own choices
  PUT/DELETE /api/preferences/{app}/{role}   {model?, params?, note?} the model and default params an app's jobs for a role get
  GET  /api/files?path=|job=             which jobs read or wrote a file, or the files a job touched (paths only)
  POST /api/models/{key}/load|unload     ask the worker to load or free a model
  POST /api/jobs                         {role, params, app, priority?, model?} queue a model job
  POST /api/run/{role}                   {params, app, priority?, model?, timeout?} queue it and wait for the result (504 with the job if it takes longer)
  GET  /api/jobs?status=&role=&app=&kind=&before=&limit=   history, newest first, cursor `before` = the last id shown
  GET  /api/jobs/{id}?wait=              one job with its result; waits up to `wait` seconds for it to finish
  DELETE /api/jobs/{id}                  withdraw a queued job or give back a lease
  POST /api/leases                       {app, vram_gb?, ttl_s?, priority?, note?, pid?} ask for vram_gb of the GPU (shared), or all of it (exclusive)
  GET  /api/leases/{id}?wait=            the lease, once granted (or as it stands after `wait`)
  POST /api/leases/{id}/heartbeat        keep a held lease alive (it expires `ttl_s` after the last one)
  DELETE /api/leases/{id}                give it back
  GET  /api/usage?hours=24               jobs and GPU seconds per app and role
  GET/PUT /api/settings
  GET  /api/images?before=&limit=        generated pictures (image-generate jobs), newest first
  GET  /api/images/file?path=            one generated or uploaded picture
  POST /api/images/upload?name=          a reference picture (body = the file) -> {path}
  /api/evals...                          experiments, analyses, write-ups, blind judging: see evals/api.py
The page at / serves `server/static/index.html` when built (`cd web && npm run build`), else `landing.html`."""

import os
import time

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from stratamodels import __version__, gpu, runtimes, settings
from stratamodels.registry import Registry, RegistryError
from stratamodels.scheduler import QueueError, Scheduler

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
LANDING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "landing.html")
MAX_WAIT_S = 60.0


class JobIn(BaseModel):
    role: str
    params: dict = Field(default_factory=dict)
    app: str = "unknown"
    priority: str = "batch"
    model: str | None = None
    options: dict | None = None


class RunIn(BaseModel):
    params: dict = Field(default_factory=dict)
    app: str = "unknown"
    priority: str = "interactive"
    model: str | None = None
    timeout: float = 300.0


class LeaseIn(BaseModel):
    app: str
    vram_gb: float | None = None
    pid: int | None = None
    ttl_s: float = 120.0
    priority: str = "batch"
    note: str | None = None


class RoleIn(BaseModel):
    model: str | None = None
    description: str | None = None


class ModelIn(BaseModel):
    runtime: str | None = None
    name: str | None = None
    vram_gb: float | None = None
    licence: str | None = None
    source: str | None = None
    used_by: list[str] | None = None
    options: dict | None = None
    roles: list[str] | None = None


class PreferenceIn(BaseModel):
    model: str | None = None
    params: dict | None = None
    note: str | None = None


class SettingsIn(BaseModel):
    data_dir: str | None = None
    models_dir: str | None = None
    vram_reserve_gb: float | None = None
    gpu_jobs: int | None = None
    cpu_jobs: int | None = None


def create_app(scheduler: Scheduler, history: gpu.History | None = None):
    app = FastAPI(title="strata-models", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")
    s = scheduler

    @app.exception_handler(RegistryError)
    @app.exception_handler(QueueError)
    def bad_request(_req: Request, e: Exception):
        return JSONResponse(status_code=400, content=dict(detail=str(e)))

    def found(j, what="job"):
        if j is None:
            raise HTTPException(404, f"no {what}")
        return j

    def latest():
        return history.latest if history is not None and history.latest is not None else s.probe()

    @app.get("/", response_class=HTMLResponse)
    def index():
        built = os.path.join(STATIC, "index.html")
        with open(built if os.path.exists(built) else LANDING, encoding="utf8") as f:
            return f.read()

    @app.get("/api/health")
    def get_health():
        return dict(ok=True, version=__version__, gpu=bool(latest()))

    @app.get("/api/status")
    def get_status():
        return s.status(latest())

    @app.get("/api/gpu/history")
    def get_history():
        return dict(every_s=history.every_s if history else None, samples=history.to_json() if history else [])

    @app.get("/api/registry")
    def get_registry():
        stats = s.store.model_stats()
        missing = runtimes.available()
        d = s.registry.to_json()
        for m in d["models"]:
            m["stats"] = stats.get(m["key"])
            m["missing"] = missing.get(m["runtime"], []) if m["served"] else []
            m["resident"] = m["key"] in s.resident
        return d

    from stratamodels import registry as reg

    @app.put("/api/roles/{key}")
    def put_role(key: str, b: RoleIn):
        reg.set_role(s.store.engine, key, b.model, b.description)
        s.reload_registry()
        return get_registry()

    @app.put("/api/models/{key}")
    def put_model(key: str, b: ModelIn):
        d = b.model_dump(exclude_none=True)
        roles = d.pop("roles", None)
        reg.save_model(s.store.engine, key, roles=roles, **d)
        s.reload_registry()
        return get_registry()

    @app.post("/api/registry/seed")
    def post_seed(follow: str | None = None):
        if follow:
            kind, _, key = follow.partition(":")
            if kind not in ("model", "role") or not key:
                raise HTTPException(400, "follow is model:KEY or role:KEY")
            reg.follow_seed(s.store.engine, kind, key)
        out = reg.seed(s.store.engine)
        s.reload_registry()
        return out

    @app.get("/api/preferences")
    def get_preferences():
        return dict(items=[p.to_json() for p in s.registry.preferences.values()])

    @app.put("/api/preferences/{app_name}/{role}")
    def put_preference(app_name: str, role: str, b: PreferenceIn):
        if b.model is None and not b.params:
            raise HTTPException(400, "give a model or params (DELETE removes a preference)")
        reg.set_preference(s.store.engine, app_name, role, b.model, b.params, b.note)
        s.reload_registry()
        return get_preferences()

    @app.delete("/api/preferences/{app_name}/{role}")
    def delete_preference(app_name: str, role: str):
        reg.set_preference(s.store.engine, app_name, role)
        s.reload_registry()
        return get_preferences()

    @app.get("/api/files")
    def get_files(path: str | None = None, job: int | None = None):
        if path is None and job is None:
            raise HTTPException(400, "give path or job")
        return dict(items=s.store.files(job_id=job, path=path))

    @app.post("/api/models/{key}/{action}", status_code=202)
    def post_model(key: str, action: str):
        s.request(action, key)
        return dict(ok=True)

    @app.post("/api/jobs", status_code=201)
    def post_job(b: JobIn):
        return s.submit(b.role, b.params, b.app, b.priority, b.model, b.options)

    @app.post("/api/run/{role}")
    def post_run(role: str, b: RunIn):
        j = s.submit(role, b.params, b.app, b.priority, b.model)
        end = time.monotonic() + max(b.timeout, 0.0)
        while True:
            j = s.wait(j["id"], min(MAX_WAIT_S, max(end - time.monotonic(), 0.0)))
            if j["status"] in ("done", "failed", "cancelled") or time.monotonic() >= end:
                break
        if j["status"] == "done":
            return j
        if j["status"] == "failed":
            return JSONResponse(status_code=422, content=dict(detail=j["error"], job=j))
        return JSONResponse(
            status_code=504,
            content=dict(detail=f"job {j['id']} is still {j['status']}; poll GET /api/jobs/{j['id']}", job=j),
        )

    @app.get("/api/jobs")
    def get_jobs(
        status: str | None = None,
        role: str | None = None,
        app_name: str | None = Query(None, alias="app"),
        kind: str | None = None,
        before: int | None = None,
        limit: int = 50,
    ):
        items = s.store.list(status=status, role=role, app=app_name, kind=kind, before=before, limit=limit)
        return dict(items=items, next=items[-1]["id"] if len(items) >= max(1, min(limit, 500)) else None)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: int, wait: float = 0.0):
        if s.store.get(job_id, result=False) is None:
            raise HTTPException(404, "no job")
        return s.wait(job_id, min(wait, MAX_WAIT_S))

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: int):
        return s.cancel(job_id)

    @app.post("/api/leases", status_code=201)
    def post_lease(b: LeaseIn):
        return s.lease(b.app, b.vram_gb, b.ttl_s, b.priority, b.note, b.pid)

    @app.get("/api/leases/{job_id}")
    def get_lease(job_id: int, wait: float = 0.0):
        s._lease_row(job_id)
        return s.wait(job_id, min(wait, MAX_WAIT_S))

    @app.post("/api/leases/{job_id}/heartbeat")
    def post_heartbeat(job_id: int):
        return s.heartbeat(job_id)

    @app.delete("/api/leases/{job_id}")
    def delete_lease(job_id: int):
        return s.release(job_id)

    @app.get("/api/usage")
    def get_usage(hours: float = 24.0):
        return dict(hours=hours, rows=s.store.usage(s.clock() - max(hours, 0.0) * 3600))

    @app.get("/api/settings")
    def get_settings():
        return settings.all_settings()

    @app.put("/api/settings")
    def put_settings(b: SettingsIn):
        changes = {}
        for k in ("data_dir", "models_dir"):
            v = getattr(b, k)
            if v is not None:
                if not os.path.isabs(os.path.expanduser(v)):
                    raise HTTPException(400, f"{k} must be an absolute path")
                changes[k] = v
        for k in ("vram_reserve_gb", "gpu_jobs", "cpu_jobs"):
            if getattr(b, k) is not None:
                changes[k] = getattr(b, k)
        try:
            settings.save(changes)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if "vram_reserve_gb" in changes:
            s.reserve_mb = settings.vram_reserve_gb() * 1024.0
        if "models_dir" in changes:
            s.models_dir = settings.models_dir()
        if "gpu_jobs" in changes or "cpu_jobs" in changes:  # takes effect at the next dispatch; jobs running now finish
            s.gpu_jobs, s.cpu_jobs = settings.jobs("gpu_jobs"), settings.jobs("cpu_jobs")
            s._notify()
        return settings.all_settings()

    @app.get("/api/images")
    def get_images(before: int | None = None, limit: int = 40):
        """Generated pictures, newest first (the image-generate jobs that finished), for the gallery."""
        items = list(s.store.list(status="done", role="image-generate", before=before, limit=limit))
        out = []
        for j in items:
            full = s.store.get(j["id"])
            r = full.get("result") or {}
            if r.get("path"):
                out.append(
                    dict(
                        id=j["id"],
                        app=j["app"],
                        model=j["model"],
                        prompt=(j["params"] or {}).get("prompt"),
                        finished_at=j["finished_at"],
                        **{
                            k: r.get(k) for k in ("path", "width", "height", "seed", "steps", "seconds", "peak_vram_mb")
                        },
                    )
                )
        return dict(items=out, next=items[-1]["id"] if len(items) >= max(1, min(limit, 500)) else None)

    def _inside(path, *roots):
        real = os.path.realpath(path)
        return any(real.startswith(os.path.realpath(r) + os.sep) for r in roots)

    @app.get("/api/images/file")
    def get_image_file(path: str):
        """A generated picture or sound (speak jobs), an uploaded reference, or an eval's reference voice; nothing else on the disk."""
        roots = [
            os.path.join(settings.data_dir(), d) for d in ("generated", "uploads", os.path.join("evals", "voices"))
        ]
        if not _inside(path, *roots) or not os.path.isfile(path):
            raise HTTPException(404, "no such file")
        return FileResponse(path)

    @app.post("/api/images/upload", status_code=201)
    async def post_upload(request: Request, name: str = "reference.png"):
        """A reference picture for an edit (the request body is the file); returns the path to give in ref_images."""
        data = await request.body()
        if not data or len(data) > 40 * 1024 * 1024:
            raise HTTPException(400, "send a picture of at most 40 MB as the request body")
        ext = os.path.splitext(name)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp"):
            raise HTTPException(400, "a reference is a .png, .jpg or .webp picture")
        import hashlib

        folder = os.path.join(settings.data_dir(), "uploads")
        os.makedirs(folder, exist_ok=True)
        p = os.path.join(folder, hashlib.sha256(data).hexdigest()[:16] + ext)
        with open(p, "wb") as f:
            f.write(data)
        return dict(path=p)

    from stratamodels.evals import api as evals_api

    evals_api.mount(app, s)

    if os.path.isdir(os.path.join(STATIC, "assets")):
        app.mount("/assets", StaticFiles(directory=os.path.join(STATIC, "assets")), name="assets")
    return app


def build(memory=False, sample_every_s=5.0):
    """The scheduler, sampler and app for this machine (the database, the registry, nvidia-smi), with the worker and sampler started."""
    from stratamodels import registry as reg
    from stratamodels.store import Store, memory_store

    store = memory_store() if memory else Store.connect()
    if not memory:
        settings.bind(store.engine)  # the settings live in the database (a demo keeps the file)
    try:
        reg.seed(store.engine, log=print)  # models.toml: new entries in, edited rows kept
    except reg.RegistryError as e:
        print(f"registry seed refused, serving the database as it is: {e}")
    history = gpu.History(gpu.query, every_s=sample_every_s)
    history.sample()
    sch = Scheduler(
        store,
        Registry.from_db(store.engine),
        settings.models_dir(),
        probe=gpu.query,
        reserve_gb=settings.vram_reserve_gb(),
        gpu_jobs=settings.jobs("gpu_jobs"),
        cpu_jobs=settings.jobs("cpu_jobs"),
    )
    sch.start()
    history.start()
    app = create_app(sch, history)
    if not memory:
        app.state.eval_summaries.warm()  # every experiment's analysis up to date in the background, before the page asks
    return app


def main(argv=None):
    import argparse

    import uvicorn

    a = argparse.ArgumentParser()
    a.add_argument("--host", default="127.0.0.1")
    a.add_argument("--port", type=int, default=8366)
    a.add_argument("--memory", action="store_true", help="keep jobs in memory (a demo without the database)")
    a = a.parse_args(argv)
    app = build(memory=a.memory)
    print(f"models folder: {settings.models_dir()}\nhttp://{a.host}:{a.port}/")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
