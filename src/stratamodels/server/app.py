"""The service: `strata-models serve` starts the GPU worker, the GPU sampler and this FastAPI app, which serves the API and the built React app.

    strata-models serve [--host 127.0.0.1] [--port 8366] [--memory]

API (JSON), for apps (through the client in client/) and for the web app:
  GET  /api/health                       version, and whether a GPU was found
  GET  /api/status                       the GPU now, the VRAM budget, loaded models, the running job, the queue, recent events
  GET  /api/gpu/history                  samples of the last hour (util %, memory used)
  GET  /api/registry                     roles and models (with load stats and which runtimes are installed)
  POST /api/models/{key}/load|unload     ask the worker to load or free a model
  POST /api/jobs                         {role, params, app, priority?, model?} queue a model job
  POST /api/run/{role}                   {params, app, priority?, model?, timeout?} queue it and wait for the result (504 with the job if it takes longer)
  GET  /api/jobs?status=&role=&app=&kind=&before=&limit=   history, newest first, cursor `before` = the last id shown
  GET  /api/jobs/{id}?wait=              one job with its result; waits up to `wait` seconds for it to finish
  DELETE /api/jobs/{id}                  withdraw a queued job or give back a lease
  POST /api/leases                       {app, vram_gb?, ttl_s?, priority?, note?} ask for the whole GPU
  GET  /api/leases/{id}?wait=            the lease, once granted (or as it stands after `wait`)
  POST /api/leases/{id}/heartbeat        keep a held lease alive (it expires `ttl_s` after the last one)
  DELETE /api/leases/{id}                give it back
  GET  /api/usage?hours=24               jobs and GPU seconds per app and role
  GET/PUT /api/settings
  /api/evals...                          experiments, analyses, write-ups, blind judging: see evals/api.py
The page at / serves `server/static/index.html` when built (`cd web && npm run build`), else `landing.html`."""
import os, time

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from stratamodels import __version__, gpu, runtimes, settings
from stratamodels.registry import Registry, RegistryError
from stratamodels.scheduler import QueueError, Scheduler

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
LANDING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'landing.html')
MAX_WAIT_S = 60.0


class JobIn(BaseModel):
    role: str
    params: dict = Field(default_factory=dict)
    app: str = 'unknown'
    priority: str = 'batch'
    model: str | None = None
    options: dict | None = None


class RunIn(BaseModel):
    params: dict = Field(default_factory=dict)
    app: str = 'unknown'
    priority: str = 'interactive'
    model: str | None = None
    timeout: float = 300.0


class LeaseIn(BaseModel):
    app: str
    vram_gb: float | None = None
    ttl_s: float = 120.0
    priority: str = 'batch'
    note: str | None = None


class SettingsIn(BaseModel):
    data_dir: str | None = None
    models_dir: str | None = None
    vram_reserve_gb: float | None = None


def create_app(scheduler: Scheduler, history: gpu.History | None = None):
    app = FastAPI(title='strata-models', version=__version__, docs_url='/api/docs', openapi_url='/api/openapi.json')
    s = scheduler

    @app.exception_handler(RegistryError)
    @app.exception_handler(QueueError)
    def bad_request(_req: Request, e: Exception): return JSONResponse(status_code=400, content=dict(detail=str(e)))

    def found(j, what='job'):
        if j is None: raise HTTPException(404, f'no {what}')
        return j

    def latest():
        return history.latest if history is not None and history.latest is not None else s.probe()

    @app.get('/', response_class=HTMLResponse)
    def index():
        built = os.path.join(STATIC, 'index.html')
        with open(built if os.path.exists(built) else LANDING, encoding='utf8') as f: return f.read()

    @app.get('/api/health')
    def get_health(): return dict(ok=True, version=__version__, gpu=bool(latest()))

    @app.get('/api/status')
    def get_status(): return s.status(latest())

    @app.get('/api/gpu/history')
    def get_history(): return dict(every_s=history.every_s if history else None, samples=history.to_json() if history else [])

    @app.get('/api/registry')
    def get_registry():
        stats = s.store.model_stats(); missing = runtimes.available(); d = s.registry.to_json()
        for m in d['models']: m['stats'] = stats.get(m['key']); m['missing'] = missing.get(m['runtime'], []) if m['served'] else []; m['resident'] = m['key'] in s.resident
        return d

    @app.post('/api/models/{key}/{action}', status_code=202)
    def post_model(key: str, action: str): s.request(action, key); return dict(ok=True)

    @app.post('/api/jobs', status_code=201)
    def post_job(b: JobIn): return s.submit(b.role, b.params, b.app, b.priority, b.model, b.options)

    @app.post('/api/run/{role}')
    def post_run(role: str, b: RunIn):
        j = s.submit(role, b.params, b.app, b.priority, b.model)
        end = time.monotonic() + max(b.timeout, 0.0)
        while True:
            j = s.wait(j['id'], min(MAX_WAIT_S, max(end - time.monotonic(), 0.0)))
            if j['status'] in ('done', 'failed', 'cancelled') or time.monotonic() >= end: break
        if j['status'] == 'done': return j
        if j['status'] == 'failed': return JSONResponse(status_code=422, content=dict(detail=j['error'], job=j))
        return JSONResponse(status_code=504, content=dict(detail=f'job {j["id"]} is still {j["status"]}; poll GET /api/jobs/{j["id"]}', job=j))

    @app.get('/api/jobs')
    def get_jobs(status: str | None = None, role: str | None = None, app_name: str | None = Query(None, alias='app'), kind: str | None = None, before: int | None = None, limit: int = 50):
        items = s.store.list(status=status, role=role, app=app_name, kind=kind, before=before, limit=limit)
        return dict(items=items, next=items[-1]['id'] if len(items) >= max(1, min(limit, 500)) else None)

    @app.get('/api/jobs/{job_id}')
    def get_job(job_id: int, wait: float = 0.0):
        if s.store.get(job_id, result=False) is None: raise HTTPException(404, 'no job')
        return s.wait(job_id, min(wait, MAX_WAIT_S))

    @app.delete('/api/jobs/{job_id}')
    def delete_job(job_id: int): return s.cancel(job_id)

    @app.post('/api/leases', status_code=201)
    def post_lease(b: LeaseIn): return s.lease(b.app, b.vram_gb, b.ttl_s, b.priority, b.note)

    @app.get('/api/leases/{job_id}')
    def get_lease(job_id: int, wait: float = 0.0):
        s._lease_row(job_id); return s.wait(job_id, min(wait, MAX_WAIT_S))

    @app.post('/api/leases/{job_id}/heartbeat')
    def post_heartbeat(job_id: int): return s.heartbeat(job_id)

    @app.delete('/api/leases/{job_id}')
    def delete_lease(job_id: int): return s.release(job_id)

    @app.get('/api/usage')
    def get_usage(hours: float = 24.0): return dict(hours=hours, rows=s.store.usage(s.clock() - max(hours, 0.0) * 3600))

    @app.get('/api/settings')
    def get_settings(): return settings.all_settings()

    @app.put('/api/settings')
    def put_settings(b: SettingsIn):
        changes = {}
        for k in ('data_dir', 'models_dir'):
            v = getattr(b, k)
            if v is not None:
                if not os.path.isabs(os.path.expanduser(v)): raise HTTPException(400, f'{k} must be an absolute path')
                changes[k] = v
        if b.vram_reserve_gb is not None: changes['vram_reserve_gb'] = b.vram_reserve_gb
        try: settings.save(changes)
        except ValueError as e: raise HTTPException(400, str(e))
        if 'vram_reserve_gb' in changes: s.reserve_mb = settings.vram_reserve_gb() * 1024.0
        if 'models_dir' in changes: s.models_dir = settings.models_dir()
        return settings.all_settings()

    from stratamodels.evals import api as evals_api
    evals_api.mount(app, s)

    if os.path.isdir(os.path.join(STATIC, 'assets')): app.mount('/assets', StaticFiles(directory=os.path.join(STATIC, 'assets')), name='assets')
    return app


def build(memory=False, sample_every_s=5.0):
    """The scheduler, sampler and app for this machine (the database, the registry, nvidia-smi), with the worker and sampler started."""
    from stratamodels.store import Store, memory_store
    store = memory_store() if memory else Store.connect()
    history = gpu.History(gpu.query, every_s=sample_every_s); history.sample()
    sch = Scheduler(store, Registry.load(), settings.models_dir(), probe=gpu.query, reserve_gb=settings.vram_reserve_gb())
    sch.start(); history.start(); return create_app(sch, history)


def main(argv=None):
    import argparse, uvicorn
    a = argparse.ArgumentParser(); a.add_argument('--host', default='127.0.0.1'); a.add_argument('--port', type=int, default=8366); a.add_argument('--memory', action='store_true', help='keep jobs in memory (a demo without the database)'); a = a.parse_args(argv)
    app = build(memory=a.memory)
    print(f'models folder: {settings.models_dir()}\nhttp://{a.host}:{a.port}/'); uvicorn.run(app, host=a.host, port=a.port, log_level='warning')
