"""Job rows in the database (`jobs`, with each run's output in `results` and the paths it read or wrote in `files`): a thin, thread-safe layer the
scheduler and the API share. A row comes back as one dict, `result` included, as it did when the output was a column of the job. Every call opens its own session, so the worker thread
and the web requests never share one; calls are also serialised by a lock, which costs nothing at this volume and makes the in-memory SQLite
store (one shared connection) safe across threads."""

import functools
import threading
import time

import sqlalchemy as sa
from sqlalchemy.orm import Session

from stratamodels.db.models import ApiCall, Job, JobFile, JobResult

FIELDS = [c.name for c in Job.__table__.columns]
FINISHED = ("done", "failed", "cancelled", "expired")
PATH_KEYS = ("path", "paths", "ref_images", "images", "audio", "files")


def row(j, result=True, output=None):
    d = {k: getattr(j, k) for k in FIELDS}
    if result:
        d["result"] = output
    if d["started_at"] is not None:
        d["wait_s"] = round(d["started_at"] - d["submitted_at"], 3)
    return d


def paths(x, key=None):
    """The absolute file paths in a job's params or result: values under keys named path/paths/ref_images/..., or ending in _path(s)."""
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            if k.startswith("__"):
                continue
            out += paths(v, k)
    elif isinstance(x, (list, tuple)):
        for v in x:
            out += paths(v, key)
    elif (
        isinstance(x, str)
        and key
        and (key in PATH_KEYS or key.endswith(("_path", "_paths")))
        and (x.startswith("/") or (len(x) > 2 and x[1] == ":" and x[2] in "\\/"))
    ):
        out.append(x)
    return out


def _files(s, job_id, direction, data, at):
    seen = set()
    for p in paths(data):
        if p not in seen and len(p) <= 1024:
            seen.add(p)
            s.add(JobFile(job_id=job_id, direction=direction, path=p, created_at=at))


def _output(s, job_id):
    r = s.get(JobResult, job_id)
    return r.output if r else None


def locked(fn):
    @functools.wraps(fn)
    def run(self, *a, **kw):
        with self.lock:
            return fn(self, *a, **kw)

    return run


class Store:
    def __init__(self, engine, clock=time.time):
        self.engine, self.clock, self.lock = engine, clock, threading.RLock()
        engine.stratamodels_lock = (
            self.lock
        )  # evals open their own sessions on this engine: they take the same lock (see evals.core.session)

    @classmethod
    def connect(cls, url_or_app="stratamodels", migrate=True):
        """The service's database (omarchy-db: STRATAMODELS_DATABASE_URL, OMARCHY_DB_URL or the shared server). `migrate`: True brings it to
        head (the service, at start); 'new' only creates the tables of a database that has none and refuses one at another revision (the CLI:
        a command run from a branch must never migrate the live database under the running service); False leaves it as it is."""
        from omarchy_db import connect

        e = connect(url_or_app)
        if e.dialect.name == "postgresql":
            fast_commits(e)
        if migrate == "new":
            have, head = revisions(e)
            if have is None:
                project().upgrade(e)
            elif have != head:
                raise StaleDatabase(
                    f"the database is at revision {have}, this code expects {head}: deploy the matching service (it migrates at start) or run `dbtool migrate`"
                )
        elif migrate:
            project().upgrade(e)
        return cls(e)

    @locked
    def add(self, **kw):
        kw.setdefault("submitted_at", self.clock())
        kw.setdefault("status", "queued")
        with Session(self.engine, expire_on_commit=False) as s:
            j = Job(**kw)
            s.add(j)
            s.flush()
            if j.kind == "run":
                _files(s, j.id, "input", j.params, j.submitted_at)
            s.commit()
            return row(j)

    @locked
    def update(self, job_id, **kw):
        """Change a job; `result=` stores its output (in results, with the paths it wrote in files)."""
        has, result = "result" in kw, kw.pop("result", None)
        with Session(self.engine, expire_on_commit=False) as s:
            j = s.get(Job, job_id)
            if j is None:
                return None
            for k, v in kw.items():
                setattr(j, k, v)
            if has:
                at = j.finished_at or self.clock()
                r = s.get(JobResult, job_id)
                if r is None:
                    s.add(JobResult(job_id=job_id, output=result, created_at=at))
                else:
                    r.output = result
                _files(s, job_id, "output", result, at)
            s.commit()
            return row(j, output=_output(s, job_id))

    @locked
    def claim(self, job_id, **kw):
        """Start a queued job: the update happens only while it is still queued (a lease withdrawn meanwhile stays withdrawn). Returns the row, or None."""
        with Session(self.engine, expire_on_commit=False) as s:
            n = s.execute(sa.update(Job).where(Job.id == job_id, Job.status == "queued").values(**kw)).rowcount
            s.commit()
            return row(s.get(Job, job_id), output=_output(s, job_id)) if n else None

    @locked
    def get(self, job_id, result=True):
        with Session(self.engine) as s:
            j = s.get(Job, job_id)
            return row(j, result, _output(s, job_id) if result else None) if j else None

    @locked
    def files(self, job_id=None, path=None):
        """Paths a job read or wrote, or the jobs that read or wrote `path`."""
        q = sa.select(JobFile).order_by(JobFile.id)
        if job_id is not None:
            q = q.where(JobFile.job_id == job_id)
        if path is not None:
            q = q.where(JobFile.path == path)
        with Session(self.engine) as s:
            return [
                dict(job_id=f.job_id, direction=f.direction, path=f.path, created_at=f.created_at) for f in s.scalars(q)
            ]

    @locked
    def queued(self):
        with Session(self.engine) as s:
            return [row(j, False) for j in s.scalars(sa.select(Job).where(Job.status == "queued").order_by(Job.id))]

    @locked
    def running(self):
        with Session(self.engine) as s:
            return [row(j, False) for j in s.scalars(sa.select(Job).where(Job.status == "running").order_by(Job.id))]

    @locked
    def list(self, status=None, role=None, app=None, kind=None, before=None, limit=50):
        """Newest first, `limit` rows before job id `before` (the cursor), without results."""
        q = sa.select(Job).order_by(Job.id.desc()).limit(max(1, min(int(limit), 500)))
        for col, v in ((Job.status, status), (Job.role, role), (Job.app, app), (Job.kind, kind)):
            if v:
                q = q.where(col == v)
        if before:
            q = q.where(Job.id < int(before))
        with Session(self.engine) as s:
            return [row(j, False) for j in s.scalars(q)]

    @locked
    def recover(self):
        """At start: jobs a previous run left running cannot finish, so they fail (the apps that asked have long stopped waiting); returns how many."""
        with Session(self.engine) as s:
            n = s.execute(
                sa.update(Job)
                .where(Job.status == "running")
                .values(status="failed", error="the service restarted while this ran", finished_at=self.clock())
            ).rowcount
            s.commit()
            return n

    @locked
    def usage(self, since):
        """Per app and role since `since`: jobs, failures, GPU seconds (run time, plus lease hold time)."""
        held = sa.func.coalesce(Job.finished_at, self.clock()) - Job.started_at
        q = (
            sa.select(
                Job.app,
                Job.kind,
                Job.role,
                sa.func.count(),
                sa.func.sum(sa.case((Job.status == "failed", 1), else_=0)),
                sa.func.sum(sa.case((Job.kind == "lease", held), else_=sa.func.coalesce(Job.run_s, 0.0))),
            )
            .where(Job.submitted_at >= since, Job.started_at.is_not(None))
            .group_by(Job.app, Job.kind, Job.role)
            .order_by(Job.app, Job.kind, Job.role)
        )
        with Session(self.engine) as s:
            return [
                dict(app=a, kind=k, role=r, jobs=int(n), failed=int(f or 0), gpu_s=round(float(g or 0.0), 1))
                for a, k, r, n, f, g in s.execute(q)
            ]

    @locked
    def add_api_call(self, **kw):
        """One row of the cloud API ledger (spend.record); returns it as a dict."""
        with Session(self.engine) as s:
            r = ApiCall(**kw)
            s.add(r)
            s.commit()
            return {c.name: getattr(r, c.name) for c in ApiCall.__table__.columns}

    @locked
    def api_spent(self, provider, since):
        """(USD, calls) a provider's key spent since `since`."""
        q = sa.select(sa.func.coalesce(sa.func.sum(ApiCall.cost_usd), 0.0), sa.func.count()).where(
            ApiCall.provider == provider, ApiCall.at >= since
        )
        with Session(self.engine) as s:
            usd, n = s.execute(q).one()
            return float(usd or 0.0), int(n)

    @locked
    def api_breakdown(self, since):
        """Per provider, model and app since `since`: calls, failures, tokens and USD."""
        q = (
            sa.select(
                ApiCall.provider,
                ApiCall.model,
                ApiCall.app,
                sa.func.count(),
                sa.func.sum(sa.case((ApiCall.ok.is_(False), 1), else_=0)),
                sa.func.sum(ApiCall.tokens_in + ApiCall.cache_read + ApiCall.cache_write),
                sa.func.sum(ApiCall.tokens_out),
                sa.func.sum(ApiCall.cost_usd),
            )
            .where(ApiCall.at >= since)
            .group_by(ApiCall.provider, ApiCall.model, ApiCall.app)
            .order_by(ApiCall.provider, ApiCall.model, ApiCall.app)
        )
        with Session(self.engine) as s:
            return [
                dict(
                    provider=p,
                    model=m,
                    app=a,
                    calls=int(n),
                    failed=int(f or 0),
                    tokens_in=int(ti or 0),
                    tokens_out=int(to or 0),
                    cost_usd=float(c or 0.0),
                )
                for p, m, a, n, f, ti, to, c in s.execute(q)
            ]

    @locked
    def api_log(self, provider=None, model=None, app=None, ok=None, since=None, before=None, limit=50):
        """Ledger rows newest first, filtered, without the request and response bodies (`api_call` has them); cursor `before` = the last id."""
        q = sa.select(*[c for c in ApiCall.__table__.columns if c.name not in ("request", "response")]).order_by(
            ApiCall.id.desc()
        )
        for col, v in ((ApiCall.provider, provider), (ApiCall.model, model), (ApiCall.app, app), (ApiCall.ok, ok)):
            if v is not None:
                q = q.where(col == v)
        if since is not None:
            q = q.where(ApiCall.at >= since)
        if before is not None:
            q = q.where(ApiCall.id < before)
        with Session(self.engine) as s:
            return [dict(r._mapping) for r in s.execute(q.limit(max(1, min(int(limit), 500))))]

    @locked
    def api_per_task(self, since=0.0):
        """Per (role, model key): jobs that called a cloud API since `since`, their mean USD and tokens (a job's calls summed first, so a
        Gemini job that fell from the free key to the paid one counts once), and how many of those jobs failed."""
        per_job = (
            sa.select(
                ApiCall.job_id.label("job_id"),
                sa.func.max(ApiCall.model_key).label("model_key"),
                sa.func.sum(ApiCall.cost_usd).label("usd"),
                sa.func.sum(ApiCall.tokens_in + ApiCall.cache_read + ApiCall.cache_write).label("tin"),
                sa.func.sum(ApiCall.tokens_out).label("tout"),
            )
            .where(ApiCall.at >= since, ApiCall.job_id.is_not(None))
            .group_by(ApiCall.job_id)
            .subquery()
        )
        q = (
            sa.select(
                Job.role,
                per_job.c.model_key,
                sa.func.count(),
                sa.func.avg(per_job.c.usd),
                sa.func.avg(per_job.c.tin),
                sa.func.avg(per_job.c.tout),
                sa.func.sum(sa.case((Job.status == "failed", 1), else_=0)),
            )
            .join(Job, Job.id == per_job.c.job_id)
            .group_by(Job.role, per_job.c.model_key)
        )
        with Session(self.engine) as s:
            return {
                (r, m): dict(
                    n=int(n),
                    usd=float(u or 0.0),
                    tokens_in=float(ti or 0.0),
                    tokens_out=float(to or 0.0),
                    failed=int(f or 0),
                )
                for r, m, n, u, ti, to, f in s.execute(q)
            }

    @locked
    def api_call(self, call_id):
        """One ledger row with what was sent and what came back, or None."""
        with Session(self.engine) as s:
            r = s.get(ApiCall, call_id)
            return None if r is None else {c.name: getattr(r, c.name) for c in ApiCall.__table__.columns}

    @locked
    def api_points(self, since):
        """(at, provider, model, app, cost_usd, tokens, ok) of every call since `since`, oldest first: what the charts bucket."""
        q = (
            sa.select(
                ApiCall.at,
                ApiCall.provider,
                ApiCall.model,
                ApiCall.app,
                ApiCall.cost_usd,
                ApiCall.tokens_in + ApiCall.cache_read + ApiCall.cache_write + ApiCall.tokens_out,
                ApiCall.ok,
            )
            .where(ApiCall.at >= since)
            .order_by(ApiCall.at)
        )
        with Session(self.engine) as s:
            return [tuple(r) for r in s.execute(q)]

    @locked
    def model_stats(self):
        """Per model: jobs run, mean run seconds, last finished time."""
        q = (
            sa.select(Job.model, sa.func.count(), sa.func.avg(Job.run_s), sa.func.max(Job.finished_at))
            .where(Job.kind == "run", Job.status == "done")
            .group_by(Job.model)
        )
        with Session(self.engine) as s:
            return {
                m: dict(jobs=int(n), mean_run_s=round(float(a), 3) if a is not None else None, last_at=l)
                for m, n, a, l in s.execute(q)
            }

    @locked
    def prune(self, keep_s=7 * 86400):
        """Drop results older than `keep_s` (embeddings and transcripts are the apps' to keep); the job rows and their file paths stay for the history."""
        with Session(self.engine) as s:
            n = s.execute(sa.delete(JobResult).where(JobResult.created_at < self.clock() - keep_s)).rowcount
            s.commit()
            return n


class StaleDatabase(RuntimeError):
    """The database is at another schema revision than this code."""


def project():
    import os

    from omarchy_db import Project

    return Project.load(
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "pyproject.toml")
    )


def revisions(engine):
    """(the database's revision or None, this code's head)."""
    from alembic.script import ScriptDirectory
    from omarchy_db.migrations import current, make_config

    p = project()
    head = ScriptDirectory.from_config(make_config(p.migrations_dir, p.metadata, None)).get_current_head()
    return current(engine, p.schema if engine.dialect.name == "postgresql" else None), head


def fast_commits(engine):
    """Commit without waiting for the WAL flush (synchronous_commit off) on the service's own connections: each commit was 17 ms on omarchy,
    and a job takes three (queued, started, finished), which was most of a quick job's overhead. A crash can lose the last few hundred
    milliseconds of job rows, never corrupt them; the apps that asked get their answer over HTTP either way."""
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _off(dbapi_conn, _rec):
        cur = dbapi_conn.cursor()
        cur.execute("SET synchronous_commit TO OFF")
        cur.close()
        if not getattr(dbapi_conn, "autocommit", True):
            dbapi_conn.commit()

    return engine


def memory_store(clock=time.time):
    """An in-memory SQLite store built by the migrations, not create_all, so the tests exercise them (tests, and `serve --memory` for a demo
    without a database)."""
    from sqlalchemy.pool import StaticPool

    e = sa.create_engine("sqlite://", connect_args=dict(check_same_thread=False), poolclass=StaticPool)
    project().upgrade(e)
    return Store(e, clock)
