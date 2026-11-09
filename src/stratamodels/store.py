"""Job rows in the database: a thin, thread-safe layer the scheduler and the API share. Every call opens its own session, so the worker thread
and the web requests never share one; calls are also serialised by a lock, which costs nothing at this volume and makes the in-memory SQLite
store (one shared connection) safe across threads."""

import functools, threading, time

import sqlalchemy as sa
from sqlalchemy.orm import Session

from stratamodels.db.models import Base, Job

FIELDS = [c.name for c in Job.__table__.columns]
FINISHED = ("done", "failed", "cancelled", "expired")


def row(j, result=True):
    d = {k: getattr(j, k) for k in FIELDS}
    if not result:
        d.pop("result")
    if d["started_at"] is not None:
        d["wait_s"] = round(d["started_at"] - d["submitted_at"], 3)
    return d


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
        """The service's database (omarchy-db: STRATAMODELS_DATABASE_URL, OMARCHY_DB_URL or the shared server), migrated to head."""
        from omarchy_db import Project, connect
        import os

        e = connect(url_or_app)
        if e.dialect.name == "postgresql":
            fast_commits(e)
        if migrate:
            Project.load(
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "pyproject.toml"
                )
            ).upgrade(e)
        return cls(e)

    @locked
    def add(self, **kw):
        kw.setdefault("submitted_at", self.clock())
        kw.setdefault("status", "queued")
        with Session(self.engine, expire_on_commit=False) as s:
            j = Job(**kw)
            s.add(j)
            s.commit()
            return row(j)

    @locked
    def update(self, job_id, **kw):
        with Session(self.engine, expire_on_commit=False) as s:
            j = s.get(Job, job_id)
            if j is None:
                return None
            for k, v in kw.items():
                setattr(j, k, v)
            s.commit()
            return row(j)

    @locked
    def claim(self, job_id, **kw):
        """Start a queued job: the update happens only while it is still queued (a lease withdrawn meanwhile stays withdrawn). Returns the row, or None."""
        with Session(self.engine, expire_on_commit=False) as s:
            n = s.execute(sa.update(Job).where(Job.id == job_id, Job.status == "queued").values(**kw)).rowcount
            s.commit()
            return row(s.get(Job, job_id)) if n else None

    @locked
    def get(self, job_id, result=True):
        with Session(self.engine) as s:
            j = s.get(Job, job_id)
            return row(j, result) if j else None

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
        """Drop results older than `keep_s` (embeddings and transcripts are the apps' to keep); the rows stay for the history."""
        with Session(self.engine) as s:
            n = s.execute(
                sa.update(Job)
                .where(Job.finished_at < self.clock() - keep_s, Job.result.is_not(None))
                .values(result=None)
            ).rowcount
            s.commit()
            return n


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
    """An in-memory SQLite store with the tables created from the models (tests, and `serve --memory` for a demo without a database)."""
    from sqlalchemy.pool import StaticPool

    e = sa.create_engine("sqlite://", connect_args=dict(check_same_thread=False), poolclass=StaticPool)
    Base.metadata.create_all(e)
    return Store(e, clock)
