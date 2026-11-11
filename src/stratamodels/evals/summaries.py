"""Each experiment's analysis worked out ahead of the page, kept in the database (eval_summaries).

The analysis bootstraps every metric (seconds to a minute for a large dataset), so the page must not wait for it. It is worked out in the
background when something it reads changes (a run finishes, a judgment comes in, the spec is edited) and at the service's start, and kept with
the signature of what it read. The page gets the kept one: when it is out of date it gets it anyway, marked stale, while a fresh one is
worked out. Only an experiment never analysed, or one whose analysis took under QUICK_S, is worked out while the request waits."""

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import func, select

from stratamodels.db.models import EvalJudgment, EvalResult, EvalSummary
from stratamodels.evals import core as C


def _code_hash():
    """The analysis code itself: a deploy that changes how a metric is worked out makes every kept summary out of date."""
    h = hashlib.sha256()
    here = os.path.dirname(__file__)
    for f in ("core.py", "metrics.py", "stats.py", "summaries.py"):
        with open(os.path.join(here, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


CODE = _code_hash()
QUICK_S = 1.5  # an analysis that took less is redone while the page waits rather than shown out of date


def finite(x):
    """NaN and infinities as null: JSON (and Postgres) has no NaN, and a metric with no observations (every job failed) has no value."""
    if isinstance(x, float):
        return x if x == x and x not in (float("inf"), float("-inf")) else None
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    return x


def signature(spec, engine):
    """What an analysis of `spec` reads, as a hash: the spec, the dataset's version, the outputs (count and newest id) and judgments, the code.
    Cheap: a few counts, no outputs read. Readings a [score] pass writes into outputs in place do not change it: the run that writes them
    refreshes its experiment (Summaries.refresh), and POST .../refresh is there for anything written from outside the service."""
    with C.session(engine) as s:
        parts = [
            C._canon(spec.to_json()),
            s.scalar(select(func.count()).select_from(EvalResult).where(EvalResult.dataset == spec.dataset)),
            s.scalar(select(func.max(EvalResult.id)).where(EvalResult.dataset == spec.dataset)),
            s.scalar(select(func.count()).select_from(EvalJudgment).where(EvalJudgment.experiment == spec.name)),
            s.scalar(select(func.max(EvalJudgment.id)).where(EvalJudgment.experiment == spec.name)),
            C.load_dataset(spec.dataset, engine=engine).version,
            CODE,
        ]
    return hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()


class Summaries:
    """The kept analyses: get() for the page, refresh() when something changed, warm() at start. One background worker works them out, one
    experiment at a time, so a burst of judgments makes one or two analyses, not one each."""

    def __init__(self, engine, analyse=None, log=print):
        self.engine = engine  # a function -> the engine (the scheduler's store may be swapped in tests)
        self.analyse = analyse  # None: evals.core.analyse, looked up when used
        self.log = log
        self.memo = {}  # name -> (signature, summary): saves reading the row back on every request
        self.locks = {}
        self.guard = threading.Lock()
        self.pending = {}  # name -> force, of refreshes queued and not started
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval-summaries")

    def _lock(self, name):
        with self.guard:
            return self.locks.setdefault(name, threading.Lock())

    def _stored(self, name):
        with C.session(self.engine()) as s:
            r = s.get(EvalSummary, name)
            return None if r is None else (r.signature, r.result, r.computed_at, r.seconds)

    def compute(self, spec, sig=None, force=False):
        """Work out spec's analysis now and keep it (unless the kept one already matches and not `force`). Returns the summary."""
        with self._lock(spec.name):
            sig = sig or signature(spec, self.engine())
            if not force:
                hit = self.memo.get(spec.name)
                if hit and hit[0] == sig:
                    return hit[1]
                row = self._stored(spec.name)
                if row and row[0] == sig:
                    self.memo[spec.name] = (sig, row[1])
                    return row[1]
            t = time.time()
            res = finite((self.analyse or C.analyse)(spec, self.engine()))
            took = time.time() - t
            res["summary"] = dict(computed_at=time.time(), seconds=round(took, 3), stale=False)
            with C.session(self.engine()) as s:
                r = s.get(EvalSummary, spec.name)
                if r is None:
                    s.add(
                        EvalSummary(
                            experiment=spec.name, signature=sig, result=res, computed_at=time.time(), seconds=took
                        )
                    )
                else:
                    r.signature, r.result, r.computed_at, r.seconds = sig, res, time.time(), took
                s.commit()
            self.memo[spec.name] = (sig, res)
            if took > 2:
                self.log(f"eval summary {spec.name}: {took:.1f} s")
            return res

    def get(self, spec):
        """The page's analysis: the kept one when it is current; the kept one marked stale (a fresh one on its way) when it is not; worked out
        while waiting only when there is none."""
        sig = signature(spec, self.engine())
        hit = self.memo.get(spec.name)
        if hit and hit[0] == sig:
            return hit[1]
        row = self._stored(spec.name)
        if row and row[0] == sig:
            self.memo[spec.name] = (sig, row[1])
            return row[1]
        old = row[1] if row else (hit[1] if hit else None)
        took = row[3] if row else ((old or {}).get("summary") or {}).get("seconds")
        if old is None or (took is not None and took <= QUICK_S):
            return self.compute(spec, sig)  # none yet, or a quick one: no reason to show an old one
        self.refresh(spec.name)
        return {**old, "summary": {**(old.get("summary") or {}), "stale": True}}

    def refresh(self, name, force=False):
        """Work out name's analysis in the background (once, however often asked before it starts; forced if any of those asks was)."""
        with self.guard:
            if name in self.pending:
                self.pending[name] = (
                    self.pending[name] or force
                )  # a forced ask joining a queued ordinary one keeps its force
                return
            self.pending[name] = force
        self.pool.submit(self._run, name)

    def refresh_dataset(self, dataset, force=False):
        """Every experiment on `dataset`: outputs are shared by key across experiments, and a [score] pass rewrites them in place."""
        for name, s in C.list_specs(self.engine()).items():
            if not isinstance(s, C.EvalError) and s.dataset == dataset:
                self.refresh(name, force)

    def _run(self, name):
        with self.guard:
            force = self.pending.pop(name, False)
        try:
            self.compute(C.load_spec(name, self.engine()), force=force)
        except (
            Exception
        ) as e:  # an experiment that cannot be analysed (bad spec, missing dataset) answers its error on request
            self.log(f"eval summary {name}: {type(e).__name__}: {e}")

    def warm(self):
        """Bring every experiment's kept analysis up to date, in the background (at the service's start)."""
        for name, s in C.list_specs(self.engine()).items():
            if not isinstance(s, C.EvalError):
                self.refresh(name)
