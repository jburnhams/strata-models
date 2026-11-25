"""An experiment's analysis in a process of its own: `run(spec, engine, progress)` is evals.core.analyse, worked out by a low-priority child.

The large ones take 10 to 20 minutes of one core (page-layout, page-rebuild). In the service's process that time was the GIL's: every endpoint
waited, and what they allocated stayed with the service. A child costs about a second to start, gives all its memory back when it ends, and
yields the CPU to anything else (nice 15, one numerical thread, no GPU).

The child reads the database as it stands (never migrates it) and writes lines of JSON: {"progress": [step, done, total]} and, last,
{"result": ...} or {"error": ..., "type": ...}. Anything a library prints goes to stderr, so it cannot break the lines."""

import json
import os
import subprocess
import sys
import threading
from typing import IO, cast

from stratamodels.evals import core as C

NICE = 15
ONE_THREAD = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


class AnalysisFailed(RuntimeError):
    """The child died or answered nothing."""


class Runner:
    """Runs analyses in children, and can stop them all (the service is closing: a restart must not wait twenty minutes)."""

    def __init__(self, popen=subprocess.Popen, argv=None):
        self.popen = popen
        self.argv = argv or [sys.executable, "-m", "stratamodels.evals.process"]
        self.live = set()
        self.stopped = False
        self.guard = threading.Lock()

    def __call__(self, spec, engine=None, progress=None):
        env = {**os.environ, **dict.fromkeys(ONE_THREAD, "1"), "CUDA_VISIBLE_DEVICES": ""}
        with self.guard:
            if self.stopped:
                raise AnalysisFailed("the service is stopping")
            p = self.popen(
                [*self.argv, spec.name], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, env=env, text=True
            )
            self.live.add(p)
        last = {}
        lines = cast("IO[str]", p.stdout)  # it is a pipe
        try:
            for line in lines:
                try:
                    last = json.loads(line)
                except ValueError:
                    continue  # not one of ours
                if "progress" in last and progress is not None:
                    progress(*last["progress"])
            code = p.wait()
        finally:
            if p.poll() is None:
                p.kill()
            lines.close()
            with self.guard:
                self.live.discard(p)
        if "error" in last:  # an experiment that cannot be analysed says why, as it did in this process
            raise (C.EvalError if last.get("type") == "EvalError" else AnalysisFailed)(last["error"])
        if "result" not in last:
            raise AnalysisFailed(f"the analysis of {spec.name} ended without an answer (exit {code})")
        return last["result"]

    def stop(self):
        with self.guard:
            self.stopped = True
            for p in self.live:
                p.kill()


def main(argv=None):
    from strata_python.oslib import lower_priority

    from stratamodels import settings
    from stratamodels.db.models import EvalExperiment
    from stratamodels.evals.summaries import finite
    from stratamodels.store import Store

    (name,) = argv if argv is not None else sys.argv[1:]
    lower_priority(NICE)
    out, sys.stdout = sys.stdout, sys.stderr

    def say(**kw):
        out.write(json.dumps(kw, default=str) + "\n")
        out.flush()

    try:
        store = Store.connect(migrate=False)
        settings.bind(store.engine)
        with C.session(
            store.engine
        ) as s:  # its row, not load_spec: importing changed spec files is the service's to do
            row = s.get(EvalExperiment, name)
            if row is None:
                raise C.EvalError(f"no experiment {name!r}")
            spec = C.parse_spec(row.spec, row.source or "")
        res = C.analyse(spec, store.engine, progress=lambda step, done, total: say(progress=[step, done, total]))
        say(result=finite(res))
    except Exception as e:
        say(error=str(e) if isinstance(e, C.EvalError) else f"{type(e).__name__}: {e}", type=type(e).__name__)
        return 1
    finally:
        sys.stdout = out
    return 0


if __name__ == "__main__":
    sys.exit(main())
