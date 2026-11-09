"""The JSON-lines loop every venv-worker runs (see stratamodels/runtimes/worker.py), so a worker script only says how to load and run its
model. Standalone: imports nothing from stratamodels, so it works in any venv.

    from protocol import ParamError, serve
    serve(load, run)          # load(model: dict, models_dir) -> info; run(params) -> result; raise ParamError for a bad request
"""

import json
import sys
import traceback


class ParamError(ValueError):
    """The request was wrong (a missing file, an unknown option): the job fails, the model stays loaded."""


def serve(load, run, stdin=None, stdout=None):
    out = stdout or sys.stdout
    sys.stdout = sys.stderr  # libraries that print must not corrupt the answers
    for line in stdin or sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        op = msg.get("op")
        if op == "quit":
            break
        try:
            res = load(msg["model"], msg["models_dir"]) if op == "load" else run(msg.get("params") or {})
            ans = dict(ok=True, result=res)
        except ParamError as e:
            ans = dict(ok=False, params=True, error=str(e))
        except Exception as e:
            traceback.print_exc()
            ans = dict(ok=False, error=f"{type(e).__name__}: {e}")
        out.write(json.dumps(ans) + "\n")
        out.flush()
