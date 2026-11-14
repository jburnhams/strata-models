"""A trial of the `font-match` eval before the service has the role: each candidate's stages (a vision model's shortlist) run as jobs through
the running service's queue (the vision-language role is already there), and the fit runs here, in this process, on this checkout's
runtimes/fontfit.py. Scores are the eval's own metrics; the summary is per candidate, real and synthetic items apart. Answers are cached in
<data>/evals/font-match-trial/ so a second run only does what is new. Once the role is deployed, `./strata-models eval run font-match` is the
real run (queue, cache, statistics, judges, write-up); this is for trying the method on a branch.

    .venv/bin/python scripts/evals/font_match_trial.py                       # every candidate of evals/specs/font-match.toml
    .venv/bin/python scripts/evals/font_match_trial.py naive qwen-fit --jobs 3
"""

import argparse
import json
import os
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "client"))
from stratamodels import settings
from stratamodels.evals import core as C
from stratamodels.evals import metrics as M
from stratamodels.runtimes import fontfit as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def stage_answer(stage, item, cache, sm):
    params = C.stage_params(stage, dict(item["input"]))
    key = C.stage_key(stage, params, item)
    path = os.path.join(cache, f"stage-{key}.json")
    if os.path.isfile(path):
        with open(path, encoding="utf8") as f:
            return json.load(f)
    job = sm.submit(stage["role"], params, "evals", "low", stage["model"], stage["options"] or None)["id"]
    r = sm.wait_job(job)
    done = r["status"] == "done"
    out = dict(
        output=r.get("result") if done else None,
        error=None if done else (r.get("error") or r["status"]),
        seconds=r.get("run_s"),
    )
    with open(path, "w", encoding="utf8") as f:
        json.dump(out, f)
    return out


def fit(args):
    _name, params, item, out = args
    if os.path.isfile(out):
        with open(out, encoding="utf8") as f:
            return json.load(f)
    h = dict(key="fontfit", models_dir=settings.models_dir(), top=6, get=F._get)
    try:
        r, err = F.run(h, dict(params, **item["input"], out=out[:-5] + ".png")), None
    except Exception as e:  # a failed job is kept as a failure, as the eval does
        r, err = None, f"{type(e).__name__}: {e}"
    d = dict(output=r, error=err)
    with open(out, "w", encoding="utf8") as f:
        json.dump(d, f)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", nargs="*")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    with open(os.path.join(ROOT, "evals", "specs", "font-match.toml"), "rb") as f:
        spec = C.parse_spec(tomllib.load(f), "font-match.toml")
    ds = C.load_dataset(spec.dataset)
    cache = os.path.join(settings.data_dir(), "evals", "font-match-trial")
    os.makedirs(cache, exist_ok=True)
    cands = [c for c in spec.candidates if not a.candidates or c.name in a.candidates]
    sm = None
    todo, extra = [], {}
    for c in cands:
        items = C.sample(c, ds.items)
        for it in items:
            params = dict(c.params)
            stage_s, stage_err = 0.0, None
            for st in c.stages:
                if sm is None:
                    import strata_models_client as sm

                    if not sm.available():
                        sys.exit(f"strata-models is not running at {sm.url()}: the stages go through its queue")
                ans = stage_answer(st, it, cache, sm)
                stage_s += ans.get("seconds") or 0.0
                if ans["error"]:
                    stage_err = f"stage {st['role']}: {ans['error']}"
                    break
                try:
                    params[st["into"]] = C.stage_text(ans["output"])
                except C.EvalError as e:
                    stage_err = str(e)
                    break
            out = os.path.join(cache, f"{c.name}--{it['id']}.json")
            extra[(c.name, it["id"])] = (stage_s, stage_err)
            if stage_err is None:
                todo.append((c.name, params, it, out))
    with ProcessPoolExecutor(a.jobs) as ex:
        results = dict(zip([(t[0], t[2]["id"]) for t in todo], ex.map(fit, todo)))
    names = [m for m in spec.metrics if m in M.METRICS and m not in ("latency_s", "cost_usd")]
    by_id = {it["id"]: it for it in ds.items}
    rows = []
    for c in cands:
        for grp in ("real", "synthetic"):
            vals = {n: [] for n in names}
            secs = []
            for (cn, iid), (stage_s, err) in extra.items():
                it = by_id[iid]
                if cn != c.name or (it["tags"].get("synthetic") == "yes") != (grp == "synthetic"):
                    continue
                d = results.get((cn, iid)) or dict(output=None, error=err)
                for n in names:
                    v = M.METRICS[n].score(it, d["output"], d["error"])
                    if v is not None:
                        vals[n].append(float(v))
                if d["output"]:
                    secs.append(stage_s + d["output"]["seconds"])
            row = dict(
                candidate=c.name, items=grp, n=len(secs), seconds=round(sum(secs) / len(secs), 1) if secs else None
            )
            row.update({n: round(sum(v) / len(v), 3) if v else None for n, v in vals.items()})
            rows.append(row)
            print(json.dumps(row))
    with open(os.path.join(cache, "summary.json"), "w", encoding="utf8") as f:
        json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
