"""The Evals part of the API (mounted by server/app.py):

GET  /api/evals                              experiments (spec, how many outputs are in) and datasets
PUT  /api/evals/{name}/spec                  {spec as JSON, the TOML's fields} add or change an experiment in the database (its file no longer overwrites it)
GET  /api/evals/{name}                       the analysis: every metric with its interval, comparisons with the baseline, the decision; kept in
                                             the database and worked out in the background (summary: computed_at, seconds, stale)
POST /api/evals/{name}/run                   run what is missing, in the background, through the queue (batch); progress in GET .../status
POST /api/evals/{name}/refresh                work the kept analysis out again (outputs changed from outside the service)
GET  /api/evals/{name}/status
GET  /api/evals/{name}/report                the standard write-up, as Markdown
GET  /api/evals/{name}/pair                  a blind pair to judge: one item, two candidates' outputs as X and Y (a token says which is which);
                                             end results only: a pipeline's intermediate steps are left out
GET  /api/evals/{name}/outputs[?item_id=]    every candidate's full output per item, a pipeline's every step (what it was sent, what it answered)
POST /api/evals/{name}/judgments             {token, winner: x | y | tie, rater?, note?}
GET  /api/evals/{name}/media/{item_id}       the item's input file (audio, picture), for the judging page; only files a dataset names"""

import base64
import hashlib
import hmac
import json
import mimetypes
import os
import random
import secrets
import threading
import time

from fastapi import HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel
from sqlalchemy import select

from stratamodels.db.models import EvalJudgment
from stratamodels.evals import core as C
from stratamodels.evals import metrics as M
from stratamodels.evals import report as R
from stratamodels.evals.summaries import Summaries, finite

SECRET = secrets.token_bytes(
    16
)  # signs judging tokens for this process's life: a token says which output was X without the page knowing


class JudgmentIn(BaseModel):
    token: str
    winner: str
    rater: str = "jonathan"
    note: str | None = None


def _sign(d):
    raw = base64.urlsafe_b64encode(json.dumps(d).encode()).decode()
    return raw + "." + hmac.new(SECRET, raw.encode(), hashlib.sha256).hexdigest()[:24]


def _unsign(tok):
    raw, _, sig = tok.rpartition(".")
    if not raw or not hmac.compare_digest(sig, hmac.new(SECRET, raw.encode(), hashlib.sha256).hexdigest()[:24]):
        raise HTTPException(400, "this pair has expired: load a new one")
    return json.loads(base64.urlsafe_b64decode(raw))


def same(x, y, kind=None):
    """Whether two outputs say the same thing (similarity 1)."""
    from stratamodels.evals import similarity as SIM

    return SIM.similarity(x, y, kind) >= 0.9999


MEDIA_EXT = (".flac", ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".mp4", ".mov", ".mkv", ".osv", ".insv")
MEDIA_TYPES = {
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
}
CONTEXT_S = 2.0  # seconds either side of a phrase in a ?cut=1 answer


def mount(app, scheduler):
    runs = {}
    engine = lambda: scheduler.store.engine
    sims = {}  # sims: (result key, result key) -> similarity, outputs never change under a key
    summaries = app.state.eval_summaries = Summaries(engine)

    def analysis(s):
        """The analysis as kept in the database, worked out in the background when what it reads changes (evals.summaries)."""
        return summaries.get(s)

    def spec_or_404(name):
        try:
            return C.load_spec(name, engine())
        except C.EvalError as e:
            raise HTTPException(404 if "no experiment" in str(e) else 400, str(e))

    @app.get("/api/evals")
    def get_evals():
        out = []
        for name, s in C.list_specs(engine()).items():
            if isinstance(s, C.EvalError):
                out.append(dict(name=name, error=str(s)))
                continue
            try:
                ds = C.load_dataset(s.dataset, engine=engine())
                keys = [C.result_key(s, c, it) for c in s.candidates for it in ds.items]
                have = len(C.cached(engine(), keys))
                total = len(keys)
            except C.EvalError as e:
                out.append(dict(s.to_json(), error=str(e)))
                continue
            out.append(
                dict(s.to_json(), outputs=have, total=total, running=runs.get(name, {}).get("state") == "running")
            )
        return dict(experiments=out, datasets=C.list_datasets(engine=engine()))

    @app.put("/api/evals/{name}/spec")
    def put_spec(name: str, d: dict):
        d = {**d, "name": name}
        try:
            out = C.save_spec(engine(), d).to_json()
        except C.EvalError as e:
            raise HTTPException(400, str(e))
        summaries.refresh(name)
        return out

    @app.get("/api/evals/{name}")
    def get_eval(name: str):
        s = spec_or_404(name)
        try:
            return finite(analysis(s))
        except C.EvalError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/evals/{name}/run", status_code=202)
    def post_run(name: str, candidate: str | None = None, limit: int | None = None):
        s = spec_or_404(name)
        if runs.get(name, {}).get("state") == "running":
            return runs[name]
        st = runs[name] = dict(state="running", started_at=time.time(), log=[])

        def go():
            try:
                res, state = (
                    C.run(
                        s,
                        engine(),
                        C.SchedulerExecutor(scheduler),
                        candidates=[candidate] if candidate else None,
                        limit=limit,
                        log=lambda m: st["log"].append(m),
                    ),
                    "done",
                )
            except Exception as e:
                res, state = None, "failed"
                st["error"] = f"{type(e).__name__}: {e}"
            # queued before the state says finished, so whoever waits on the run finds the refresh queued; forced: a [score] pass changes
            # outputs in place, which the signature does not see
            summaries.refresh(name, force=True)
            if res is not None:
                st["result"] = res
            st["finished_at"] = time.time()
            st["state"] = state

        threading.Thread(target=go, name=f"eval-{name}", daemon=True).start()
        return st

    @app.post("/api/evals/{name}/refresh", status_code=202)
    def post_refresh(name: str):
        spec_or_404(name)
        summaries.refresh(name, force=True)
        return dict(ok=True)

    @app.get("/api/evals/{name}/status")
    def get_run_status(name: str):
        return runs.get(name, dict(state="idle", log=[]))

    @app.get("/api/evals/{name}/report", response_class=PlainTextResponse)
    def get_report(name: str):
        s = spec_or_404(name)
        res = analysis(s)
        ds, outs, _ = C.outputs_for(s, engine())
        return R.render(
            res,
            R.worst(
                s,
                outs,
                ds.items,
                s.primary if s.primary != "judged" else next((m for m in s.metrics if M.METRICS[m].score), s.primary),
            ),
            pipelines=R.pipelines(s, outs, ds.items),
        )

    @app.get("/api/evals/{name}/pair")
    def get_pair(name: str):
        s = spec_or_404(name)
        if len(s.candidates) < 2:
            raise HTTPException(400, "an experiment needs two candidates to judge")
        ds, outs, _ = C.outputs_for(s, engine())
        pairs = []
        from stratamodels.evals import similarity as SIM

        for i in range(len(ds.items)):
            ok = [c.name for c in s.candidates if outs[c.name][i] and not outs[c.name][i][1]]
            for n, a in enumerate(ok):
                for b in ok[n + 1 :]:
                    oa, ob = C.without_pipeline(outs[a][i][0]), C.without_pipeline(outs[b][i][0])
                    key = (
                        s.similarity,
                        hashlib.sha256(json.dumps([oa, ob], sort_keys=True, default=str).encode()).hexdigest(),
                    )  # the outputs themselves: a rerun after forget, or another kind, is scored again
                    if key not in sims:
                        sims[key] = SIM.similarity(oa, ob, s.similarity or None)
                    if sims[key] < 0.9999:
                        pairs.append((sims[key], i, a, b))  # identical outputs teach nothing: not shown
        if s.primary == "judged":
            pairs = [
                p for p in pairs if s.baseline in p[2:]
            ]  # decided head to head with the baseline: only those pairs
        if not pairs:
            return dict(pair=None)
        current = {(c.name, it["id"]): C.result_key(s, c, it) for c in s.candidates for it in ds.items}
        with C.session(engine()) as ses:  # a judgment of outputs that are no longer the candidates' does not count
            judged = {
                (j.item_id, frozenset((j.a, j.b)))
                for j in ses.scalars(select(EvalJudgment).where(EvalJudgment.experiment == name))
                if j.a_key and current.get((j.a, j.item_id)) == j.a_key and current.get((j.b, j.item_id)) == j.b_key
            }
        fresh = [
            p for p in pairs if (ds.items[p[1]]["id"], frozenset(p[2:])) not in judged
        ] or pairs  # pairs not judged yet first,
        least = min(p[0] for p in fresh)
        sim, i, a, b = random.choice([p for p in fresh if p[0] <= least + 0.02])  # and of those the most different
        if random.random() < 0.5:
            a, b = b, a  # which side is X is random
        it = ds.items[i]
        left = sum(1 for p in pairs if (ds.items[p[1]]["id"], frozenset(p[2:])) not in judged)
        return dict(
            remaining=left,
            pair=dict(
                similarity=round(sim, 3),
                item_id=it["id"],
                input={k: v for k, v in it["input"].items() if k != "path"},
                has_media=isinstance(it["input"].get("path"), str),
                reference=it.get("reference"),
                x=C.without_pipeline(outs[a][i][0]),
                y=C.without_pipeline(outs[b][i][0]),
                token=_sign(
                    dict(
                        e=name,
                        i=it["id"],
                        x=a,
                        y=b,
                        xk=C.result_key(s, s.candidate(a), it),
                        yk=C.result_key(s, s.candidate(b), it),
                    )
                ),
            ),
        )

    @app.post("/api/evals/{name}/judgments", status_code=201)
    def post_judgment(name: str, b: JudgmentIn):
        d = _unsign(b.token)
        if d["e"] != name:
            raise HTTPException(400, "this pair belongs to another experiment")
        if b.winner not in ("x", "y", "tie"):
            raise HTTPException(400, "winner is x, y or tie")
        win = d["x"] if b.winner == "x" else d["y"] if b.winner == "y" else "tie"
        with C.session(engine()) as ses:
            ses.add(
                EvalJudgment(
                    experiment=name,
                    item_id=d["i"],
                    a=d["x"],
                    b=d["y"],
                    winner=win,
                    rater=b.rater[:64],
                    at=time.time(),
                    note=(b.note or None) and b.note[:500],
                    a_key=d.get("xk"),
                    b_key=d.get("yk"),
                )
            )
            ses.commit()
        summaries.refresh(name)
        return dict(ok=True, item_id=d["i"], x=d["x"], y=d["y"], winner=win)

    @app.get("/api/evals/{name}/outputs")
    def get_outputs(name: str, item_id: str | None = None):
        s = spec_or_404(name)
        ds, outs, runs = C.outputs_for(s, engine())
        items = []
        for i, it in enumerate(ds.items):
            if item_id is not None and it["id"] != item_id:
                continue
            items.append(
                dict(
                    id=it["id"],
                    input={k: v for k, v in it["input"].items() if k != "path"},
                    has_media=isinstance(it["input"].get("path"), str),
                    reference=it.get("reference"),
                    outputs={
                        c.name: (
                            None
                            if outs[c.name][i] is None
                            else dict(output=outs[c.name][i][0], error=outs[c.name][i][1], **(runs[c.name][i] or {}))
                        )
                        for c in s.candidates
                    },
                )
            )
        if item_id is not None and not items:
            raise HTTPException(404, f"no item {item_id!r}")
        return finite(dict(candidates=[c.to_json() for c in s.candidates], items=items))

    @app.get("/api/evals/{name}/media/{item_id}")
    def get_media(name: str, item_id: str, cut: bool = False):
        s = spec_or_404(name)
        it = next((x for x in C.load_dataset(s.dataset, engine=engine()).items if x["id"] == item_id), None)
        p = it and it["input"].get("path")
        if not isinstance(p, str) or not os.path.isfile(p):
            raise HTTPException(404, "no file for this item")
        clip = it["input"].get("clip")
        ext = os.path.splitext(p)[1].lower()
        if cut and clip and ext in MEDIA_EXT:  # ?cut=1: just the phrase with a little either side
            import subprocess

            a, b = max(float(clip[0]) - CONTEXT_S, 0.0), float(clip[1]) + CONTEXT_S
            r = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-ss",
                    f"{a:.2f}",
                    "-to",
                    f"{b:.2f}",
                    "-i",
                    p,
                    "-vn",
                    "-ac",
                    "1",
                    "-f",
                    "wav",
                    "-",
                ],
                capture_output=True,
            )
            if r.returncode == 0 and r.stdout:
                return Response(content=r.stdout, media_type="audio/wav")
        # the whole file, with ranges and its real type (Python's mimetypes does not know .flac here): the page cues it to the phrase
        return FileResponse(
            p, media_type=MEDIA_TYPES.get(ext) or mimetypes.guess_type(p)[0] or "application/octet-stream"
        )
