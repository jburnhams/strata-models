"""The Evals part of the API (mounted by server/app.py):

  GET  /api/evals                              experiments (spec, how many outputs are in) and datasets
  PUT  /api/evals/{name}/spec                  {spec as JSON, the TOML's fields} add or change an experiment in the database (its file no longer overwrites it)
  GET  /api/evals/{name}                       the analysis: every metric with its interval, comparisons with the baseline, the decision
  POST /api/evals/{name}/run                   run what is missing, in the background, through the queue (batch); progress in GET .../status
  GET  /api/evals/{name}/status
  GET  /api/evals/{name}/report                the standard write-up, as Markdown
  GET  /api/evals/{name}/pair                  a blind pair to judge: one item, two candidates' outputs as X and Y (a token says which is which)
  POST /api/evals/{name}/judgments             {token, winner: x | y | tie, rater?, note?}
  GET  /api/evals/{name}/media/{item_id}       the item's input file (audio, picture), for the judging page; only files a dataset names"""
import base64, hashlib, hmac, json, mimetypes, os, random, secrets, threading, time

from fastapi import HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel
from sqlalchemy import select

from stratamodels.db.models import EvalJudgment
from stratamodels.evals import core as C, metrics as M, report as R

SECRET = secrets.token_bytes(16)          # signs judging tokens for this process's life: a token says which output was X without the page knowing


class JudgmentIn(BaseModel):
    token: str
    winner: str
    rater: str = 'jonathan'
    note: str | None = None


def _sign(d):
    raw = base64.urlsafe_b64encode(json.dumps(d).encode()).decode(); return raw + '.' + hmac.new(SECRET, raw.encode(), hashlib.sha256).hexdigest()[:24]


def _unsign(tok):
    raw, _, sig = tok.rpartition('.')
    if not raw or not hmac.compare_digest(sig, hmac.new(SECRET, raw.encode(), hashlib.sha256).hexdigest()[:24]): raise HTTPException(400, 'this pair has expired: load a new one')
    return json.loads(base64.urlsafe_b64decode(raw))


def finite(x):
    """The analysis with NaN and infinities as null: JSON has no NaN, and a metric with no observations (every job failed) has no value."""
    if isinstance(x, float): return x if x == x and x not in (float('inf'), float('-inf')) else None
    if isinstance(x, dict): return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [finite(v) for v in x]
    return x


def same(x, y, kind=None):
    """Whether two outputs say the same thing (similarity 1)."""
    from stratamodels.evals import similarity as SIM
    return SIM.similarity(x, y, kind) >= 0.9999


def mount(app, scheduler):
    runs = {}; engine = lambda: scheduler.store.engine; memo = {}; sims = {}       # sims: (result key, result key) -> similarity, outputs never change under a key

    def analysis(s):
        """The analysis, recomputed only when the spec, the outputs or the judgments changed: the bootstraps take seconds to a minute."""
        from sqlalchemy import func, select
        from stratamodels.db.models import EvalJudgment as J, EvalResult as Rr
        with C.session(engine()) as ses:
            sig = (C._canon(s.to_json()), ses.scalar(select(func.count()).select_from(Rr).where(Rr.dataset == s.dataset)), ses.scalar(select(func.max(Rr.id)).where(Rr.dataset == s.dataset)),
                   ses.scalar(select(func.count()).select_from(J).where(J.experiment == s.name)), C.load_dataset(s.dataset, engine=engine()).version, C.SCORED[0])
        hit = memo.get(s.name)
        if hit and hit[0] == sig: return hit[1]
        res = C.analyse(s, engine()); memo[s.name] = (sig, res); return res

    def spec_or_404(name):
        try: return C.load_spec(name, engine())
        except C.EvalError as e: raise HTTPException(404 if 'no experiment' in str(e) else 400, str(e))

    @app.get('/api/evals')
    def get_evals():
        out = []
        for name, s in C.list_specs(engine()).items():
            if isinstance(s, C.EvalError): out.append(dict(name=name, error=str(s))); continue
            try: ds = C.load_dataset(s.dataset, engine=engine()); keys = [C.result_key(s, c, it) for c in s.candidates for it in ds.items]; have = len(C.cached(engine(), keys)); total = len(keys)
            except C.EvalError as e: out.append(dict(s.to_json(), error=str(e))); continue
            out.append(dict(s.to_json(), outputs=have, total=total, running=runs.get(name, {}).get('state') == 'running'))
        return dict(experiments=out, datasets=C.list_datasets(engine=engine()))

    @app.put('/api/evals/{name}/spec')
    def put_spec(name: str, d: dict):
        d = {**d, 'name': name}
        try: return C.save_spec(engine(), d).to_json()
        except C.EvalError as e: raise HTTPException(400, str(e))

    @app.get('/api/evals/{name}')
    def get_eval(name: str):
        s = spec_or_404(name)
        try: return finite(analysis(s))
        except C.EvalError as e: raise HTTPException(400, str(e))

    @app.post('/api/evals/{name}/run', status_code=202)
    def post_run(name: str, candidate: str | None = None, limit: int | None = None):
        s = spec_or_404(name)
        if runs.get(name, {}).get('state') == 'running': return runs[name]
        st = runs[name] = dict(state='running', started_at=time.time(), log=[])
        def go():
            try: st['result'] = C.run(s, engine(), C.SchedulerExecutor(scheduler), candidates=[candidate] if candidate else None, limit=limit, log=lambda m: st['log'].append(m)); st['state'] = 'done'
            except Exception as e: st['state'] = 'failed'; st['error'] = f'{type(e).__name__}: {e}'
            st['finished_at'] = time.time()
        threading.Thread(target=go, name=f'eval-{name}', daemon=True).start(); return st

    @app.get('/api/evals/{name}/status')
    def get_run_status(name: str): return runs.get(name, dict(state='idle', log=[]))

    @app.get('/api/evals/{name}/report', response_class=PlainTextResponse)
    def get_report(name: str):
        s = spec_or_404(name); res = analysis(s); ds, outs, _ = C.outputs_for(s, engine())
        return R.render(res, R.worst(s, outs, ds.items, s.primary if s.primary != 'judged' else next((m for m in s.metrics if M.METRICS[m].score), s.primary)))

    @app.get('/api/evals/{name}/pair')
    def get_pair(name: str):
        s = spec_or_404(name)
        if len(s.candidates) < 2: raise HTTPException(400, 'an experiment needs two candidates to judge')
        ds, outs, _ = C.outputs_for(s, engine()); pairs = []
        from stratamodels.evals import similarity as SIM
        for i, it in enumerate(ds.items):
            ok = [c.name for c in s.candidates if outs[c.name][i] and not outs[c.name][i][1]]
            for n, a in enumerate(ok):
                for b in ok[n + 1:]:
                    oa, ob = outs[a][i][0], outs[b][i][0]
                    key = (s.similarity, hashlib.sha256(json.dumps([oa, ob], sort_keys=True, default=str).encode()).hexdigest())    # the outputs themselves: a rerun after forget, or another kind, is scored again
                    if key not in sims: sims[key] = SIM.similarity(oa, ob, s.similarity or None)
                    if sims[key] < 0.9999: pairs.append((sims[key], i, a, b))                                    # identical outputs teach nothing: not shown
        if s.primary == 'judged': pairs = [p for p in pairs if s.baseline in p[2:]]                                          # decided head to head with the baseline: only those pairs
        if not pairs: return dict(pair=None)
        current = {(c.name, it['id']): C.result_key(s, c, it) for c in s.candidates for it in ds.items}
        with C.session(engine()) as ses:                                                                     # a judgment of outputs that are no longer the candidates' does not count
            judged = {(j.item_id, frozenset((j.a, j.b))) for j in ses.scalars(select(EvalJudgment).where(EvalJudgment.experiment == name)) if j.a_key and current.get((j.a, j.item_id)) == j.a_key and current.get((j.b, j.item_id)) == j.b_key}
        fresh = [p for p in pairs if (ds.items[p[1]]['id'], frozenset(p[2:])) not in judged] or pairs                  # pairs not judged yet first,
        least = min(p[0] for p in fresh); sim, i, a, b = random.choice([p for p in fresh if p[0] <= least + 0.02])          # and of those the most different
        if random.random() < 0.5: a, b = b, a                                                                               # which side is X is random
        it = ds.items[i]
        left = sum(1 for p in pairs if (ds.items[p[1]]['id'], frozenset(p[2:])) not in judged)
        return dict(remaining=left, pair=dict(similarity=round(sim, 3), item_id=it['id'], input={k: v for k, v in it['input'].items() if k != 'path'}, has_media=isinstance(it['input'].get('path'), str), reference=it.get('reference'),
                              x=outs[a][i][0], y=outs[b][i][0], token=_sign(dict(e=name, i=it['id'], x=a, y=b, xk=C.result_key(s, s.candidate(a), it), yk=C.result_key(s, s.candidate(b), it)))))

    @app.post('/api/evals/{name}/judgments', status_code=201)
    def post_judgment(name: str, b: JudgmentIn):
        d = _unsign(b.token)
        if d['e'] != name: raise HTTPException(400, 'this pair belongs to another experiment')
        if b.winner not in ('x', 'y', 'tie'): raise HTTPException(400, 'winner is x, y or tie')
        win = d['x'] if b.winner == 'x' else d['y'] if b.winner == 'y' else 'tie'
        with C.session(engine()) as ses:
            ses.add(EvalJudgment(experiment=name, item_id=d['i'], a=d['x'], b=d['y'], winner=win, rater=b.rater[:64], at=time.time(), note=(b.note or None) and b.note[:500], a_key=d.get('xk'), b_key=d.get('yk'))); ses.commit()
        return dict(ok=True, x=d['x'], y=d['y'], winner=win)

    @app.get('/api/evals/{name}/media/{item_id}')
    def get_media(name: str, item_id: str):
        s = spec_or_404(name); it = next((x for x in C.load_dataset(s.dataset, engine=engine()).items if x['id'] == item_id), None)
        p = it and it['input'].get('path')
        if not isinstance(p, str) or not os.path.isfile(p): raise HTTPException(404, 'no file for this item')
        clip = it['input'].get('clip')
        if clip and (mimetypes.guess_type(p)[0] or '').startswith(('audio', 'video')):                 # the phrase only, not the whole recording
            import subprocess
            r = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-ss', str(float(clip[0])), '-to', str(float(clip[1])), '-i', p, '-vn', '-ac', '1', '-f', 'wav', '-'], capture_output=True)
            if r.returncode == 0 and r.stdout: return Response(content=r.stdout, media_type='audio/wav')
        return FileResponse(p, media_type=mimetypes.guess_type(p)[0] or 'application/octet-stream')
