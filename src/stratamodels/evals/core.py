"""Experiments: what is being decided, on which items, between which candidates, judged how.

An experiment is a TOML spec (`evals/specs/<name>.toml` in this repository for generic ones, or `<data>/evals/specs/` for private ones):

    name = "transcribe-en"
    question = "Which Whisper model should serve the transcribe role for English speech?"
    role = "transcribe"
    dataset = "librispeech-clean-200"
    metrics = ["wer", "cer", "latency_s", "failed"]

    [decision]
    primary = "wer"                   # the metric the decision is made on
    baseline = "large-v3"             # the candidate in use now
    min_effect = 0.005                # a smaller improvement is not worth a change (in the metric's own units: 0.005 = half a point of WER)
    guards = { latency_s = 2.0 }      # a candidate may be at most this many times the baseline's value on these metrics

    [[candidates]]
    name = "large-v3"
    model = "whisper-large-v3"        # a registry model serving the role
    params = { language = "en" }      # added to every item's input
    options = {}                      # load options over the registry's (a variant of the model, loaded apart)

A dataset is JSON lines in `<data>/evals/datasets/<name>.jsonl`, one item per line: {"id", "input": {job params, paths absolute}, "reference":
{what the metrics read}, "tags": {...}}. Its version is the hash of the file, written in every report. Datasets stay out of git: they name
private files.

Outputs are cached in eval_results under a key of everything that decides them, so adding a candidate runs only that candidate, and a rerun of
an unchanged experiment runs nothing."""
import contextlib, dataclasses, hashlib, json, os, time, tomllib

from sqlalchemy import select
from sqlalchemy.orm import Session

from stratamodels import settings
from stratamodels.db.models import EvalJudgment, EvalResult
from stratamodels.evals import metrics as M, stats as S

REPO_SPECS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'evals', 'specs')


class EvalError(ValueError):
    """A spec or dataset that cannot be used as it is."""


@contextlib.contextmanager
def session(engine):
    """A session on the service's engine, under the store's lock when there is one (the in-memory SQLite store shares one connection)."""
    with getattr(engine, 'stratamodels_lock', contextlib.nullcontext()), Session(engine) as s: yield s


def _canon(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _h(x, n=16): return hashlib.sha256(x.encode() if isinstance(x, str) else x).hexdigest()[:n]


# ---- datasets -----------------------------------------------------------------------------------------------------------------------------
def datasets_dir():
    p = os.path.join(settings.data_dir(), 'evals', 'datasets'); os.makedirs(p, exist_ok=True); return p


@dataclasses.dataclass
class Dataset:
    name: str
    version: str
    items: list
    description: str = ''

    def to_json(self, items=False):
        d = dict(name=self.name, version=self.version, n=len(self.items), description=self.description, tags=sorted({k for it in self.items for k in (it.get('tags') or {})}))
        if items: d['items'] = self.items
        return d


def load_dataset(name, folder=None):
    path = os.path.join(folder or datasets_dir(), f'{name}.jsonl')
    try: raw = open(path, 'rb').read()
    except FileNotFoundError: raise EvalError(f'no dataset {name!r} ({path})')
    items, seen, desc = [], set(), ''
    for n, line in enumerate(raw.decode('utf8').splitlines(), 1):
        if not line.strip(): continue
        if line.startswith('#'): desc = (desc + ' ' + line.lstrip('# ').strip()).strip(); continue
        try: it = json.loads(line)
        except ValueError as e: raise EvalError(f'{name}.jsonl line {n}: {e}')
        if not isinstance(it, dict) or 'id' not in it or not isinstance(it.get('input'), dict): raise EvalError(f'{name}.jsonl line {n}: an item needs "id" and an "input" object')
        if str(it['id']) in seen: raise EvalError(f'{name}.jsonl line {n}: id {it["id"]!r} twice')
        it['id'] = str(it['id']); seen.add(it['id']); items.append(it)
    if not items: raise EvalError(f'dataset {name!r} has no items')
    return Dataset(name, _h(raw, 12), items, desc)


def write_dataset(name, items, description='', folder=None):
    """Write a dataset (items as dicts) and return it loaded; `description` becomes its first, commented line."""
    path = os.path.join(folder or datasets_dir(), f'{name}.jsonl'); tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf8') as f:
        for line in description.splitlines(): f.write(f'# {line}\n')
        for it in items: f.write(_canon(it) + '\n')
    os.replace(tmp, path); return load_dataset(name, folder)


def list_datasets(folder=None):
    out = []
    for f in sorted(os.listdir(folder or datasets_dir())):
        if f.endswith('.jsonl'):
            try: out.append(load_dataset(f[:-6], folder).to_json())
            except EvalError as e: out.append(dict(name=f[:-6], error=str(e)))
    return out


# ---- specs --------------------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Candidate:
    name: str
    model: str
    params: dict = dataclasses.field(default_factory=dict)
    options: dict = dataclasses.field(default_factory=dict)
    note: str = ''

    def to_json(self): return dataclasses.asdict(self)


@dataclasses.dataclass
class Spec:
    name: str
    question: str
    role: str
    dataset: str
    metrics: list
    candidates: list
    primary: str
    baseline: str
    min_effect: float = 0.0
    guards: dict = dataclasses.field(default_factory=dict)
    judge: bool = False                    # offer the blind A/B judging page for this experiment
    similarity: str = ''                   # how outputs are compared to order judging (evals.similarity: text, image, json); '' picks from the outputs
    source: str = ''

    def candidate(self, name):
        c = next((c for c in self.candidates if c.name == name), None)
        if c is None: raise EvalError(f'{self.name}: no candidate {name!r}')
        return c

    def to_json(self):
        return dict(name=self.name, question=self.question, role=self.role, dataset=self.dataset, metrics=self.metrics, candidates=[c.to_json() for c in self.candidates],
                    primary=self.primary, baseline=self.baseline, min_effect=self.min_effect, guards=self.guards, judge=self.judge, similarity=self.similarity, source=self.source)


def parse_spec(d, source=''):
    try:
        dec = d.get('decision') or {}; cands = [Candidate(name=str(c['name']), model=str(c['model']), params=dict(c.get('params') or {}), options=dict(c.get('options') or {}), note=str(c.get('note', ''))) for c in d['candidates']]
        s = Spec(name=str(d['name']), question=str(d['question']), role=str(d['role']), dataset=str(d['dataset']), metrics=list(d['metrics']), candidates=cands,
                 primary=str(dec.get('primary') or d['metrics'][0]), baseline=str(dec.get('baseline') or cands[0].name), min_effect=float(dec.get('min_effect', 0.0)),
                 guards={k: float(v) for k, v in (dec.get('guards') or {}).items()}, judge=bool(d.get('judge', False)), similarity=str(d.get('similarity', '')), source=source)
    except (KeyError, TypeError, ValueError, IndexError) as e: raise EvalError(f'{source or "spec"}: {e!r}')
    unknown = [m for m in s.metrics + [s.primary] + list(s.guards) if m not in M.METRICS]
    if unknown: raise EvalError(f'{s.name}: unknown metrics {unknown} (known: {", ".join(M.METRICS)})')
    from stratamodels.evals import similarity as SIM
    if s.similarity and s.similarity not in SIM.KINDS: raise EvalError(f'{s.name}: unknown similarity {s.similarity!r} (known: {", ".join(SIM.KINDS)})')
    if s.primary not in s.metrics: s.metrics.insert(0, s.primary)
    for g in s.guards:
        if g not in s.metrics: s.metrics.append(g)
    names = [c.name for c in cands]
    if len(set(names)) != len(names): raise EvalError(f'{s.name}: two candidates share a name')
    s.candidate(s.baseline); return s


def spec_folders(): return [REPO_SPECS, os.path.join(settings.data_dir(), 'evals', 'specs')]


def list_specs():
    out = {}
    for folder in spec_folders():
        if not os.path.isdir(folder): continue
        for f in sorted(os.listdir(folder)):
            if f.endswith('.toml'):
                p = os.path.join(folder, f)
                try:
                    with open(p, 'rb') as fh: s = parse_spec(tomllib.load(fh), p)
                    out[s.name] = s
                except (EvalError, tomllib.TOMLDecodeError) as e: out[f[:-5]] = EvalError(str(e))
    return out


def load_spec(name):
    s = list_specs().get(name)
    if s is None: raise EvalError(f'no experiment {name!r}')
    if isinstance(s, EvalError): raise s
    return s


# ---- running ------------------------------------------------------------------------------------------------------------------------------
def job_params(spec, cand, item): return {**cand.params, **item['input']}


def result_key(spec, cand, item):
    """Everything that decides the output: role, model, load options, the job's params, the item (reference and tags left out: changing a label
    re-scores, it does not re-run)."""
    return _h(_canon(dict(role=spec.role, model=cand.model, options=cand.options, params=job_params(spec, cand, item), item=item['id'])), 32)


def cached(engine, keys):
    with session(engine) as s:
        return {r.key: r for r in s.scalars(select(EvalResult).where(EvalResult.key.in_(list(keys))))} if keys else {}


def run(spec, engine, executor, dataset=None, candidates=None, limit=None, log=print, batch=32):
    """Run what is missing: every (candidate, item) without a cached output goes to `executor` (submit(role, params, model, options) -> job id;
    wait(job id) -> job row). Returns dict(ran, cached, failed)."""
    ds = dataset or load_dataset(spec.dataset); items = ds.items[:limit] if limit else ds.items
    cands = [spec.candidate(c) for c in candidates] if candidates else spec.candidates; todo = []
    for c in cands:
        keys = {result_key(spec, c, it): it for it in items}; have = cached(engine, keys)
        queued = {k for _, _, k in todo}
        todo += [(c, it, k) for k, it in keys.items() if k not in have and k not in queued]       # two candidates that are the same job run it once
    total = len(cands) * len(items); log(f'{spec.name}: {total - len(todo)} of {total} outputs cached, {len(todo)} to run'); failed = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]; ids = [executor.submit(spec.role, job_params(spec, c, it), c.model, c.options) for c, it, _ in chunk]
        done = [(c, it, k, executor.wait(jid)) for (c, it, k), jid in zip(chunk, ids)]       # wait outside the lock: the worker needs the store meanwhile
        rows = [EvalResult(key=k, dataset=ds.name, item_id=it['id'], role=spec.role, model=c.model, created_at=time.time(), output=j.get('result') if j['status'] == 'done' else None,
                           error=None if j['status'] == 'done' else (j.get('error') or j['status']), run_s=j.get('run_s'), load_s=j.get('load_s'), job_id=j.get('id')) for c, it, k, j in done]
        failed += sum(1 for *_, j in done if j['status'] != 'done'); save(engine, rows)
        log(f'  {min(i + batch, len(todo))} of {len(todo)} run')
    return dict(ran=len(todo), cached=total - len(todo), failed=failed)


def save(engine, rows):
    """Store outputs; one another run stored meanwhile (two experiments sharing a candidate, run at once) is kept and this copy dropped."""
    from sqlalchemy.exc import IntegrityError
    with session(engine) as s:
        have = {r.key for r in s.scalars(select(EvalResult).where(EvalResult.key.in_([r.key for r in rows])))}
        s.add_all([r for r in rows if r.key not in have])
        try: s.commit(); return
        except IntegrityError: s.rollback()
        for r in rows:                                                                      # the race was lost between the check and the commit: row by row
            s.add(r)
            try: s.commit()
            except IntegrityError: s.rollback()


def forget(spec, engine, candidate=None):
    """Drop the cached outputs of an experiment (one candidate, or all) so they run again (after a change the key does not see, e.g. new weights)."""
    ds = load_dataset(spec.dataset); keys = [result_key(spec, c, it) for c in spec.candidates if candidate in (None, c.name) for it in ds.items]
    with session(engine) as s:
        n = 0
        for r in s.scalars(select(EvalResult).where(EvalResult.key.in_(keys))): s.delete(r); n += 1
        s.commit(); return n


class SchedulerExecutor:
    """Runs eval jobs in this process's scheduler (the service, and tests)."""
    def __init__(self, scheduler, app='evals'): self.s, self.app = scheduler, app
    def submit(self, role, params, model, options): return self.s.submit(role, params, self.app, 'batch', model, options or None)['id']
    def wait(self, job_id):
        while True:
            j = self.s.wait(job_id, 60)
            if j['status'] not in ('queued', 'running'): return j


class ClientExecutor:
    """Runs eval jobs through a running service (the CLI): they queue behind, and are run by, the one GPU worker like any app's."""
    def __init__(self, client, app='evals'): self.sm, self.app = client, app
    def submit(self, role, params, model, options): return self.sm.submit(role, params, self.app, 'batch', model, options or None)['id']
    def wait(self, job_id): return self.sm.wait_job(job_id)


# ---- analysis -----------------------------------------------------------------------------------------------------------------------------
def outputs_for(spec, engine, dataset=None):
    """{candidate: [(output, error) per item]} and the items (each with its run seconds under _run_s for the baseline's latency; per candidate
    in `runs`). Items not yet run by a candidate are None."""
    ds = dataset or load_dataset(spec.dataset); out, runs = {}, {}
    for c in spec.candidates:
        keys = [result_key(spec, c, it) for it in ds.items]; have = cached(engine, keys)
        out[c.name] = [(have[k].output, have[k].error) if k in have else None for k in keys]
        runs[c.name] = [dict(run_s=have[k].run_s, load_s=have[k].load_s) if k in have else None for k in keys]
    return ds, out, runs


def judgments(engine, experiment, current=None):
    """(a, b, winner) of the experiment's judgments; with `current` ({(candidate, item): result key}) only those of outputs that are still the
    candidates' (a changed model, params or dataset, or a forgotten and regenerated output, makes a new key, and the old judgments stop counting)."""
    with session(engine) as s: rows = list(s.scalars(select(EvalJudgment).where(EvalJudgment.experiment == experiment)))
    if current is None: return [(j.a, j.b, j.winner) for j in rows]
    return [(j.a, j.b, j.winner) for j in rows if j.a_key and j.b_key and current.get((j.a, j.item_id)) == j.a_key and current.get((j.b, j.item_id)) == j.b_key]


def analyse(spec, engine, dataset=None, seed=None):
    """Everything a report says, as data: per candidate the value and interval of every metric, the comparison with the baseline, the decision."""
    ds, outs, runs = outputs_for(spec, engine, dataset); names = [c.name for c in spec.candidates]
    complete = [i for i in range(len(ds.items)) if all(outs[n][i] is not None for n in names)]     # compare only on items every candidate has
    items = [ds.items[i] for i in complete]
    def its(n): return [dict(it, _run_s=(runs[n][i] or {}).get('run_s')) for it, i in zip(items, complete)]
    def oo(n): return [outs[n][i] for i in complete]
    res = dict(experiment=spec.to_json(), dataset=dict(ds.to_json(), compared=len(complete)), candidates={}, comparisons={}, metrics={m: M.METRICS[m].to_json() for m in spec.metrics})
    for n in names:
        done = sum(o is not None for o in outs[n]); loads = [r['load_s'] for r in runs[n] if r and r.get('load_s')]
        res['candidates'][n] = dict(done=done, failed=sum(1 for o in outs[n] if o and o[1]), load_s=max(loads) if loads else None, metrics={m: M.summarise(m, its(n), oo(n), seed) for m in spec.metrics} if complete else {})
    base = spec.baseline; pvals = []
    for n in names:
        if n == base or not complete: continue
        res['comparisons'][n] = {m: M.compare(m, items, oo(base), oo(n), seed) for m in spec.metrics if m != 'latency_s'}       # latency: each side's own run time, below
        if 'latency_s' in spec.metrics:
            a = [M.METRICS['latency_s'].score(it, *o) for it, o in zip(its(base), oo(base))]; b = [M.METRICS['latency_s'].score(it, *o) for it, o in zip(its(n), oo(n))]
            r = S.paired_diff([x if x is not None else float('nan') for x in a], [y if y is not None else float('nan') for y in b], seed=seed)
            r['improves'] = None if r['n'] == 0 else (True if r['high'] < 0 else False if r['low'] > 0 else None); res['comparisons'][n]['latency_s'] = r
        pvals.append((n, res['comparisons'][n][spec.primary]['p']))
    for (n, _), adj in zip(pvals, S.holm([p for _, p in pvals])): res['comparisons'][n][spec.primary]['p_holm'] = adj
    res['decision'] = decide(spec, res, items, oo, base)
    current = {(c.name, it['id']): result_key(spec, c, it) for c in spec.candidates for it in ds.items}
    j = judgments(engine, spec.name, current)
    if j: res['judgments'] = dict(n=len(j), strength={k: list(v) for k, v in S.bradley_terry_ci(j, names, b=500, seed=seed).items()})
    return res


def decide(spec, res, items, oo, base):
    """Per candidate: 'adopt' (better on the primary metric beyond min_effect with the interval clear of 0, and within every guard), 'reject'
    (worse, or a guard broken), or 'unclear' with the items an answer would need. The overall recommendation is the adoptable candidate with the
    biggest improvement, else keep the baseline."""
    m = M.METRICS[spec.primary]; sign = -1 if m.better == 'lower' else 1; out = {}
    for n, comp in res['comparisons'].items():
        c = comp[spec.primary]; gain = sign * c['diff'] if c['n'] else float('nan'); broken = []
        for g, limit in spec.guards.items():
            bv = res['candidates'][base]['metrics'][g]['value']; cv = res['candidates'][n]['metrics'][g]['value']
            if bv is None or cv is None or bv != bv or cv != cv: continue                     # no observations on one side: nothing to hold the guard to
            ratio = f'{cv / bv:.2f}x' if bv else f'{cv:g} against 0'
            if M.METRICS[g].better == 'lower' and cv > limit * bv + 1e-12: broken.append(f'{g} {ratio} the baseline (limit {limit:g}x)')                 # a baseline of 0 failures allows none
            if M.METRICS[g].better == 'higher' and cv < bv / limit - 1e-12: broken.append(f'{g} {ratio} the baseline (limit 1/{limit:g})')
        if broken: verdict, why = 'reject', '; '.join(broken)
        elif c['improves'] is True and gain >= spec.min_effect: verdict, why = 'adopt', f'{spec.primary} better by {_fmt(m, gain)} (95 % interval {_fmt(m, min(sign * c["low"], sign * c["high"]))} to {_fmt(m, max(sign * c["low"], sign * c["high"]))})'
        elif c['improves'] is True: verdict, why = 'reject', f'better by only {_fmt(m, gain)}, under the {_fmt(m, spec.min_effect)} worth a change'
        elif c['improves'] is False: verdict, why = 'reject', f'{spec.primary} worse by {_fmt(m, -gain)}'
        else:
            need = None
            if m.shape != 'dataset' and spec.min_effect:
                d = [(lambda x, y: None if x is None or y is None else (y[0] / max(y[1], 1) - x[0] / max(x[1], 1)) if m.shape == 'ratio' else float(y) - float(x))(m.score(it, *a), m.score(it, *b)) for it, a, b in zip(items, oo(base), oo(n))]
                need = S.items_needed(d, spec.min_effect)
            verdict, why = 'unclear', f'difference {_fmt(m, gain)}, interval includes no change' + (f'; about {need} items would show a {_fmt(m, spec.min_effect)} difference' if need else '')
        out[n] = dict(verdict=verdict, why=why, gain=gain)
    best = max((n for n, v in out.items() if v['verdict'] == 'adopt'), key=lambda n: out[n]['gain'], default=None)
    return dict(per_candidate=out, recommendation=best or base, change=best is not None)


def _fmt(m, v):
    if v is None or v != v: return 'n/a'
    return f'{v * 100:.2f} points' if m.unit == '%' else f'{v:.3f}{" " + m.unit if m.unit else ""}'
