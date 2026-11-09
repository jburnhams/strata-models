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

Both live in the database (eval_experiments + eval_candidates, eval_datasets + eval_items) once the service has seen them: the files are SEEDS,
imported when they are new or changed (a spec edited through the API, `edited`, is no longer overwritten by its file), and anything with an engine
reads the tables. Without an engine (scripts, the offline CLI) the files are read as before.

Outputs are cached in eval_results under a key of everything that decides them, so adding a candidate runs only that candidate, and a rerun of
an unchanged experiment runs nothing."""

import contextlib, dataclasses, hashlib, json, math, os, time, tomllib

from sqlalchemy import select

from stratamodels import settings
from stratamodels.db.models import EvalCandidate, EvalDataset, EvalExperiment, EvalItem, EvalJudgment, EvalResult
from stratamodels.evals import metrics as M, stats as S

REPO_SPECS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "evals", "specs"
)


class EvalError(ValueError):
    """A spec or dataset that cannot be used as it is."""


from stratamodels.db import session  # noqa: E402 (kept here: evals code and tests use core.session)


def _canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _h(x, n=16):
    return hashlib.sha256(x.encode() if isinstance(x, str) else x).hexdigest()[:n]


# ---- datasets -----------------------------------------------------------------------------------------------------------------------------
def datasets_dir():
    p = os.path.join(settings.data_dir(), "evals", "datasets")
    os.makedirs(p, exist_ok=True)
    return p


@dataclasses.dataclass
class Dataset:
    name: str
    version: str
    items: list
    description: str = ""

    def to_json(self, items=False):
        d = dict(
            name=self.name,
            version=self.version,
            n=len(self.items),
            description=self.description,
            tags=sorted({k for it in self.items for k in (it.get("tags") or {})}),
        )
        if items:
            d["items"] = self.items
        return d


ITEM_KEYS = ("id", "input", "reference", "tags")
_DS_CACHE = {}


def load_dataset(name, folder=None, engine=None):
    """The dataset: from the database when an engine is given (its .jsonl, if there is one and it changed, imported first), else from the file."""
    if engine is None:
        return _file_dataset(name, folder)
    try:
        ds = _file_dataset(name, folder)
    except EvalError as e:
        if "no dataset" not in str(e):
            raise
        ds = None
    if ds is not None:
        import_dataset(engine, ds, os.path.join(folder or datasets_dir(), f"{name}.jsonl"))
    return _db_dataset(engine, name)


def import_dataset(engine, ds, source=None):
    """Store a dataset in the tables, replacing its items, unless that version is there already. Returns True when it wrote."""
    with session(engine) as s:
        row = s.get(EvalDataset, ds.name)
        if row is not None and row.version == ds.version:
            return False
        if row is None:
            row = EvalDataset(
                name=ds.name,
                version=ds.version,
                n=len(ds.items),
                updated_at=time.time(),
                description=ds.description,
                source=source,
            )
            s.add(row)
        else:
            from sqlalchemy import delete

            s.execute(delete(EvalItem).where(EvalItem.dataset == ds.name))
            row.version, row.n, row.updated_at, row.description, row.source = (
                ds.version,
                len(ds.items),
                time.time(),
                ds.description,
                source or row.source,
            )
        s.flush()
        for i, it in enumerate(ds.items):
            s.add(
                EvalItem(
                    dataset=ds.name,
                    item_id=it["id"],
                    position=i,
                    input=it["input"],
                    reference=it.get("reference"),
                    tags=it.get("tags"),
                    extra={k: v for k, v in it.items() if k not in ITEM_KEYS} or None,
                )
            )
        s.commit()
        return True


def _db_dataset(engine, name):
    with session(engine) as s:
        row = s.get(EvalDataset, name)
        if row is None:
            raise EvalError(f"no dataset {name!r} (not in the database, and no {name}.jsonl in {datasets_dir()})")
        hit = _DS_CACHE.get((id(engine), name))
        if hit and hit.version == row.version:
            return hit
        items = []
        for it in s.scalars(select(EvalItem).where(EvalItem.dataset == name).order_by(EvalItem.position)):
            d = dict(id=it.item_id, input=it.input)
            if it.reference is not None:
                d["reference"] = it.reference
            if it.tags is not None:
                d["tags"] = it.tags
            d.update(it.extra or {})
            items.append(d)
        ds = _DS_CACHE[(id(engine), name)] = Dataset(row.name, row.version, items, row.description or "")
        return ds


def _file_dataset(name, folder=None):
    path = os.path.join(folder or datasets_dir(), f"{name}.jsonl")
    try:
        raw = open(path, "rb").read()
    except FileNotFoundError:
        raise EvalError(f"no dataset {name!r} ({path})")
    items, seen, desc = [], set(), ""
    for n, line in enumerate(raw.decode("utf8").splitlines(), 1):
        if not line.strip():
            continue
        if line.startswith("#"):
            desc = (desc + " " + line.lstrip("# ").strip()).strip()
            continue
        try:
            it = json.loads(line)
        except ValueError as e:
            raise EvalError(f"{name}.jsonl line {n}: {e}")
        if not isinstance(it, dict) or "id" not in it or not isinstance(it.get("input"), dict):
            raise EvalError(f'{name}.jsonl line {n}: an item needs "id" and an "input" object')
        if str(it["id"]) in seen:
            raise EvalError(f"{name}.jsonl line {n}: id {it['id']!r} twice")
        it["id"] = str(it["id"])
        seen.add(it["id"])
        items.append(it)
    if not items:
        raise EvalError(f"dataset {name!r} has no items")
    return Dataset(name, _h(raw, 12), items, desc)


def write_dataset(name, items, description="", folder=None, engine=None):
    """Write a dataset (items as dicts) and return it loaded; `description` becomes its first, commented line. The .jsonl is written either way
    (the dataset builders in scripts/ run without the database); with an engine it goes into the tables too, else the service imports it when it
    next reads it."""
    path = os.path.join(folder or datasets_dir(), f"{name}.jsonl")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        for line in description.splitlines():
            f.write(f"# {line}\n")
        for it in items:
            f.write(_canon(it) + "\n")
    os.replace(tmp, path)
    return load_dataset(name, folder, engine)


def list_datasets(folder=None, engine=None):
    """Every dataset: the .jsonl files (imported when an engine is given) and those only in the database."""
    out, seen = [], set()
    for f in sorted(os.listdir(folder or datasets_dir())):
        if f.endswith(".jsonl"):
            seen.add(f[:-6])
            try:
                out.append(load_dataset(f[:-6], folder, engine).to_json())
            except EvalError as e:
                out.append(dict(name=f[:-6], error=str(e)))
    if engine is not None:
        with session(engine) as s:
            rest = [n for n in s.scalars(select(EvalDataset.name).order_by(EvalDataset.name)) if n not in seen]
        out += [_db_dataset(engine, n).to_json() for n in rest]
    return out


# ---- specs --------------------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Candidate:
    name: str
    model: str
    params: dict = dataclasses.field(default_factory=dict)
    options: dict = dataclasses.field(default_factory=dict)
    note: str = ""

    def to_json(self):
        return dataclasses.asdict(self)


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
    judge: bool = False  # offer the blind A/B judging page for this experiment
    similarity: str = (
        ""  # how outputs are compared to order judging (evals.similarity: text, image, json); '' picks from the outputs
    )
    source: str = ""
    score: dict = dataclasses.field(
        default_factory=dict
    )  # a second model that reads each output for the metrics (TTS: transcribe the audio back): {role, model?, params}

    def candidate(self, name):
        c = next((c for c in self.candidates if c.name == name), None)
        if c is None:
            raise EvalError(f"{self.name}: no candidate {name!r}")
        return c

    def to_json(self):
        return dict(
            name=self.name,
            question=self.question,
            role=self.role,
            dataset=self.dataset,
            metrics=self.metrics,
            candidates=[c.to_json() for c in self.candidates],
            primary=self.primary,
            baseline=self.baseline,
            min_effect=self.min_effect,
            guards=self.guards,
            judge=self.judge,
            similarity=self.similarity,
            source=self.source,
            score=self.score,
        )


def parse_spec(d, source=""):
    try:
        dec = d.get("decision") or {}
        cands = [
            Candidate(
                name=str(c["name"]),
                model=str(c["model"]),
                params=dict(c.get("params") or {}),
                options=dict(c.get("options") or {}),
                note=str(c.get("note", "")),
            )
            for c in d["candidates"]
        ]
        s = Spec(
            name=str(d["name"]),
            question=str(d["question"]),
            role=str(d["role"]),
            dataset=str(d["dataset"]),
            metrics=list(d["metrics"]),
            candidates=cands,
            primary=str(dec.get("primary") or d["metrics"][0]),
            baseline=str(dec.get("baseline") or cands[0].name),
            min_effect=float(dec.get("min_effect", 0.0)),
            guards={k: float(v) for k, v in (dec.get("guards") or {}).items()},
            judge=bool(d.get("judge", False)),
            similarity=str(d.get("similarity", "")),
            source=source,
            score=dict(d.get("score") or {}),
        )
        if s.score and not s.score.get("role"):
            raise ValueError("[score] names no role")
    except (KeyError, TypeError, ValueError, IndexError) as e:
        raise EvalError(f"{source or 'spec'}: {e!r}")
    unknown = [m for m in s.metrics + [s.primary] + list(s.guards) if m not in M.METRICS]
    if unknown:
        raise EvalError(f"{s.name}: unknown metrics {unknown} (known: {', '.join(M.METRICS)})")
    from stratamodels.evals import similarity as SIM

    if s.similarity and s.similarity not in SIM.KINDS:
        raise EvalError(f"{s.name}: unknown similarity {s.similarity!r} (known: {', '.join(SIM.KINDS)})")
    if s.primary == "judged":
        s.judge = True  # decided by people: the Judge tab must be on
    if s.primary not in s.metrics:
        s.metrics.insert(0, s.primary)
    for g in s.guards:
        if g not in s.metrics:
            s.metrics.append(g)
    names = [c.name for c in cands]
    if len(set(names)) != len(names):
        raise EvalError(f"{s.name}: two candidates share a name")
    s.candidate(s.baseline)
    return s


def spec_folders():
    return [REPO_SPECS, os.path.join(settings.data_dir(), "evals", "specs")]


def _file_specs():
    """{name: (Spec, raw dict, hash of the file) or EvalError} of the spec files."""
    out = {}
    for folder in spec_folders():
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if f.endswith(".toml"):
                p = os.path.join(folder, f)
                try:
                    raw = open(p, "rb").read()
                    d = tomllib.loads(raw.decode("utf8"))
                    s = parse_spec(d, p)
                    out[s.name] = (s, d, _h(raw, 32))
                except (EvalError, tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
                    out[f[:-5]] = EvalError(str(e))
    return out


def save_spec(engine, d, source=None, seed_hash=None, edited=True):
    """Store an experiment (a spec as a dict, validated by parse_spec) with its candidates; returns the Spec. `edited` (the default, an edit
    through the API) stops its file from overwriting it."""
    spec = parse_spec(d, source or "")
    with session(engine) as s:
        row = s.get(EvalExperiment, spec.name)
        now = time.time()
        if row is None:
            row = EvalExperiment(
                name=spec.name,
                question=spec.question,
                role=spec.role,
                dataset=spec.dataset,
                baseline=spec.baseline,
                spec=d,
                updated_at=now,
                source=source,
                seed_hash=seed_hash,
                edited=edited,
            )
            s.add(row)
        else:
            (
                row.question,
                row.role,
                row.dataset,
                row.baseline,
                row.spec,
                row.updated_at,
                row.source,
                row.seed_hash,
                row.edited,
            ) = spec.question, spec.role, spec.dataset, spec.baseline, d, now, source or row.source, seed_hash, edited
        from sqlalchemy import delete

        s.flush()
        s.execute(delete(EvalCandidate).where(EvalCandidate.experiment == spec.name))
        for i, c in enumerate(spec.candidates):
            s.add(
                EvalCandidate(
                    experiment=spec.name,
                    name=c.name,
                    model=c.model,
                    position=i,
                    params=c.params,
                    options=c.options,
                    note=c.note,
                )
            )
        s.commit()
    return spec


def list_specs(engine=None):
    """{name: Spec, or EvalError for one that cannot be used}. With an engine: the spec files are imported (new or changed, and not edited
    here) and the experiments come from the tables."""
    files = _file_specs()
    if engine is None:
        return {n: (v if isinstance(v, EvalError) else v[0]) for n, v in files.items()}
    with session(engine) as s:
        have = {r.name: (r.seed_hash, r.edited) for r in s.scalars(select(EvalExperiment))}
    for n, v in files.items():
        if isinstance(v, EvalError):
            continue
        spec, d, h = v
        if n not in have or (have[n][0] != h and not have[n][1]):
            save_spec(engine, json.loads(_canon(d)), spec.source, h, edited=False)
    out = {n: v for n, v in files.items() if isinstance(v, EvalError)}
    with session(engine) as s:
        rows = [(r.name, r.spec, r.source) for r in s.scalars(select(EvalExperiment).order_by(EvalExperiment.name))]
    for n, d, src in rows:
        try:
            out[n] = parse_spec(d, src or "")
        except EvalError as e:
            out[n] = e
    return out


def load_spec(name, engine=None):
    s = list_specs(engine).get(name)
    if s is None:
        raise EvalError(f"no experiment {name!r}")
    if isinstance(s, EvalError):
        raise s
    return s


# ---- running ------------------------------------------------------------------------------------------------------------------------------
def job_params(spec, cand, item):
    return {**cand.params, **item["input"]}


def result_key(spec, cand, item):
    """Everything that decides the output: role, model, load options, the job's params, the item (reference and tags left out: changing a label
    re-scores, it does not re-run)."""
    return _h(
        _canon(
            dict(
                role=spec.role,
                model=cand.model,
                options=cand.options,
                params=job_params(spec, cand, item),
                item=item["id"],
            )
        ),
        32,
    )


def cached(engine, keys):
    with session(engine) as s:
        return {r.key: r for r in s.scalars(select(EvalResult).where(EvalResult.key.in_(list(keys))))} if keys else {}


def run(spec, engine, executor, dataset=None, candidates=None, limit=None, log=print, batch=32):
    """Run what is missing: every (candidate, item) without a cached output goes to `executor` (submit(role, params, model, options) -> job id;
    wait(job id) -> job row). Returns dict(ran, cached, failed)."""
    ds = dataset or load_dataset(spec.dataset, engine=engine)
    items = ds.items[:limit] if limit else ds.items
    cands = [spec.candidate(c) for c in candidates] if candidates else spec.candidates
    todo = []
    for c in cands:
        keys = {result_key(spec, c, it): it for it in items}
        have = cached(engine, keys)
        queued = {k for _, _, k in todo}
        todo += [
            (c, it, k) for k, it in keys.items() if k not in have and k not in queued
        ]  # two candidates that are the same job run it once
    total = len(cands) * len(items)
    log(f"{spec.name}: {total - len(todo)} of {total} outputs cached, {len(todo)} to run")
    failed = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        ids = [executor.submit(spec.role, job_params(spec, c, it), c.model, c.options) for c, it, _ in chunk]
        done = [
            (c, it, k, executor.wait(jid)) for (c, it, k), jid in zip(chunk, ids)
        ]  # wait outside the lock: the worker needs the store meanwhile
        rows = [
            EvalResult(
                key=k,
                dataset=ds.name,
                item_id=it["id"],
                role=spec.role,
                model=c.model,
                created_at=time.time(),
                output=j.get("result") if j["status"] == "done" else None,
                error=None if j["status"] == "done" else (j.get("error") or j["status"]),
                run_s=j.get("run_s"),
                load_s=j.get("load_s"),
                job_id=j.get("id"),
            )
            for c, it, k, j in done
        ]
        failed += sum(1 for *_, j in done if j["status"] != "done")
        save(engine, rows)
        log(f"  {min(i + batch, len(todo))} of {len(todo)} run")
    out = dict(ran=len(todo), cached=total - len(todo), failed=failed)
    if spec.score:
        out["scored"] = score_outputs(
            spec, engine, executor, [result_key(spec, c, it) for c in cands for it in items], log, batch
        )
    return out


def score_key(spec):
    return _h(_canon(spec.score), 16)


SCORED = [
    0
]  # bumped each time the [score] pass writes readings: they change outputs in place (same rows, same ids), so the API's analysis memo watches this


def readings(output):
    """{score key: {result} or {error}} of an output: every [score] config's reading is kept apart (an output is shared by every experiment whose
    candidate made it, and two experiments may read it with different transcribers). A reading from before they were kept apart counts under its key."""
    out = dict((output or {}).get("scores") or {})
    old = (output or {}).get("scored")
    if isinstance(old, dict) and old.get("key") and old["key"] not in out:
        out[old["key"]] = {k: v for k, v in old.items() if k != "key"}
    return out


def score_outputs(spec, engine, executor, keys, log=print, batch=32):
    """The [score] pass: every stored output with a file (`path`) and no reading by the current scoring config gets one: a job on the scoring
    role with path = the output's file, whose result is kept in the output under `scores` = {score key: {result} or {error}}. Returns how many ran."""
    sk = score_key(spec)
    rows = [
        r
        for r in cached(engine, keys).values()
        if r.output and isinstance(r.output.get("path"), str) and sk not in readings(r.output)
    ]
    if rows:
        log(f"{spec.name}: scoring {len(rows)} outputs with {spec.score['role']}")
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        ids = [
            executor.submit(
                spec.score["role"],
                {**(spec.score.get("params") or {}), "path": r.output["path"]},
                spec.score.get("model"),
                None,
            )
            for r in chunk
        ]
        done = {r.key: executor.wait(jid) for r, jid in zip(chunk, ids)}
        with session(engine) as s:
            for r in s.scalars(select(EvalResult).where(EvalResult.key.in_(list(done)))):
                j = done[r.key]
                reading = (
                    dict(result=j.get("result")) if j["status"] == "done" else dict(error=j.get("error") or j["status"])
                )
                r.output = {
                    **{k: v for k, v in r.output.items() if k != "scored"},
                    "scores": {**readings(r.output), sk: reading},
                }
            s.commit()
        SCORED[0] += 1
        log(f"  {min(i + batch, len(rows))} of {len(rows)} scored")
    return len(rows)


def scored_view(spec, output):
    """The output as the metrics see it for this spec: `scored` is the reading by this spec's [score] config (None when not read yet)."""
    if not spec.score or not isinstance(output, dict):
        return output
    return {**output, "scored": readings(output).get(score_key(spec))}


def save(engine, rows):
    """Store outputs; one another run stored meanwhile (two experiments sharing a candidate, run at once) is kept and this copy dropped."""
    from sqlalchemy.exc import IntegrityError

    with session(engine) as s:
        have = {r.key for r in s.scalars(select(EvalResult).where(EvalResult.key.in_([r.key for r in rows])))}
        s.add_all([r for r in rows if r.key not in have])
        try:
            s.commit()
            return
        except IntegrityError:
            s.rollback()
        for r in rows:  # the race was lost between the check and the commit: row by row
            s.add(r)
            try:
                s.commit()
            except IntegrityError:
                s.rollback()


def forget(spec, engine, candidate=None):
    """Drop the cached outputs of an experiment (one candidate, or all) so they run again (after a change the key does not see, e.g. new weights)."""
    ds = load_dataset(spec.dataset, engine=engine)
    keys = [result_key(spec, c, it) for c in spec.candidates if candidate in (None, c.name) for it in ds.items]
    with session(engine) as s:
        n = 0
        for r in s.scalars(select(EvalResult).where(EvalResult.key.in_(keys))):
            s.delete(r)
            n += 1
        s.commit()
        return n


class SchedulerExecutor:
    """Runs eval jobs in this process's scheduler (the service, and tests)."""

    def __init__(self, scheduler, app="evals"):
        self.s, self.app = scheduler, app

    def submit(self, role, params, model, options):
        return self.s.submit(role, params, self.app, "batch", model, options or None)["id"]

    def wait(self, job_id):
        while True:
            j = self.s.wait(job_id, 60)
            if j["status"] not in ("queued", "running"):
                return j


class ClientExecutor:
    """Runs eval jobs through a running service (the CLI): they queue behind, and are run by, the one GPU worker like any app's."""

    def __init__(self, client, app="evals"):
        self.sm, self.app = client, app

    def submit(self, role, params, model, options):
        return self.sm.submit(role, params, self.app, "batch", model, options or None)["id"]

    def wait(self, job_id):
        return self.sm.wait_job(job_id)


# ---- analysis -----------------------------------------------------------------------------------------------------------------------------
def outputs_for(spec, engine, dataset=None):
    """{candidate: [(output, error) per item]} and the items (each with its run seconds under _run_s for the baseline's latency; per candidate
    in `runs`). Items not yet run by a candidate are None."""
    ds = dataset or load_dataset(spec.dataset, engine=engine)
    out, runs = {}, {}
    for c in spec.candidates:
        keys = [result_key(spec, c, it) for it in ds.items]
        have = cached(engine, keys)
        out[c.name] = [(scored_view(spec, have[k].output), have[k].error) if k in have else None for k in keys]
        runs[c.name] = [dict(run_s=have[k].run_s, load_s=have[k].load_s) if k in have else None for k in keys]
    return ds, out, runs


def judgments(engine, experiment, current=None):
    """(a, b, winner) of the experiment's judgments; with `current` ({(candidate, item): result key}) only those of outputs that are still the
    candidates' (a changed model, params or dataset, or a forgotten and regenerated output, makes a new key, and the old judgments stop counting)."""
    with session(engine) as s:
        rows = list(s.scalars(select(EvalJudgment).where(EvalJudgment.experiment == experiment)))
    if current is None:
        return [(j.a, j.b, j.winner) for j in rows]
    return [
        (j.a, j.b, j.winner)
        for j in rows
        if j.a_key and j.b_key and current.get((j.a, j.item_id)) == j.a_key and current.get((j.b, j.item_id)) == j.b_key
    ]


def analyse(spec, engine, dataset=None, seed=None):
    """Everything a report says, as data: per candidate the value and interval of every metric, the comparison with the baseline, the decision."""
    ds, outs, runs = outputs_for(spec, engine, dataset)
    names = [c.name for c in spec.candidates]
    complete = [
        i for i in range(len(ds.items)) if all(outs[n][i] is not None for n in names)
    ]  # compare only on items every candidate has
    items = [ds.items[i] for i in complete]

    def its(n):
        return [dict(it, _run_s=(runs[n][i] or {}).get("run_s")) for it, i in zip(items, complete)]

    def oo(n):
        return [outs[n][i] for i in complete]

    res = dict(
        experiment=spec.to_json(),
        dataset=dict(ds.to_json(), compared=len(complete)),
        candidates={},
        comparisons={},
        metrics={m: M.METRICS[m].to_json() for m in spec.metrics},
    )
    for n in names:
        done = sum(o is not None for o in outs[n])
        loads = [r["load_s"] for r in runs[n] if r and r.get("load_s")]
        res["candidates"][n] = dict(
            done=done,
            failed=sum(1 for o in outs[n] if o and o[1]),
            load_s=max(loads) if loads else None,
            metrics={m: M.summarise(m, its(n), oo(n), seed) for m in spec.metrics if m != "judged"} if complete else {},
        )
    current = {(c.name, it["id"]): result_key(spec, c, it) for c in spec.candidates for it in ds.items}
    j = judgments(engine, spec.name, current)
    jcomp = {n: judged_vs(j, n, spec.baseline) for n in names if n != spec.baseline}
    if "judged" in spec.metrics:
        for n in names:
            res["candidates"][n]["metrics"]["judged"] = judged_share(j, n)
    base = spec.baseline
    pvals = []
    for n in names:
        if n == base or not complete:
            continue
        res["comparisons"][n] = {
            m: M.compare(m, items, oo(base), oo(n), seed) for m in spec.metrics if m not in ("latency_s", "judged")
        }  # latency: each side's own run time, below
        if "latency_s" in spec.metrics:
            a = [M.METRICS["latency_s"].score(it, *o) for it, o in zip(its(base), oo(base))]
            b = [M.METRICS["latency_s"].score(it, *o) for it, o in zip(its(n), oo(n))]
            r = S.paired_diff(
                [x if x is not None else float("nan") for x in a],
                [y if y is not None else float("nan") for y in b],
                seed=seed,
            )
            r["improves"] = None if r["n"] == 0 else (True if r["high"] < 0 else False if r["low"] > 0 else None)
            res["comparisons"][n]["latency_s"] = r
        if "judged" in spec.metrics:
            res["comparisons"][n]["judged"] = jcomp.get(n) or judged_vs([], n, base)
        pvals.append((n, res["comparisons"][n][spec.primary]["p"]))
    for (n, _), adj in zip(pvals, S.holm([p for _, p in pvals])):
        res["comparisons"][n][spec.primary]["p_holm"] = adj
    res["decision"] = decide(spec, res, items, oo, base)
    if j:
        res["judgments"] = dict(
            n=len(j), strength={k: list(v) for k, v in S.bradley_terry_ci(j, names, b=500, seed=seed).items()}
        )
    return res


def _tally(j, n, against=None):
    """(wins, losses, ties) of candidate n in judgments j, against one other candidate or any."""
    w = l = t = 0
    for a, b, win in j:
        if n not in (a, b) or a == b or (against is not None and against not in (a, b)):
            continue
        if win == n:
            w += 1
        elif win == "tie":
            t += 1
        else:
            l += 1
    return w, l, t


def judged_share(j, n):
    """Share of all its judged pairs a candidate won (ties half), with its Wilson interval."""
    w, l, t = _tally(j, n)
    k = w + l + t
    v, lo, hi = S.wilson(w + t / 2, k) if k else (float("nan"),) * 3
    return dict(value=v, low=lo, high=hi, n=k)


def judged_vs(j, n, base):
    """Head to head with the baseline: the share of their pairs n won (ties half) minus one half, its Wilson interval, a two-sided sign test on
    the decided pairs; improves when the whole interval is above one half."""
    import math

    w, l, t = _tally(j, n, base)
    k = w + l + t
    if not k:
        return dict(
            n=0, wins=0, losses=0, ties=0, diff=float("nan"), low=float("nan"), high=float("nan"), p=1.0, improves=None
        )
    v, lo, hi = S.wilson(w + t / 2, k)
    d = w + l
    p = min(1.0, 2 * sum(math.comb(d, i) for i in range(0, min(w, l) + 1)) / 2**d) if d else 1.0
    return dict(
        n=k,
        wins=w,
        losses=l,
        ties=t,
        mean_a=1 - v,
        mean_b=v,
        diff=v - 0.5,
        low=lo - 0.5,
        high=hi - 0.5,
        p=p,
        improves=True if lo > 0.5 else False if hi < 0.5 else None,
    )


def decide(spec, res, items, oo, base):
    """Per candidate: 'adopt' (better on the primary metric beyond min_effect with the interval clear of 0, and within every guard), 'reject'
    (worse, or a guard broken), or 'unclear' with the items an answer would need. The overall recommendation is the adoptable candidate with the
    biggest improvement, else keep the baseline."""
    m = M.METRICS[spec.primary]
    sign = -1 if m.better == "lower" else 1
    out = {}
    for n, comp in res["comparisons"].items():
        c = comp[spec.primary]
        gain = sign * c["diff"] if c["n"] else float("nan")
        broken = []
        for g, limit in spec.guards.items():
            bv = res["candidates"][base]["metrics"][g]["value"]
            cv = res["candidates"][n]["metrics"][g]["value"]
            if bv is None or cv is None or bv != bv or cv != cv:
                continue  # no observations on one side: nothing to hold the guard to
            ratio = f"{cv / bv:.2f}x" if bv else f"{cv:g} against 0"
            if M.METRICS[g].better == "lower" and cv > limit * bv + 1e-12:
                broken.append(f"{g} {ratio} the baseline (limit {limit:g}x)")  # a baseline of 0 failures allows none
            if M.METRICS[g].better == "higher" and cv < bv / limit - 1e-12:
                broken.append(f"{g} {ratio} the baseline (limit 1/{limit:g})")
        if broken:
            verdict, why = "reject", "; ".join(broken)
        elif c["improves"] is True and gain >= spec.min_effect:
            verdict, why = (
                "adopt",
                f"{spec.primary} better by {_fmt(m, gain)} (95 % interval {_fmt(m, min(sign * c['low'], sign * c['high']))} to {_fmt(m, max(sign * c['low'], sign * c['high']))})",
            )
        elif c["improves"] is True:
            verdict, why = (
                "reject",
                f"better by only {_fmt(m, gain)}, under the {_fmt(m, spec.min_effect)} worth a change",
            )
        elif c["improves"] is False:
            verdict, why = "reject", f"{spec.primary} worse by {_fmt(m, -gain)}"
        else:
            need = None
            if m.shape == "judged" and spec.min_effect:
                need = math.ceil(
                    (0.98 / spec.min_effect) ** 2
                )  # pairs with the baseline for a 95 % interval of +-min_effect
            elif m.shape != "dataset" and spec.min_effect:
                d = [
                    (
                        lambda x, y: (
                            None
                            if x is None or y is None
                            else (y[0] / max(y[1], 1) - x[0] / max(x[1], 1))
                            if m.shape == "ratio"
                            else float(y) - float(x)
                        )
                    )(m.score(it, *a), m.score(it, *b))
                    for it, a, b in zip(items, oo(base), oo(n))
                ]
                need = S.items_needed(d, spec.min_effect)
            unit = "judged pairs with the baseline" if m.shape == "judged" else "items"
            verdict, why = (
                "unclear",
                f"difference {_fmt(m, gain)}, interval includes no change"
                + (f"; about {need} {unit} would show a {_fmt(m, spec.min_effect)} difference" if need else ""),
            )
        out[n] = dict(verdict=verdict, why=why, gain=gain)
    best = max((n for n, v in out.items() if v["verdict"] == "adopt"), key=lambda n: out[n]["gain"], default=None)
    return dict(per_candidate=out, recommendation=best or base, change=best is not None)


def _fmt(m, v):
    if v is None or v != v:
        return "n/a"
    return f"{v * 100:.2f} points" if m.unit == "%" else f"{v:.3f}{' ' + m.unit if m.unit else ''}"
