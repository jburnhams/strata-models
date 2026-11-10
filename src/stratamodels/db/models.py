"""The service's tables in its schema (`stratamodels`) of the shared database. Times are epoch seconds (float): the same on SQLite and Postgres, and
what the web app formats anyway.

  registry      models, roles, model_roles (which roles a model may serve; a role's `model` is the one chosen for it)
  config        settings (the service's own: folders, VRAM reserve), preferences (an app's own choice of model or default params for a role)
  requests      jobs (every run and lease), results (a run's output, dropped after a week), files (paths a job read or wrote: never the bytes)
  evals         eval_datasets + eval_items, eval_experiments + eval_candidates, eval_results (cached outputs), eval_judgments (blind A/B choices)

Files in the repository (models.toml, evals/specs/*.toml, dataset .jsonl files) are SEEDS: imported at start, and followed while the row has not
been edited here (`edited`); see registry.seed and evals.core."""

import sqlalchemy as sa
from omarchy_db import make_base
from sqlalchemy.orm import Mapped, mapped_column

Base = make_base()


class Job(Base):
    """One request to the GPU: `kind` run (a model job for `role`) or lease (an app holding the GPU for its own kernels). The queue is the rows with
    status queued; history is everything else, kept so the web app can show who used the GPU, for what and how long."""

    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    kind: Mapped[str] = mapped_column(sa.String(16), index=True)  # run | lease
    app: Mapped[str] = mapped_column(sa.String(64), index=True)  # who asked: bright-horizons, strata-360, ...
    priority: Mapped[str] = mapped_column(sa.String(16))  # interactive | batch
    status: Mapped[str] = mapped_column(
        sa.String(16), index=True
    )  # queued | running | done | failed | cancelled | expired
    submitted_at: Mapped[float]
    role: Mapped[str | None] = mapped_column(sa.String(64), default=None, index=True)
    model: Mapped[str | None] = mapped_column(sa.String(128), default=None)
    note: Mapped[str | None] = mapped_column(
        sa.String(300), default=None
    )  # a lease's reason ("arena build"), shown in the queue
    params: Mapped[dict | None] = mapped_column(sa.JSON, default=None)
    error: Mapped[str | None] = mapped_column(sa.Text, default=None)
    vram_gb: Mapped[float | None] = mapped_column(default=None)  # a lease: what the app said it needs
    ttl_s: Mapped[float | None] = mapped_column(default=None)  # a lease: expires this long after its last heartbeat
    started_at: Mapped[float | None] = mapped_column(default=None)
    finished_at: Mapped[float | None] = mapped_column(default=None)
    heartbeat_at: Mapped[float | None] = mapped_column(default=None)
    load_s: Mapped[float | None] = mapped_column(
        default=None
    )  # time spent loading the model for this job (0 when it was resident)
    run_s: Mapped[float | None] = mapped_column(default=None)


class EvalResult(Base):
    """One model output for one eval item, keyed by everything that decides it (role, model, load options, the job's params, the item): the same
    candidate on the same item is never run twice, across experiments. Outputs are kept (they are what a report's examples and the judging page show)."""

    __tablename__ = "eval_results"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    key: Mapped[str] = mapped_column(sa.String(64), unique=True)
    dataset: Mapped[str] = mapped_column(sa.String(128), index=True)
    item_id: Mapped[str] = mapped_column(sa.String(256))
    role: Mapped[str] = mapped_column(sa.String(64))
    model: Mapped[str] = mapped_column(sa.String(128))
    created_at: Mapped[float]
    output: Mapped[dict | None] = mapped_column(sa.JSON, default=None)
    error: Mapped[str | None] = mapped_column(sa.Text, default=None)
    run_s: Mapped[float | None] = mapped_column(default=None)
    load_s: Mapped[float | None] = mapped_column(default=None)
    job_id: Mapped[int | None] = mapped_column(default=None)


class EvalJudgment(Base):
    """A person's blind choice between two candidates' outputs for one item (or a tie): what metrics cannot measure (is this transcript better,
    is this caption right). Analysed with Bradley-Terry."""

    __tablename__ = "eval_judgments"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    experiment: Mapped[str] = mapped_column(sa.String(128), index=True)
    item_id: Mapped[str] = mapped_column(sa.String(256))
    a: Mapped[str] = mapped_column(sa.String(128))
    b: Mapped[str] = mapped_column(sa.String(128))
    winner: Mapped[str] = mapped_column(sa.String(128))  # a's name, b's name, or 'tie'
    rater: Mapped[str] = mapped_column(sa.String(64))
    at: Mapped[float]
    note: Mapped[str | None] = mapped_column(sa.String(500), default=None)
    a_key: Mapped[str | None] = mapped_column(
        sa.String(64), default=None
    )  # the result keys of the two outputs judged: a judgment counts only
    b_key: Mapped[str | None] = mapped_column(
        sa.String(64), default=None
    )  # while they are still the candidates' outputs


class EvalSummary(Base):
    """An experiment's analysis (evals.core.analyse, NaN as null), worked out in the background when its outputs, judgments or spec change, so the
    page reads it instead of waiting for the bootstraps. `signature` is what it was worked out from (evals.summaries.signature)."""

    __tablename__ = "eval_summaries"
    experiment: Mapped[str] = mapped_column(sa.String(128), primary_key=True)
    signature: Mapped[str] = mapped_column(sa.String(64))
    result: Mapped[dict] = mapped_column(sa.JSON)
    computed_at: Mapped[float]
    seconds: Mapped[float]


class JobResult(Base):
    """A run's output (JSON: embeddings, transcripts, the path of a generated picture). Kept apart from the job row so the history stays small:
    results go after a week (Store.prune), the job rows stay."""

    __tablename__ = "results"
    job_id: Mapped[int] = mapped_column(sa.ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True)
    output: Mapped[dict | None] = mapped_column(sa.JSON)
    created_at: Mapped[float]


class JobFile(Base):
    """A file a job read (`input`: the picture or audio it was given) or wrote (`output`: a generated picture). Only the path: the bytes stay on
    disk where the app or the service put them. Lets anyone ask which job made a file, or what a job touched."""

    __tablename__ = "files"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    job_id: Mapped[int] = mapped_column(sa.ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    direction: Mapped[str] = mapped_column(sa.String(8))  # input | output
    path: Mapped[str] = mapped_column(sa.String(1024), index=True)
    created_at: Mapped[float]


# ---- registry and config -----------------------------------------------------------------------------------------------------------------
class ModelRow(Base):
    """A model the service knows: how it is loaded (runtime, name, options) and what it costs. A runtime this service does not implement marks one
    that is known (an app loads it itself today) but not served here yet."""

    __tablename__ = "models"
    key: Mapped[str] = mapped_column(sa.String(128), primary_key=True)
    runtime: Mapped[str] = mapped_column(sa.String(64))
    name: Mapped[str] = mapped_column(sa.String(300))
    vram_gb: Mapped[float]
    updated_at: Mapped[float]
    licence: Mapped[str] = mapped_column(sa.String(300), default="")
    source: Mapped[str] = mapped_column(sa.String(500), default="")
    used_by: Mapped[list] = mapped_column(
        sa.JSON, default_factory=list
    )  # apps that use it (or loaded it themselves before)
    options: Mapped[dict] = mapped_column(sa.JSON, default_factory=dict)  # passed to the runtime's load
    seed_hash: Mapped[str | None] = mapped_column(
        sa.String(32), default=None
    )  # the models.toml entry it was last imported from
    edited: Mapped[bool] = mapped_column(
        default=False, server_default=sa.false()
    )  # changed here: models.toml no longer overwrites it


class RoleRow(Base):
    """A job kind apps ask for (`faces`, `transcribe`), never a file: `model` is the one chosen for it, by an eval where there was one."""

    __tablename__ = "roles"
    key: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    model: Mapped[str] = mapped_column(sa.ForeignKey("models.key", onupdate="CASCADE"))
    updated_at: Mapped[float]
    description: Mapped[str] = mapped_column(sa.String(500), default="")
    seed_hash: Mapped[str | None] = mapped_column(sa.String(32), default=None)
    edited: Mapped[bool] = mapped_column(default=False, server_default=sa.false())


class ModelRole(Base):
    """Which roles a model may serve (a job may name any model listed under its role)."""

    __tablename__ = "model_roles"
    model: Mapped[str] = mapped_column(
        sa.ForeignKey("models.key", ondelete="CASCADE", onupdate="CASCADE"), primary_key=True
    )
    role: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    position: Mapped[int] = mapped_column(default=0)


class Setting(Base):
    """The service's settings (data_dir, models_dir, vram_reserve_gb); environment variables still override them."""

    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    value: Mapped[str] = mapped_column(sa.String(1024))
    updated_at: Mapped[float]


class Preference(Base):
    """An app's own choice for a role: the model its jobs run on when they name none, and params added under every job's own (e.g. strata-360's
    transcribe jobs default to language "en"). The role's model stays everyone else's."""

    __tablename__ = "preferences"
    __table_args__ = (sa.UniqueConstraint("app", "role"),)
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    app: Mapped[str] = mapped_column(sa.String(64))
    role: Mapped[str] = mapped_column(sa.ForeignKey("roles.key", ondelete="CASCADE", onupdate="CASCADE"))
    updated_at: Mapped[float]
    model: Mapped[str | None] = mapped_column(
        sa.ForeignKey("models.key", ondelete="SET NULL", onupdate="CASCADE"), default=None
    )
    params: Mapped[dict | None] = mapped_column(sa.JSON, default=None)
    note: Mapped[str | None] = mapped_column(sa.String(500), default=None)


# ---- evals: datasets and experiments -----------------------------------------------------------------------------------------------------
class EvalDataset(Base):
    """A versioned set of items (version = hash of the items as imported or written). Items name files by path; the files stay where they are."""

    __tablename__ = "eval_datasets"
    name: Mapped[str] = mapped_column(sa.String(128), primary_key=True)
    version: Mapped[str] = mapped_column(sa.String(32))
    n: Mapped[int]
    updated_at: Mapped[float]
    description: Mapped[str] = mapped_column(sa.Text, default="")
    source: Mapped[str | None] = mapped_column(sa.String(1024), default=None)  # the .jsonl it was imported from


class EvalItem(Base):
    __tablename__ = "eval_items"
    __table_args__ = (sa.UniqueConstraint("dataset", "item_id"),)
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    dataset: Mapped[str] = mapped_column(sa.ForeignKey("eval_datasets.name", ondelete="CASCADE"), index=True)
    item_id: Mapped[str] = mapped_column(sa.String(256))
    position: Mapped[int]
    input: Mapped[dict] = mapped_column(sa.JSON)  # job params; paths absolute
    reference: Mapped[dict | None] = mapped_column(sa.JSON, default=None)
    tags: Mapped[dict | None] = mapped_column(sa.JSON, default=None)
    extra: Mapped[dict | None] = mapped_column(sa.JSON, default=None)  # any other fields of the item, kept as they were


class EvalExperiment(Base):
    """An experiment: the spec as a whole (`spec`, validated by evals.core.parse_spec, so new spec fields need no migration) and the fields worth
    querying by in columns."""

    __tablename__ = "eval_experiments"
    name: Mapped[str] = mapped_column(sa.String(128), primary_key=True)
    question: Mapped[str] = mapped_column(sa.Text)
    role: Mapped[str] = mapped_column(sa.String(64), index=True)
    dataset: Mapped[str] = mapped_column(sa.String(128))
    baseline: Mapped[str] = mapped_column(sa.String(128))
    spec: Mapped[dict] = mapped_column(sa.JSON)
    updated_at: Mapped[float]
    source: Mapped[str | None] = mapped_column(sa.String(1024), default=None)  # the TOML it was imported from
    seed_hash: Mapped[str | None] = mapped_column(sa.String(32), default=None)
    edited: Mapped[bool] = mapped_column(default=False, server_default=sa.false())


class EvalCandidate(Base):
    __tablename__ = "eval_candidates"
    experiment: Mapped[str] = mapped_column(
        sa.ForeignKey("eval_experiments.name", ondelete="CASCADE"), primary_key=True
    )
    name: Mapped[str] = mapped_column(sa.String(128), primary_key=True)
    model: Mapped[str] = mapped_column(
        sa.String(128), index=True
    )  # not a foreign key: a candidate may be a model not registered yet
    position: Mapped[int]
    params: Mapped[dict] = mapped_column(sa.JSON, default_factory=dict)
    options: Mapped[dict] = mapped_column(sa.JSON, default_factory=dict)
    note: Mapped[str] = mapped_column(sa.String(500), default="")
