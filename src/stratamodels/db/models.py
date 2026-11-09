"""The service's tables in its schema (`stratamodels`) of the shared database. Times are epoch seconds (float): the same on SQLite and Postgres, and
what the web app formats anyway."""
import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from omarchy_db import make_base

Base = make_base()


class Job(Base):
    """One request to the GPU: `kind` run (a model job for `role`) or lease (an app holding the GPU for its own kernels). The queue is the rows with
    status queued; history is everything else, kept so the web app can show who used the GPU, for what and how long."""
    __tablename__ = 'jobs'
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    kind: Mapped[str] = mapped_column(sa.String(16), index=True)                    # run | lease
    app: Mapped[str] = mapped_column(sa.String(64), index=True)                     # who asked: bright-horizons, strata-360, ...
    priority: Mapped[str] = mapped_column(sa.String(16))                            # interactive | batch
    status: Mapped[str] = mapped_column(sa.String(16), index=True)                  # queued | running | done | failed | cancelled | expired
    submitted_at: Mapped[float]
    role: Mapped[str | None] = mapped_column(sa.String(64), default=None, index=True)
    model: Mapped[str | None] = mapped_column(sa.String(128), default=None)
    note: Mapped[str | None] = mapped_column(sa.String(300), default=None)          # a lease's reason ("arena build"), shown in the queue
    params: Mapped[dict | None] = mapped_column(sa.JSON, default=None)
    result: Mapped[dict | None] = mapped_column(sa.JSON, default=None)              # dropped after a week (prune), the row stays
    error: Mapped[str | None] = mapped_column(sa.Text, default=None)
    vram_gb: Mapped[float | None] = mapped_column(default=None)                     # a lease: what the app said it needs
    ttl_s: Mapped[float | None] = mapped_column(default=None)                       # a lease: expires this long after its last heartbeat
    started_at: Mapped[float | None] = mapped_column(default=None)
    finished_at: Mapped[float | None] = mapped_column(default=None)
    heartbeat_at: Mapped[float | None] = mapped_column(default=None)
    load_s: Mapped[float | None] = mapped_column(default=None)                      # time spent loading the model for this job (0 when it was resident)
    run_s: Mapped[float | None] = mapped_column(default=None)
