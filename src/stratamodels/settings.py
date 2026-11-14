"""The service's settings: rows of the `settings` table once the service has its database (`bind(engine)`, done by Store.connect); before that,
and for a run without the database (`serve --memory`, the CLI's offline commands, tests), one JSON file, `~/.strata-models/settings.json` (the
folder can be moved with STRATAMODELS_HOME). The first bind copies that file's values into an empty table, once (a marker row records it).

  data_dir        where the service keeps its own files (registry override, logs). Default: `data/` in this repository (gitignored).
  models_dir      where model weights are downloaded and read from, shared by every runtime. Default: `~/models`.
  vram_reserve_gb VRAM left for the desktop and for processes outside the service, on top of what they already use. Default: 1.0.
  gpu_jobs        GPU jobs that may run at once (on different models, when they fit in the budget together). Default: 2. 1 is one at a time.
  cpu_jobs        jobs for zero-VRAM models (recorded outputs, CPU models, cloud CLIs) that may run at once, beside the GPU's. Default: 4.
  whole_card_gb   a model or lease that needs more than this runs with the whole card to itself, the reserve ignored (to try models too big
                  to budget for, at their own risk). Default: 10.0.
  claude_budget_usd, gemini_budget_usd   what a cloud API key may spend per period (spend.py); a call whose worst case does not fit what is
                  left is refused. Default: 0 (nothing runs until a budget is set).
  claude_rollover, gemini_rollover       YYYY-MM-DD: a day the budget starts again, repeated monthly on that day of the month. Unset: the
                  budget is one pot that only goes down. (NVIDIA NIM is free and only rate-limited: no budget.)

Environment variables override the file: STRATAMODELS_DATA, STRATAMODELS_MODELS, STRATAMODELS_VRAM_RESERVE_GB, STRATAMODELS_GPU_JOBS,
STRATAMODELS_CPU_JOBS, STRATAMODELS_WHOLE_CARD_GB. The database is chosen by omarchy-db:
STRATAMODELS_DATABASE_URL, else OMARCHY_DB_URL, else schema `stratamodels` of the shared `omarchy` database over the local socket."""

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULTS = dict(
    data_dir=os.path.join(REPO, "data"),
    models_dir=os.path.join("~", "models"),
    vram_reserve_gb="1.0",
    gpu_jobs="2",
    cpu_jobs="4",
    whole_card_gb="10.0",
    claude_budget_usd="0",
    claude_rollover="",
    gemini_budget_usd="0",
    gemini_rollover="",
)
ENV = dict(
    data_dir="STRATAMODELS_DATA",
    models_dir="STRATAMODELS_MODELS",
    vram_reserve_gb="STRATAMODELS_VRAM_RESERVE_GB",
    gpu_jobs="STRATAMODELS_GPU_JOBS",
    cpu_jobs="STRATAMODELS_CPU_JOBS",
    whole_card_gb="STRATAMODELS_WHOLE_CARD_GB",
    claude_budget_usd="STRATAMODELS_CLAUDE_BUDGET_USD",
    claude_rollover="STRATAMODELS_CLAUDE_ROLLOVER",
    gemini_budget_usd="STRATAMODELS_GEMINI_BUDGET_USD",
    gemini_rollover="STRATAMODELS_GEMINI_ROLLOVER",
)
BUDGETED = ("claude", "gemini")  # the keys that cost money
KNOWN = tuple(DEFAULTS)


def home():
    return os.environ.get("STRATAMODELS_HOME") or os.path.join(os.path.expanduser("~"), ".strata-models")


def _path():
    return os.path.join(home(), "settings.json")


_engine = None
IMPORTED = "_imported"  # a row (not a setting) marking that the file was copied in


def bind(engine):
    """Keep the settings in this database from now on (None: back to the file). Copies the file's settings in when the table is empty."""
    global _engine
    _engine = None
    if engine is None:
        return
    from sqlalchemy import func, select
    from sqlalchemy.orm import Session

    from stratamodels.db.models import Setting

    with Session(engine) as s:
        if (
            s.get(Setting, IMPORTED) is None
        ):  # the file is copied in once: settings reset to their defaults later stay reset
            import time

            if not s.scalar(
                select(func.count()).select_from(Setting)
            ):  # a table with rows from before this marker was imported already
                for k, v in _file().items():
                    if k in KNOWN:
                        s.add(Setting(key=k, value=str(v), updated_at=time.time()))
            s.add(Setting(key=IMPORTED, value=_path(), updated_at=time.time()))
            s.commit()
    _engine = engine


def _db_rows():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from stratamodels.db.models import Setting

    with _lock(), Session(_engine) as s:
        return {r.key: r.value for r in s.scalars(select(Setting)) if r.key != IMPORTED}


def _lock():
    import contextlib

    return (
        getattr(_engine, "stratamodels_lock", None) or contextlib.nullcontext()
    )  # the in-memory store shares one connection: take its lock


def load():
    if _engine is not None:
        return _db_rows()
    return _file()


def _file():
    try:
        with open(_path(), encoding="utf8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        d = {}
    return d if isinstance(d, dict) else {}


def save(changes):
    """Merge `changes` (known keys only) into the settings file; returns the saved settings. A None value removes the setting (back to its default)."""
    d = load()
    for k, v in changes.items():
        if k not in KNOWN:
            raise KeyError(f"unknown setting {k!r}")
        if k in ("vram_reserve_gb", "whole_card_gb") and v is not None:
            _gb(k, v)  # refuse a bad number before it is written
        if k in ("gpu_jobs", "cpu_jobs") and v is not None:
            v = _slots(k, v)
        if k.endswith("_budget_usd") and v is not None:
            v = _usd(k, v)
        if k.endswith("_rollover") and v is not None:
            v = _date(k, v) or None  # "" clears it
        if v is None:
            d.pop(k, None)
        else:
            d[k] = str(v)
    if _engine is not None:
        import time

        from sqlalchemy.orm import Session

        from stratamodels.db.models import Setting

        with _lock(), Session(_engine) as s:
            for k in changes:
                row = s.get(Setting, k)
                if k not in d:
                    if row is not None:
                        s.delete(row)
                elif row is None:
                    s.add(Setting(key=k, value=d[k], updated_at=time.time()))
                else:
                    row.value, row.updated_at = d[k], time.time()
            s.commit()
        return d
    os.makedirs(home(), exist_ok=True)
    tmp = _path() + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, _path())
    return d


def _raw(key):
    return os.environ.get(ENV[key]) or load().get(key) or DEFAULTS[key]


def _folder(key):
    p = os.path.abspath(os.path.expanduser(_raw(key)))
    os.makedirs(p, exist_ok=True)
    return p


def data_dir():
    """The folder that holds the service's own files (created when asked for)."""
    return _folder("data_dir")


def models_dir():
    """The one folder every runtime downloads weights into (created when asked for)."""
    return _folder("models_dir")


def _gb(key, v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number of GB, not {v!r}")
    if not 0 <= f <= 64:
        raise ValueError(f"{key} must be between 0 and 64")
    return f


def _gb_setting(key):
    try:
        return _gb(key, _raw(key))
    except ValueError:
        return float(DEFAULTS[key])


def vram_reserve_gb():
    """GB of VRAM the service never fills, so the desktop keeps working."""
    return _gb_setting("vram_reserve_gb")


def whole_card_gb():
    """GB above which a model or lease gets the whole card to itself instead of a share of the budget."""
    return _gb_setting("whole_card_gb")


def _slots(key, v):
    try:
        n = int(str(v).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a whole number of jobs, not {v!r}")
    if not 1 <= n <= 16:
        raise ValueError(f"{key} must be between 1 and 16")
    return n


def jobs(key):
    """How many jobs of a lane (`gpu_jobs` or `cpu_jobs`) the scheduler runs at once."""
    try:
        return _slots(key, _raw(key))
    except ValueError:
        return int(DEFAULTS[key])


def _usd(key, v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number of US dollars, not {v!r}")
    if not 0 <= f <= 100000:
        raise ValueError(f"{key} must be between 0 and 100000")
    return f


def _date(key, v):
    import datetime

    v = str(v).strip()
    if not v:
        return ""
    try:
        return datetime.date.fromisoformat(v).isoformat()
    except ValueError:
        raise ValueError(f"{key} must be a date, YYYY-MM-DD, not {v!r}")


def budget(provider):
    """(USD a key may spend per period, its rollover date YYYY-MM-DD or None)."""
    try:
        usd = _usd(f"{provider}_budget_usd", _raw(f"{provider}_budget_usd"))
    except ValueError:
        usd = 0.0
    try:
        day = _date(f"{provider}_rollover", _raw(f"{provider}_rollover")) or None
    except ValueError:
        day = None
    return usd, day


def all_settings():
    return dict(
        data_dir=data_dir(),
        models_dir=models_dir(),
        vram_reserve_gb=vram_reserve_gb(),
        gpu_jobs=jobs("gpu_jobs"),
        cpu_jobs=jobs("cpu_jobs"),
        whole_card_gb=whole_card_gb(),
        **{k: v for p in BUDGETED for k, v in zip((f"{p}_budget_usd", f"{p}_rollover"), budget(p))},
    )
