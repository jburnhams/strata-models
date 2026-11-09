"""User settings: one JSON file, `~/.strata-models/settings.json` (the folder can be moved with STRATAMODELS_HOME).

  data_dir        where the service keeps its own files (registry override, logs). Default: `data/` in this repository (gitignored).
  models_dir      where model weights are downloaded and read from, shared by every runtime. Default: `~/models`.
  vram_reserve_gb VRAM left for the desktop and for processes outside the service, on top of what they already use. Default: 1.0.

Environment variables override the file: STRATAMODELS_DATA, STRATAMODELS_MODELS, STRATAMODELS_VRAM_RESERVE_GB. The database is chosen by omarchy-db:
STRATAMODELS_DATABASE_URL, else OMARCHY_DB_URL, else schema `stratamodels` of the shared `omarchy` database over the local socket."""

import json, os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULTS = dict(data_dir=os.path.join(REPO, "data"), models_dir=os.path.join("~", "models"), vram_reserve_gb="1.0")
ENV = dict(
    data_dir="STRATAMODELS_DATA", models_dir="STRATAMODELS_MODELS", vram_reserve_gb="STRATAMODELS_VRAM_RESERVE_GB"
)
KNOWN = tuple(DEFAULTS)


def home():
    return os.environ.get("STRATAMODELS_HOME") or os.path.join(os.path.expanduser("~"), ".strata-models")


def _path():
    return os.path.join(home(), "settings.json")


def load():
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
        if k == "vram_reserve_gb" and v is not None:
            _reserve(v)  # refuse a bad number before it is written
        if v is None:
            d.pop(k, None)
        else:
            d[k] = str(v)
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


def _reserve(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"vram_reserve_gb must be a number of GB, not {v!r}")
    if not 0 <= f <= 64:
        raise ValueError("vram_reserve_gb must be between 0 and 64")
    return f


def vram_reserve_gb():
    """GB of VRAM the service never fills, so the desktop keeps working."""
    try:
        return _reserve(_raw("vram_reserve_gb"))
    except ValueError:
        return float(DEFAULTS["vram_reserve_gb"])


def all_settings():
    return dict(data_dir=data_dir(), models_dir=models_dir(), vram_reserve_gb=vram_reserve_gb())
