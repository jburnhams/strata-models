"""User settings: one JSON file, `~/.strata-models/settings.json` (the folder can be moved with STRATAMODELS_HOME).

  data_dir        where the service keeps its own files (registry override, logs). Default: `data/` in this repository (gitignored).
  models_dir      where model weights are downloaded and read from, shared by every runtime. Default: `~/models`.
  vram_reserve_gb VRAM left for the desktop and for processes outside the service, on top of what they already use. Default: 1.0.

Environment variables override the file: STRATAMODELS_DATA, STRATAMODELS_MODELS, STRATAMODELS_VRAM_RESERVE_GB. The database is chosen by omarchy-db:
STRATAMODELS_DATABASE_URL, else OMARCHY_DB_URL, else schema `stratamodels` of the shared `omarchy` database over the local socket.
The file handling, precedence and validation are [strata_python.settings](https://github.com/jburnhams/strata-python); this module declares the settings."""

import os
from collections.abc import Mapping
from typing import Any

from strata_python.settings import Setting, Settings, absolute_path

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULTS = {"data_dir": os.path.join(REPO, "data"), "models_dir": os.path.join("~", "models"), "vram_reserve_gb": "1.0"}


def _reserve(v: object) -> float:
    try:
        f = float(str(v))
    except ValueError as exc:
        raise ValueError(f"vram_reserve_gb must be a number of GB, not {v!r}") from exc
    if not 0 <= f <= 64:
        raise ValueError("vram_reserve_gb must be between 0 and 64")
    return f


def _reserve_text(v: str) -> str:
    _reserve(v)
    return v.strip()


SETTINGS = Settings(
    "strata-models",
    [
        Setting("data_dir", default=lambda: DEFAULTS["data_dir"], normalise=absolute_path, env="STRATAMODELS_DATA"),
        Setting(
            "models_dir", default=lambda: DEFAULTS["models_dir"], normalise=absolute_path, env="STRATAMODELS_MODELS"
        ),
        Setting("vram_reserve_gb", default=lambda: DEFAULTS["vram_reserve_gb"], normalise=_reserve_text),
    ],
    prefix="STRATAMODELS",
)
KNOWN = SETTINGS.keys


def home() -> str:
    return SETTINGS.home()


def load() -> dict[str, Any]:
    return SETTINGS.load()


def save(changes: Mapping[str, object]) -> dict[str, Any]:
    """Merge `changes` (known keys only) into the settings file; returns the saved settings. A None value removes the setting (back to its default)."""
    return SETTINGS.save(changes)


def data_dir() -> str:
    """The folder that holds the service's own files (created when asked for)."""
    return SETTINGS.folder("data_dir")


def models_dir() -> str:
    """The one folder every runtime downloads weights into (created when asked for)."""
    return SETTINGS.folder("models_dir")


def vram_reserve_gb() -> float:
    """GB of VRAM the service never fills, so the desktop keeps working."""
    return _reserve(SETTINGS.get("vram_reserve_gb"))


def all_settings() -> dict[str, Any]:
    return {"data_dir": data_dir(), "models_dir": models_dir(), "vram_reserve_gb": vram_reserve_gb()}
