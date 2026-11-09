# Changelog

Newest first, grouped by day. One entry per change, in the same commit or PR as the change.

## 2026-11-09

**Move the Python side to uv, ruff, pyright and the shared strata-python library (branch `claude/project-thread-iw7199`)**
Jonathan asked for every Python project to use the same tooling as [python-app-template](https://github.com/jburnhams/python-app-template): uv, ruff with `ruff format`, pyright and ty, one `scripts/check` gate, and the server code shared through [strata-python](https://github.com/jburnhams/strata-python) instead of copied. Behaviour of the service is unchanged except where noted.

- `pyproject.toml`, `uv.lock`: **new** `[project]`, `dev` and `gpu` dependency groups (the GPU stack is `uv sync --group gpu`), git sources for `omarchy-db` and `strata-python` (tracking its `main`; the lock pins the commit), tool configuration (ruff `E,F,I,B,UP,SIM,RUF,C4,PT,S,TID` with `pytest.importorskip` banned, pyright strict ratchet, coverage). `requirements*.txt` and `scripts/setup_env.*`: **removed**.
- `src/stratamodels/oslib.py`, `tests/unit/test_oslib.py`: **removed**, now `strata_python.oslib` (tested in that library).
- `src/stratamodels/settings.py`: declared with `strata_python.settings.Settings` (prefix `STRATAMODELS`, so `STRATAMODELS_DATA` and `STRATAMODELS_MODELS` still work); same settings file. A bad stored value now falls back to the default instead of crashing.
- `src/stratamodels/server/app.py`, `cli.py`: page serving, `<base href>`, `serve` and `set` come from `strata_python.web` and `strata_python.cli`; `/api/health` (with the GPU), `PUT /api/settings` (scheduler side effects) and the service's own `doctor` stay here.
- `src/stratamodels/runtimes/{clip,faces,sdcpp,whisper}.py`: ruff format; a file-level pyright directive for the GPU libraries that are not installed in CI, each with its reason. `whisper.py`: `noqa: S607` for the ffmpeg call.
- All Python files: `ruff format` at line length 120 (a mechanical commit of its own), and the ruff rules above.
- `.github/workflows/unit.yml`, `integration.yml`: uv; `strata-check --fast` once on Linux 3.13. `upgrade-strata-python.yml`: **new**, weekly PR that moves the library pin after the gate. `.github/dependabot.yml`: **new**, daily, minor and patch grouped, no ignores.
- `tests/conftest.py`: the isolation and no-network fixtures now come from the strata-python pytest plugin. The fake-ASR eval spec's latency guard is 1e9, not 1000: the fake runs in microseconds, so latency ratios were scheduler noise and `test_run_caches_outputs_and_analysis_decides` failed about one run in seven (it asserted the recommendation, and the noise rejected the candidate). Correction to the earlier suite: that test was flaky, not seed-dependent. 0 failures in 30 runs after the change.
- `README.md`, `AGENTS.md`, `tests/README.md`, `strata-models` launcher: commands and comments for uv.
