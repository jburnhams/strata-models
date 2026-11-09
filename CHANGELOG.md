# Changelog

Newest first, grouped by day. One entry per change, in the same commit or PR as the change.

## 2026-11-09

**One CUDA per venv: Whisper as a venv-worker, the service's venv on CUDA 13 (branch `whisper-worker`)**
Jonathan asked to move Whisper into its own venv rather than keep onnxruntime pinned to CUDA 12, building on the venv-worker runtime the TTS work added and without duplicating it. onnxruntime 1.24+ needs CUDA 13 and CTranslate2 (newest 4.8.5) CUDA 12; both load cuDNN 9 by the same file name, so they cannot share a process.

- `src/stratamodels/workers/protocol.py`: **new**, the JSON-lines loop and `ParamError` every worker uses. `workers/tts.py`: uses it (its own copy removed).
- `src/stratamodels/workers/whisper.py`: **new**, Whisper (faster-whisper) as a worker: the load/decode/run code of `runtimes/whisper.py`, which is **removed** (and the `faster-whisper` runtime with it). A `clip` is now decoded alone (ffmpeg -ss/-to) with times kept those of the whole file.
- `src/stratamodels/runtimes/worker.py`: a worker's venv's own `nvidia/*/lib` folders go first on its library path, and the service's venv's are left off it.
- `src/stratamodels/models.toml`: the Whisper models run as `venv-worker` (`worker = "whisper"`, `venv = "whisper"`).
- `scripts/setup_worker_venv.sh`: a `whisper` recipe (faster-whisper with the CUDA 12 cuBLAS and cuDNN wheels).
- `pyproject.toml`, `uv.lock`: the `gpu` group drops faster-whisper and the CUDA 12 wheels, takes onnxruntime-gpu 1.31 (CUDA 13, served by torch's CUDA 13 wheels) and nvidia-curand (which torch lacks). `strata-models` launcher: every `nvidia/*/lib` of the venv on the path again (it is all CUDA 13).
- Tests: `tests/unit/test_whisper_worker.py` **new** (the worker with a fake model and ffmpeg, the shared loop); `tests/gpu` runs Whisper through the worker (and a clip). Measured on the RTX 3060: faces 0.04 s a photo warm on onnxruntime 1.31; Whisper turbo 2.0 GB, 0.41 s for 11 s of speech.
- After merging, on omarchy: `scripts/setup_worker_venv.sh whisper` (done) and `uv sync --group gpu`, then restart the service.

**Move the Python side to uv, ruff, pyright and the shared strata-python library (branch `claude/project-thread-iw7199`)**
Jonathan asked for every Python project to use the same tooling as [python-app-template](https://github.com/jburnhams/python-app-template): uv, ruff with `ruff format`, pyright and ty, one `scripts/check` gate, and the server code shared through [strata-python](https://github.com/jburnhams/strata-python) instead of copied. Behaviour of the service is unchanged except where noted.

- `pyproject.toml`, `uv.lock`: **new** `[project]`, `dev` and `gpu` dependency groups (the GPU stack is `uv sync --group gpu`), git sources for `omarchy-db` and `strata-python` (tracking its `main`; the lock pins the commit), tool configuration (ruff `E,F,I,B,UP,SIM,RUF,C4,PT,S,TID` with `pytest.importorskip` banned, pyright strict ratchet, coverage). `requirements*.txt` and `scripts/setup_env.*`: **removed**.
- `src/stratamodels/oslib.py`, `tests/unit/test_oslib.py`: **removed**, now `strata_python.oslib` (tested in that library).
- `src/stratamodels/settings.py`, `server/app.py`, `cli.py`: **main's database-backed versions are kept as they are** (merged 2026-11-09); an earlier version of this change moved settings, page serving and the `set` command onto `strata_python`, and that is dropped for this repo until its settings can be expressed there. `oslib` still comes from the library.
- `src/stratamodels/runtimes/{clip,faces,sdcpp,whisper}.py`: ruff format; a file-level pyright directive for the GPU libraries that are not installed in CI, each with its reason. `whisper.py`: `noqa: S607` for the ffmpeg call.
- All Python files: `ruff format` at line length 120 (a mechanical commit of its own), and the ruff rules above.
- `.github/workflows/unit.yml`, `integration.yml`: uv; `strata-check --fast` once on Linux 3.13. `upgrade-strata-python.yml`: **new**, weekly PR that moves the library pin after the gate. `.github/dependabot.yml`: **new**, daily, minor and patch grouped, no ignores.
- `tests/conftest.py`: the isolation and no-network fixtures come from the strata-python pytest plugin; a small autouse fixture keeps the database URL on a tmp SQLite file. `pyproject.toml` adds `pillow` (evals) and `opencv-python-headless` (dev, faces view tests), which main took from the old requirements files; ruff ignores B904, B905, S110, S310, S603, S607 and SIM105 for main's code, and pyright's strict entry for `settings.py` is dropped because main's rewrite is untyped. The fake-ASR eval spec's latency guard is 1e9, not 1000: the fake runs in microseconds, so latency ratios were scheduler noise and `test_run_caches_outputs_and_analysis_decides` failed about one run in seven (it asserted the recommendation, and the noise rejected the candidate). Correction to the earlier suite: that test was flaky, not seed-dependent. 0 failures in 30 runs after the change.
- `README.md`, `AGENTS.md`, `tests/README.md`, `strata-models` launcher: commands and comments for uv.
