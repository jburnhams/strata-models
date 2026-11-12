# Changelog

Newest first, grouped by day. One entry per change, in the same commit or PR as the change.

## 2026-11-12

**Evals: model judges beside people, and how far each agrees with them (branch `claude/project-thread-kjwkpt`)**
Jonathan asked whether models judge eval outputs (they did not: only his blind A/B picks) and wanted a model's verdict recorded beside his on the same pairs, to learn how good a judge it is before relying on it more; and judges sorted by kind, each with its own model.

- `src/stratamodels/evals/judges.py`: **new**. A spec's `[[judges]]`: kind `text` or `vision` (registry roles `judge-text`, `judge-vision`, each with its chosen model; a judge may name another model to trial it), or a cloud model via `cli = "codex" | "agy"` on the plan. A judge reads the item (and its pictures, for vision), the two answers as X and Y with every field that names the model removed, and answers {reason, winner}; asked both ways round by default, and a judge that changes its mind with the order gives a tie. `pairs = "judged"` (default: the pairs people judged) or `"all"`, `limit` per run, `auto` (judge each pair right after a person does). The analysis gains `judges`: verdicts, agreement with people (tie included), same winner where both picked one, Cohen's kappa, order consistency, disagreements, and what the judge alone would say against the baseline.
- `db/models.py`, `migrations/20261112_7b3e9d2c4f18_judgment_judges.py`: `eval_judgments.judge` (the judge's config hash; null for a person) and `detail` (each order's answer and reason). Nullable: every existing row stays a person's. A changed judge config or a new model for its kind is a new judge.
- `core.judgments` and the Judge tab count people only: a judge never decides a `judged` experiment or makes a pair look judged.
- `evals/api.py`: `POST /api/evals/{name}/judges/{judge}/run`, `GET .../judges/status`, `GET .../verdicts/{item_id}`; a person's judgment starts the `auto` judges on that pair. CLI `strata-models eval judge NAME [--judge J] [--limit N]` (exit 75 when a plan's allowance runs out).
- `evals/report.py`: a **Model judges** section after Human judgments. `evals/cloud.py`: the codex/agy calls moved out of `scripts/evals/record_cloud.py` (which imports them) so judges use the same official-CLI path.
- `models.toml`: roles `judge-text` and `judge-vision` (qwen3.5-9b-vl to start, unmeasured), served by the llama-server models.
- `web/`: the experiment page's Model judges table; after a judgment the unblinded pair shows what each judge said about it.

**Scheduler: several jobs at once, cloud and CPU jobs beside the GPU's (branch `claude/project-thread-kf7tb7`)**
The one worker thread ran one job at a time, so a 30 s cloud CLI call (the subscription-backends idea) or a CPU model would hold up the GPU queue, and two small GPU models that fit together still took turns.

- `src/stratamodels/scheduler.py`: a dispatcher decides (`dispatch`), each job runs on its own thread (`execute`; `step` still does both inline for the tests). Two lanes: zero-VRAM models (vram_gb 0) run up to `cpu_jobs` at once and never wait for the GPU or a lease, exclusive ones included; GPU jobs run up to `gpu_jobs` at once on different models, when they fit together. One job per model at a time; a running or loading model is never evicted; a loading model's estimate, and a running one's estimate beyond what it measured at load (sd-cpp allocates per job), count as promised, so a second job never starts into room the first is about to take. A GPU job that does not fit beside running ones waits and nothing on the GPU passes it (no starving big jobs); with nothing else running it starts as before. Leases wait for the room running jobs hold; an exclusive one waits until GPU jobs finish. A load is measured from the card only when no other GPU work overlapped it; otherwise the estimate stands. Load requests from the Models page run like jobs; an unload of a busy model waits for its job.
- `settings.py`, `/api/settings`, `strata-models set`, Settings page: `gpu_jobs` (default 2) and `cpu_jobs` (default 4), 1-16, applied at once (`STRATAMODELS_GPU_JOBS`, `STRATAMODELS_CPU_JOBS`). `gpu_jobs = 1` is the old one-at-a-time behaviour for the GPU.
- `/api/status`: `running` (every job running, with its lane), `slots`, `budget.pending_mb`, `resident[].busy`; `current` stays (the oldest running). CLI `status` and the Overview list every running job.

## 2026-11-10

**Evals page: analyses worked out ahead and kept in the database (branch `evals-precompute`)**
Jonathan found the evals page slow to the point of timing out. faces-bh took 81 to 97 s per analysis: the face-verification bootstrap re-read every embedding for each of its 2,800 samples. The analysis was memoised in memory only, so every restart, new output or judgment made the page wait again.

- `src/stratamodels/evals/metrics.py`: a `dataset` metric may have `prepare`/`on`: verification reads embeddings and works out the similarity matrix once, a bootstrap sample only picks rows and columns. faces-bh 97 s -> 11 s, every value and interval the same as before.
- `src/stratamodels/evals/summaries.py`: **new**. Each experiment's analysis kept in `eval_summaries` with the signature of what it read (spec, dataset version, outputs count and newest id, judgments, the analysis code). Worked out in the background when a run finishes, a judgment comes in, a spec is saved, and at the service's start. The page gets the kept one; when it is out of date and took over 1.5 s it gets it marked `summary.stale` while a fresh one is worked out (one worker, requests coalesce).
- `src/stratamodels/evals/api.py`: uses it; `POST /api/evals/{name}/refresh` for outputs changed from outside the service.
- `src/stratamodels/db/models.py`, `migrations/20261110_5c1d2e7f9a30_eval_summaries.py`: table `eval_summaries`.
- `src/stratamodels/evals/similarity.py`: `phonetic` no longer crashes on a word its rules empty ("gh"), which made the describe-photos Judge tab answer 500.
- `web/`: the experiment page says when its results are being updated and checks every 3 s until they are.
- `src/stratamodels/evals/core.py` `analyse`: a candidate with no output at all is "not run yet" (`dataset.not_run`) and no longer holds every comparison at 0 items. describe-photos showed "No outputs yet" with 7 of 9 candidates done because the two qwen3.8-27b candidates had not run; now it compares the 7 on 60 items. The page marks such a candidate "Not run yet".
- `web/` Judge tab: after a vote the buttons no longer stay disabled while the results tab's analysis reloads (it refreshes behind), and a picture can be clicked to choose it.

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
