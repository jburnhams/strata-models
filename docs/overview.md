# Design brief

## Goal
Several apps on omarchy (bright-horizons, strata-360, strata-stills, strata-listory, machine-bench) loaded their own copies of the same models in their own venvs, with nothing deciding who uses the GPU when (two processes at once once corrupted strata-stills' arena build). strata-models is the one owner of the GPU: it picks the best model for each role, keeps the loaded ones warm within a VRAM budget, runs every app's jobs through one queue, and lends the whole GPU to apps that run their own kernels.

The inventory and proposal behind this (2026-11-09) are in the project's doc "Local models on omarchy: inventory and proposal"; the stack research recommended llama.cpp behind llama-swap for LLMs and VLMs, which is the plan for the `vision-language` role (a runtime that starts llama-server, through the same queue).

## Architecture
- `registry.py` + `models.toml`: roles -> models (runtime, name, VRAM estimate, licence, which apps use it). Models whose runtime is not implemented are listed as "not served yet".
- `scheduler.py`: one worker thread. Picks the next job (interactive first; loaded model first unless it would pass a lease or the oldest job has waited `max_skip_s`); makes room by freeing least recently used models; measures each model's real VRAM with nvidia-smi on load; grants leases (exclusive, heartbeat or expire).
- `store.py` / `db/models.py`: one `jobs` table (runs and leases), the queue and the history. omarchy-db migrations, run at start.
- `runtimes/`: `faces` (insightface), `worker` (venv-worker: a model in a venv of its own, run by a script in `workers/` over JSON lines: Whisper on CUDA 12, TTS), `fake` (tests). The service's own venv is CUDA 13 throughout.
- `gpu.py`: nvidia-smi parsing, processes with their command lines, an hour of samples.
- `server/app.py`: the API (see its docstring) and the built web app. Inputs are file paths.
- `client/`: what apps import (stdlib only), with fallbacks so an app still works without the service.
- `web/`: strata-react app (Overview, Models, Jobs, Usage, Settings).

## Principles
- Apps ask for roles; the service chooses models.
- One process touches the GPU for models; others take a lease.
- An app must keep working when the service is down: the client raises `Unavailable` (fall back to loading the model in-process) and `gpu_lease` yields None.
