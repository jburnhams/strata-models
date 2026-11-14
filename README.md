# strata-models

One service on the machine with the GPU (omarchy, RTX 3060 12 GB) that every app uses for local models, instead of each app loading its own copy.

- **Roles, not files.** An app asks for `faces` or `transcribe`; [`models.toml`](src/stratamodels/models.toml) says which model serves each role, so the best model for a role is one edit. The service reads it from its database (`strata-models role ROLE MODEL`, or the API), with `models.toml` (and one in the data folder) as the seed: see [docs/database.md](docs/database.md).
- **One queue for the GPU.** Every job goes through one dispatcher: interactive before batch, jobs for an already loaded model first (but never ahead of a lease or past one that has waited two minutes). Jobs run on threads of their own: up to `gpu_jobs` (2) GPU jobs at once on different models when they fit in the VRAM budget together, and up to `cpu_jobs` (4) zero-VRAM jobs (recorded outputs, CPU models, cloud CLIs) beside them, which never wait for the card. Jobs and their history are in the shared Postgres (schema `stratamodels`).
- **A VRAM budget.** Models stay loaded until the room is needed, then the least recently used one is freed. A reserve (1 GB by default) is never filled, so the desktop keeps working.
- **GPU leases.** An app that runs its own CUDA code (strata-stills' develop) asks for a lease: the worker frees the memory, runs nothing else until the lease is given back, and expires it if the app stops sending heartbeats.
- **The whole card for big models.** A model or lease that needs more than `whole_card_gb` (10 GB), or more than the budget could ever give, does not share: it waits until the card is empty, every idle model is freed, the VRAM reserve is ignored and nothing else on the GPU starts until it ends (cloud and CPU jobs carry on). A way to try models too big to budget for, at their own risk; only a lease bigger than the card is refused.
- **Cloud APIs, on a budget.** Claude (`claude-api`), Google AI Studio (`gemini-api`: the free-tier key first, the paid key after) and NVIDIA NIM (`nvidia-nim`) serve the same params as the local vision-language models, so they stand in evals and jobs beside them. The keys live outside the repository in `~/.config/strata-models/secrets.env` (chmod 600, see [`keys.py`](src/stratamodels/keys.py)). No vendor shows a key's remaining credit to the key, so every call is a row of the ledger (`api_calls`: tokens, cost at the model's `price`, the request and the answer) and each paid key has a budget and a monthly rollover date (`strata-models set claude_budget_usd 100`, `set claude_rollover 2026-12-08`): a call whose worst case does not fit what is left is refused before it is sent ([`spend.py`](src/stratamodels/spend.py)). The Cloud page shows each key's budget, spend and calls per day, and every request.
- **A web app** (strata-react): what the GPU is doing now and where its memory is, every model and role, every job with its params, result and error, and GPU time per app.

## Use it from an app

```
pip install "strata-models-client @ git+https://github.com/jburnhams/strata-models#subdirectory=client"
```
```python
import strata_models_client as sm

sm.faces("/photos/a.jpg", app="bright-horizons")  # boxes, keypoints, 512-d ArcFace embeddings
sm.transcribe("/clips/a.wav", app="strata-360", language="en")  # segments with word timings
sm.ocr("/scans/p012.png", app="strata-scan")  # lines and words with boxes
with sm.gpu_lease("strata-stills", vram_gb=4, note="arena build"):
    ...  # the GPU is yours until the block ends
```
Standard library only; see [client/README.md](client/README.md). Inputs are file paths: the service runs on the same machine.

## Run it

```bash
uv sync --group gpu                  # .venv with the service, the test tools and the model runtimes (CPU-only machines: uv sync)
./strata-models doctor               # libraries, runtimes, GPU
dbtool new-app stratamodels          # once: the schema in the shared database (migrations run at start)
./strata-models serve                # http://127.0.0.1:8366/ (--memory: keep jobs in memory, no database)
./strata-models status | jobs | registry
```
Weights live in `~/models` (`./strata-models set models_dir ...`). The launcher puts the CUDA 12 libraries from the venv on the library path.

## Roles served now

| Role | Model | Measured on the RTX 3060 |
|---|---|---|
| `faces` | insightface `buffalo_l` (SCRFD, ArcFace, 3D landmarks) | 0.6 GB, 0.05 s a photo once loaded |
| `transcribe` | Whisper large-v3-turbo (faster-whisper, float16, in its own CUDA 12 worker venv) | 2.0 GB, 0.4 s for 11 s of speech |
| `transcribe-fast` | Whisper small | 0.6 GB, 0.3 s for 11 s |
| `denoise-audio` | DeepFilterNet3 capped at 18 dB with strata-360's speech finish and -14 LUFS (its `audio_clean`), in its own worker venv; MossFormer2, MossFormerGAN, FRCRN, Resemble Enhance, noisereduce and afftdn as eval candidates | not yet measured here |
| `ocr` | Tesseract 5 (`eng`, LSTM), on the CPU: lines and words with boxes, and a searchable PDF page on request; `clean` cleans the page first (`flatten`, `contrast`, `sharpen`, `denoise`, `otsu`, `adaptive`, `upscale`, see `runtimes/pageclean.py`). The vision-language models also serve `ocr`, for the `ocr-pages` eval only (no boxes) | CPU, about 1-3 s a book page |
| `audio-quality` | DNSMOS P.835 and torchaudio SQUIM (and SI-SDR, PESQ, STOI against a reference), on the CPU: an eval's `[score]` | CPU |

The other roles in `models.toml` (vision-language, sound events, speaker embeddings, people detection, upscale, denoise, NSFW) are listed with the apps that still load them themselves; the Models page shows that list.

## Tests

```bash
pytest tests/unit --cov              # scheduler, store, registry, API, CLI (fake runtime and fake GPU)
pytest tests/integration             # the real service process, driven through the client package
pytest tests/gpu -s                  # the real runtimes on the GPU (STRATAMODELS_TEST_IMAGE, STRATAMODELS_TEST_AUDIO)
cd web && npm run test:coverage:gate
```

Docs: [docs/overview.md](docs/overview.md) (design), [docs/progress.md](docs/progress.md) (findings), [AGENTS.md](AGENTS.md) (conventions).
