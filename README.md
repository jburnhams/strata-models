# strata-models

One service on the machine with the GPU (omarchy, RTX 3060 12 GB) that every app uses for local models, instead of each app loading its own copy.

- **Roles, not files.** An app asks for `faces` or `transcribe`; [`models.toml`](src/stratamodels/models.toml) says which model serves each role, so the best model for a role is one edit. The service reads it from its database (`strata-models role ROLE MODEL`, or the API), with `models.toml` (and one in the data folder) as the seed: see [docs/database.md](docs/database.md).
- **One queue for the GPU.** Every job goes through one dispatcher: priority `high` before `normal` (the default) before `low` (evals),  jobs for an already loaded model first (but never ahead of a lease or past one that has waited two minutes). Jobs run on threads of their own: up to `gpu_jobs` (2) GPU jobs at once on different models when they fit in the VRAM budget together, and up to `cpu_jobs` (4) zero-VRAM jobs (recorded outputs, CPU models, cloud CLIs) beside them, which never wait for the card. Jobs and their history are in the shared Postgres (schema `stratamodels`).
- **A VRAM budget.** Models stay loaded until the room is needed, then the least recently used one is freed. A reserve (1 GB by default) is never filled, so the desktop keeps working.
- **GPU leases.** An app that runs its own CUDA code (strata-stills' develop) asks for a lease: the worker frees the memory, runs nothing else until the lease is given back, and expires it if the app stops sending heartbeats.
- **Idle models give their memory back.** A loaded model no job has used for `idle_unload_min` (30 minutes; 0 keeps it until the room is needed) is unloaded; the next job loads it again.
- **The whole card for big models.** A model or lease that needs more than `whole_card_gb` (10 GB), or more than the budget could ever give, does not share: it waits until the card is empty, every idle model is freed, the VRAM reserve is ignored and nothing else on the GPU starts until it ends (cloud and CPU jobs carry on). A way to try models too big to budget for, at their own risk; only a lease bigger than the card is refused.
- **Cloud APIs, on a budget.** Claude (`claude-api`), Google AI Studio (`gemini-api`: the free-tier key, the paid key only for a model that names it) and NVIDIA NIM (`nvidia-nim`) serve the same params as the local vision-language models, so they stand in evals and jobs beside them. The keys live outside the repository in `~/.config/strata-models/secrets.env` (chmod 600, see [`keys.py`](src/stratamodels/keys.py)). No vendor shows a key's remaining credit to the key, so every call is a row of the ledger (`api_calls`: tokens, cost at the model's `price`, the request and the answer) and each paid key has a budget and a monthly rollover date (`strata-models set claude_budget_usd 100`, `set claude_rollover 2026-12-08`): a call whose worst case does not fit what is left is refused before it is sent ([`spend.py`](src/stratamodels/spend.py)). The Cloud page shows each key's budget, spend and calls per day, and every request. Every Claude job goes through the **Claude Message Batches API** at half price, however it was queued (`sm.submit_many`, `POST /api/jobs/bulk`, a normal or low `sm.submit`, an eval, a job a runtime asks for), with the same result as a call made at once, usually within minutes; the batcher sends what is marked every few seconds. The one exception is a single high-priority job (`POST /api/jobs`, someone waiting on it), sent at once ([`batches.py`](src/stratamodels/batches.py)). A call's `batch` flag is accepted and not read.
- **Cloud or local.** [`catalogue.toml`](src/stratamodels/catalogue.toml) maps every cloud backend (the agy and codex subscriptions, the Claude, NVIDIA NIM and Gemini keys) and this machine's GPU and CPU to roles, power (tier) and cost (GPU seconds, quota a call, dollars); `strata-models route ROLE` ranks them with the eval verdicts and what is left of each allowance: see [docs/routing.md](docs/routing.md).
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
Standard library only; see [client/README.md](client/README.md). The **face store** (one database of pictures, faces and people for every
app: `strata_python.clients.faces.index`, `search`, `cluster`, `assign`, ...) is described in [docs/database.md](docs/database.md#faces); the web app's **Faces** screens (`/#/faces`) are where a person names groups, merges duplicates and sets tags. Inputs are file paths: the service runs on the same machine.

## Run it

```bash
uv sync --group gpu                  # .venv with the service, the test tools and the model runtimes (CPU-only machines: uv sync)
./strata-models doctor               # libraries, runtimes, GPU
uv run strata-db new-app stratamodels  # once: the schema in the shared database (migrations run at start); the service needs STRATA_DB_LIVE=1 and STRATA_DB_DATABASE
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
| `shot-boundaries` | ffmpeg's scene score after yadif at 160 px (`runtimes/scenes.py`), on the CPU: the cuts between shots and the black stretches of a video (a tape's lost signal counts as black), for strata-reels; the baseline evals `shots-spliced` and `shots-vhs` measure other settings and the rivals against: TransNet V2 and PySceneDetect's adaptive and content detectors (`workers/shots.py`, venv `shots`) | CPU; on one real tape (eval `shots-vhs`, 21 min of PAL) 49 of 50 cuts and none false, 2 to 2.5 min, about a tenth of the tape's length on a busy machine |
| `face-tracks` | insightface `buffalo_l` behind a tracker (`runtimes/facetracks.py`): ffmpeg decodes at 10 fps, faces are joined into one track per person per shot by position and embedding, with ByteTrack's second pass for unsure faces, each with its span, boxes, a quality-weighted embedding and its 3 best stills (size, sharpness, pose); which tracks were seen together. For strata-360 and strata-reels; fps and tracker chosen by eval `face-tracks-home` (2026-11-27), still quality not yet | about 100 s for a 2-minute home video clip (0.8x real time; 720p to 4K) |
| `scene-describe` | Qwen3.5-9B (llama-server, the same residency as `vision-language`) with preset `scene-describe-v2` (or `-v3`, given the people the family named, which also says who does what): one scene of a home video from a few of its frames and its speech, what is seen (`observed`) kept apart from what is guessed (`inferred`, each with its evidence), people and children counted, tags and the picture's faults, for strata-reels. Claude Haiku is the other candidate | not yet measured |
| `chapter-name` | Qwen3.5-9B (llama-server, text only) with preset `chapter-name-v1`: a title, a summary and guesses with their evidence for a run of a home video's scenes, from their descriptions and speech, for strata-reels' chapters. Claude Haiku is the other candidate | not yet measured |
| `speaker-embedding` | ECAPA-TDNN on VoxCeleb (SpeechBrain, `workers/speaker.py`, venv `speaker`): one normalised 192-number vector per stretch of speech (`items: [{path, clip?}]`), for strata-voice to keep only the speaker's own sentences from a found recording | not yet measured |
| `image-kind` | What kind a picture from a book is (20 kinds: engraving, wood engraving, woodcut, halftone, hand-coloured, marbled pattern, binding, handwriting...) and its group (illustration, photo or other), and whether it is line art to vectorise: `rules` (numpy), `clip` (zero-shot, CPU) or any vision-language model with preset `image-kind-v1` (`runtimes/imagekind.py`); eval `image-kind` decides which, for strata-scan | rules and CLIP on the CPU |
| `vectorise` | A picture redrawn as an SVG over its own pixels (`runtimes/vectorise.py`): potrace (the command, GPL, as a separate program), stacked potrace tones, VTracer, VTracer's colour under potrace's ink, or a vision-language model drawing, steering, colouring or tidying; answers the SVG file and its rendering. Eval `vectorise-plates` decides which, for strata-scan's illustrations | CPU, under a second to a few seconds a plate (tracers) |
| `nsfw` | how sensitive a picture is: the probability (0-1) of nudity or sexual content, from Marqo's NSFW ViT classifier at 384 px (about 40 pictures a second); `paths` in, `{items: [{path, sensitive}]}` out | GPU, 0.3 GB |
| `audio-quality` | DNSMOS P.835 and torchaudio SQUIM (and SI-SDR, PESQ, STOI against a reference), on the CPU: an eval's `[score]` | CPU |

The other roles in `models.toml` (vision-language, sound events, people detection, upscale, denoise) are listed with the apps that still load them themselves; the Models page shows that list.

## Tests

```bash
pytest tests/unit --cov              # scheduler, store, registry, API, CLI (fake runtime and fake GPU)
pytest tests/integration             # the real service process, driven through the client package
pytest tests/gpu -s                  # the real runtimes on the GPU (STRATAMODELS_TEST_IMAGE, STRATAMODELS_TEST_AUDIO)
cd web && npm run test:coverage:gate
```

Docs: [docs/overview.md](docs/overview.md) (design), [docs/routing.md](docs/routing.md) (cloud or local), [docs/progress.md](docs/progress.md) (findings), [AGENTS.md](AGENTS.md) (conventions).
