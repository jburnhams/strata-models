# Changelog

Newest first, grouped by day. One entry per change, in the same commit or PR as the change.

## 2026-11-14

**A routing catalogue: every cloud backend and local model by role, power and cost (branch `claude/project-thread-2aatsx`)**
Jonathan asked for every cloud API mapped to capabilities, power and cost, so jobs can be routed between cloud and local models.

- `catalogue.toml` (**new**): the backends (local GPU and CPU lanes, agy, codex, Claude API, NVIDIA NIM, the free and paid Gemini keys) with Jonathan's order and use rules and their reserves; their models (roles, family, tier, price per Mtok or measured quota per call, the recorded models that stand for them in evals). NIM ids from its public model list.
- `routing.py` (**new**): `Catalogue`, eval evidence from `eval_summaries`, `Allowances` (agy and codex usage from their CLIs, the API keys from `stratamodels.spend` once it lands, cached 5 min), `options()` and `route()` (cheapest or best; private, tier floor, reserve and budget rules; `would_pick`).
- `server/app.py`: `GET /api/catalogue`, `GET /api/route/{role}`, `GET /api/allowances`. `cli.py`: `catalogue`, `route ROLE`.
- `tests/unit/test_routing.py` (**new**); server and CLI tests. `docs/routing.md` (**new**), README.

**Model judges for OCR, scored against the hand-typed text (branch `claude/project-thread-0n1ijc`)**
Jonathan asked whether qwen or Gemini could judge how good each OCR setting is. On the ten typed pages a judge can be checked against the character error rate, and one that agrees well can then rank settings on the pages nobody has typed (the rest of both books).

- `evals/judges.py`: `reference = false` keeps the item's reference from a judge (for a video pair, the tape's crops too: review) (in the judge's key only when hidden, so older judges keep their verdicts); `pairs = "baseline"` judges each candidate against the baseline only (12 pairs a page in ocr-pages, not 78); in an eval decided by a scored metric, each judge's verdicts are also scored against that metric (`metric` in the analysis: the output that scored better on the item is the right answer).
- `evals/report.py`: the Model judges section gives that agreement.
- `evals/specs/ocr-pages.toml`: judges qwen, gemma (local, judge-vision) and gemini-flash (agy, 40 pairs a run), reference hidden, baseline pairs.
- `tests/unit/test_eval_judges.py`: the hidden reference, baseline pairs and agreement with the metric. `docs/evals.md`: the options and the OCR judges.

**An OCR eval on hard book pages, and page cleaning before OCR (branch `claude/project-thread-0n1ijc`)**
Jonathan asked for evals of strata-scan's trickier pages to see what helps, and to look at cleanup like diary-study's, before and after OCR. Ten hard pages from the two Punch and Judy books, typed out by hand, measure Tesseract with each cleaning step and page mode, a vision-language model reading the page, and one correcting Tesseract's text.

- `runtimes/pageclean.py` (**new**, in pyright's strict list): seven named cleaning steps in Pillow and numpy (`flatten`, from diary-study's `remove_background_and_lighten_fold`; `contrast`, `sharpen`, `denoise`, `otsu`, `adaptive` (Sauvola), `upscale`).
- `runtimes/tesseract.py`: param `clean` (a list or "flatten,sharpen") runs those steps first, keeping the picture's dpi (doubled by `upscale`), and the result names them; `image` is accepted for `path`, so an eval item's picture reaches Tesseract and a vision-language model alike. A multipage picture is refused before cleaning (review: cleaning kept only its first frame).
- `models.toml`: `qwen3.5-9b-vl` and `gemma-4-12b-vl` also serve `ocr`, as eval candidates only (they give no boxes); `tesseract-eng` stays the role's model.
- `runtimes/presets/ocr-read-v1.md`, `ocr-correct-v1.md` (**new**): transcribe a page as printed; correct an OCR text against the page.
- `evals/specs/ocr-pages.toml` (**new**): 13 candidates (Tesseract with cleaning and psm; two vision-language readers; Tesseract then a corrector). `scripts/evals/ocr_pages.py` (**new**): builds the dataset from strata-scan or a folder of PNGs; `scripts/evals/ocr_pages/*.txt` (**new**): the ten references.
- `tests/unit/test_pageclean.py` (**new**): each step on a synthetic shaded page. `tests/integration/test_tesseract_runtime.py`: `image` and `clean` on the real command.
- `docs/evals.md`: the OCR eval. README, client docstring: `clean`.
- `evals/metrics.py`: `transcript` (what `cer` and `wer` read) takes an output's `text` when it has no `segments`; an `ocr` page or a vision-language answer scored 100 % wrong before. `tests/unit/test_eval_metrics.py`: that case.
- First numbers, the nine Tesseract candidates through a local service (CER, %): no cleaning 4.93, psm 4 3.48, flatten 3.60, denoise 3.83, adaptive 4.27, upscale 4.62, flatten+sharpen 2.87 (the report's pick: 2.06 points better, interval 0.28 to 6.93), flatten with psm 4 3.13, flatten+denoise with psm 4 3.04. Tried locally and left out: otsu 5.11, flatten then upscale 5.56. The vision-language candidates need the GPU.

**FlashVSR+ gets its Triton attention kernel (branch `claude/flashvsr-triton`)**
The first FlashVSR+ Turbo run (one clip, 2026-11-13) stopped before any frame: with neither Block-Sparse-Attention nor FlashVSR's own Triton kernel installed it falls back to PyTorch SDPA, which refuses a 1536 x 1152 clip ("would allocate too many mask entries"); the card had peaked at 4.2 GB. `scripts/setup_worker_venv.sh flashvsr-plus` now also installs flash-attn 2.7.4.post1 (the prebuilt wheel for cp310, torch 2.7, CUDA 12, C++11 ABI: no CUDA compile) and the repo's `operators/flashvsr_attn` (pure Triton), the backend FlashVSR+'s README recommends for consumer cards.

## 2026-11-13

**SeedVR2 decodes in 512 px tiles (branch `claude/seedvr2-decode-tiles`)**
The first SeedVR2 3B run on the RTX 3060 (one clip, 2026-11-13) finished its DiT pass (80 batches of 5 frames, ~6.5 min) and ran out of memory at the first VAE decode batch: 1024 px tiles took allocation from 8.8 to 10.7 GiB. `models.toml`: VAE encode and decode tiles 512 px with 64 px overlap.

**An `ocr` role on Tesseract, for strata-scan (branch `claude/project-thread-0n1ijc`)**
Jonathan is turning strata-scan into an app for scanned books (rotate, crop, text and pictures out) and asked for the model work to live here. A page needs its lines and words with boxes, which Tesseract gives on the CPU, so the role uses the CPU lane and never waits for the GPU; a vision-language reader is the eval to run against it later.

- `runtimes/tesseract.py` (**new**, in pyright's strict list): runs the `tesseract` command with TSV output (and `pdf`, a searchable PDF page, when a job asks), and turns it into lines of words with boxes and confidences; params `lang`, `psm`, `dpi`, `pdf`. `runtimes/__init__.py`: registered, and `available()` names a missing command.
- `runtimes/tesseract.py`: each job runs with `OMP_THREAD_LIMIT=1`. Tesseract's own threads fought the CPU lane's four jobs: a 360 dpi page took 80 s under load and 2 s with one thread each.
- `models.toml`: role `ocr`, model `tesseract-eng` (vram 0).
- `client`: `sm.ocr(path, app, pdf=...)`; README.
- `tests/unit/test_tesseract_runtime.py` (**new**): TSV parsing. `tests/integration/test_tesseract_runtime.py` (**new**): the real command on a page drawn with Pillow, and its failures (moved out of the unit suite after review: unit tests run no external processes). `tests/integration/test_service.py`: the role through the service and client.
- `runtimes/tesseract.py`: a picture with more than one page (a multipage TIFF) is refused, since Tesseract restarts its block and line numbers on each page and lines would merge across pages.
- `.github/workflows/integration.yml`: installs Tesseract (apt, brew, choco) so the tests run the real command. README: the role in the table.

**Samples can name their items; SwinIR once, on the face close-up, with 4 hours (branch `claude/swinir-one-clip`)**
SwinIR-L needs over 2 hours an 8 s clip on the RTX 3060 (two runs timed out, at 30 min then 2 h). Jonathan chose to see it once, on one hard clip.

- `evals/core.py`: a candidate's `items` may be a list of item ids (beside a count or another dataset's name): the candidate runs on those items, in the dataset's order, and is compared with the baseline on them.
- `evals/specs/video-enhance-vhs.toml`: swinir-real-x4 runs on 07-face-zoom-closeup only; `models.toml`: its job limit 4 h.

**A venv-worker model sets how long one job may run (branch `claude/video-worker-timeout`)**
SwinIR (video-enhance-vhs) failed on all 8 clips, about 6 hours of GPU time: each 8 s clip takes 35-45 min on the RTX 3060 and the venv-worker runtime stopped every job at its fixed 1800 s.

- `runtimes/worker.py`: models.toml `options.timeout_s` (default 1800, as before) is how long one job may run before the worker is stopped.
- `models.toml`: SwinIR 7200, SeedVR2 3B 10800, FlashVSR+ Turbo 7200 and Base 10800 (not yet measured here: generous), Real-ESRGAN x4plus, RealPLKSR and the 1x cleanup model 3600 (they took 14, 20 and 11 min a clip: too close to the old limit).

## 2026-11-12

**Video pairs play on their own, round and round (branch `claude/video-autoplay`)**
Jonathan: "when video compares load then autoplay them on full loop immediately".

- Judge tab: a video pair starts playing as soon as its videos can (muted, `audio="none"`: browsers block autoplay with sound), and loops the whole clip (0 to the shorter output's last frame) unless a range is set with `i`/`o`; `c` goes back to the whole clip. `CompareVideo` has no autoPlay yet (strata-react#203), so the page presses its own Play once (`useAutoPlay`); remove that when #203 ships. Someone with reduced motion set is left to press Play.
- Outputs tab: `VideoCompare` takes `autoPlay` (muted, once every video can play); it already loops.
- Judge tab: the answer buttons are above the pair as well as below it (groups "Answers above" and "Answers below"), for pictures, video and text alike: no scrolling past a tall frame to vote (Jonathan).

**Scheduler: models and leases above 10 GB get the whole card (branch `claude/project-thread-ecj9cv`)**
Jonathan: "above 10gb any model can run as long as it gets whole card reservation", so bigger models can be tried at their own risk instead of waiting forever or being refused.

- `scheduler.py`: a model job, a load from the Models page or a sized lease that needs more than `whole_card_gb`, or more than the card less the reserve, waits until no GPU job runs and no lease is held (nothing on the GPU passes it), frees every idle GPU model, ignores the reserve, and holds the card alone until it ends (the CPU lane carries on). An event says it runs on the whole GPU at its own risk. Only a lease bigger than the card itself fails; before, a lease above the card less the reserve failed and one that fit that but not beside other programs waited forever.
- `settings.py`, `PUT /api/settings`, `strata-models set`, Settings page: `whole_card_gb` (default 10, env `STRATAMODELS_WHOLE_CARD_GB`), applied at once.

**Sound: noise reduction moved here from strata-360, with evals scored by people, models and measures (branch `claude/project-thread-ecj9cv`)**
Jonathan asked to move strata-360's audio noise reduction into this service, look at other denoisers, and build evals on real audio (race speech from strata-360, a digitised tape's soundtrack), judged by programmatic measures and model judges as well as people.

- `src/stratamodels/workers/audio.py`: **new** venv-worker for role `denoise-audio`. strata-360's `dsp.clean_for_playback` (high-pass 80 Hz, DeepFilterNet3 capped at 18 dB, speech EQ and speechnorm, -14 LUFS with a limiter; its BS.1770 loudness) as one chain every engine runs, so candidates differ only in the denoiser: `none`, `afftdn`, `noisereduce`, `dfn3`, `clearvoice` (MossFormer2_SE_48K, MossFormerGAN_SE_16K, FRCRN_SE_16K) and `resemble` (Resemble Enhance, denoise or enhance). The cap (`atten_lim_db`) mixes back a share of what was taken out for engines without one of their own, as strata-360's MossFormer2 stand-in did. Every output is levelled to the same loudness, so no candidate wins by being louder.
- `src/stratamodels/workers/audio_quality.py`: **new** venv-worker for role `audio-quality` (CPU): DNSMOS P.835 (sig, bak, ovrl) and SQUIM objective estimates without a reference; SI-SDR, PESQ (wide band) and extended STOI against one, aligned first.
- `models.toml`: roles `denoise-audio` (deepfilternet3 with strata-360's settings as `options.defaults`), `audio-quality` (dnsmos-squim) and `judge-audio` (qwen3-omni-30b-a3b; gemma-4-e4b to trial); one venv per model family: `scripts/setup_worker_venv.sh dfn | clearvoice | resemble | audio | audio-quality` (dfn installs deepfilternet without its numpy pin and patches its torchaudio import, as strata-360 did).
- `evals/judges.py`: judge kind `audio`: the item's recording, then X's and Y's, as 16 kHz WAV; local through `runtimes/llama.py` (new `audio`/`audios` params, sent as `input_audio`), or Gemini through agy (`cloud.prompt_of` tells it to open the recordings with its file viewer; codex refused).
- `evals/core.py`: `[score] with = {param = "item.field"}` passes fields of the item to the scoring job (the clean reference of a mix). `evals/metrics.py`: `dnsmos_ovrl`, `dnsmos_sig`, `dnsmos_bak`, `squim_stoi`, `squim_pesq`, `squim_si_sdr`, `si_sdr`, `pesq`, `estoi`, `removed_db`.
- `evals/specs/denoise-audio.toml` (real recordings, judged by people against strata-360's chain; model judges Qwen3-Omni, Gemma 4 E4B and Gemini Flash; DNSMOS/SQUIM as context and a guard) and `denoise-audio-mix.toml` (clean speech mixed into real noise, decided by PESQ). `scripts/evals/denoise_audio.py` builds both datasets (private: they stay in the data folder).
- Dev dependencies: scipy and soundfile (the workers' loudness and file writing, tested without the model libraries).

**Video quality readings never fail a job (branch `claude/video-quality-optional`)**
A worker venv without numpy (or a reading ffmpeg cannot take) failed the whole job at its last step, after the video was made.

- `workers/video.py`: without numpy, `quality` keeps the ffmpeg readings (fidelity, flicker) and leaves out sharpness and noise, with `quality.skipped` saying why; any other error in the readings gives `quality = {skipped: "<error>"}` and the job still succeeds. The metrics read a missing reading as not measured (None), never as a failure.

**VHS cleanup: a stage before any upscaler, and one eval per cleanup problem (branch `claude/vhs-cleanup-evals`)**
Jonathan asked for VHS-specific noise removal, as separate evals for the different cleanup options.

- `workers/video.py`: params `cleanup = { chroma_shift = [cb_h, cb_v, cr_h, cr_v], filters = "...", model = "1x file", tile }`, the same for every engine. The shift and filters run on the deinterlaced, cropped picture in the tape's own pixels (before the resize); the 1x model (spandrel) runs on the prepared frames, then the engine (lanczos, an upscaler, or the prepared file a CLI engine reads). So the winners chain in front of any upscaler. `pipe_frames` puts frames through any list of spandrel models (cleanup, then upscaler) and kills both ffmpegs on a failure.
- `scripts/setup_worker_venv.sh video`: also the 1x models from OpenModelDB, checked by sha256: BleedOut Compact (WTFPL), BaldrickVHSFix v0.2 and BroadcastToStudio Lite (CC-BY-NC-SA-4.0).
- Cleanup `planes = "luma" | "chroma"`: the filters run on a copy and mergeplanes takes the other planes from the original, bit for bit (hqdn3d reads a strength of 0 as its default, so "hqdn3d=0:4:0:6" denoised luma too). A 1x cleanup model is GPU work: such candidates run on the new `video-lanczos-cleanup` (vram_gb 1.5), and a model registered with no VRAM (the CPU lane) keeps its cleanup model on the CPU. The seedvr2 and flashvsr-plus venvs also get spandrel, for a cleanup model before those engines.
- Measured beside people's verdicts: the worker returns `quality` (SSIM and PSNR of the result resized back against the tape with no cleanup, flicker against the tape's own, and on stills luma sharpness, luma and chroma noise), read by new metrics `fidelity_ssim`, `fidelity_psnr`, `flicker_ratio`, `sharpness`, `luma_noise`, `chroma_noise`. Each region's crop also gets a still (`region_stills`, default 3 s), and a vision judge of a video pair is shown, per region, the tape's crop, X's and Y's at the same moment (`judges._video_stills`); a video item's reference (the tape) is not shown as a "right answer".
- A VapourSynth stage before everything else in the worker, for what ffmpeg has no equal of: `deinterlace = "qtgmc"` (vsjetpack's QTempGaussMC) and `cleanup.vs = { degrain = { tr, thsad, blksize } }` (vsjetpack's mc_degrain, SMDegrain's motion-compensated temporal average, on the cleanup's `planes`). vspipe writes a lossless intermediate (FFV1, the audio copied) and the rest runs on it; quality is still measured against the tape as bwdif deinterlaces it. `scripts/setup_worker_venv.sh vapoursynth`: VapourSynth R82, vsjetpack and the plugins (mvtools, mvutensils (vsjetpack 3 calls `core.mvu`), znedi3, zsmooth, bestsource, bwdif, resize2, akarin, vszip) as PyPI wheels in a venv of their own, no system packages; then `vapoursynth config` (a pip VapourSynth finds its Python through `~/.config/vapoursynth/vapoursynth.toml`, keyed by the venv's own libvsscript, so a system install keeps its entry). Candidates: `qtgmc` in deinterlace-vhs, `smdegrain` and `smdegrain-strong` in vhs-grain.
- The five video specs (video-enhance-vhs, deinterlace-vhs, vhs-chroma-shift, vhs-chroma-cleanup, vhs-grain) list those metrics beside `judged` and three vision judges with criteria for each problem: qwen-vision (judge-vision's model), gemma-vision (gemma-4-12b-vl on trial) and gemini-flash (agy, 20 pairs a run), all `auto`. People's verdicts still decide.
- `evals/specs/vhs-chroma-shift.toml` (none, left 2 and 4 px, right 2, up 1 line), `vhs-chroma-cleanup.toml` (none, colour-only hqdn3d light and strong, BleedOut, BaldrickVHSFix, BroadcastToStudio) and `vhs-grain.toml` (none, luma hqdn3d light and medium, atadenoise, nlmeans; a motion-compensated denoise joins when its VapourSynth plugins are installed). Each judged blind against no cleanup, with the same lanczos after, so only the cleanup differs.

**Video enhancement: role `video-enhance`, VHS evals with crops, video in the Judge and Outputs tabs (branch `claude/project-thread-rea1ol`)**
Jonathan digitised a PAL VHS tape (720x576i TFF) and wants it upscaled and enhanced; the right model is to be chosen by eye, on crops where the differences show.

- `src/stratamodels/workers/video.py`: **new** venv-worker. Every engine gets the same preparation (bwdif or yadif send_field to 50p with the field order forced, the active picture cropped, 768 x 576 square pixels) and the same finish (lanczos to 1440 x 1080, x264, the source's audio), so candidates differ only in the enhancement. Engines: `ffmpeg` (classical, the baseline; optional `filters` such as hqdn3d), `spandrel` (per-frame models: Real-ESRGAN, RealPLKSR, SwinIR..., optional tiles), `seedvr2` (SeedVR2 3B FP8 through numz's standalone CLI, block swap) and `flashvsr` (official FlashVSR+, which falls back to PyTorch SDPA, so no Block-Sparse-Attention build). For judging, each `regions` box (source pixels) is also written as a crop of the result enlarged nearest-neighbour, and a few `stills` are kept as PNG. The peak VRAM of a CLI engine's processes is polled from nvidia-smi.
- `models.toml`: role `video-enhance` (video-lanczos until the eval decides) and 10 models. `scripts/setup_worker_venv.sh`: recipes `video` (torch + spandrel, and the five weights files), `seedvr2` and `flashvsr-plus` (the CLI repos pinned to a commit under `<models>/src/`, and their weights).
- `evals/similarity.py`: kind `video` (the stills' image similarity, frame for frame). `evals/api.py`: `GET /api/evals/{name}/media/{item}?file=` serves an item's reference previews (`reference.preview`, `reference.crops[].path`) and no other file.
- `scripts/evals/video_clips.py`: **new** dataset builder from a folder of clips and its private `clips.toml` (crop, field order, regions per clip); writes the source enlarged nearest-neighbour (whole frame and each region) into `<data>/evals/media/<dataset>/` as the item's reference.
- `evals/specs/video-enhance-vhs.toml` (judged against plain lanczos: hqdn3d, Real-ESRGAN x4plus/x2plus/general-v3, RealPLKSR, SwinIR, SeedVR2 3B, FlashVSR+ Turbo/Base; guards latency 100x and failures, since VRAM against a 0-VRAM baseline cannot be a ratio) and `evals/specs/deinterlace-vhs.toml` (bwdif against yadif; QTGMC waits for VapourSynth plugins).
- `web/`: strata-react 1.0.31. The Judge tab draws a video pair in strata-react's `CompareVideo` (strata-react#156-158: one clock, side by side, swipe, fade and blink, zoom locked, the item's regions as boxes in the cropped frame, nearest-neighbour pixels, frame stepping, a loop kept across pairs of the same item, speeds down to 0.1x), with the answers in its legend and footer and the tape itself below. The Outputs tab keeps `VideoCompare` (videos on one clock, with the worker's region crops and the source) until `CompareVideoGrid` (#159, merged) is released.

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
