# Progress

## 2026-11-09: first version
- faces: buffalo_l on the RTX 3060 loads in 0.7 s, 0.6 GB, 0.05 s per photo once warm (first call 0.4-1 s while CUDA kernels compile).
- transcribe: Whisper large-v3 float16 3.7 GB, 1.2 s warm for the 11 s JFK sample; small 0.6 GB, 0.3 s.
- onnxruntime-gpu 1.24+ is built for CUDA 13 (`libcublasLt.so.13`) while CTranslate2 needs CUDA 12: pinned onnxruntime-gpu 1.23.2 so one set of CUDA 12 wheels serves both. Its CUDA provider also needs cudart, curand and cufft; without them it falls back to the CPU silently (VRAM 0 MB is the tell).
- faster-whisper 1.2.1's decoder calls `av.open(metadata_errors=...)`, gone in PyAV 15+: audio is decoded with ffmpeg instead and passed as an array.
- The in-memory SQLite store shares one connection across threads: store calls are serialised by a lock (also harmless on Postgres).
- PgQueuer (the stack research's pick) was not used: one process owns the GPU, so the queue only needs a table the worker reads, and omarchy-db's models/migrations/test fixtures (SQLite and Postgres) cover it. Revisit if jobs ever need to run on a second machine.

## 2026-11-09: face-kind role
`face-kind` says whether a detected face is a real person or a statue, picture, toy or animal (runtime `face-kind`, `runtimes/facekind.py`): the face cut out with a 0.3 margin, embedded by CLIP ViT-B/32, then a logistic head trained by `scripts/face_kind_train.py` from the apps' labelled crops (the head lives in `~/models/face-kind/head-v1.npz`, not in git; without it the runtime falls back to zero-shot prompts). First head: 447 faces from strata-stills' catalogue labelled by hand (333 real, 44 statues, 24 pictures, 20 toys, 13 animals, 13 other), 5-fold real-or-not AUC 0.994, at 0.5 it keeps 98.2 % of real faces and drops 89.5 % of the rest; zero-shot 0.92; the detector's own score 0.69. bright-horizons' "not a face" labels (16 so far) can be added with a second `--crops/--labels` pair.

## 2026-11-09: faces `pad` view
A view can now pad the picture (`{pad: 2}`: centred on a mid-grey canvas twice its size, after any crop, before any turn). buffalo_l at det_size 640 misses faces that fill the frame: on strata-stills' pictures 1 in 5 faces over 20 % of the frame went unfound. On 37 pictures where a vision model saw a medium or large face and the detector did not, a pad 2 look found one in 59 % (a plain second look 16 %), with nothing lost on 60 that were already found. Geometry tests: `tests/unit/test_faces_views.py`.

## 2026-11-09: text to speech (role `speak`)
- strata-360 spoke its voice-over with Kokoro-82M v1.0 (kokoro-onnx, CPU), voice bm_lewis at speed 1.0 (Legends' saved choice); it also had an
  ElevenLabs path, switched off. Those are the baseline of eval `speak-narration`.
- Chatterbox (`chatterbox-tts` 0.1.7) pins torch 2.6 and Qwen3-TTS (`qwen-tts` 0.1.1) brings torch 2.15 (CUDA 13.2): neither can share the
  service's venv (torch 2.11, onnxruntime-gpu pinned for CUDA 12). New runtime `venv-worker`: a long-lived worker process per model in its own
  venv (`scripts/setup_worker_venv.sh`), JSON lines over stdin/stdout. The worker's GPU memory counts as the service's own (gpu.query treats
  child processes as ours).
- Chatterbox Turbo refuses a reference recording under 5 s. Qwen3-TTS clones closer with the reference's words (`voice_text`).
- The runner's voice for cloning is cut from Legends' race audio (wearer segments, `audio_clean.flac`): about 21 s, wind and breathing
  included. A clean 20-30 s recording read indoors would be a better reference.

## 2026-11-09: one CUDA per venv: Whisper in a worker venv, the service on CUDA 13
onnxruntime 1.24+ is built for CUDA 13 and CTranslate2 (4.8.5, the newest) for CUDA 12; both need cuDNN 9, whose CUDA 12 and CUDA 13 builds share the file name libcudnn.so.9, so one process cannot have both. Whisper now runs as a venv-worker (`workers/whisper.py`, venv `whisper` from `scripts/setup_worker_venv.sh whisper`, its own CUDA 12 wheels first on its library path), and the service's venv moved to onnxruntime-gpu 1.31 on torch's CUDA 13 wheels (plus nvidia-curand, which torch lacks and onnxruntime's CUDA provider needs). Measured on the RTX 3060 under a lease: faces 0.72 GB, 0.04 s a photo warm (as before); Whisper turbo in its worker 2.0 GB, 0.41 s for the 11 s sample warm. A clip is now decoded alone (ffmpeg -ss/-to), not the whole file. The worker loop is shared (`workers/protocol.py`), used by the TTS and Whisper workers.

## 2026-11-12: strata-react 1.0
- Web app moved from strata-react 0.1.16 to 1.0.20 (its UPGRADING.md). `DataTable` is now a React Aria `grid`, so tests query `grid`, not `table`.
- Judge tab: two pictures are a blind `Compare` (side by side, swipe, fade, blink, difference; one zoom and pan; full screen) named X and Y only. Answers are buttons and keys `[` (X), `]` (Y), `t` (about the same), shown in a `KeysLegend`: not `x`/`y` or digits, which `Compare` binds for its views. Clicking a picture no longer answers (a click there pans and zooms). Spoken outputs play in `MediaPlayer` (speech rates, no page keys, so two players do not fight over Space). A failed answer is an inline `Alert`.
- Outputs tab: candidates with a picture are one `CompareGrid` (same zoom for all, full screen, a pipeline's steps under its tile); failures and text stay cards. `j`/`k` step through items.
- Elsewhere: errors are `Alert`s, the reference picture is a `FileButton` (keyboard reachable), and `index.html` inlines `colorSchemeScript()` so dark mode does not flash light.

## 2026-11-12: even pace for narration (speak)
- Jonathan heard the speak-narration clips vary a lot in pace, with some long pauses. Measured on the 360 clips: the same text with the same
  params gives the same clip (seed 0), so the variation is line to line. Kokoro (a duration predictor) keeps each line within about +-10 % of
  its median rate. Chatterbox and Qwen3-TTS sample their timing token by token and spread about twice as much (CV 0.15 to 0.24), with
  short lines at half or double speed. Chatterbox leaves pauses of up to 3.9 s and about 0.5 s of silence at the ends.
- workers/pacing.py measures rate in syllables of the words as said (numbers and capital abbreviations read out), shown as words a minute
  at 1.4 syllables a word, so short and long words even out. Every clip now has pauses over max_pause (0.45 s) cut and 0.05 s kept at each
  end. `wpm` or `fit_s` sets a target: Kokoro re-speaks at the speed that lands on it, and sampled models make `tries` takes and keep the
  nearest. A pitch-keeping stretch (0.75 to 1.35x) then closes the rest, and `fit` says when it could not.
- Eval speak-pace (primary pace_spread, baseline unpaced kokoro-lewis) asks whether 160 wpm evens the lines without hurting them.

## 2026-11-12: video enhancement for digitised VHS
- The tape (H.264 10 Mb/s in MP4) is PAL 720x576 at 25 fps, SAR 787:720 (4:3 from the 702-pixel active line), coded
  interlaced; ffmpeg idet on 3 x 1000 frames: 100 % TFF. The file's own field order says nothing, so the worker forces it. Active picture: about
  4 px of black on the left, 10 on the right and 8 head-switching lines at the bottom: crop 692x568 at (14, 0), which is 4:3 at 1440 x 1080.
- The classical baseline (bwdif, hqdn3d, lanczos) runs at ~20 s per 8 s clip on the CPU. The worker's CPU-only engines were run end to end on
  a scratch instance (SQLite, scratch data and models folders): deinterlace-vhs 16 outputs, video-enhance-vhs's two ffmpeg candidates.
- FlashVSR (v1/v1.1) needs Block-Sparse-Attention built from source; FlashVSR+ (Oct 2026, same lab, Apache-2.0) has a Triton kernel and a
  PyTorch SDPA fallback, and output sides must be multiples of 128 (1536 x 1152 for 4:3). SeedVR2 runs through numz's standalone CLI
  (v2.5.24): 3B FP8 with block swap for the 12 GB card; FP8 is upcast on Ampere (the RTX 3060 has no FP8 units).
- VapourSynth R80 is installed on omarchy with only its core plugins: QTGMC needs havsfunc (or vsjetpack), mvtools, nnedi3/znedi3, fmtconv,
  misc, rgvs/zsmooth and a source filter; not attempted.
- Guards are ratios to the baseline: against plain lanczos (no GPU) peak VRAM cannot be a guard, so the spec guards run time (100x) and failures.
- VHS cleanup (2026-11-12, later): the chroma offset could not be measured. A correlation of chroma gradients against luma gradients
  (luma blurred to chroma's bandwidth, sub-pixel peak) on 7 clips found peaks of 0.02-0.15 that disagree from clip to clip: the colour is too
  soft for its edges to locate. vhs-chroma-shift therefore tries both directions. ffmpeg's nlmeans has no plane choice (it denoises colour
  too); atadenoise's `p` selects planes.

