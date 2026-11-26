# Progress

## 2026-11-26: Faces screens
The web app has a Faces area (`/#/faces`) where a person makes the face store's decisions, built from strata-react's shared parts so any app
can show the same things on its own screens: People (everyone named, a search, each person's faces as crops), a face opened lands zoomed on it
in its photo with everyone beside it named (`BoxedPicture` + `BoxList` over the kept 1600 px copy), Unnamed groups one at a time (all faces
chosen; name them, add them to someone, or say they are not faces), Possible duplicates (merge into the one with more faces, or keep apart) and
Tags (which keep their pictures out of searches unless asked for). Every decision waits out an Undo window before it is sent, and is recorded
in the app name `strata-models`. Taking a face off a person records "not this person", so it does not come back to them in a group. strata-react
moved to 1.3.22, and the app now uses Strata's standard look (`strata.css`) instead of a palette of its own (the owner: this app is not special).

## 2026-11-25: one face store for every app
The faces of bright-horizons, strata-360, strata-stills and later strata-reels move into one store here, decided with the owner (one global
person per human, apps add their own fields; sensitive tags opt-in; merges suggested, never automatic; clustering and crops here; a 1600 px
copy of each picture kept). Built on strata-python's new `db.vectors` (pgvector column and nearest search), `images.crop` and
`clients.faces` (strata-python PR #81). Tables and the store are in [database.md](database.md#faces); apps migrate one by one, bright-horizons
first. Nothing is indexed yet: the live database gets the tables when this is deployed.

## 2026-11-24: shot-boundaries and scene-describe on a real tape
strata-reels' whole analysis ran on a real digitised family tape (PAL, 21 min, interlaced, torn restarts, a lost signal at the end). Every frame was scored and every candidate down to a scene score of 0.09 looked at by eye: 50 cuts, two of them where the signal comes and goes, one only six frames after another (under `min_shot_s`, so no setting with 0.6 s finds it).

| Setting | Tape: found of 50 | Tape: false | Spliced race files: found of 84 (files, far, near) | false |
|---|---|---|---|---|
| 0.3, as served before | 46 | 7 | 54 (26, 14, 14) | 0 |
| 0.3, lost signal found | 46 | 0 | 54 (26, 14, 14) | 0 |
| 0.25 | 47 | 0 | 66 (28, 20, 18) | 0 |
| **0.2** | 49 | 0 | 70 (28, 21, 21) | 0 |
| 0.15 | 49 | 8 | 74 (28, 22, 24) | 0 |
| 0.2, `min_shot_s` 0.2 | 50 | 0 | 70 (28, 21, 21) | 0 |

(Run in process from the branch, not yet through the evals; the spliced files are 28 cuts of each kind.)

- 0.2 serves now. The three cuts 0.3 missed scored 0.21 (a torn first frame), 0.28 and 0.25; the highest frame that is not a cut (fast motion over stripes, 0.19) is close under it, so 0.2 has no room to spare: another tape may want another number.
- The 7 false cuts at 0.3 are all in the last 7 s: rolling grey bars between black frames where the recording has ended. blackdetect calls none of it black for long. Frames that do not vary along their rows are no picture; shrunk to 16x12 grey the lost signal measured up to 1.1 (standard deviation along a row), the flattest real picture 5.8, and 2 was chosen: at 4, head-torch footage of night races (the spliced files) was taken for lost signal for a second at a time. Generated snow measured 2.8 a frame and 1.3 over the mean of five, hence the mean. A stretch that does not flash adds no cut, so pitch-dark footage is marked black and not cut out.
- Head-switching noise along the bottom lines, chroma flashes on a restart's first frames and fast pans over stripes made no false cut at 0.2.
- `scene-describe-v1` called all 40 real scenes "noisy" with "tracking lines", 37 "blurry" and 32 "poor": its own prompt said tapes often are. v2 says what each fault means, what a tape looks like anyway, and that the frames are of one scene the camera moves through. On 11 scenes: v1 listed tracking lines on all 11, v2 on none (the tape has none), and people in photographs no longer count as people.

## 2026-11-24: rival shot detectors
TransNet V2 (the PyTorch port with the authors' weights) and PySceneDetect's AdaptiveDetector and ContentDetector now run as one venv worker (`workers/shots.py`, venv `shots`) and answer in the baseline's shape, so eval `shots-spliced` can compare them and strata-reels could switch without a code change. To keep the comparison about the detector, every engine sees frames from the same ffmpeg pass (yadif first), gets the same black stretches, and has the baseline's minimum-shot rule applied afterwards (the libraries' own minimum lengths are off). TransNet V2 puts a hard cut's peak on the old shot's last frame (frames 49, 74 and 124 on the four-shot test video), so its cut is the frame after the peak. On that video all three find the cuts at 2, 3 and 5 s to the frame, on the CPU, TransNet V2 in 1.9 s for 7 s of video (cold) against the adaptive detector's 0.4 s. 
Scored on omarchy the same day (worker and runtime called directly, not through the eval runner; the machine was busy, so times are high). Tape: 50 cuts of one real 21-minute PAL VHS tape marked by eye; spliced: the 84 cuts of `shots-spliced` (28 per kind); both matched within 0.5 s.

| Detector | Tape found / false | s on the tape | Spliced found (files, far, near) / false |
|---|---|---|---|
| ffmpeg 0.3 | 46 / 7 | 137 | 54 (26, 14, 14) / 0 |
| ffmpeg 0.2, lost signal found (#124) | 49 / 0 | 113 | 70 (28, 21, 21) / 0 |
| ffmpeg 0.15, lost signal found | 49 / 8 | 116 | 74 (28, 22, 24) / 0 |
| TransNet V2 0.5 | 47 / 2 | 477 CPU, 134 GPU | 78 (28, 26, 24) / 1 |
| TransNet V2 0.3 | 49 / 7 | 480 CPU | 82 (28, 27, 27) / 2 |
| adaptive 3 | 43 / 1 | 119 | 64 (27, 20, 17) / 0 |
| adaptive 2 | 49 / 11 | 115 | 82 (28, 26, 28) / 2 |
| content 27 | 49 / 134 | 115 | 83 (28, 27, 28) / 84 |

The learned and adaptive detectors find far more of the spliced near and far cuts (TransNet V2 0.3 and adaptive 2: 82 of 84 against the tuned baseline's 70), but on the tape the tuned baseline is the only one with no false cut; TransNet V2's are fast pans, something passing the lens and the colour locking as the tape starts, and at 0.5 it missed two restarts in almost the same framing. The baseline's threshold was tuned on this tape, so its tape row flatters it, and the rivals have no search for lost signal yet (with #124's they would lose their false cuts in the tape's last 7 s). So the role stays on the retuned ffmpeg for tapes; one tape is too few to choose on, and eval `shots-vhs` (#124) is where these candidates go next. Two fixes from the run: the PySceneDetect engines' size probe failed on the spliced MP4s (ffprobe's csv listed the stream twice; the probe is JSON now, taking the first stream), and TransNet V2's `vram_gb` was 0.5 where torch reserved 1.47 GB on the tape (now 1.6).

## 2026-11-23: shot-boundaries role, for strata-reels
strata-reels (the video editor that will replace strata-360; family VHS tapes first) splits a tape into shots before anything else, and Jonathan's rule is that all its model work lives here. The role starts with the plain baseline: ffmpeg's `select` scene score, every frame against the one before, after yadif (one frame per frame, so an interlaced tape's two fields do not read as motion) at 160 px across, cut at 0.3, of cuts under 0.6 s apart the strongest; blackdetect's stretches come back beside the cuts. On a generated 7 s video (test pattern, black, fractal, bars) it finds the three cuts to the frame, in 0.16 s. Its eval does not need cuts marked by hand: Jonathan's idea was to splice uncut camera files (the race footage) so every cut is known, and to take cuts from within one file so both sides look alike (`scripts/evals/spliced_shots.py`, eval `shots-spliced`). On two generated camera files the baseline finds every `files` cut to the frame. Learned detectors (TransNetV2, PySceneDetect's adaptive detector) come next as venv workers, so their libraries never join this service's (Jonathan: no domain-specific libraries in strata-models).

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

## 2026-11-12: noise reduction from strata-360 (role `denoise-audio`)
- What strata-360 does today (`audio/dsp.py clean_for_playback`, stage `audio_clean`): high-pass 80 Hz, DeepFilterNet3 with
  `atten_lim_db=18` (it takes out at most 18 dB, so the crowd and the place stay and artefacts stay small), EQ -2 dB at 250 Hz and +2 dB at
  3 kHz, ffmpeg speechnorm, then -14 LUFS with alimiter. Recognition uses the raw audio: enhancement raised Whisper's word error rate there
  (0.16 to 0.19-0.22 with small), so this role is for listening, not for transcribing. On Windows DeepFilterNet needed a Rust build and
  MossFormer2_SE_48K stood in with the same cap.
- strata-360's own comparison on the Windows GPU (76 English segments, DNSMOS overall): original 2.10, its capped DeepFilterNet 2.47,
  MossFormer2 2.94, MossFormerGAN 3.04, FRCRN 3.02; but the others are uncapped and remove the place's sound, which DNSMOS rewards (bak).
  That is why `denoise-audio` is decided by people with the original heard first, and DNSMOS's speech score (sig) is only a guard.
- Other options looked at (2026-11): DPDFNet (streaming DeepFilterNet with a dual-path RNN, Dec 2025), GTCRN (tiny), MP-SENet, MaTSE,
  DriftSE: no head-to-head on real recordings found; worth adding as candidates when one is packaged. Generative restorers (Resemble
  Enhance's enhance mode) can change a voice: in the eval, but judged with that in mind.
- Model judges that listen: llama.cpp hears through the mmproj of Qwen3-Omni-30B-A3B (ggml-org GGUF, 18.6 GB Q4_K_M, experts partly on the
  CPU with `--fit`) and Gemma 4 E4B; Gemini through agy. All of them hear 16 kHz, so tape hiss above 8 kHz is for people.
- Measured on omarchy (10 s of Legends speech with white noise at 0 dB, under a lease): taken out / first run / VRAM: DeepFilterNet3 8.8 dB,
  4.7 s, 0.15 GB; MossFormerGAN 9.3 dB, 3.5 s, 0.13 GB; FRCRN 8.6 dB, 4.8 s, 0.17 GB; MossFormer2 7.0 dB, 1.8 s, 0.33 GB (26 s first load);
  Resemble Enhance (denoise) 5.0 dB, 2.3 s, 1.5 GB; noisereduce 8.6 dB, 0.7 s, CPU; afftdn 0.1 dB (barely anything, at any setting tried).
  Resemble Enhance fetches its weights with git-lfs (not installed): the worker downloads them with huggingface_hub into
  <models>/audio/resemble-enhance; ClearVoice wrote its checkpoints into the working folder: now <models>/audio/clearvoice.
- Audio judges on llama.cpp b12224 (audio input logged as experimental) fail a clear-cut pair (clean against the same with noise at -5 dB):
  Gemma 4 E4B answers X in both orders (1.5 s a verdict, 4.8 GB); Qwen3-Omni-30B-A3B says X and Y are both the input (12 s). Asked to rate
  one recording's noise 1-5, neither tells clean from noisy. The denoise-audio spec keeps Gemini (agy) as its only model judge; DNSMOS,
  SQUIM and people carry the rest. Retry the local ones on a newer llama.cpp.
- huggingface_hub is now a declared dependency: the llama-server and sd-cpp runtimes fetch weights with it, and the live venv only had it
  because it was installed by hand (a plain `uv sync` would have removed it).

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

