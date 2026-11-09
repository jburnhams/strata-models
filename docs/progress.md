# Progress

## 2026-11-09: first version
- faces: buffalo_l on the RTX 3060 loads in 0.7 s, 0.6 GB, 0.05 s per photo once warm (first call 0.4-1 s while CUDA kernels compile).
- transcribe: Whisper large-v3 float16 3.7 GB, 1.2 s warm for the 11 s JFK sample; small 0.6 GB, 0.3 s.
- onnxruntime-gpu 1.24+ is built for CUDA 13 (`libcublasLt.so.13`) while CTranslate2 needs CUDA 12: pinned onnxruntime-gpu 1.23.2 so one set of CUDA 12 wheels serves both. Its CUDA provider also needs cudart, curand and cufft; without them it falls back to the CPU silently (VRAM 0 MB is the tell).
- faster-whisper 1.2.1's decoder calls `av.open(metadata_errors=...)`, gone in PyAV 15+: audio is decoded with ffmpeg instead and passed as an array.
- The in-memory SQLite store shares one connection across threads: store calls are serialised by a lock (also harmless on Postgres).
- PgQueuer (the stack research's pick) was not used: one process owns the GPU, so the queue only needs a table the worker reads, and omarchy-db's models/migrations/test fixtures (SQLite and Postgres) cover it. Revisit if jobs ever need to run on a second machine.

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
