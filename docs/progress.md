# Progress

## 2026-11-09: first version
- faces: buffalo_l on the RTX 3060 loads in 0.7 s, 0.6 GB, 0.05 s per photo once warm (first call 0.4-1 s while CUDA kernels compile).
- transcribe: Whisper large-v3 float16 3.7 GB, 1.2 s warm for the 11 s JFK sample; small 0.6 GB, 0.3 s.
- onnxruntime-gpu 1.24+ is built for CUDA 13 (`libcublasLt.so.13`) while CTranslate2 needs CUDA 12: pinned onnxruntime-gpu 1.23.2 so one set of CUDA 12 wheels serves both. Its CUDA provider also needs cudart, curand and cufft; without them it falls back to the CPU silently (VRAM 0 MB is the tell).
- faster-whisper 1.2.1's decoder calls `av.open(metadata_errors=...)`, gone in PyAV 15+: audio is decoded with ffmpeg instead and passed as an array.
- The in-memory SQLite store shares one connection across threads: store calls are serialised by a lock (also harmless on Postgres).
- PgQueuer (the stack research's pick) was not used: one process owns the GPU, so the queue only needs a table the worker reads, and omarchy-db's models/migrations/test fixtures (SQLite and Postgres) cover it. Revisit if jobs ever need to run on a second machine.

## 2026-11-09: faces `pad` view
A view can now pad the picture (`{pad: 2}`: centred on a mid-grey canvas twice its size, after any crop, before any turn). buffalo_l at det_size 640 misses faces that fill the frame: on strata-stills' pictures 1 in 5 faces over 20 % of the frame went unfound. On 37 pictures where a vision model saw a medium or large face and the detector did not, a pad 2 look found one in 59 % (a plain second look 16 %), with nothing lost on 60 that were already found. Geometry tests: `tests/unit/test_faces_views.py`.
