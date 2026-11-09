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
