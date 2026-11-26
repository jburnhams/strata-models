# Implementation plan

- [x] **M1 service**: registry, queue with VRAM budget, leases, Postgres history, faces + Whisper runtimes, client, CLI.
- [x] **M2 web app** on strata-react: Overview, Models, Jobs, Usage, Settings.
- [ ] **M3 on omarchy**: systemd user unit, Caddy path `/strata-models/`, docs in the docs repo.
- [ ] **M4 apps**: bright-horizons face ingest on `faces`; strata-stills GPU work behind `gpu_lease`; strata-360 speech, lyrics and faces on the service.
- [ ] **M5 more roles**: `vision-language` (llama-server runtime, Qwen3.5-9B now, Gemma 4 12B / Qwen3.8-27B to compare), `detect-people` (ultralytics), `sound-events`, `speaker-embedding` (served 2026-11-26: ECAPA in worker venv `speaker`, for strata-voice), `upscale` / `denoise` (spandrel), `nsfw`.
- [ ] **M5b roles for strata-reels** (the video editor; its plan is strata-reels `docs/implementation-plan.md`): `shot-boundaries` (ffmpeg scene-score baseline served, eval `shots-spliced` on spliced race footage with known cuts; TransNetV2 and PySceneDetect added as candidates (venv worker `shots`, 2026-11-24), to be scored on omarchy before the role's model is chosen); `cut-points` (subjective cuts: where to cut inside one shot, scored on whether a cut breaks a spoken phrase or interrupts a movement, and judged blind A/B by people in the Judge tab); then, as reels needs them, face recognition across tapes, speaker diarization, word alignment, sound events and translation, each with an eval.
- [ ] **M6 bake-offs**: per role, on omarchy's own data, the result recorded in models.toml.
