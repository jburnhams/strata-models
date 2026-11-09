# Implementation plan

- [x] **M1 service**: registry, queue with VRAM budget, leases, Postgres history, faces + Whisper runtimes, client, CLI.
- [x] **M2 web app** on strata-react: Overview, Models, Jobs, Usage, Settings.
- [ ] **M3 on omarchy**: systemd user unit, Caddy path `/strata-models/`, docs in the docs repo.
- [ ] **M4 apps**: bright-horizons face ingest on `faces`; strata-stills GPU work behind `gpu_lease`; strata-360 speech, lyrics and faces on the service.
- [ ] **M5 more roles**: `vision-language` (llama-server runtime, Qwen3.5-9B now, Gemma 4 12B / Qwen3.8-27B to compare), `detect-people` (ultralytics), `sound-events`, `speaker-embedding`, `upscale` / `denoise` (spandrel), `nsfw`.
- [ ] **M6 bake-offs**: per role, on omarchy's own data, the result recorded in models.toml.
