# Repository Guidelines

Template extracted from `strata-360` and `strata-stills`: Python backend (FastAPI + CLI) and React frontend (TypeScript + Vite + Tailwind CSS v4), with hermetic unit and integration test suites, coverage ratchets and multi-OS GitHub Actions CI.

- Python application code lives in `src/app/`; the React web app lives in `web/src/` and builds into `src/app/server/static/`, which `./app serve` serves at `/`.
- Python tests live under `tests/`, grouped by type:
  - `tests/unit/`: fast, isolated tests of pure logic and mocked boundaries (synthetic data, fakes). No external processes or network.
  - `tests/integration/`: end-to-end tests that run the CLI (in-process and as a subprocess) and the FastAPI HTTP API over real on-disk state.
  - `tests/utils/`: shared helpers (`fakes.py` for HTTP/subprocess doubles). `tests/conftest.py` holds shared fixtures and makes every suite hermetic (isolated home and data dir, no network).
  - Read `tests/README.md` before writing any Python test: fixtures, conventions, file-ownership rules and the per-module coverage ledger.
- Web tests live in `web/tests/{unit,integration,utils}` (vitest; unit runs in node, integration in jsdom). Backend calls are mocked with msw (`web/tests/utils/server.ts`: every endpoint the app calls needs a handler there); read `web/tests/README.md` before writing any.
- Commands (Python, from the repo root, with `.venv` active or via `./app` / `app.cmd`; `pip install -r requirements-test.txt` is enough for the test suites):
  - `pytest tests/unit --cov` runs the unit suite with coverage (random order via pytest-randomly; `-n auto` runs in parallel; `diff-cover` checks the lines a change touches).
  - `pytest tests/integration -n auto` runs the integration suite.
  - `ruff check src tests` lints for syntax errors and undefined names.
  - `./app doctor` checks the environment; `./app hello [name]` prints and records a greeting; `./app serve` runs the web server on port 8360.
- Commands (web, from `web/`): `npm install`, `npm run dev` (proxies `/api` to `./app serve` on 8360), `npm run build` (writes into `src/app/server/static/`), `npm run typecheck`, `npm run test:unit` (node), `npm run test:integration` (jsdom + msw), `npm run test:coverage:gate` (unit + integration together, what CI gates on: floors in `web/coverage-floor.json`), `npm run test:coverage` (unit only, informational), `npm run test:coverage:all` (unit + integration, no gate).
- Python coverage is calculated from the unit suite only (`fail_under` in `pyproject.toml`). Web coverage is gated on unit + integration together (`web/coverage-floor.json`, read by `web/vitest.gate.config.ts`). Both floors only ratchet up.
- Style: dense, comment-the-why, long lines are fine (ruff line-length 400), pytest style tests (`def test_...` with `assert`; use `monkeypatch`/`tmp_path` instead of patching globals by hand).
- Cross-platform (Linux, macOS, Windows): OS differences (file locks, process liveness and termination, priority, how to start the CLI) live in `src/app/oslib.py`; do not import `fcntl` or call `os.kill(pid, 0)` elsewhere (on Windows that terminates the process).
- Reverse-proxy support: the web build uses relative asset paths (`base: './'` in `vite.config.ts`) and `BASE` from the page URL (`web/src/api.ts` `withBase`): any URL the app builds for `fetch`, `<img>` or a link must go through `withBase`.
- CI is `.github/workflows/unit.yml` (lint, unit tests with coverage on Linux Python 3.11-3.13, macOS and Windows; web typecheck, coverage gate, build) and `integration.yml` (integration tests on Linux, macOS and Windows, web component tests, weekly). `scripts/setup_env.ps1` / `setup_env.sh` build the environment from scratch.
- Docs live in `docs/`: `overview.md` (design brief), `implementation-plan.md` (milestones and next steps), `progress.md` (log of findings and decisions), `prompts/` (unattended agent prompts for backend unit, integration and web tests).
