# Implementation plan

- [x] **M0 scaffold**: repository layout, `.venv` setup scripts (`scripts/setup_env.sh`, `scripts/setup_env.ps1`), `./app` and `app.cmd` launchers, `oslib.py`, `settings.py`, `doctor` CLI command.
- [x] **M1 hello-world backend & CLI**: `greet.py` domain module, `hello` and `messages` CLI commands, FastAPI server (`server/app.py`) with `landing.html` fallback.
- [x] **M2 React web app**: Vite + React 19 + Tailwind CSS v4 frontend (`web/`), hash router (`route.ts`), reverse-proxy-safe API client (`api.ts`), greeting and settings pages.
- [x] **M3 test suites & CI**: hermetic pytest unit and integration suites, Vitest unit (node) and integration (jsdom + msw) suites, coverage gates, and GitHub Actions workflows (`unit.yml`, `integration.yml`).
