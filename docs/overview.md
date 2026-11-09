# Design brief

## 1. Goal

Provide a clean, minimal starting point for full-stack local-first Python + React applications with the exact build, test, coverage-gate and multi-OS CI setup proven in `strata-360` and `strata-stills`.

## 2. Architecture

- **Python CLI and HTTP server (`src/app/`)**:
  - `cli.py`: command-line interface (`doctor`, `hello`, `messages`, `set`, `serve`), invoked via `./app` (POSIX) or `app.cmd` (Windows) without needing to activate `.venv` manually.
  - `server/app.py`: FastAPI application serving JSON endpoints under `/api/*` and the built React single-page app at `/` (falling back to `server/landing.html` when `web/` has not been built yet).
  - `settings.py`: persistent user configuration in `~/.app/settings.json` (`APP_HOME` override) with atomic file writes.
  - `oslib.py`: cross-platform primitives (file locks, process liveness, process tree termination, priority, CLI command builder) isolating POSIX vs Windows differences in one place.
  - `greet.py`: minimal "Hello, world!" domain module with pure formatting functions and a file-locked JSON message log in the configured data directory.
- **React frontend (`web/`)**:
  - React 19, TypeScript, Vite 7, Tailwind CSS v4 (`@tailwindcss/vite`).
  - Hash-based routing (`web/src/route.ts`) and a typed `fetch` client (`web/src/api.ts`) with `withBase()` so the same build works at `/` or behind a reverse proxy at `/<subpath>/`.
  - `npm run build` outputs directly to `src/app/server/static/`.

## 3. Principles

- **Hermetic tests by default**: every test run gets an isolated home and data directory outside `tmp_path`, stripped environment overrides, and a socket guard that blocks outbound network calls.
- **Two-tier test suites**: fast unit tests (`tests/unit/`, `web/tests/unit/`) and end-to-end integration tests (`tests/integration/`, `web/tests/integration/`), with clear file ownership so parallel agent branches never conflict.
- **Ratcheting coverage floors**: Python unit coverage (`fail_under` in `pyproject.toml`) and web combined coverage (`web/coverage-floor.json`) only ratchet up.
