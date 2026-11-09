# Progress log

## 2026-10-29: template extraction

- Extracted the shared Python + React full-stack structure from `strata-360` and `strata-stills` into `python-app-template`.
- Included cross-platform `oslib.py`, atomic JSON `settings.py`, FastAPI server with `landing.html` fallback and static asset mounting from `src/app/server/static/`, React 19 + Vite 7 + Tailwind v4 web app with `withBase()` reverse-proxy path resolution, hermetic pytest and Vitest suites with MSW v3, ratcheting coverage floors, and multi-OS GitHub Actions workflows (`unit.yml` and `integration.yml`).
