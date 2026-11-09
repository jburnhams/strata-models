# python-app-template

A full-stack Python + React project template extracted from [strata-360](https://github.com/jburnhams/strata-360) and [strata-stills](https://github.com/jburnhams/strata-stills). Includes a FastAPI backend and CLI (`src/app/`), a React 19 + TypeScript + Tailwind CSS v4 frontend (`web/`) that builds directly into the Python package, hermetic unit and integration test suites (pytest + hypothesis on Python, Vitest + Testing Library + MSW on web), ratcheting coverage gates, and multi-OS GitHub Actions CI.

## Quick start

```bash
scripts/setup_env.sh                 # macOS / Linux (or scripts\setup_env.ps1 on Windows)
./app doctor                         # check the environment
./app hello World                    # CLI hello world
./app serve                          # starts the FastAPI + React app on http://127.0.0.1:8360/
```

For hot-reload frontend development, run `./app serve` in one terminal and `cd web && npm run dev` in another (Vite proxies `/api` to `127.0.0.1:8360`).

## Using this template for a new project

1. Create a new repository from `python-app-template` on GitHub (or clone it).
2. Rename `src/app` and the `./app` / `app.cmd` launchers if you want a project-specific package name, and update `pyproject.toml` (`source = ["src/<pkg>"]`), `web/vite.config.ts` (`outDir`), and `APP_HOME` / `APP_DATA` in `src/<pkg>/settings.py`.
3. Replace the sample greeting routes (`src/app/greet.py`, `web/src/components/GreetingPage.tsx`) with your domain modules and components.

## Documentation

| Document | What it is |
|---|---|
| [docs/overview.md](docs/overview.md) | The design brief: goals, architecture, layout. |
| [docs/implementation-plan.md](docs/implementation-plan.md) | Milestones and what to build next, in order. |
| [docs/progress.md](docs/progress.md) | The log of what was tried, measured and decided. |
| [AGENTS.md](AGENTS.md) | Repository conventions and commands (for people and coding agents). |

## Tests

```bash
pip install -r requirements-test.txt
pytest tests/unit --cov                 # unit tests + coverage (CI gates on fail_under in pyproject.toml)
pytest tests/integration -n auto        # CLI and HTTP API end-to-end
ruff check src tests
cd web && npm run test:coverage:gate    # web unit + jsdom integration tests against coverage-floor.json
```

See [tests/README.md](tests/README.md) and [web/tests/README.md](web/tests/README.md) for the test layout, hermetic fixtures, and file-ownership conventions.
