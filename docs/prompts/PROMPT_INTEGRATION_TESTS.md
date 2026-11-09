# Task: add thorough integration tests (one PR, then stop)

You are a coding agent new to this repo. Find a coherent slice of **CLI or server behaviour that has no integration test**, cover it with thorough, best-practice end-to-end tests, and extend and refactor the shared fixtures as you go. Open **one PR**, drive CI to green, then stop. This prompt is run many times: each run continues from the state the last one left, so choose work from the *current* ledger and leave it up to date.

## Unattended operation (read first)

You run without a human: nobody will answer questions, approve a plan or review before you open the PR. Never ask for clarification or confirmation, never stop to wait for input, and never end your turn with a question. When something is ambiguous, pick the most reasonable option, note the decision in the PR description and carry on.

## 0. Orient yourself (read before writing anything)

1. Read `AGENTS.md` and `CLAUDE.md` at the repo root. They are authoritative. The web app (`web/`) is out of scope.
2. Read **`tests/README.md`** (shared commands, hermetic setup, ownership rules) and **`tests/integration/README.md`** (fixtures, conventions, coverage table, known issues) in full. **This task runs in parallel with the unit-test and web-test tasks on other branches, so obey the ownership table in `tests/README.md`**: you may edit `tests/integration/**`, new `tests/utils/integration_<topic>.py` files, and `tests/integration/README.md`. Don't touch `tests/unit/**`, `tests/conftest.py`, `tests/README.md`, `tests/unit/README.md`, `pyproject.toml`, the workflows or `web/`.
3. What integration means here: a test that needs a real file layout on disk, several modules together, a real process, or the CLI/HTTP boundary. Pure logic is a *unit* test (see `docs/prompts/PROMPT_BACKEND_UNIT_TESTS.md`); put it in `tests/unit/`, where it counts toward the coverage gate.
4. Setup: `pip install -r requirements-test.txt`. Commands from the repo root: `pytest tests/integration -n auto`, `pytest tests/integration -p randomly --randomly-seed=N`, `ruff check src tests`.
5. **GitHub Actions is the source of truth.** `integration.yml` runs this suite on Linux, macOS and Windows (all blocking), plus the web component tests.

## 1. Choose the slice (data-driven, so reruns don't collide)

1. Open the coverage table in `tests/integration/README.md` and cross-check it against a fresh run: `pytest tests/integration -n auto --cov --cov-fail-under=0 --cov-report=term-missing` (informational).
2. Check what's in flight: `git log --oneline -20`, `git branch -a`, open PRs. Don't pick what another branch or PR covers.
3. Pick **one cohesive slice**: aim for 15-30 tests across 1-3 files covering CLI commands, exit codes, persistence on disk, and HTTP API endpoints.
4. **Stop condition:** if the table shows nothing valuable left to test, change nothing, open no PR, say so in your final message, and don't invent work.

## 2. How to write the tests

- Use `cli` (in process) for commands, `cli_subprocess` for entry-point and process isolation tests, and `client` for the HTTP API over real on-disk state.
- Assert on **artifacts** (JSON written or returned, HTTP status and body, exit codes and error messages), not just printed text.
- Keep each test under ~10 s; no `time.sleep`; no fixed ports (use `TestClient`); parallel-safe (`-n auto`).
- Update `tests/integration/README.md` with every area your tests exercise.
