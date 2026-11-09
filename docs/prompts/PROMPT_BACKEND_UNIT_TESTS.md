# Task: add thorough backend unit tests (one PR, then stop)

You are a coding agent new to this repo. Find a coherent slice of **under-tested Python code**, cover it with thorough, best-practice unit tests, and extend and refactor the shared test helpers as you go. Open **one PR**, drive CI to green, then stop. This prompt is run many times: each run continues from the state the last one left, so choose work from the *current* coverage ledger and leave it up to date.

## Unattended operation (read first)

You run without a human: nobody will answer questions, approve a plan or review before you open the PR. Never ask for clarification or confirmation, never stop to wait for input, and never end your turn with a question. When something is ambiguous, pick the most reasonable option, note the decision in the PR description and carry on. Specifically:

- If the branch name, the slice, a convention or a tool is unclear, decide using this prompt, `AGENTS.md` and the READMEs; record the choice under "Decisions" in the PR.
- If a command fails or a tool is missing, work around it or rely on CI as the runner; do not stop. If you truly cannot make progress on part of the slice, drop that part, list it under "Not done" in the PR, and finish the rest.
- If you find a bug or something out of scope, record it under "Findings" in the PR (as an xfail where this prompt says so); don't ask whether to fix it.
- Whenever you have made changes, finish by opening the PR (or, if one already exists for your branch, pushing to it) and driving CI to green as described below; a PR with a clear "Not done" list beats no PR. If the stop condition applies and there is nothing to change, open no PR. Don't merge, and don't wait for a reviewer.

## 0. Orient yourself (read before writing anything)

1. Read `AGENTS.md` and `CLAUDE.md` at the repo root. They are authoritative. Web code (`web/`) is out of scope for this task.
2. Read **`tests/README.md`** (shared commands, hermetic setup, file-ownership rules) and **`tests/unit/README.md`** (unit conventions and the "Where things are tested" table, which is the ledger) in full. **This task runs in parallel with the integration-test and web-test tasks on other branches, so obey the ownership table in `tests/README.md`**: you may edit `tests/unit/**`, `tests/conftest.py`, `tests/unit/conftest.py`, `tests/utils/<topic>.py` (not `integration_*.py`), `tests/unit/README.md` and the `fail_under` line in `pyproject.toml`. Don't touch `tests/integration/**`, `tests/README.md`, `web/`, `requirements-test.txt` (unless a new test dependency is unavoidable, and say so in the PR) or the workflows.
3. Layout: application code is in `src/app/`. Unit tests are in `tests/unit/`, fast and hermetic (no external services, no network, no real home folder: enforced by the autouse fixtures in `tests/conftest.py`). Integration tests (`tests/integration/`) are out of scope except to avoid breaking them.
4. Setup and commands, from the repo root: `pip install -r requirements-test.txt`, then
   - `pytest tests/unit --cov --cov-report=term-missing --cov-report=xml` (coverage gate: `fail_under` in `pyproject.toml`)
   - `pytest tests/unit -n auto` (parallel), `pytest tests/unit -p randomly --randomly-seed=N` (reproduce an order)
   - `ruff check src tests`
   - `diff-cover coverage.xml --compare-branch origin/main` (coverage of just your changed lines)
5. **GitHub Actions is the source of truth.** `unit.yml` runs the unit suite with coverage and ruff on Python 3.11-3.13 on Linux plus macOS and Windows (all blocking), and the web checks. The PR is done only when CI is green on the head commit. Code must be cross-platform: build paths with `os.path`/`pathlib`, don't assume symlinks or POSIX permissions (skip with a reason), and put OS differences only in `oslib.py`.

## 1. Choose the slice (data-driven, so reruns don't collide)

1. Open the ledger in `tests/unit/README.md` and cross-check it against a fresh `pytest tests/unit --cov --cov-report=term-missing` (or the latest CI log). The report wins on conflict.
2. Check what's in flight: `git log --oneline -20`, `git branch -a`, and the open PRs. Don't pick modules another branch or PR already covers.
3. Pick **one cohesive slice**, about 2-5 related modules or 300-600 statements of missing coverage, ranked by logic density, pure code at 0-50%, and bug-prone areas (boundaries, error branches, concurrency, HTTP endpoints through `TestClient`).
4. State the chosen slice and why at the top of the PR description.
5. **Stop condition:** if every module is `done` (>= 90%, or a justified gap in the ledger), change nothing, open no PR, say so in your final message, and don't invent work.

## 2. Reuse and extend the shared helpers (don't bypass them)

- Use `tmp_path` and `monkeypatch`, not `tempfile.mkdtemp()` or assignment to module globals.
- If a helper is missing, add it in `tests/conftest.py` or `tests/utils/<topic>.py`. Anything two test files share moves into `tests/utils/`.
- Use `pytest-randomly` as a bug-finder: run your slice and the whole suite under several seeds (`--randomly-seed=1..5`) and with `-n 4`.

## 3. How to write the tests

- Test behaviour through the public function (or the HTTP endpoint via `TestClient`), asserting exact values or tight bounds. Cover the happy path, empty/missing input, malformed input, boundaries, each error branch, and idempotence/repeat calls.
- `@pytest.mark.parametrize` for tables of cases; classes to group related tests; names that read as a sentence.
- **hypothesis** for pure logic where an invariant is easy to state. Keep strategies bounded and fast.
- Don't change `src/` behaviour in a test-only PR. For a bug you find, write the test for the *correct* behaviour with `@pytest.mark.xfail(strict=True, reason='...')`, and list it under "Findings". Never skip, delete, or weaken an existing test, and never lower `fail_under`.

## 4. Update the ledger and the ratchet

- In `tests/unit/README.md`, update the "Where things are tested" rows for every module you touched.
- You are the only task that edits `fail_under`. After CI is green and the total coverage is known, raise `fail_under` in `pyproject.toml` to the floor of the new total, rounded down (it only ratchets up).
