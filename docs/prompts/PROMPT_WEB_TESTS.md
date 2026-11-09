# Task: add thorough front-end tests (one PR, then stop)

You are a coding agent new to this repo. Find a coherent slice of **untested front-end code**, cover it with thorough, best-practice tests, and extend and refactor the shared test helpers as you go. Open **one PR**, drive CI to green, then stop. This prompt is run many times: each run continues from the state the last one left, so choose work from the *current* coverage report and leave the notes up to date.

## Unattended operation (read first)

You run without a human: nobody will answer questions, approve a plan or review before you open the PR. Never ask for clarification or confirmation, never stop to wait for input, and never end your turn with a question. When something is ambiguous, pick the most reasonable option, note the decision in the PR description and carry on.

## 0. Orient yourself (read before writing anything)

1. Read `AGENTS.md` and `CLAUDE.md` at the repo root. They are authoritative. Python code is out of scope: don't touch `src/app/`, the Python `tests/`, `pyproject.toml` or the workflows. **This task runs in parallel with the backend unit-test and integration-test tasks on other branches**, which never touch `web/`; keep your changes inside `web/` so nothing collides. Your notes live only in `web/tests/README.md`.
2. Read **`web/tests/README.md`** in full, then everything in `web/tests/utils/` (`server.ts`: the one msw server and its default happy-path handlers; `setup.ts`: the jsdom setup; `render.tsx`: the Testing Library + user-event setup helper) and the existing tests in `unit/` and `integration/`.
3. Stack: React 19, TypeScript, Vite 7, vitest 4, Testing Library, user-event, msw v3. Source is in `web/src/` (`api.ts` is the typed client). The app calls the FastAPI backend through same-origin relative `/api/...` URLs via `withBase()`.
4. Test projects: `unit` runs in node (`tests/unit/**`); `integration` runs in jsdom (`tests/integration/**`, setup `utils/setup.ts`). Commands, from `web/`: `npm ci`, `npm run test:unit`, `npm run test:integration`, `npm run test:coverage:gate` (unit + integration together, what CI gates on; floors in `web/coverage-floor.json`), `npm run test:coverage:all` (no gate), `npm run test:coverage` (unit only, informational), `npm run typecheck`, `npm run build`.

## 1. Choose the slice (data-driven, so reruns don't collide)

1. Check the "Coverage status" table in `web/tests/README.md` against a fresh `npm run test:coverage:all`.
2. Pick **one cohesive slice** from the files marked `todo`/`partial`, ranked by logic density, interaction with the backend through `api.ts`, and user flows.
3. **Stop condition:** if every file is `done` (>= 90% lines, or a justified gap noted in the table), change nothing, open no PR, say so in your final message, and don't invent work.

## 2. How to write the tests

- Every endpoint the app calls needs a default happy-path handler in `utils/server.ts`; a request without a handler fails the test (`onUnhandledFrame: 'error'`). Per-test variations use `server.use(...)` inside the test. Never stub `fetch` by hand.
- **Behaviour, not implementation.** Query by role, label or text (`getByRole`, `findByText`). No class-name queries, no snapshots.
- Cover happy path, loading state, empty state, server error (4xx/5xx with `{ detail }`), edge values, and user interactions.
- Placement: pure/no-DOM -> `tests/unit/<name>.test.ts`; renders React or needs jsdom -> `tests/integration/<Name>.test.tsx`.
- Update `web/tests/README.md` and ratchet `web/coverage-floor.json` up (never lower a floor).
