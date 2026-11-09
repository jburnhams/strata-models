# Web tests

vitest 4, Testing Library, user-event, msw (backend mocks). Run from `web/`: `npm run test:unit`, `npm run test:integration`, `npm run test:coverage:gate` (unit + integration together, the number CI gates on: floors in `web/coverage-floor.json`, which only ratchet up), `npm run test:coverage:all` (the same run without the gate), `npm run test:coverage` (unit only, informational), `npm run typecheck`, `npm run build`.

## Layout

- `unit/`: node environment, no DOM. Pure functions (`logic.test.ts`).
- `integration/`: jsdom. Components rendered with Testing Library against a mocked backend (`App.test.tsx`, `GreetingPage.test.tsx`, `SettingsPage.test.tsx`).
- `utils/`: `server.ts` is the one msw server and its default happy-path handlers: **every endpoint the app calls needs a handler there** (an unmocked request fails the test); `setup.ts` is the jsdom setup; `render.tsx` provides `setup()` (Testing Library `render` + `userEvent.setup()`).
- One test file per area of the app; add to the existing file if there is one.

## Conventions

- Test what a user sees: query by role, label or text; no class-name queries, no snapshots.
- One concept per `it`, named as a sentence. Cover the happy path, loading, empty, a server error and the edge values.
- No arbitrary sleeps: `findBy*`, `waitFor`, or fake timers.
- Render the real children and let msw answer their requests; mock a module only for something jsdom cannot run.
- A bug found while testing: write `it.todo` or `it.fails`, comment it as a finding, and don't change app behaviour in a test-only change.

## Coverage floors

`coverage-floor.json` holds the floors of `npm run test:coverage:gate` (lines, statements, functions, branches). After CI is green, raise them to the new totals; never lower them.

## Coverage status

| File | % Lines | Status | Notes |
|---|---|---|---|
| `src/App.tsx` | 100 | done | Covered by integration/App.test.tsx |
| `src/api.ts` | 100 | done | Covered by unit/logic.test.ts and integration tests |
| `src/route.ts` | 100 | done | Covered by unit/logic.test.ts and navigation tests |
| `src/components/GreetingPage.tsx` | 100 | done | Covered by integration/GreetingPage.test.tsx |
| `src/components/SettingsPage.tsx` | 100 | done | Covered by integration/SettingsPage.test.tsx |
| `src/components/ui.tsx` | 100 | done | Shared UI primitives |
