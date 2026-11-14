# The database

strata-models keeps everything it decides and does in its schema, `stratamodels`, of the shared `omarchy` PostgreSQL database, through the
`strata_python.db` (the `[db]` extra of [strata-python](https://github.com/jburnhams/strata-python), which replaced the git-pinned `omarchy-db` on
2026-11-14 with the same naming convention, so no revision drifts): models in `src/stratamodels/db/models.py`, Alembic revisions in `migrations/`
(`strata-db revision`, `strata-db check` on SQLite and Postgres), migrated by the service at start.

**Service environment.** A real Postgres is opt-in: the service unit sets `STRATA_DB_LIVE=1` and `STRATA_DB_DATABASE` (the shared database's
name; strata-python is public, so it has no built-in name). Without them the service refuses to start rather than guess. `STRATA_DB_URL` or
`STRATAMODELS_DATABASE_URL` point it anywhere else (a path means SQLite).

| Area | Tables | Filled by |
|---|---|---|
| Registry | `models`, `roles`, `model_roles` | `registry.seed` from `models.toml` at start; edits from `PUT /api/roles/{key}`, `PUT /api/models/{key}`, `strata-models role` |
| Config | `settings` | `settings.bind` at start (copies `~/.strata-models/settings.json` into an empty table); `PUT /api/settings`, `strata-models set` |
| Preferences | `preferences` | an app's own model and default params for a role: `PUT/DELETE /api/preferences/{app}/{role}`, `strata-models prefer` |
| Requests | `jobs`, `results`, `files` | the scheduler: every run and lease, each run's output (dropped after a week), the paths a job read or wrote (never the bytes) |
| Evals | `eval_datasets`, `eval_items`, `eval_experiments`, `eval_candidates`, `eval_results`, `eval_judgments` (people's, and model judges' with `judge` set), `eval_summaries` | spec TOMLs and dataset JSONLs imported when new or changed; `PUT /api/evals/{name}/spec`; runs and the Judge tab; `eval_summaries` worked out in the background after runs, judgments, spec edits and at start |

**Files are seeds.** `models.toml`, `evals/specs/*.toml` and the dataset `.jsonl` files are imported when they are new or changed. A row edited
here (`edited`) is no longer overwritten by its file: the choice made here stands. `strata-models seed --follow role:KEY` (or
`POST /api/registry/seed?follow=role:KEY`) hands a row back to its file. So a pull request that adds a model or an experiment still takes effect
at the next start.

**Never against the live database from a branch.** Tests pin `STRATAMODELS_DATABASE_URL` to a temporary SQLite file (`strata_env` in `pyproject.toml`), and strata-python's guard refuses a real Postgres during tests. The CLI
only creates the tables of an empty database and refuses one at another revision (`Store.connect(migrate='new')`); only the service migrates, at
start, so the live schema changes when matching code is deployed.
