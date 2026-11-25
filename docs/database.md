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
| Requests | `jobs`, `results`, `files` | the scheduler: every run and lease (a run with the load options and `settings_hash` it started with), each run's output (dropped after a week), the paths a job read or wrote (never the bytes) |
| Evals | `eval_datasets`, `eval_items`, `eval_experiments`, `eval_candidates`, `eval_results`, `eval_judgments` (people's, and model judges' with `judge` set), `eval_summaries` | spec TOMLs and dataset JSONLs imported when new or changed; `PUT /api/evals/{name}/spec`; runs and the Judge tab; `eval_summaries` worked out in the background after runs, judgments, spec edits and at start |
| Faces | `face_images`, `face_image_tags`, `face_tags`, `faces`, `face_embeddings` (pgvector `public.vector(512)`, HNSW cosine index), `identities`, `identity_app_metadata`, `face_batches`, `face_assignments`, `identity_not_same` | every app's indexed pictures (`POST /api/faces/images`), the people in them and the decisions about them: see [Faces](#faces) |

**Files are seeds.** `models.toml`, `evals/specs/*.toml` and the dataset `.jsonl` files are imported when they are new or changed. A row edited
here (`edited`) is no longer overwritten by its file: the choice made here stands. `strata-models seed --follow role:KEY` (or
`POST /api/registry/seed?follow=role:KEY`) hands a row back to its file. So a pull request that adds a model or an experiment still takes effect
at the next start.

**Never against the live database from a branch.** Tests pin `STRATAMODELS_DATABASE_URL` to a temporary SQLite file (`strata_env` in `pyproject.toml`), and strata-python's guard refuses a real Postgres during tests. The CLI
only creates the tables of an empty database and refuses one at another revision (`Store.connect(migrate='new')`); only the service migrates, at
start, so the live schema changes when matching code is deployed.

## Faces

One face store for every app (design, survey and decisions: the face database proposal of 2026-11-25). Code: `faces/store.py` (the store),
`faces/api.py` (the routes, listed in its docstring); apps call it through `strata_python.clients.faces`.

- **A picture** (`face_images`) is one per `app` + `ref` (the app's own id), with the path or URL of the original and `medium`, a 1600 px WebP
  copy kept under the data folder so a face can still be shown in its picture when the original is out of reach. Crops are cut on demand and
  cached (`faces/crops`), from the original when it is readable, else from the copy.
- **Tags** (`face_image_tags`): every picture carries its app's name and any tags the app gave. A tag marked `opt_in` in `face_tags` (a sensitive
  set, such as a nursery's) is left out of every search and clustering unless the caller names it in `include`.
- **Embeddings** (`face_embeddings`) are per face and model: indexing again with another model keeps the faces (matched by box overlap, IoU 0.5)
  and their decisions, and adds the new model's embedding. Searches compare only one model's embeddings.
- **People** (`identities`) are global; `identity_app_metadata` holds an app's own fields and its id for the person, which override the global
  fields when that app reads them. A merge points `merged_into` at the surviving person and is undone with an unmerge; nothing merges by itself:
  `merge-suggestions` ranks pairs whose centroids are close, minus pairs marked `identity_not_same`.
- **Decisions** (`face_assignments`) are append-only, grouped in `face_batches`; a face's newest decision whose batch is not undone wins, so any
  batch is undone in one step.
- **pgvector** lives in `public` and the service's connection searches only `stratamodels`, so the type, operator and operator class are
  schema-qualified (`strata_python.db.vectors`). SQLite (tests) keeps vectors as text and searches in Python.
