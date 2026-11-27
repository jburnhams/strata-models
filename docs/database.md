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
| Faces | `face_images`, `face_image_tags`, `face_tags`, `faces`, `face_embeddings` (pgvector `public.vector(512)`, HNSW cosine index), `identities`, `identity_app_metadata`, `face_batches`, `face_assignments`, `face_hints`, `identity_not_same` | every app's indexed pictures (`POST /api/faces/images`), the people in them and the decisions about them: see [Faces](#faces) |

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
  Any format strata-python's `images` reads is a picture: JPEG, PNG, WebP, AVIF, TIFF, HEIC/HEIF, JPEG XL and camera raw files
  (developed by LibRaw), always turned by their EXIF orientation, so `width` and `height` and every box are in the upright picture's pixels.
- **Tags** (`face_image_tags`): every picture carries its app's name and any tags the app gave. A tag marked `opt_in` in `face_tags` (a sensitive
  set, such as a nursery's) is left out of every search and clustering unless the caller names it in `include`.
- **The bar** (owner, 2026-11-27): every face the detector finds is stored, but only faces at least `faces_min_score` (detector score,
  default 0.7) and `faces_min_px` (the shorter side of the box in the picture's pixels, default 40) are ever searched, clustered, reviewed,
  proposed or used for merge suggestions. Both are service settings (Settings page), read on every query, so a change applies at once and
  a lower bar brings stored faces back; no app filters tiny or doubtful faces itself. A clustering's `min_score` can only raise it.
  Faces a person already decided about stay on their person's page whatever the bar.
- **Sources:** every tag is a source (an app's name, or a tag it gave, such as a video eval's stills). `GET /api/identities?tag=` lists
  only people with faces there, counting those; `GET /api/identities/{id}/review?tag=` offers only that source's suggestions (scored
  against all the person's confirmed faces); `cluster`'s `tags` groups only its faces. The web app's Faces screens share one Source
  picker (`?source=`).
- **Deleting** (`DELETE /api/faces/images/{id}`, `POST /api/faces/images/delete {ids?, app?, tag?, dry_run}`): pictures go with their
  faces, embeddings, decisions and hints, and their kept copy when no other picture uses it (cached crops stay until the cache is cleared).
  The bulk form counts first (`dry_run` is the default) and says how many confirmed faces would go; it needs at least one of ids, app and
  tag, all of which must match. For pictures an eval or a trial put in, under their own app or tag.
- **Embeddings** (`face_embeddings`) are per face and model: indexing again with another model keeps the faces (matched by box overlap, IoU 0.5)
  and their decisions, and adds the new model's embedding. Searches compare only one model's embeddings.
- **People** (`identities`) are global; `identity_app_metadata` holds an app's own fields and its id for the person, which override the global
  fields when that app reads them. A merge points `merged_into` at the surviving person and is undone with an unmerge; nothing merges by itself:
  `merge-suggestions` ranks pairs whose centroids are close, minus pairs marked `identity_not_same`.
- **Decisions** (`face_assignments`) are append-only, grouped in `face_batches`; a face's newest decision whose batch is not undone wins, so any
  batch is undone in one step.
- **One global layer of decisions** (owner, 2026-11-26): a confirmation counts for every app, with its app kept as the source; there is no
  per-app confirmed layer.
- **Possible labels** (`face_hints`) are what an app knows before anyone picks a face: "these people are in this picture" (a Google Photos
  name tag: `face_id` empty) or "this face is this person" in the app's terms (bright-horizons' approved faces: `face_id` set). Each names the
  app's key for the person (`app_ref`, the same key as `identity_app_metadata.app_ref`) and how sure the app is (`possible`, or `confirmed`
  by a person in that app). Stored in `face_batches` of kind `hint`, so an import is undone in one step. Hints never decide a face.
- **Grouping a video's tracks** (`POST /api/faces/group-tracks`, `FaceStore.group_tracks`): an app sends a video's face tracks (from the
  `face-tracks` role: a key and the track's embedding each, and the pairs seen in the same frame as `apart`) and gets them grouped into
  people to name a group at a time: average linkage at the clustering threshold, never two tracks seen together in one group, each group
  with the known person most of the 10 confirmed faces nearest it (within 0.5, 60 %) are, as a suggestion. A track that names its
  stored stills (`faces`) and has none over the bar is answered in `hidden`, not grouped. Stateless: the app keeps its
  tracks and stores their stills as pictures when it wants them named (strata-reels does both).
- **Proposing** (`POST /api/faces/propose`, `FaceStore.propose`, planned in `faces/propose.py`) turns an app's hints into `suggested`
  decisions for a person to confirm: seeds are the faces that can only be one person (a face hint; the one face of a picture whose only
  hinted person is them; faces already confirmed for the global person their app_ref maps to), faces the detector scored under 0.7
  left out of everything; a person qualifies with 10+ seeds, 85 %+ of
  them within 0.5 of their centroid; in each of their hinted pictures the clearly closest face (within 0.5, 0.05 ahead of the next face and
  of any other hinted person) is suggested; with `groups`, an unnamed group mostly in one person's pictures is suggested whole. A person
  with no global person yet gets a new one named from the hint, mapped by app_ref. A dry run (the default) only counts; a real run is one
  `assign` batch (source `hint`), undone as one. Faces anyone already decided about are left alone, so a second run adds only what is new.
  Merge suggestions compare a person who has no confirmed face yet by their suggested faces.
  A person confirms suggestions on the person's page in the web app (Suggested: a page of 200 at a time, confirm the chosen or all, or
  "not this person", which records a rejection); `GET /api/identities` counts `suggested` beside `faces` (confirmed), and the Unnamed
  groups screen asks `cluster` with `suggested: false`, so a face waiting on a suggestion is not shown as unnamed.
  Thousands of suggestions for one person are reviewed in rounds (`GET /api/identities/{id}/review`): the most certain first, scored
  afresh on every call against the person's own confirmed faces once there are 3 (centre of their densest bunch), before that against
  the largest tight group (average linkage at 0.35) of their 2000 most confident suggestions; a suggestion nearer a face rejected for
  them than to that centre is left out. So each round confirmed makes the next one surer, with nothing to re-run.
  `min_likeness` (likeness = 1 - distance) returns every suggestion at least that alike at once (up to 10,000), and `counts` says how many
  there are at each likeness from 50 to 95 %; the web app switches to that once a person has more than 40 confirmed faces (default 70 %).
- **pgvector** lives in `public` and the service's connection searches only `stratamodels`, so the type, operator and operator class are
  schema-qualified (`strata_python.db.vectors`). SQLite (tests) keeps vectors as text and searches in Python.
