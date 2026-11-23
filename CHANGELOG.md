# Changelog

Newest first, grouped by day. One entry per change, in the same commit or PR as the change.

## 2026-11-23

**Page layout default: Opus regroups Tesseract's lines when a page is rebuilt (branch `claude/project-thread-dvu0cz`)**
The page-layout eval's pick (opus-fix-v2, 2026-11-15: block_f1 96.5 % on 20 pages against Tesseract's 64 %, $0.033 a page) was never made the default, so an app sending a plain OCR answer still got Tesseract's blocks.
- `runtimes/pagebuild.py`: a model whose options name a layout model (`group`, with `group_preset` and `group_tokens`) asks it for the `groups` of a plain `ocr` answer (the numbered lines and the picture), then draws as with given groups. `group = false` in a job draws Tesseract's blocks; a failed call (budget, network, no blocks, no picture) does too, as does an answer that fits the preset's schema but not the page (line numbers out of range, a malformed figure box: Codex review), and the answer's `grouper` says why. The answer carries the call's `cost_usd`.
- `models.toml`: model `pagebuild-opus-layout` (`group = "claude-opus-5-5-api"`, preset page-layout-fix-v2) is the `page-build` role's model. `pagebuild` stays, drawing only: every eval candidate names it, so the page-layout and page-rebuild baselines and runs are unchanged.
- Tests in `test_pagebuild.py`; `docs/evals.md` (Page rebuild).

**Jobs record the model settings they ran with; the client returns them (issue #103, branch `claude/project-thread-3e79cp`)**
strata-reels keeps the model, version and config on every inferred item and keys its cache on them. The job row had only the model key, which stays the same when `models.toml` changes its options (`ffmpeg-scene-score`'s `threshold`), and `run()` dropped which model served the job.
- `jobs.options` and `jobs.settings_hash` (**migration** `d32954a09201`): set when a run starts (`Scheduler.dispatch`), from the model as it runs: its load options (`models.toml`'s merged with the job's own variant `options`) and `registry.settings_hash` (12 hex digits of sha256 over runtime, model file and options). Recorded at start, not submit, so an edit while the job waits counts. Leases and older rows keep null. The job row in the API carries both.
- `GET /api/registry`: every model and role has `settings_hash` (a role: its model's).
- `client/strata_models_client`: `run_with_provenance(...)` returns `(result, {model, options, settings_hash, job_id})`; `provenance(job_row)` reads the same from a row; `role_settings(role)` returns `{model, settings_hash}`. New names, so `run` and `role_model` callers are untouched. strata-python's `clients.models` has the same calls, typed (strata-python PR #49).
- Tests: `test_scheduler.py` (recorded at start, changes with options under the same key, variants, leases), `test_registry.py` (what the hash covers), `test_service.py` (the client against the real service, on the shot-boundaries role).

**Shot-boundaries role: ffmpeg scene-score baseline, cut metrics and the `shots-spliced` eval (branch `claude/project-thread-1p0er6`)**
strata-reels, the new video editor, needs its tapes split into shots, and its model work belongs here. This serves the plain baseline on the CPU and sets up the eval that learned detectors will be measured by; no new dependencies.
- `runtimes/scenes.py`: **new** runtime `ffmpeg-scenes`: per-frame scene scores from ffmpeg (yadif, 160 px), cuts at a threshold with a minimum shot length, black stretches, an optional clip and every frame's score on request. Answer times are the file's seconds.
- `runtimes/__init__.py`: registered, with no Python needs; `available()` reports a missing ffmpeg. `models.toml`: role `shot-boundaries`, model `ffmpeg-scene-score` (threshold 0.3, min_shot_s 0.6, width 160, deinterlace on). `pyproject.toml`: the runtime is in the strict pyright list.
- `evals/metrics.py`: `cut_miss`, `cut_false`, `cut_f1` (cuts matched one to one within 0.5 s or the item's `tolerance_s`), and `match_cuts`. After review: the matching is the largest one (an in-order walk), not nearest pairs first, which could strand two cuts that both had an answer in reach; a failed job scores F1 0 on a one-shot item too. The runtime drops black stretches shorter than `black_min_s`, which blackdetect's metadata does not (one still black when the clip ends is kept: it may go on).
- `evals/specs/shots-spliced.toml`: **new** eval: the baseline against threshold 0.2 and 0.4 and 320 px; primary `cut_f1`.
- `scripts/evals/spliced_shots.py`: **new** dataset builder. Jonathan asked for known cuts instead of hand-marked ones, so it splices stretches of uncut camera files (the race footage) into videos at exact frame counts: items of `files`, `far` and `near` cuts (`tags.kind`), the near ones a jump of 2 to 12 s within one file, which looks almost the same on both sides. This replaces a first builder that read hand-marked tape cuts from a `shots.toml` (not kept).
- `client/strata_models_client`: `shots(path, app, ...)` shortcut; `client/README.md` lists it. `README.md`: the role in the served table.
- Tests: `tests/unit/test_scenes_runtime.py` and `tests/unit/test_spliced_shots_builder.py` **new**, cut metrics in `test_eval_metrics.py`; `tests/integration/test_scenes_runtime.py` and `tests/integration/test_spliced_shots.py` **new** (real ffmpeg: generated video, and two generated camera files spliced, whose cuts the runtime finds to the frame; paths compared by parts, so the builder tests pass on Windows) and a client call in `test_service.py`. `.github/workflows/integration.yml` installs ffmpeg beside Tesseract.
- Docs: `docs/evals.md` (Shots section, and subjective cuts as a later eval), `docs/implementation-plan.md` (M5b: learned detectors as venv workers, subjective cuts), `docs/progress.md`.


**An eval page never hangs on its first analysis, and shows where a run stands (branch `claude/project-thread-dvu0cz`)**
Jonathan: the page-rebuild page hung on open. Its first analysis worked out while the request waited, and page_ink_all and page_ink_f1 each binarised every scan (seconds a page, for every candidate).
- `evals/summaries.py`: an experiment with no kept analysis (or a quick one) is worked out beside the request, which waits QUICK_S; past that the page gets a stub marked `pending` with what the analysis is doing (`progress`: step, done, total), and the same work is joined, not started again, when it asks again. Errors still answer the request.
- `evals/core.py`: `analyse(..., progress=)` says which candidate it is scoring or comparing.
- `evals/metrics.py`: one binarised scan a page (`_page_binarised`, cached) for both page ink metrics; masks are made per call, so only the ink is held.
- `evals/api.py`: `GET /api/evals/{name}/progress`: per candidate its outputs in and failed and, for a pipeline, how many items have each step; the queue's eval jobs on the dataset's items (running and queued, model, item, how long); the run started here; the analysis step. It works for runs started from the CLI too.
- Web: a Progress panel on the experiment page (outputs per candidate, pipeline steps, the jobs running and queued with their items and times, the analysis step), polled while there is work; the Results and Write-up tabs say the results are being worked out instead of waiting (the write-up is fetched again once the analysis is in, keyed by when it was worked out); progress counts a sampled candidate over its sample. Static build updated.
- Tests in `test_eval_api.py` and `web/tests/integration/Evals.test.tsx`.

**End-to-end page rebuild eval: layout, corrected text, fitted type, pictures and borders (branch `claude/project-thread-dvu0cz`)**
Jonathan asked for the full page rebuild once the stage evals had winners. Spec `page-rebuild` is a ladder over the page-layout pages, Tesseract up to the full recipe, so the report says what each step is worth; `full-every-face` tries every face instead of Sonnet's shortlist (the font-match run's ink winner, wrong class half the time), for the judge to settle (docs/evals.md "Page rebuild").
- `runtimes/pagebuild.py`: with `ocr` and `groups`, `texts` (a model's corrections of the numbered lines; an empty text drops a line), `font_fit` (a font-match answer: each word in its fitted face, cut, size and place; the answer's font files drawn with), `styles` (fitted styles to keep), `snap` (sizes near the block's median take it, widths kept), `crop` (figures cut from the scan, paper whitened) and `rules` (the scan's ruled lines near the print, the border among them, cut from the scan as figures of kind rule).
- `pyproject.toml`: scipy moves from the dev group to the runtime dependencies (page-build's `rules`; already installed wherever `uv sync` ran with the dev group, so a deploy's `uv sync` changes nothing there).
- Preset `page-text-fix-v1` (+ schema): Haiku corrects Tesseract's numbered lines one by one, answering only those it changes.
- `evals/core.py`: a stage may set `answer = "json"` to hand on its whole answer (font-match's) instead of its text; only kept in the stage when given, so no key changes. Candidates sharing one stage job each apply their own `answer` and `into` to its answer (Codex review). A stage's cost and `cost_usd` count a runtime's own cloud call (font-match's `proposer.cost_usd`).
- `evals/metrics.py`: `page_ink_all`, the rebuilt page's ink against the scan's over the whole printed area, border included.
- `scripts/evals/ocr_reference.py --pages DIR` reads local pictures (the page-layout pages, into `DIR/text/`); `page_layout.py` adds a reading as the item's `reference.text`, for the rebuild's cer_exact.
- Tests in `test_pagebuild.py`, `test_eval_pipeline.py`, `test_page_metrics.py`, `test_ocr_reference.py`.

**Cull a candidate from blind judging (branch `claude/project-thread-dvu0cz`)**
A candidate with `judge = false` is in no new pair on the Judge tab or for a model judge; its outputs, metrics and judgments made already still count (`evals/core.py`, `evals/api.py`, `evals/judges.py`, docs/evals.md "Judged experiments"). page-layout culls `sonnet-layout` (lost all 12 of its pairs) and `gemma-layout` (2 wins, 4 losses, block_f1 12 %), so judging time goes to the candidates still in contention. Test in `test_eval_api.py`.

**The item's own picture beside picture outputs (branch `claude/project-thread-dvu0cz`)**
After #93 the page-layout Judge tab showed the rebuilt pages but not the scan they rebuild, which the judge needs. `/api/evals/{name}/media/{item_id}` now serves the item's `image` input when it has no `path` (still only a file the dataset names); the Judge tab shows it above a pair of pictures (open by default), and the Outputs tab puts it first in the grid as "Source". Tests in `test_eval_api.py` and `web/tests/integration/Evals.test.tsx`; static build updated.

**Evals API: an experiment that cannot be read is a 400 everywhere (branch `claude/project-thread-01gj43`)**
The font-match page on omarchy broke because its dataset was never built into the service's data folder (the trial ran in a scratch clone): `/api/evals/font-match` said so (400), but `/report`, `/outputs` and `/pair` raised the same `EvalError` as a bare 500. `evals/api.py` now turns any `EvalError` into a 400 with the reason.

**Page-build pictures show on the eval tabs (branch `claude/project-thread-dvu0cz`)**
The page-layout Judge tab said "Picture not found" for every pair: page-build writes its PDFs and rasters to `<data>/pages`, which `/api/images/file` did not serve (it served `generated`, `uploads` and the eval voices only). It now serves `<data>/pages` too; test in `test_server.py`.

**font-match: Claude Sonnet proposes the shortlist (branch `claude/project-thread-01gj43`)**
Jonathan chose Sonnet from the 2026-11-15 trial (class right on 75 % of real blocks against 22 % for every face, 8 s and about a cent a block).
- `runtimes/fontfit.py`: the model's `propose` option (a registry model key) is asked for the shortlist (preset `font-propose-v1`, the block cut out when `box` is given) when a job gives no candidates; loaded once in-process, its spend counted by its own runtime. A failed call falls back to screening every face; the answer's `proposer` gives the model, seconds and cost or the error. `propose = false` turns it off.
- `models.toml`: `fontfit` options `propose = "claude-sonnet-5-5-api"`, `propose_tokens = 1500`. Spec `font-match`: `exhaustive` sets `propose = false`.

## 2026-11-15

**font-match: fonts from Fontsource, 17 more period faces, Claude shortlists (branch `claude/project-thread-01gj43`)**
Jonathan (2026-11-15): prefer Claude models (far more budget there); Fontsource as one list with one fetch method.
- `runtimes/fontfit.py`: `fetch` reads Fontsource's JSON API (`api.fontsource.org/v1/fonts/<id>`): versioned latin-subset TrueType files (kerning and ligatures kept) instead of parsing Google's CSS; a face in `fonts.toml` may give its files' URLs instead. Cuts the metadata lists but the CDN lacks (a bold italic) are left out; only a missing roman fails.
- `runtimes/fonts.toml`: 57 → 74 faces. From Fontsource: Besley (Clarendon), Goudy Bookletter 1911, Castoro, Source Serif 4, Newsreader, Unna, Antic Didone, Bigshot One, Bevan, Chunk Five, Wellfleet, Vast Shadow, Grenze Gotisch, Petit Formal Script. From CTAN (GUST licence): TeX Gyre Schola (Century Schoolbook: the Scotch/Century gap), TeX Gyre Bonum (Bookman, Old Style Antique 1858), Latin Modern (Modern Extended 1896). Preset `font-propose-v1` lists them.
- Spec `font-match`: `haiku-fit` on every block, `sonnet-fit` and `opus-fit` (new), a Sonnet judge; Claude shortlists get max_tokens 1500 (Haiku ran out at 600 once).
- Trial (omarchy, 2026-11-15, dataset 5cbdb78d2f8d, 36 real / 29 synthetic). Real, class right: Flash 81 %, Sonnet 75 %, Opus 72 %, Haiku 56 %, every face 22 %; ink_f1 0.84 Sonnet and Opus, 0.81 Flash and Haiku, 0.86 every face, 0.58 naive. Synthetic, exact face top-1: every face 83 %, Opus 79 %, Sonnet 72 %, Haiku 66 %, Flash 52 %. Seconds a block: Sonnet 7.9, Haiku 8.6, Opus 9.1, Flash 27, every face 26. Claude spend $1.58 for all three over 65 blocks. Of the new faces TeX Gyre Schola (Century) matters: Sonnet's most frequent pick on real blocks (9 of 36); Latin Modern and Bonum never won.
**The paid Gemini key is a deliberate last resort (branch `claude/project-thread-gl1u9l`)**
A font-match trial sent 62 of 64 `gemini-3.8-flash-api` calls to the paid key (about $0.38). Each came after a free-tier 429 for the daily quota ("requests per day", limit 20 for gemini-3.8-flash), which the runtime turned into a silent fall-over to the paid key inside the same call.

Now:
- A `gemini-api` model uses the free key alone, unless its `keys` name the paid one. Argon keeps `keys = ["gemini"]`.
- A free-tier 429 or 503 that names a wait of a minute or less is waited out and tried again on the free key, up to three times.
- A daily-quota 429 is remembered per model until midnight Pacific, so later calls fail at once, without sending a request.
- When the free tier refuses a call, the error names agy, or a model with the paid key, as the way forward.
- `routing` ranks a Gemini model by its first key, with the free key as the default.

**FlashVSR+ Turbo fits its memory cap again (branch `claude/project-thread-rea1ol`)**
Since the 16 GB cap (#62) every Turbo job was OOM-killed at about 2 minutes. Measured on the title card under `systemd-run --scope -p MemoryMax=16G`: decode-every 12 killed at 16 GB, 8 completes (program memory ~14.4 GB at each segment's decode, 10.8 GB VRAM, 484 s a clip, as fast as before). `video-flashvsr-plus-turbo`: decode-every 8, memory_max_gb 20.

## 2026-11-15

**page-layout: preset page-layout-fix-v2 for Claude (branch `claude/project-thread-dvu0cz`)**
sonnet-fix (85.5 %) and haiku-fix (83.7 %) trailed flash-fix (95.5 %) by over-splitting: a play's speeches, a footnote's notes and a title page's byline each became several blocks, and a smudged running head was dropped. `page-layout-fix-v2` (**new**, same schema) says to split only by space or type size and to keep badly read lines; candidates `haiku-fix-v2`, `sonnet-fix-v2`, `opus-fix-v2` (Claude first: the larger budget). Tuned on these ten pages, so it needs pages it was not tuned on before it is adopted.

**page-layout: ten more labelled pages, held out from the prompt tuning (branch `claude/project-thread-dvu0cz`)**
`scripts/evals/page_layout/pages.toml` has ten more pages (Ward 1874 dialogue, songs and advertisements; Collier 1828 prose, verse, footnotes and a play), `set = "check"`: labelled after `page-layout-fix-v2` was written, so they test it on pages it was not tuned on (items carry the `set` tag: `tune` or `check`). They were grouped from Tesseract's numbered lines and fitted to the ink, so the fix approach's ceiling there is 99.8 % (97.6 % on the first ten) and the comparison between fix candidates is the fair one. Tesseract alone scores block_f1 53 % on them (77 % on the first ten). Rebuild the dataset on omarchy (`.venv/bin/python scripts/evals/page_layout.py`); only the new items run.

## 2026-11-14

**The tape's horizontal smear: a cleanup step and the vhs-smear eval (branch `claude/vhs-smear`)**
High-contrast edges (handwriting on the title card) carry a bright trail to their right, which the upscalers then enlarge. Measured on the tape it is one-sided and the same on every line: an overshoot about 2 px after each dark-to-light edge (+27 levels on average over its first 5 px), then a faint trail.

- `workers/video.py`: the VapourSynth stage's `cleanup.vs` also takes `desmear` (`desmear_rows`, numpy in a ModifyFrame on luma: a Wiener filter that maps the line's fitted impulse response to a plain Gaussian, so the overshoot goes and the noise is not raised; an optional exponential `trail` term) and `dehalo` (vsjetpack's `dehalo_alpha` or `fine_dehalo`), run in the order desmear, dehalo, degrain, right after the deinterlace.
- `scripts/evals/smear_kernel.py` (**new**): fits the impulse response from a high-contrast region of a clip (two-level ideal rows, least squares, smoothed tail) and prints the TOML.
- `evals/specs/vhs-smear.toml`: none, de-ring soft (overshoot +27 -> +4, strokes lighter, less noise), de-ring sharp (+27 -> +9, strokes keep more contrast) and dehalo_alpha (+27 -> +24: made for short halos), each with bwdif and the same lanczos after. The trail term was tried and left out (it drew dark lines along rows); fine_dehalo needs the edgemasks plugin, not installed.
- Review fixes: `desmear_rows` pads both ends of each row (edge replication) before the FFT, so a row whose ends differ no longer rings from its right end into its start, and normalises the filter to unit gain at zero frequency (lam dimmed flat areas by 1 / (1 + lam): about 2.7 levels on the title card). Re-measured: de-ring soft overshoot +27 -> +5, sharp +27 -> +10, mean brightness within 0.2 of none.

**Whole VHS pipelines judged against each other (branch `claude/project-thread-rea1ol`, on top of the refine branch)**
Every stage's eval has its shortlist; stages interact, so the winners are now combined into whole pipelines and judged head to head.

- `evals/core.py`, `api.py`, `judges.py`: `[decision] pairs = "all"` makes a judged eval's Judge tab (and a judge's `pairs = "all"`) offer every pair, not only pairs with the baseline. The decision against the baseline still uses the pairs with it; the write-up's Bradley-Terry strengths rank the rest.
- `workers/video.py`: `cleanup.vs.planes` gives the VapourSynth degrain its own planes, so one job can run SMDegrain on luma and hqdn3d on chroma; `refine.against = "cleaned"` holds a refinement to the tape after the job's own cleanup, so back-projecting a denoised pipeline's upscale does not pull the grain back in.
- `evals/specs/video-pipeline-vhs.toml`: P1 QTGMC + lanczos; P2 QTGMC + SMDegrain + chroma hqdn3d + Real-ESRGAN x2; P3 the same with FlashVSR+ Turbo (baseline); P4 P3 back-projected at half strength; P5 without the grain step, with and without back-projection. Judges: qwen and Gemini (gemma, which tied nearly every pair, dropped).

**Refining an upscale toward the tape: candidates made from other candidates' outputs (branch `claude/video-refine`)**
FlashVSR+ Turbo won every judged pair against lanczos but invents detail (Jonathan: "bad on the handwriting"), and its fidelity to the tape is the lowest of the candidates.

- `evals/core.py`: a candidate's `from = [...]` names other candidates; its job params get their output paths as `sources`. It runs after them, waits while one is missing or failed (`waiting` in the result), and its result key includes theirs.
- `workers/video.py`: engine `refine` (numpy and Pillow, CPU): `backproject` (correct by the difference between the tape and the result shrunk back, `iterations` x `strength`), `blend` (weight of the first source over the second), `fallback` (the second source where the first contradicts the tape locally). Measured on the title card: Turbo SSIM 0.938, one pass at half strength 0.970, full strength 0.990 (and back to lanczos's softness, with the tape's noise), a 60/40 blend with Real-ESRGAN x2 0.957.
- `models.toml`: `video-refine` (vram 0). `video-enhance-vhs`: candidates turbo+backproject, turbo-blend-x2, turbo-fallback-x2.
- Review fixes: the refine engine fails the job when the tape or a source decoder stopped with an error (a decoder that errors part-way otherwise looked like a short clip); `forget(candidate=X)` also forgets the outputs of candidates made from X (a rerun gives X a new output under the same key, and its refinements would have stayed cached).

**font-match: the first trial of the shortlists, and two fixes it found (branch `claude/project-thread-01gj43`)**
Trial on omarchy (scripts/evals/font_match_trial.py, 36 real and 29 synthetic blocks, 2026-11-14). Real blocks: class right 19 % trying every face, 56 % with qwen's shortlist then the fit, 69 % gemma, 78 % Gemini Flash; ink_f1 0.85 / 0.82 / 0.82 / 0.83 (naive 0.58). Synthetic (face in the catalogue): every face wins, exact face 76 % (top 3 83 %) against Flash's 45 %, gemma 28 %, qwen 31 %. A shortlist decides the class; the fit adds about 0.1 ink_f1 and the size. Seconds a block: every face 19, gemma 6.5, qwen 8, Flash 22 (about $0.006 a call if charged).
- `fetch` wrote every download to `<face>.ttf.part`, so parallel fits fetching the same face collided (52 failed fits at 6 jobs): each writer now has its own temporary name.
- `parse_candidates` takes the closest id for a near miss (gemma wrote `libra-bodoni` for libre-bodoni in 26 of 65 answers).

**page-layout: Tesseract's lines regrouped by a vision model (branch `claude/project-thread-dvu0cz`)**
- `runtimes/pagebuild.py`: `list_lines` (Tesseract's lines numbered, in thousandths, split at wide gaps between words) and `groups` (a model's blocks named by those numbers, drawn with Tesseract's measured lines).
- `evals/core.py`: a stage's `also` passes item fields as they are beside the one sent (the page image beside a text); kept out of the parsed stage when not given, so existing pipelines keep their result keys.
- `runtimes/llama.py`, `runtimes/cloud.py`: a preset's `{context}` takes a text as it is (a dict is still JSON).
- Preset `page-layout-fix-v1` (**new**, with a schema); candidates `qwen-fix`, `flash-fix`, `haiku-fix`, `sonnet-fix` in `evals/specs/page-layout.toml`. The labels' own grouping of Tesseract's lines scores block_f1 97.6 %, the approach's ceiling; OCR's best preprocessing (psm 4, flatten) finds fewer blocks than psm 3, so psm 3 stays.

**page-layout: Sonnet as a candidate (branch `claude/project-thread-dvu0cz`)**
Haiku found 24 % of the labelled blocks (Flash 83 %, Tesseract 77 %); `sonnet-layout` asks whether the larger Claude reads page layouts well enough. Only it runs on the next `eval run page-layout` (the others are cached).

**cost_usd counts a pipeline's cloud steps (branch `claude/project-thread-dvu0cz`)**
The page-layout run on omarchy (2026-11-14) showed cost_usd 0 for every candidate, though Gemini Flash's and Haiku's layout steps cost $0.088 and $0.011: the metric read only the last job's output (page-build, local). Each step in a pipeline's trace now keeps its answer's `cost_usd` (or the mark of a subscription CLI or recording), `cost_usd` adds the steps to the last job, and `outputs_for` fills the cost of steps saved before this from their own cached rows, so the existing run's report is right without running anything again. `runtimes/fake.py`'s text model takes a `cost_usd` param for the test (`test_eval_pipeline.py`).

**Page rebuild: the `page-layout` eval, ten labelled pages and the page ink score (#67, branch `claude/project-thread-dvu0cz`)**
- `runtimes/pagebuild.py`: a third input, `layout` (a vision model's reading, preset `page-layout-v1`: boxes in thousandths of the page, blocks in reading order, lines without boxes, figures), drawn in the picture's pixels; `page`, `ocr` and `layout` each also come as JSON text (a model's answer, code fences allowed) or a JSON file's path. The raster is put back on the scan's own size (pdfium rounds a page's size in points up, so a 2660 px scan at 500 dpi came back 2661 wide and matched none of its ink), and the answer names it as `path` too, for the Judge and Outputs tabs.
- `evals/metrics.py`: `page_ink_f1`, the rebuilt page's raster against the scan within the reference text blocks (evals/ink.py).
- Preset `page-layout-v1` (**new**, with a schema); `scripts/evals/page_layout.py` and `scripts/evals/page_layout/pages.toml` (**new**: ten pages of Ward 1874 and Collier 1828 from archive.org, layout labelled by Claude, checked by eye); spec `evals/specs/page-layout.toml` (**new**: Tesseract's blocks against qwen, gemma, Gemini Flash and Haiku layouts).
- Baseline (cloud container, 2026-11-14): Tesseract's blocks score block_f1 77 % [65, 88], role_acc 40 % (every block is body), order_tau 0.96, figure_f1 0, page_ink_f1 39 % on average.
- Tests: layout input, JSON-text and file params, raster size, `page_ink_f1`, and a check that every labelled page is drawable and matches itself.

**Font identification: role `font-match`, a catalogue of period faces, and its eval (branch `claude/project-thread-01gj43`)**
Jonathan: font identification is a skill of its own; a vision model suggests candidates, then an optimisation finds the exact face and settings. The page rebuild (strata-scan: real text in a matching face) consumes it.

- `runtimes/fontfit.py` (**new**, role `font-match`, model `fontfit`, CPU): render-and-fit. Each line of a block is set from Tesseract's words in a candidate face; size, width, tracking and weight are searched to match the scan's ink, then each word keeps its best style (roman, italic, bold, small capitals, real or made from capitals). Answer: the face and class, a ranked list of the others, and per line the size in px and pt, hscale, tracking, weight and baseline, per word the style, and a rendering for judging. Modes: `fit` (candidates given, or every face screened), `measure` (a proposer's first face, no search), `naive` (Tinos stretched to Tesseract's boxes).
- `runtimes/fonts.toml` (**new**): 57 open faces from Google Fonts in 10 classes, each with what historical type it is like; fetched once into `<models_dir>/fonts/`.
- `evals/ink.py` (**new**, shared with the page rebuild): binarise (flatten, deskew, Sauvola, despeckle) and ink F1 with a tolerance, by box or mask.
- Metrics `ink_f1`, `font_class`, `font_close`, `font_top1`, `font_top3`, `font_style`, `font_size_err`; preset `font-propose-v1`; spec `evals/specs/font-match.toml`; dataset builder `scripts/evals/font_match.py` and `font_match_trial.py` (trial before deploy).
- First numbers (this machine, CPU only, no vision model yet): on 29 synthetic blocks `exhaustive` names the exact face 83 % of the time (top 3: 90 %), size within 5 %; on 38 real blocks ink F1 0.84 against 0.58 for `naive`, but the class is right on only 26 % (heavy display faces fit blurred print too well): what a vision model's shortlist should fix.

**Page rebuild, first part: the `page-build` role and layout metrics (#67, branch `claude/project-thread-dvu0cz`)**
- `runtimes/pagebuild.py` (**new**, role `page-build`, model `pagebuild`, CPU lane): a page description in (JSON or a path; or a Tesseract `ocr` answer, turned into the plain-overlay baseline), the PDF out, drawn by `strata_python.pages.render`; also writes the raster at the scan's dpi and returns the text read back from the PDF. Output defaults to `<data>/pages/<hash>.pdf`. In the pyright strict list.
- Dependency: `strata-python[db,pdf]>=0.2.2,<0.3` (the `pdf` extra brings ReportLab and pdfium).
- `evals/metrics.py`: `block_f1`, `figure_f1`, `role_acc`, `order_tau` against `reference.page`, and `match_boxes` (greedy IoU matching of `[x, y, w, h]` boxes). docs/evals.md says how a page-rebuild eval is scored.
- Tests: `tests/unit/test_pagebuild.py`, `tests/unit/test_page_metrics.py` (**new**).

**strata-python 0.2.1 from PyPI: its database layer, test fakes, dev tools and web serving replace this repo's copies (branch `claude/project-thread-dvu0cz`)**
strata-python is on PyPI now: the dependency is `strata-python[db]>=0.2.1,<0.3` instead of a git pin of its main, so a breaking change there reaches the service only through a deliberate edit of the range. **Deploy needs the service unit's environment changed first** (below).

- Database: `strata_python.db` replaces the git-pinned `omarchy-db` (same naming convention, so no revision drifts; `strata-db check` passes from empty on SQLite). `db/models.py`, `store.py`, `test_database.py`, `test_cli.py` import from it; `[tool.omarchy-db]` is `[tool.strata-db]`. Both packages registered a pytest `--pg` option, so they could not be installed together. **The service now needs `STRATA_DB_LIVE=1` and `STRATA_DB_DATABASE=<the shared database>` in its unit** (strata-python is public, so the name is no longer built in); without them it refuses the shared database instead of guessing. Keep `OMARCHY_DB_LIVE=1` beside them until every app has moved. `STRATA_DB_URL` works like `OMARCHY_DB_URL` did. The README and docs/database.md say `strata-db` where they said `dbtool`.
- `.github/workflows/database.yml`: its own job (Postgres 18 service container, `strata-db check`, the database tests on SQLite and with `--pg`) instead of jburnhams/database's reusable workflow, which installs omarchy-db.
- Tests: `tests/utils/fakes.py` deleted and the `fake_urlopen` / `fake_popen` / `fake_run` fixtures and the integration marker hook dropped from `conftest.py`: strata-python's plugin provides them. The tmp SQLite URL is `strata_env` in `pyproject.toml` instead of an autouse fixture, and `STRATA_DB_URL`, `STRATA_DB_LIVE` and `STRATA_DB_DATABASE` are cleared for every test.
- Dev group: `strata-python[dev]` (ruff, pyright, ty, httpx and the pytest plugins at the versions strata-check is tested with) instead of listing them here.
- `server/app.py`: the page and assets come from `strata_python.web.add_web`. The built app (or the landing page) is now served at every path outside `/api` with a relative `<base href>`, so a reload or a shared link deep in the app opens it; unknown `/api/...` paths stay 404.
- `runtimes/worker.py`: a worker venv's interpreter from `strata_python.oslib.venv_python`.
- `upgrade-strata-python.yml`, `AGENTS.md`: the weekly upgrade moves to the newest release within the range; AGENTS.md says to use strata-python's modules before writing a copy and to file what is missing there.
- Not adopted yet, filed on strata-python instead: settings stored in the database (#16), extra checks in `doctor` and extra health fields (#17), `strata-public-check` (#13: npm versions read as private IPs; it also found home-folder paths in the speak-* specs), the client moving there (#18), page rendering for the PDF rebuild (#9), the `--pg` clash (#11).

**A resumed eval run waits on the jobs its lost run left queued (branch `claude/eval-resume-waits`)**
After the 17:20 restart on 2026-11-14 the queue still held the lost FlashVSR+ Base run's jobs, nothing waited on them, and the resumed run (`recover`) queued copies: the same clip ran twice. `evals/core.py`: with `recover`, a queued or running job of the same role, model (or a variant of it) and params is waited on (`pending_job`), after a finished one is looked for (`done_job`), before a new job is queued.

**FlashVSR+ reads and writes in segments, and heavy video workers get a memory cap (branch `claude/flashvsr-decode-every`)**
On 2026-11-14 at 17:20 systemd-oomd killed strata-models (29.1 GB memory peak, 27.1 GB swap) while FlashVSR+ Base ran: with `--decode-every auto` its CLI holds the whole input and the whole 1536 x 1152 output of an 8 s clip in RAM (20.6 GB for one process on a 46 GB machine). `models.toml`: Turbo and Base pass `decode-every = 28`, which reads, decodes and writes in segments (its IncrementalVideoWriter). Base also needs more than Turbo because its KV cache does not fit the card (it estimates 16-17 GB of VRAM against 10.8 free) and 30 of its layers' caches go to host RAM; Turbo offloads none.
- `runtimes/worker.py`: models.toml `options.memory_max_gb` starts the venv worker (and the CLIs it runs) in a systemd user scope with MemoryMax and no swap (Linux with a user bus; elsewhere as before), so a worker that outgrows it is killed alone and its job fails, instead of systemd-oomd killing the service. FlashVSR+ Turbo, Base and SeedVR2: 16 GB.

**OCR presets v2 (plain text, no Markdown) and preprocessing under Haiku's correction pass (branch `claude/project-thread-gl1u9l`)**
On ocr-hard, the cloud readers' cer_exact was mostly formatting rather than misreads: Haiku wrapped italics in asterisks, and Flash-Lite added Markdown table rules. `ocr-read-v2` and `ocr-correct-v2` are v1 plus "as plain text: no Markdown". They are new presets, not edits, because outputs are keyed by preset name and the v1 results must stay comparable. ocr-hard gains v2 runs of haiku-read, flash-lite-read and tesseract-haiku-fix, every Tesseract preprocessing setting from ocr-pages on its own, and six of them under Haiku's v2 correction pass, to see how much the cleaning still matters once Haiku corrects the text.

**Cloud runtimes send a schema without its value bounds (branch `claude/project-thread-gl1u9l`)**
The describe-photos stand-outs failed on every Claude model (400: "For 'integer' type, property 'minimum' is not supported") and on Gemini (400: "Request contains an invalid argument"): photo-describe-v1's schema bounds its integers and arrays, which llama.cpp's grammar takes and the vendors' structured outputs do not. `cloud.portable` drops `minimum`, `maximum`, `exclusive*`, `multipleOf`, `minLength`, `maxLength`, `pattern`, `format`, `maxItems`, `uniqueItems`, `$schema` and a `minItems` above 1 before Claude, Gemini or NIM sees the schema; the shape (types, properties, required, enum, items, additionalProperties) stays enforced, and the bounds are in the preset's prompt as before.

**Better OCR scoring, and a harder OCR eval scored against Opus (branch `claude/project-thread-gl1u9l`)**
The ocr-pages eval put qwen-read at 0.16 % CER, yet Jonathan saw unclear and plainly wrong text on most pages: `cer` ignores case and punctuation (small capitals typed as PUNCH, a stop read as a comma cost nothing) and the ten typed pages are not the worst ones. His plan: score the OCR better first, take the worst pages by kind of error as the test, and have a few strong-model calls (Opus) read them as the reference.

- Metrics: `cer_exact` (case and punctuation kept, only curly quotes, dashes, whitespace and a space before a stop unified), `wer_bag` (word order ignored: separates misreads from lines read out of order, which is most of ward1874-p011's 47 %), and two that need no reference, for ranking pages nobody has typed: `garbled` (words that look like OCR junk: a stray symbol, a digit among letters, a capital inside a word, five consonants running, a lone letter) and `unclear` (words Tesseract gave a confidence under 60).
- `scripts/evals/ocr_reference.py` (**new**): one strong model (default Opus 5.5 on the Claude API) reads each picked page once into `scripts/evals/ocr_hard/<item>.txt`; `SOURCES.json` keeps why the page was picked, the model, the cost, and on hand-typed pages its `cer_exact` against the typed text. A spend cap (`--max-usd`, default $1).
- `ocr_pages.py`: `--refs` builds a dataset from another reference folder (`--refs scripts/evals/ocr_hard --name ocr-hard`), its tags from SOURCES.json.
- `evals/specs/ocr-hard.toml` (**new**): Tesseract, the best cleaning, qwen and gemma (read and fix), and the cloud readers (Claude Haiku and Sonnet, Gemini Flash and Flash-Lite, NIM Nemotron and Gemma 4 31B, and Tesseract corrected by Haiku), on `cer_exact` with `cost_usd` per page.
- `models.toml`: the cloud vision models also serve `ocr`, as eval candidates (they give no boxes).
- `runtimes/cloud.py` `encode`: a picture over 5 MB (base64) is sent scaled to 2576 px on its long side, as PNG or else JPEG: a strata-scan page at full size is about 16 MB and the Claude API refuses one over 10 MB.
- `scripts/evals/ocr_hard/` (**new**): Opus 5.5's readings of the ten worst pages, $0.28 in all; on the four of them that were hand-typed, `cer_exact` against the typing is 0.00 %.

**Eval runs save each output as it finishes, and can recover a run lost to a restart (branch `claude/eval-save-each`)**
A service restart at 12:26 on 2026-11-14 (a deploy) came in the middle of a 7-clip SeedVR2 batch: five clips had finished (about 32 min each) but none reached eval_results, because a batch of up to 32 jobs was saved only once all of them had been waited for.

- `evals/core.py` `run`: each output is saved the moment its job is waited for; the batch's jobs are still all queued at once. A restart now loses only the job running then.
- `run(..., recover=True)` (`POST /api/evals/{name}/run?recover=1`, CLI `eval run --recover`): before queuing a job, a finished job of the same role and model with exactly the same params (load options included) whose result is still kept (a week) is taken instead (`done_job`). Off by default, so `forget` then `run` still makes new outputs.

**Cloud API clients (Claude, Google AI Studio, NVIDIA NIM) with budgets and a Cloud page (branch `claude/project-thread-gl1u9l`)**
Jonathan has a Claude API key, two Google AI Studio keys (free tier and a paid one) and an NVIDIA NIM key, and wants their spend watched closely so nothing is wasted. Checked 2026-11-14 with one tiny call each: none of the four shows its remaining credit to the key (the Claude key gets 401 on every Admin endpoint: usage, cost and spend limits need an Admin or organization-scoped key; NIM's credits endpoint is 403; Google's balance is only in Cloud Billing). So the service keeps its own ledger and budgets.

- `keys.py` (**new**): keys from the environment or `~/.config/strata-models/secrets.env` (`STRATAMODELS_SECRETS`), refused when the file is readable by others; `redact` takes them out of any error text.
- `spend.py` (**new**): the ledger (table `api_calls`, migration `a5fc50ef0727`: tokens, cost, the request with pictures as paths, the answer, the app and job), prices from each model's `price` option (cache and long-prompt tiers, dated price changes), a budget and optional monthly rollover per paid key (settings `claude_budget_usd`, `claude_rollover`, `gemini_budget_usd`, `gemini_rollover`; default 0, so nothing paid runs until a budget is set), and `reserve`: the worst case (counted input plus max_tokens) must fit what is left, or the call is refused before it is sent.
- `runtimes/claude.py` (`claude-api`, anthropic SDK: count_tokens, then the call), `runtimes/gemini.py` (`gemini-api`: free key first, the paid key on 429/503, private pictures can opt out of the free tier with `keys = ["gemini"]`), `runtimes/nim.py` (`nvidia-nim`: OpenAI-compatible, retries 429), sharing `runtimes/cloud.py` (llama-server's params: prompt or preset, system, images, schema, max_tokens, effort).
- `models.toml`: Claude Haiku, Sonnet and Opus 5.5; Gemini 3.8 Flash, 3.5 Flash-Lite and 4 Argon (paid key only: no free tier); NIM Nemotron 3.5 Super VL, Mistral Large 4, Gemma 4 31B. None is a role's model.
- Job priorities (Jonathan): `high`, `normal` (the default) and `low`. Evals submit `low`, so they wait while any other app's job is queued; an app that is waiting on a job sends `high` (the client's `priority=`; `POST /api/run` defaults to it, as before). The old names `interactive` and `batch` are still accepted, stored as high and normal; old rows keep them (the column is free text, no migration). Within a priority the order is unchanged (loaded model first, `max_skip_s`). A job already running is never stopped for a higher one.
- Evals: metric `cost_usd` (dollars per item: a cloud model's `cost_usd`, 0 for a local model, none for subscription recordings); `scripts/evals/standouts.py` (**new**) picks an eval's stand-out items for the cloud candidates (scored, local models reasonable, most room left).
- `spend.per_task()`: the mean cost of one job per (role, model key), from the ledger (for routing and the cost-per-task evals).
- API: `GET /api/spend`, `/api/spend/series`, `/api/spend/calls`, `/api/spend/calls/{id}`; the scheduler runs each job inside `spend.job(app, id)` so its calls name them.
- Web: the Cloud page (budget meters warning at 50 % and 80 %, the budgets form, filters by key, model, app, outcome and period, spend and calls per day or hour, the breakdown table, and the request log, whose rows open on the prompt, schema and answer).
- Tests: `tests/unit/test_spend.py`, `test_keys.py`, `test_cloud_runtimes.py` (**new**), and the settings, server and scheduler additions; `web/tests/integration/Cloud.test.tsx` (**new**) and logic tests.

**A routing catalogue: every cloud backend and local model by role, power and cost (branch `claude/project-thread-2aatsx`)**
Jonathan asked for every cloud API mapped to capabilities, power and cost, so jobs can be routed between cloud and local models.

- `catalogue.toml` (**new**): the backends (local GPU and CPU lanes, agy, codex, Claude API, NVIDIA NIM, the free and paid Gemini keys) with Jonathan's order and use rules and their reserves; their models (roles, family, tier, price per Mtok or measured quota per call, the recorded models that stand for them in evals). NIM ids from its public model list.
- `routing.py` (**new**): `Catalogue`, eval evidence from `eval_summaries`, `Allowances` (agy and codex usage from their CLIs, the API keys from `stratamodels.spend` once it lands, cached 5 min), `options()` and `route()` (cheapest or best; private, tier floor, reserve and budget rules; `would_pick`).
- `server/app.py`: `GET /api/catalogue`, `GET /api/route/{role}`, `GET /api/allowances`. `cli.py`: `catalogue`, `route ROLE`.
- `tests/unit/test_routing.py` (**new**); server and CLI tests. `docs/routing.md` (**new**), README.

**Model judges for OCR, scored against the hand-typed text (branch `claude/project-thread-0n1ijc`)**
Jonathan asked whether qwen or Gemini could judge how good each OCR setting is. On the ten typed pages a judge can be checked against the character error rate, and one that agrees well can then rank settings on the pages nobody has typed (the rest of both books).

- `evals/judges.py`: `reference = false` keeps the item's reference from a judge (for a video pair, the tape's crops too: review) (in the judge's key only when hidden, so older judges keep their verdicts); `pairs = "baseline"` judges each candidate against the baseline only (12 pairs a page in ocr-pages, not 78); in an eval decided by a scored metric, each judge's verdicts are also scored against that metric (`metric` in the analysis: the output that scored better on the item is the right answer).
- `evals/report.py`: the Model judges section gives that agreement.
- `evals/specs/ocr-pages.toml`: judges qwen, gemma (local, judge-vision) and gemini-flash (agy, 40 pairs a run), reference hidden, baseline pairs.
- `tests/unit/test_eval_judges.py`: the hidden reference, baseline pairs and agreement with the metric. `docs/evals.md`: the options and the OCR judges.

**An OCR eval on hard book pages, and page cleaning before OCR (branch `claude/project-thread-0n1ijc`)**
Jonathan asked for evals of strata-scan's trickier pages to see what helps, and to look at cleanup like diary-study's, before and after OCR. Ten hard pages from the two Punch and Judy books, typed out by hand, measure Tesseract with each cleaning step and page mode, a vision-language model reading the page, and one correcting Tesseract's text.

- `runtimes/pageclean.py` (**new**, in pyright's strict list): seven named cleaning steps in Pillow and numpy (`flatten`, from diary-study's `remove_background_and_lighten_fold`; `contrast`, `sharpen`, `denoise`, `otsu`, `adaptive` (Sauvola), `upscale`).
- `runtimes/tesseract.py`: param `clean` (a list or "flatten,sharpen") runs those steps first, keeping the picture's dpi (doubled by `upscale`), and the result names them; `image` is accepted for `path`, so an eval item's picture reaches Tesseract and a vision-language model alike. A multipage picture is refused before cleaning (review: cleaning kept only its first frame).
- `models.toml`: `qwen3.5-9b-vl` and `gemma-4-12b-vl` also serve `ocr`, as eval candidates only (they give no boxes); `tesseract-eng` stays the role's model.
- `runtimes/presets/ocr-read-v1.md`, `ocr-correct-v1.md` (**new**): transcribe a page as printed; correct an OCR text against the page.
- `evals/specs/ocr-pages.toml` (**new**): 13 candidates (Tesseract with cleaning and psm; two vision-language readers; Tesseract then a corrector). `scripts/evals/ocr_pages.py` (**new**): builds the dataset from strata-scan or a folder of PNGs; `scripts/evals/ocr_pages/*.txt` (**new**): the ten references.
- `tests/unit/test_pageclean.py` (**new**): each step on a synthetic shaded page. `tests/integration/test_tesseract_runtime.py`: `image` and `clean` on the real command.
- `docs/evals.md`: the OCR eval. README, client docstring: `clean`.
- `evals/metrics.py`: `transcript` (what `cer` and `wer` read) takes an output's `text` when it has no `segments`; an `ocr` page or a vision-language answer scored 100 % wrong before. `tests/unit/test_eval_metrics.py`: that case.
- First numbers, the nine Tesseract candidates through a local service (CER, %): no cleaning 4.93, psm 4 3.48, flatten 3.60, denoise 3.83, adaptive 4.27, upscale 4.62, flatten+sharpen 2.87 (the report's pick: 2.06 points better, interval 0.28 to 6.93), flatten with psm 4 3.13, flatten+denoise with psm 4 3.04. Tried locally and left out: otsu 5.11, flatten then upscale 5.56. The vision-language candidates need the GPU.

**FlashVSR+ gets its Triton attention kernel (branch `claude/flashvsr-triton`)**
The first FlashVSR+ Turbo run (one clip, 2026-11-13) stopped before any frame: with neither Block-Sparse-Attention nor FlashVSR's own Triton kernel installed it falls back to PyTorch SDPA, which refuses a 1536 x 1152 clip ("would allocate too many mask entries"); the card had peaked at 4.2 GB. `scripts/setup_worker_venv.sh flashvsr-plus` now also installs flash-attn 2.7.4.post1 (the prebuilt wheel for cp310, torch 2.7, CUDA 12, C++11 ABI: no CUDA compile) and the repo's `operators/flashvsr_attn` (pure Triton), the backend FlashVSR+'s README recommends for consumer cards.

## 2026-11-13

**SeedVR2 decodes in 512 px tiles (branch `claude/seedvr2-decode-tiles`)**
The first SeedVR2 3B run on the RTX 3060 (one clip, 2026-11-13) finished its DiT pass (80 batches of 5 frames, ~6.5 min) and ran out of memory at the first VAE decode batch: 1024 px tiles took allocation from 8.8 to 10.7 GiB. `models.toml`: VAE encode and decode tiles 512 px with 64 px overlap.

**An `ocr` role on Tesseract, for strata-scan (branch `claude/project-thread-0n1ijc`)**
Jonathan is turning strata-scan into an app for scanned books (rotate, crop, text and pictures out) and asked for the model work to live here. A page needs its lines and words with boxes, which Tesseract gives on the CPU, so the role uses the CPU lane and never waits for the GPU; a vision-language reader is the eval to run against it later.

- `runtimes/tesseract.py` (**new**, in pyright's strict list): runs the `tesseract` command with TSV output (and `pdf`, a searchable PDF page, when a job asks), and turns it into lines of words with boxes and confidences; params `lang`, `psm`, `dpi`, `pdf`. `runtimes/__init__.py`: registered, and `available()` names a missing command.
- `runtimes/tesseract.py`: each job runs with `OMP_THREAD_LIMIT=1`. Tesseract's own threads fought the CPU lane's four jobs: a 360 dpi page took 80 s under load and 2 s with one thread each.
- `models.toml`: role `ocr`, model `tesseract-eng` (vram 0).
- `client`: `sm.ocr(path, app, pdf=...)`; README.
- `tests/unit/test_tesseract_runtime.py` (**new**): TSV parsing. `tests/integration/test_tesseract_runtime.py` (**new**): the real command on a page drawn with Pillow, and its failures (moved out of the unit suite after review: unit tests run no external processes). `tests/integration/test_service.py`: the role through the service and client.
- `runtimes/tesseract.py`: a picture with more than one page (a multipage TIFF) is refused, since Tesseract restarts its block and line numbers on each page and lines would merge across pages.
- `.github/workflows/integration.yml`: installs Tesseract (apt, brew, choco) so the tests run the real command. README: the role in the table.

**Samples can name their items; SwinIR once, on the face close-up, with 4 hours (branch `claude/swinir-one-clip`)**
SwinIR-L needs over 2 hours an 8 s clip on the RTX 3060 (two runs timed out, at 30 min then 2 h). Jonathan chose to see it once, on one hard clip.

- `evals/core.py`: a candidate's `items` may be a list of item ids (beside a count or another dataset's name): the candidate runs on those items, in the dataset's order, and is compared with the baseline on them.
- `evals/specs/video-enhance-vhs.toml`: swinir-real-x4 runs on 07-face-zoom-closeup only; `models.toml`: its job limit 4 h.

**A venv-worker model sets how long one job may run (branch `claude/video-worker-timeout`)**
SwinIR (video-enhance-vhs) failed on all 8 clips, about 6 hours of GPU time: each 8 s clip takes 35-45 min on the RTX 3060 and the venv-worker runtime stopped every job at its fixed 1800 s.

- `runtimes/worker.py`: models.toml `options.timeout_s` (default 1800, as before) is how long one job may run before the worker is stopped.
- `models.toml`: SwinIR 7200, SeedVR2 3B 10800, FlashVSR+ Turbo 7200 and Base 10800 (not yet measured here: generous), Real-ESRGAN x4plus, RealPLKSR and the 1x cleanup model 3600 (they took 14, 20 and 11 min a clip: too close to the old limit).

## 2026-11-12

**Video pairs play on their own, round and round (branch `claude/video-autoplay`)**
Jonathan: "when video compares load then autoplay them on full loop immediately".

- Judge tab: a video pair starts playing as soon as its videos can (muted, `audio="none"`: browsers block autoplay with sound), and loops the whole clip (0 to the shorter output's last frame) unless a range is set with `i`/`o`; `c` goes back to the whole clip. `CompareVideo` has no autoPlay yet (strata-react#203), so the page presses its own Play once (`useAutoPlay`); remove that when #203 ships. Someone with reduced motion set is left to press Play.
- Outputs tab: `VideoCompare` takes `autoPlay` (muted, once every video can play); it already loops.
- Judge tab: the answer buttons are above the pair as well as below it (groups "Answers above" and "Answers below"), for pictures, video and text alike: no scrolling past a tall frame to vote (Jonathan).

**Scheduler: models and leases above 10 GB get the whole card (branch `claude/project-thread-ecj9cv`)**
Jonathan: "above 10gb any model can run as long as it gets whole card reservation", so bigger models can be tried at their own risk instead of waiting forever or being refused.

- `scheduler.py`: a model job, a load from the Models page or a sized lease that needs more than `whole_card_gb`, or more than the card less the reserve, waits until no GPU job runs and no lease is held (nothing on the GPU passes it), frees every idle GPU model, ignores the reserve, and holds the card alone until it ends (the CPU lane carries on). An event says it runs on the whole GPU at its own risk. Only a lease bigger than the card itself fails; before, a lease above the card less the reserve failed and one that fit that but not beside other programs waited forever.
- `settings.py`, `PUT /api/settings`, `strata-models set`, Settings page: `whole_card_gb` (default 10, env `STRATAMODELS_WHOLE_CARD_GB`), applied at once.

**Sound: noise reduction moved here from strata-360, with evals scored by people, models and measures (branch `claude/project-thread-ecj9cv`)**
Jonathan asked to move strata-360's audio noise reduction into this service, look at other denoisers, and build evals on real audio (race speech from strata-360, a digitised tape's soundtrack), judged by programmatic measures and model judges as well as people.

- `src/stratamodels/workers/audio.py`: **new** venv-worker for role `denoise-audio`. strata-360's `dsp.clean_for_playback` (high-pass 80 Hz, DeepFilterNet3 capped at 18 dB, speech EQ and speechnorm, -14 LUFS with a limiter; its BS.1770 loudness) as one chain every engine runs, so candidates differ only in the denoiser: `none`, `afftdn`, `noisereduce`, `dfn3`, `clearvoice` (MossFormer2_SE_48K, MossFormerGAN_SE_16K, FRCRN_SE_16K) and `resemble` (Resemble Enhance, denoise or enhance). The cap (`atten_lim_db`) mixes back a share of what was taken out for engines without one of their own, as strata-360's MossFormer2 stand-in did. Every output is levelled to the same loudness, so no candidate wins by being louder.
- `src/stratamodels/workers/audio_quality.py`: **new** venv-worker for role `audio-quality` (CPU): DNSMOS P.835 (sig, bak, ovrl) and SQUIM objective estimates without a reference; SI-SDR, PESQ (wide band) and extended STOI against one, aligned first.
- `models.toml`: roles `denoise-audio` (deepfilternet3 with strata-360's settings as `options.defaults`), `audio-quality` (dnsmos-squim) and `judge-audio` (qwen3-omni-30b-a3b; gemma-4-e4b to trial); one venv per model family: `scripts/setup_worker_venv.sh dfn | clearvoice | resemble | audio | audio-quality` (dfn installs deepfilternet without its numpy pin and patches its torchaudio import, as strata-360 did).
- `evals/judges.py`: judge kind `audio`: the item's recording, then X's and Y's, as 16 kHz WAV; local through `runtimes/llama.py` (new `audio`/`audios` params, sent as `input_audio`), or Gemini through agy (`cloud.prompt_of` tells it to open the recordings with its file viewer; codex refused).
- `evals/core.py`: `[score] with = {param = "item.field"}` passes fields of the item to the scoring job (the clean reference of a mix). `evals/metrics.py`: `dnsmos_ovrl`, `dnsmos_sig`, `dnsmos_bak`, `squim_stoi`, `squim_pesq`, `squim_si_sdr`, `si_sdr`, `pesq`, `estoi`, `removed_db`.
- `evals/specs/denoise-audio.toml` (real recordings, judged by people against strata-360's chain; model judges Qwen3-Omni, Gemma 4 E4B and Gemini Flash; DNSMOS/SQUIM as context and a guard) and `denoise-audio-mix.toml` (clean speech mixed into real noise, decided by PESQ). `scripts/evals/denoise_audio.py` builds both datasets (private: they stay in the data folder).
- Dev dependencies: scipy and soundfile (the workers' loudness and file writing, tested without the model libraries).

**Video quality readings never fail a job (branch `claude/video-quality-optional`)**
A worker venv without numpy (or a reading ffmpeg cannot take) failed the whole job at its last step, after the video was made.

- `workers/video.py`: without numpy, `quality` keeps the ffmpeg readings (fidelity, flicker) and leaves out sharpness and noise, with `quality.skipped` saying why; any other error in the readings gives `quality = {skipped: "<error>"}` and the job still succeeds. The metrics read a missing reading as not measured (None), never as a failure.

**VHS cleanup: a stage before any upscaler, and one eval per cleanup problem (branch `claude/vhs-cleanup-evals`)**
Jonathan asked for VHS-specific noise removal, as separate evals for the different cleanup options.

- `workers/video.py`: params `cleanup = { chroma_shift = [cb_h, cb_v, cr_h, cr_v], filters = "...", model = "1x file", tile }`, the same for every engine. The shift and filters run on the deinterlaced, cropped picture in the tape's own pixels (before the resize); the 1x model (spandrel) runs on the prepared frames, then the engine (lanczos, an upscaler, or the prepared file a CLI engine reads). So the winners chain in front of any upscaler. `pipe_frames` puts frames through any list of spandrel models (cleanup, then upscaler) and kills both ffmpegs on a failure.
- `scripts/setup_worker_venv.sh video`: also the 1x models from OpenModelDB, checked by sha256: BleedOut Compact (WTFPL), BaldrickVHSFix v0.2 and BroadcastToStudio Lite (CC-BY-NC-SA-4.0).
- Cleanup `planes = "luma" | "chroma"`: the filters run on a copy and mergeplanes takes the other planes from the original, bit for bit (hqdn3d reads a strength of 0 as its default, so "hqdn3d=0:4:0:6" denoised luma too). A 1x cleanup model is GPU work: such candidates run on the new `video-lanczos-cleanup` (vram_gb 1.5), and a model registered with no VRAM (the CPU lane) keeps its cleanup model on the CPU. The seedvr2 and flashvsr-plus venvs also get spandrel, for a cleanup model before those engines.
- Measured beside people's verdicts: the worker returns `quality` (SSIM and PSNR of the result resized back against the tape with no cleanup, flicker against the tape's own, and on stills luma sharpness, luma and chroma noise), read by new metrics `fidelity_ssim`, `fidelity_psnr`, `flicker_ratio`, `sharpness`, `luma_noise`, `chroma_noise`. Each region's crop also gets a still (`region_stills`, default 3 s), and a vision judge of a video pair is shown, per region, the tape's crop, X's and Y's at the same moment (`judges._video_stills`); a video item's reference (the tape) is not shown as a "right answer".
- A VapourSynth stage before everything else in the worker, for what ffmpeg has no equal of: `deinterlace = "qtgmc"` (vsjetpack's QTempGaussMC) and `cleanup.vs = { degrain = { tr, thsad, blksize } }` (vsjetpack's mc_degrain, SMDegrain's motion-compensated temporal average, on the cleanup's `planes`). vspipe writes a lossless intermediate (FFV1, the audio copied) and the rest runs on it; quality is still measured against the tape as bwdif deinterlaces it. `scripts/setup_worker_venv.sh vapoursynth`: VapourSynth R82, vsjetpack and the plugins (mvtools, mvutensils (vsjetpack 3 calls `core.mvu`), znedi3, zsmooth, bestsource, bwdif, resize2, akarin, vszip) as PyPI wheels in a venv of their own, no system packages; then `vapoursynth config` (a pip VapourSynth finds its Python through `~/.config/vapoursynth/vapoursynth.toml`, keyed by the venv's own libvsscript, so a system install keeps its entry). Candidates: `qtgmc` in deinterlace-vhs, `smdegrain` and `smdegrain-strong` in vhs-grain.
- The five video specs (video-enhance-vhs, deinterlace-vhs, vhs-chroma-shift, vhs-chroma-cleanup, vhs-grain) list those metrics beside `judged` and three vision judges with criteria for each problem: qwen-vision (judge-vision's model), gemma-vision (gemma-4-12b-vl on trial) and gemini-flash (agy, 20 pairs a run), all `auto`. People's verdicts still decide.
- `evals/specs/vhs-chroma-shift.toml` (none, left 2 and 4 px, right 2, up 1 line), `vhs-chroma-cleanup.toml` (none, colour-only hqdn3d light and strong, BleedOut, BaldrickVHSFix, BroadcastToStudio) and `vhs-grain.toml` (none, luma hqdn3d light and medium, atadenoise, nlmeans; a motion-compensated denoise joins when its VapourSynth plugins are installed). Each judged blind against no cleanup, with the same lanczos after, so only the cleanup differs.

**Video enhancement: role `video-enhance`, VHS evals with crops, video in the Judge and Outputs tabs (branch `claude/project-thread-rea1ol`)**
Jonathan digitised a PAL VHS tape (720x576i TFF) and wants it upscaled and enhanced; the right model is to be chosen by eye, on crops where the differences show.

- `src/stratamodels/workers/video.py`: **new** venv-worker. Every engine gets the same preparation (bwdif or yadif send_field to 50p with the field order forced, the active picture cropped, 768 x 576 square pixels) and the same finish (lanczos to 1440 x 1080, x264, the source's audio), so candidates differ only in the enhancement. Engines: `ffmpeg` (classical, the baseline; optional `filters` such as hqdn3d), `spandrel` (per-frame models: Real-ESRGAN, RealPLKSR, SwinIR..., optional tiles), `seedvr2` (SeedVR2 3B FP8 through numz's standalone CLI, block swap) and `flashvsr` (official FlashVSR+, which falls back to PyTorch SDPA, so no Block-Sparse-Attention build). For judging, each `regions` box (source pixels) is also written as a crop of the result enlarged nearest-neighbour, and a few `stills` are kept as PNG. The peak VRAM of a CLI engine's processes is polled from nvidia-smi.
- `models.toml`: role `video-enhance` (video-lanczos until the eval decides) and 10 models. `scripts/setup_worker_venv.sh`: recipes `video` (torch + spandrel, and the five weights files), `seedvr2` and `flashvsr-plus` (the CLI repos pinned to a commit under `<models>/src/`, and their weights).
- `evals/similarity.py`: kind `video` (the stills' image similarity, frame for frame). `evals/api.py`: `GET /api/evals/{name}/media/{item}?file=` serves an item's reference previews (`reference.preview`, `reference.crops[].path`) and no other file.
- `scripts/evals/video_clips.py`: **new** dataset builder from a folder of clips and its private `clips.toml` (crop, field order, regions per clip); writes the source enlarged nearest-neighbour (whole frame and each region) into `<data>/evals/media/<dataset>/` as the item's reference.
- `evals/specs/video-enhance-vhs.toml` (judged against plain lanczos: hqdn3d, Real-ESRGAN x4plus/x2plus/general-v3, RealPLKSR, SwinIR, SeedVR2 3B, FlashVSR+ Turbo/Base; guards latency 100x and failures, since VRAM against a 0-VRAM baseline cannot be a ratio) and `evals/specs/deinterlace-vhs.toml` (bwdif against yadif; QTGMC waits for VapourSynth plugins).
- `web/`: strata-react 1.0.31. The Judge tab draws a video pair in strata-react's `CompareVideo` (strata-react#156-158: one clock, side by side, swipe, fade and blink, zoom locked, the item's regions as boxes in the cropped frame, nearest-neighbour pixels, frame stepping, a loop kept across pairs of the same item, speeds down to 0.1x), with the answers in its legend and footer and the tape itself below. The Outputs tab keeps `VideoCompare` (videos on one clock, with the worker's region crops and the source) until `CompareVideoGrid` (#159, merged) is released.

**Evals: model judges beside people, and how far each agrees with them (branch `claude/project-thread-kjwkpt`)**
Jonathan asked whether models judge eval outputs (they did not: only his blind A/B picks) and wanted a model's verdict recorded beside his on the same pairs, to learn how good a judge it is before relying on it more; and judges sorted by kind, each with its own model.

- `src/stratamodels/evals/judges.py`: **new**. A spec's `[[judges]]`: kind `text` or `vision` (registry roles `judge-text`, `judge-vision`, each with its chosen model; a judge may name another model to trial it), or a cloud model via `cli = "codex" | "agy"` on the plan. A judge reads the item (and its pictures, for vision), the two answers as X and Y with every field that names the model removed, and answers {reason, winner}; asked both ways round by default, and a judge that changes its mind with the order gives a tie. `pairs = "judged"` (default: the pairs people judged) or `"all"`, `limit` per run, `auto` (judge each pair right after a person does). The analysis gains `judges`: verdicts, agreement with people (tie included), same winner where both picked one, Cohen's kappa, order consistency, disagreements, and what the judge alone would say against the baseline.
- `db/models.py`, `migrations/20261112_7b3e9d2c4f18_judgment_judges.py`: `eval_judgments.judge` (the judge's config hash; null for a person) and `detail` (each order's answer and reason). Nullable: every existing row stays a person's. A changed judge config or a new model for its kind is a new judge.
- `core.judgments` and the Judge tab count people only: a judge never decides a `judged` experiment or makes a pair look judged.
- `evals/api.py`: `POST /api/evals/{name}/judges/{judge}/run`, `GET .../judges/status`, `GET .../verdicts/{item_id}`; a person's judgment starts the `auto` judges on that pair. CLI `strata-models eval judge NAME [--judge J] [--limit N]` (exit 75 when a plan's allowance runs out).
- `evals/report.py`: a **Model judges** section after Human judgments. `evals/cloud.py`: the codex/agy calls moved out of `scripts/evals/record_cloud.py` (which imports them) so judges use the same official-CLI path.
- `models.toml`: roles `judge-text` and `judge-vision` (qwen3.5-9b-vl to start, unmeasured), served by the llama-server models.
- `web/`: the experiment page's Model judges table; after a judgment the unblinded pair shows what each judge said about it.

**Scheduler: several jobs at once, cloud and CPU jobs beside the GPU's (branch `claude/project-thread-kf7tb7`)**
The one worker thread ran one job at a time, so a 30 s cloud CLI call (the subscription-backends idea) or a CPU model would hold up the GPU queue, and two small GPU models that fit together still took turns.

- `src/stratamodels/scheduler.py`: a dispatcher decides (`dispatch`), each job runs on its own thread (`execute`; `step` still does both inline for the tests). Two lanes: zero-VRAM models (vram_gb 0) run up to `cpu_jobs` at once and never wait for the GPU or a lease, exclusive ones included; GPU jobs run up to `gpu_jobs` at once on different models, when they fit together. One job per model at a time; a running or loading model is never evicted; a loading model's estimate, and a running one's estimate beyond what it measured at load (sd-cpp allocates per job), count as promised, so a second job never starts into room the first is about to take. A GPU job that does not fit beside running ones waits and nothing on the GPU passes it (no starving big jobs); with nothing else running it starts as before. Leases wait for the room running jobs hold; an exclusive one waits until GPU jobs finish. A load is measured from the card only when no other GPU work overlapped it; otherwise the estimate stands. Load requests from the Models page run like jobs; an unload of a busy model waits for its job.
- `settings.py`, `/api/settings`, `strata-models set`, Settings page: `gpu_jobs` (default 2) and `cpu_jobs` (default 4), 1-16, applied at once (`STRATAMODELS_GPU_JOBS`, `STRATAMODELS_CPU_JOBS`). `gpu_jobs = 1` is the old one-at-a-time behaviour for the GPU.
- `/api/status`: `running` (every job running, with its lane), `slots`, `budget.pending_mb`, `resident[].busy`; `current` stays (the oldest running). CLI `status` and the Overview list every running job.

## 2026-11-10

**Evals page: analyses worked out ahead and kept in the database (branch `evals-precompute`)**
Jonathan found the evals page slow to the point of timing out. faces-bh took 81 to 97 s per analysis: the face-verification bootstrap re-read every embedding for each of its 2,800 samples. The analysis was memoised in memory only, so every restart, new output or judgment made the page wait again.

- `src/stratamodels/evals/metrics.py`: a `dataset` metric may have `prepare`/`on`: verification reads embeddings and works out the similarity matrix once, a bootstrap sample only picks rows and columns. faces-bh 97 s -> 11 s, every value and interval the same as before.
- `src/stratamodels/evals/summaries.py`: **new**. Each experiment's analysis kept in `eval_summaries` with the signature of what it read (spec, dataset version, outputs count and newest id, judgments, the analysis code). Worked out in the background when a run finishes, a judgment comes in, a spec is saved, and at the service's start. The page gets the kept one; when it is out of date and took over 1.5 s it gets it marked `summary.stale` while a fresh one is worked out (one worker, requests coalesce).
- `src/stratamodels/evals/api.py`: uses it; `POST /api/evals/{name}/refresh` for outputs changed from outside the service.
- `src/stratamodels/db/models.py`, `migrations/20261110_5c1d2e7f9a30_eval_summaries.py`: table `eval_summaries`.
- `src/stratamodels/evals/similarity.py`: `phonetic` no longer crashes on a word its rules empty ("gh"), which made the describe-photos Judge tab answer 500.
- `web/`: the experiment page says when its results are being updated and checks every 3 s until they are.
- `src/stratamodels/evals/core.py` `analyse`: a candidate with no output at all is "not run yet" (`dataset.not_run`) and no longer holds every comparison at 0 items. describe-photos showed "No outputs yet" with 7 of 9 candidates done because the two qwen3.8-27b candidates had not run; now it compares the 7 on 60 items. The page marks such a candidate "Not run yet".
- `web/` Judge tab: after a vote the buttons no longer stay disabled while the results tab's analysis reloads (it refreshes behind), and a picture can be clicked to choose it.

## 2026-11-09

**One CUDA per venv: Whisper as a venv-worker, the service's venv on CUDA 13 (branch `whisper-worker`)**
Jonathan asked to move Whisper into its own venv rather than keep onnxruntime pinned to CUDA 12, building on the venv-worker runtime the TTS work added and without duplicating it. onnxruntime 1.24+ needs CUDA 13 and CTranslate2 (newest 4.8.5) CUDA 12; both load cuDNN 9 by the same file name, so they cannot share a process.

- `src/stratamodels/workers/protocol.py`: **new**, the JSON-lines loop and `ParamError` every worker uses. `workers/tts.py`: uses it (its own copy removed).
- `src/stratamodels/workers/whisper.py`: **new**, Whisper (faster-whisper) as a worker: the load/decode/run code of `runtimes/whisper.py`, which is **removed** (and the `faster-whisper` runtime with it). A `clip` is now decoded alone (ffmpeg -ss/-to) with times kept those of the whole file.
- `src/stratamodels/runtimes/worker.py`: a worker's venv's own `nvidia/*/lib` folders go first on its library path, and the service's venv's are left off it.
- `src/stratamodels/models.toml`: the Whisper models run as `venv-worker` (`worker = "whisper"`, `venv = "whisper"`).
- `scripts/setup_worker_venv.sh`: a `whisper` recipe (faster-whisper with the CUDA 12 cuBLAS and cuDNN wheels).
- `pyproject.toml`, `uv.lock`: the `gpu` group drops faster-whisper and the CUDA 12 wheels, takes onnxruntime-gpu 1.31 (CUDA 13, served by torch's CUDA 13 wheels) and nvidia-curand (which torch lacks). `strata-models` launcher: every `nvidia/*/lib` of the venv on the path again (it is all CUDA 13).
- Tests: `tests/unit/test_whisper_worker.py` **new** (the worker with a fake model and ffmpeg, the shared loop); `tests/gpu` runs Whisper through the worker (and a clip). Measured on the RTX 3060: faces 0.04 s a photo warm on onnxruntime 1.31; Whisper turbo 2.0 GB, 0.41 s for 11 s of speech.
- After merging, on omarchy: `scripts/setup_worker_venv.sh whisper` (done) and `uv sync --group gpu`, then restart the service.

**Move the Python side to uv, ruff, pyright and the shared strata-python library (branch `claude/project-thread-iw7199`)**
Jonathan asked for every Python project to use the same tooling as [python-app-template](https://github.com/jburnhams/python-app-template): uv, ruff with `ruff format`, pyright and ty, one `scripts/check` gate, and the server code shared through [strata-python](https://github.com/jburnhams/strata-python) instead of copied. Behaviour of the service is unchanged except where noted.

- `pyproject.toml`, `uv.lock`: **new** `[project]`, `dev` and `gpu` dependency groups (the GPU stack is `uv sync --group gpu`), git sources for `omarchy-db` and `strata-python` (tracking its `main`; the lock pins the commit), tool configuration (ruff `E,F,I,B,UP,SIM,RUF,C4,PT,S,TID` with `pytest.importorskip` banned, pyright strict ratchet, coverage). `requirements*.txt` and `scripts/setup_env.*`: **removed**.
- `src/stratamodels/oslib.py`, `tests/unit/test_oslib.py`: **removed**, now `strata_python.oslib` (tested in that library).
- `src/stratamodels/settings.py`, `server/app.py`, `cli.py`: **main's database-backed versions are kept as they are** (merged 2026-11-09); an earlier version of this change moved settings, page serving and the `set` command onto `strata_python`, and that is dropped for this repo until its settings can be expressed there. `oslib` still comes from the library.
- `src/stratamodels/runtimes/{clip,faces,sdcpp,whisper}.py`: ruff format; a file-level pyright directive for the GPU libraries that are not installed in CI, each with its reason. `whisper.py`: `noqa: S607` for the ffmpeg call.
- All Python files: `ruff format` at line length 120 (a mechanical commit of its own), and the ruff rules above.
- `.github/workflows/unit.yml`, `integration.yml`: uv; `strata-check --fast` once on Linux 3.13. `upgrade-strata-python.yml`: **new**, weekly PR that moves the library pin after the gate. `.github/dependabot.yml`: **new**, daily, minor and patch grouped, no ignores.
- `tests/conftest.py`: the isolation and no-network fixtures come from the strata-python pytest plugin; a small autouse fixture keeps the database URL on a tmp SQLite file. `pyproject.toml` adds `pillow` (evals) and `opencv-python-headless` (dev, faces view tests), which main took from the old requirements files; ruff ignores B904, B905, S110, S310, S603, S607 and SIM105 for main's code, and pyright's strict entry for `settings.py` is dropped because main's rewrite is untyped. The fake-ASR eval spec's latency guard is 1e9, not 1000: the fake runs in microseconds, so latency ratios were scheduler noise and `test_run_caches_outputs_and_analysis_decides` failed about one run in seven (it asserted the recommendation, and the noise rejected the candidate). Correction to the earlier suite: that test was flaky, not seed-dependent. 0 failures in 30 runs after the change.
- `README.md`, `AGENTS.md`, `tests/README.md`, `strata-models` launcher: commands and comments for uv.
