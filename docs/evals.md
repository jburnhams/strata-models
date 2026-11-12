# Evals: how a model decision is made here

Every choice of model, setting or prompt for a role is made by an experiment, run the same way and written up the same way, so decisions
can be compared and revisited when a new model appears.

## The parts
1. **A question and a decision rule, written first** (`evals/specs/<name>.toml`, or the data folder's `evals/specs/` when the dataset is
   private): the role, the dataset, the candidates (model + params + load options), the metrics, the **primary** metric, the **baseline** (what
   runs today), the **minimum effect** worth a change, and **guards** (e.g. at most 2x the run time). Writing the rule before seeing the results
   is what keeps the result honest.
2. **A versioned dataset** (`<data>/evals/datasets/<name>.jsonl`): items with an input, a reference where there is one, and tags. Its
   version is the hash of the file and is printed in every report. Datasets stay out of git (they name private files); builders that make
   them reproducibly live in `scripts/evals/` (public data) or the owning app's repo (private data, e.g. bright-horizons'
   `scripts/eval_faces_dataset.py`).
3. **Runs through the GPU queue** (`./strata-models eval run NAME`, or Run on the experiment's page): batch jobs like any app's, so an eval
   never fights other work for the card. Outputs are cached by everything that decides them; adding a candidate runs only that candidate.
4. **Metrics** (`src/stratamodels/evals/metrics.py`): each knows which way is better. Text: WER, CER (normalised the same way every time).
   Faces: found, missed, verification AUC and TAR at 1 % FAR. Generic: exact match, JSON parse rate, failure rate, run time, peak VRAM.
5. **Statistics** (`stats.py`), the same everywhere: Wilson intervals for shares, item bootstrap for means and corpus rates, **paired**
   comparisons with the baseline (bootstrap interval, permutation p, exact McNemar for right/wrong), Holm's correction across candidates,
   Bradley-Terry with bootstrap intervals for blind human judgments, Cohen's kappa for a model judge's agreement with people, and an estimate of the items needed when a result is unclear.
6. **A decision per candidate**: *adopt* (the whole interval is on the better side, the gain reaches the minimum effect, every guard holds),
   *reject* (worse, too small, or a guard broken), or *unclear* (the interval includes no change; the report says roughly how many items
   would settle it).
7. **Blind judging** (the Judge tab) where no reference exists (pictures, captions, transcripts of real speech): two outputs for the same item,
   candidates hidden until answered. **Most different first**: `evals/similarity.py` scores how alike two outputs are (text: word and phonetic
   edit similarity; images: perceptual hash, structure, colour; anything else: JSON), pairs that say the same thing are never shown, and the
   unjudged pair that differs most comes next, so a judge's time goes where the candidates really disagree. A spec can name the kind
   (`similarity = "text" | "image" | "json"`).
8. **The write-up** (`./strata-models eval report NAME --out FILE`, or the Write-up tab), always in this order: Answer, Setup, Results,
   Compared with the baseline, Human judgments, Cost, Failures and worst items, What the numbers do not show (a person writes it), Decision and
   follow-ups (a person writes it), Method, Reproduce. Final reports go in the docs repo under `tech/code/strata-models/evals/`
   (private: they may name private items); the decision goes into `models.toml` with the report linked.

## Samples: candidates whose calls are dear
A candidate can run on a sample: `items = N` (the dataset's first N items; datasets are shuffled with a fixed seed when built) or
`items = "<dataset>"` (the items of another dataset, e.g. the judged 20). Items outside the sample are never run, so they are not failures.
Each candidate is compared with the baseline **on the items both have** (paired), so a sample gets the wider interval its size earns, and an
"unclear" verdict says how many items would settle it, while the full candidates keep their whole comparison. The guards hold a candidate
to the baseline's value on the same items. This is how cloud models on a plan's allowance stand in an eval: calls in proportion to what the
plan allows (cheap models on every item, dear ones on a few), recorded by `scripts/evals/record_cloud.py` and replayed by the `recorded`
runtime.

## Judged experiments and a second reader
- **`primary = "judged"`** decides by blind human judgment where nothing can be scored automatically (a voice, a picture): the Judge tab then
  offers only pairs with the baseline, and a candidate is adopted when it wins enough of them (ties count half): the Wilson interval of its
  win share must be clear of 50 % and the share at least 50 % + `min_effect`. Until then the decision says roughly how many judged pairs would
  settle it. Judgments count only while both outputs are current (a changed model, params or dataset makes new outputs).
- **`[score]`** names a second model that reads every output with a file (`role`, optional `model`, `params`): after a run, each output gets
  a `scored` reading (kept in the output, redone when the `[score]` table changes). `tts_wer` uses it: Whisper transcribes the speech back and
  the words are scored against the text that was to be spoken, an intelligibility guard for text to speech.

## Model judges: a model's verdict beside a person's
People's blind judgments are slow to gather, so a spec can name **model judges** (`[[judges]]`, see `src/stratamodels/evals/judges.py`) that
judge the same pairs. Their verdicts are kept beside people's and **never counted as theirs**: a `judged` decision and the Judge tab's order
use people only. What the judges are for, for now, is measuring them: the analysis (and the write-up's **Model judges** section) gives each
judge's agreement with people on the pairs both judged (tie included), the same-winner share where both picked one, Cohen's kappa (agreement
beyond chance) and how often it gave the same answer with X and Y swapped. A judge that agrees well across experiments is the one to trust with
more later (pairs nobody has time for, or a first pass before people).
- **Kinds**, by what a judge must read: `text` (transcripts, captions, JSON) and `vision` (photo descriptions, generated pictures: it sees the
  item's pictures and the candidates'). Each kind is a registry role (`judge-text`, `judge-vision`) with its chosen model, picked like any model:
  by an eval, here on agreement with people. A judge may name another model serving the role to trial it. Voices have no judge kind yet.
- **Cloud judges** (`cli = "codex" | "agy"`, `model` the CLI's slug) call the vendor's own CLI on the plan, never its sign-in token
  elsewhere. Give a dear one a `limit`, like a sampled candidate. A judge that is also a candidate tends to prefer its own answers.
- Which pairs: `pairs = "judged"` (default; the ones agreement is measured on) or `"all"`; `auto = true` judges each pair right after a person
  does. Run with `./strata-models eval judge NAME`, or POST `/api/evals/{name}/judges/{judge}/run`. A changed prompt, criteria, params or model
  makes a new judge, whose verdicts start again.

## Pipelines: a step before the candidate
A candidate can run **stages** before its own job (`[[candidates.stages]]`: role, optional model, params, `input` template over the item's
fields, the `send` param that carries it, the `into` param its answer replaces). The first use: a text model turns a short image prompt into a
detailed caption (`role = "text"`, `system_preset = "image-prompt-*"`, see `runtimes/presets/`), then the image model draws it.
- Each stage's output is cached under its own key, so candidates that share a first step run it once; stages run level by level, grouped by
  model, so a text model loads once and not once per item between pictures.
- The end result keeps every step under `pipeline` (sent, answered, model, run time). Its `latency_s` is all of the steps' run time together.
- **Blind judging shows only the end result**: the steps would tell the candidates apart. After a judgment the Judge tab unblinds the last
  pair with its steps; the **Outputs** tab (`GET /api/evals/{name}/outputs`) shows every candidate's output for every item with each step's
  input and output, and the write-up has a **Pipelines** section (words in and out, step time, examples).
- A failed stage fails the output (the error names the stage) without running the candidate's job.
- `forget` keeps shared stage outputs unless asked (`stages=True`).

## Rules of thumb
- Compare on the same items, always; never compare numbers from two datasets or two dataset versions.
- One primary metric. Others are guards or context; they do not decide.
- A difference whose interval includes zero is "no clear difference", not "slightly better".
- Change the dataset, not the method, when results look wrong; then the version changes and the old report stays true for the old version.
- Keep failures in: a failed job is all errors for a rate and wrong for a share, never dropped.
