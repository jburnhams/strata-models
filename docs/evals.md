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
   Bradley-Terry with bootstrap intervals for blind human judgments, and an estimate of the items needed when a result is unclear.
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

## Rules of thumb
- Compare on the same items, always; never compare numbers from two datasets or two dataset versions.
- One primary metric. Others are guards or context; they do not decide.
- A difference whose interval includes zero is "no clear difference", not "slightly better".
- Change the dataset, not the method, when results look wrong; then the version changes and the old report stays true for the old version.
- Keep failures in: a failed job is all errors for a rate and wrong for a share, never dropped.
