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
A candidate can run on a sample: `items = N` (the dataset's first N items; datasets are shuffled with a fixed seed when built),
`items = "<dataset>"` (the items of another dataset, e.g. the judged 20) or `items = ["id", ...]` (the items named: a slow model on the
hard cases only, e.g. SwinIR, over 2 h a clip, on the VHS face close-up). Items outside the sample are never run, so they are not failures.
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
  `[decision] pairs = "all"` offers every pair instead, for candidates that all beat one baseline and need ranking against each other
  (whole pipelines, `video-pipeline-vhs`): the decision against the baseline still uses the pairs with it, and the write-up's
  Bradley-Terry strengths rank the lot.
- **Culling**: a candidate with `judge = false` is in no new pair (people's or a model judge's) once the judgments so far put it clearly
  behind (its Bradley-Terry interval below the leaders', or it loses nearly every pair it is in). Its outputs, metrics and the judgments
  already made still count; only the Judge tab stops spending time on it.
- **`[score]`** names a second model that reads every output with a file (`role`, optional `model`, `params`): after a run, each output gets
  a `scored` reading (kept in the output, redone when the `[score]` table changes). `tts_wer` uses it: Whisper transcribes the speech back and
  the words are scored against the text that was to be spoken, an intelligibility guard for text to speech. `with = {param = "item.field"}`
  gives the scoring job fields of the output's own item: the denoise mixes send `reference = "reference.audio"`, the clean speech, so the
  `audio-quality` role can measure SI-SDR, PESQ and STOI against it (`denoise-audio-mix`). Without a reference the same role gives DNSMOS
  P.835 and SQUIM, learned estimates of what listeners would say (`dnsmos_*`, `squim_*` metrics).

## Model judges: a model's verdict beside a person's
People's blind judgments are slow to gather, so a spec can name **model judges** (`[[judges]]`, see `src/stratamodels/evals/judges.py`) that
judge the same pairs. Their verdicts are kept beside people's and **never counted as theirs**: a `judged` decision and the Judge tab's order
use people only. What the judges are for, for now, is measuring them: the analysis (and the write-up's **Model judges** section) gives each
judge's agreement with people on the pairs both judged (tie included), the same-winner share where both picked one, Cohen's kappa (agreement
beyond chance) and how often it gave the same answer with X and Y swapped. A judge that agrees well across experiments is the one to trust with
more later (pairs nobody has time for, or a first pass before people).
- **Ranking agreement**: many pairs are close calls where any answer is fair, so each judge also gets how alike it *orders* the candidates:
  Bradley-Terry strengths from the person's answers and from the judge's on the same pairs, compared by Spearman's rho (with a bootstrap
  interval over pairs), Kendall's tau-b and the places a candidate moves. A judge can differ on many close calls and still rank alike.
- **Looking at the differences**: on the Results tab, Model judges, *Show* lists every pair a judge answered differently from the person, with
  both outputs, each side's pick and the judge's reason. **Borderline: either is fine** marks the person's judgment (kept in its `detail`;
  POST `/api/evals/{name}/judgments/{id}/borderline`): that pair leaves the judge's agreement figures and counts as a tie in the ranking
  comparison. The person's own choice still counts in the experiment's results.
- **Kinds**, by what a judge must read: `text` (transcripts, captions, JSON) and `vision` (photo descriptions, generated pictures: it sees the
  item's pictures and the candidates'). Each kind is a registry role (`judge-text`, `judge-vision`) with its chosen model, picked like any model:
  by an eval, here on agreement with people. A judge may name another model serving the role to trial it. `audio` (`judge-audio`: cleaned
  sound, speech) hears the item's own recording, then X's and Y's, each as 16 kHz mono WAV (made once into `<data>/judges/audio/`): a local
  model whose projector hears (Qwen3-Omni-30B-A3B, Gemma 4 E4B, through llama-server's `input_audio`) or Gemini through agy, which opens the
  files with its file viewer (codex has no audio). 16 kHz is what these models hear anyway, so hiss above 8 kHz is for people to judge.
- **Cloud judges** (`cli = "codex" | "agy"`, `model` the CLI's slug) call the vendor's own CLI on the plan, never its sign-in token
  elsewhere. Give a dear one a `limit`, like a sampled candidate. A judge that is also a candidate tends to prefer its own answers.
- Which pairs: `pairs = "judged"` (default; the ones agreement is measured on), `"all"`, or `"baseline"` (each candidate against the
  baseline only: a scored eval with many candidates); `auto = true` judges each pair right after a person
  does. Run with `./strata-models eval judge NAME`, or POST `/api/evals/{name}/judges/{judge}/run`. A changed prompt, criteria, params or model
  makes a new judge, whose verdicts start again.
- **Against a metric**: in an eval decided by a scored metric, each judge is also scored against it (on each pair it judged, the output that
  scored better on that item is the right answer; a tie when equal), in the write-up's Model judges section. With `reference = false` the
  judge does not see the item's reference, so this measures it for items that have none.

## Pipelines: a step before the candidate
A candidate can run **stages** before its own job (`[[candidates.stages]]`: role, optional model, params, `input` template over the item's
fields, the `send` param that carries it, the `into` param its answer replaces, and `also`, item fields passed as they are beside it, such as
the `image` a vision model is shown with a text; a stage hands on its answer's text, or with `answer = "json"` its whole answer as JSON, for a
role whose answer is data, such as font-match's fitted words). The first use: a text model turns a short image prompt into a
detailed caption (`role = "text"`, `system_preset = "image-prompt-*"`, see `runtimes/presets/`), then the image model draws it.
- Each stage's output is cached under its own key, so candidates that share a first step run it once; stages run level by level, grouped by
  model, so a text model loads once and not once per item between pictures.
- The end result keeps every step under `pipeline` (sent, answered, model, run time). Its `latency_s` is all of the steps' run time together.
- **Blind judging shows only the end result**: the steps would tell the candidates apart. After a judgment the Judge tab unblinds the last
  pair with its steps; the **Outputs** tab (`GET /api/evals/{name}/outputs`) shows every candidate's output for every item with each step's
  input and output, and the write-up has a **Pipelines** section (words in and out, step time, examples).
- A failed stage fails the output (the error names the stage) without running the candidate's job.
- `forget` keeps shared stage outputs unless asked (`stages=True`).

## Video: judging upscalers and deinterlacers on crops
`video-enhance` outputs are MP4s at the asked size with `crops` (each of the item's `regions`, the same box of every candidate, enlarged
nearest-neighbour) and `stills` (PNG frames at fixed times, for `similarity = "video"`). The dataset builder (`scripts/evals/video_clips.py`)
also writes the source enlarged the same way, as the item's reference, so the Judge tab shows X, Y and the Source on one clock, and the region
chips switch all three to the face, the text or the stripes, where an upscaler's gains and inventions show; the whole frame is one chip away.
Every engine gets the same deinterlaced, cropped, square-pixel clip and the same finish, so a pair differs only in the enhancement.
Cleanup evals (`vhs-chroma-shift`, `vhs-chroma-cleanup`, `vhs-grain`) hold the upscale fixed (lanczos) and vary only the worker's `cleanup`
stage, one problem per eval; a winner is then given as `cleanup` to every upscaler's candidate. `vhs-smear` does the same for the
tape's horizontal trail after high-contrast edges (`cleanup.vs.desmear`, kernel from `scripts/evals/smear_kernel.py`).

## Shots: where a video cuts (`shots-spliced`)
`shot-boundaries` answers the cuts of a video as times; `shots-spliced` scores them against cuts that are known because the builder made them
(`scripts/evals/spliced_shots.py`): stretches of uncut camera files (the race footage: each file one continuous shot) are spliced end to end
at exact frame counts, so cut k is at the frames before it over the frame rate, with nobody marking anything by hand. Each item uses one
kind of cut, in `tags.kind`: `files` (another file: the easy cut), `far` (the same file at least a minute away) and `near` (the same file, 2
to 12 s away, forward or back: both sides show nearly the same picture, the cut a detector most often misses). A reference cut is found
when an answered cut lies within 0.5 s of it (the item's `tolerance_s` overrides), one to one, nearest pairs first: `cut_miss` (reference
cuts not found), `cut_false` (answered cuts matching none) and `cut_f1` per item, the primary. A missed cut is the costlier error for the
editor (two shots it cannot split), so read `cut_miss` beside F1. The spliced video is progressive, so deinterlacing is not compared here.
The first candidates tune the ffmpeg baseline (threshold, width); learned detectors (TransNetV2, PySceneDetect) join as venv workers
(`runtimes/worker.py`), their libraries in a venv of their own, never in this service's dependencies.

**Subjective cuts** (later, role `cut-points`): where to cut inside one shot, which has no single right answer. Two ways to measure it: a
score from what the cut lands on (does it break a spoken phrase: word times from transcription; does it interrupt a movement: motion just
before and after the cut), and blind A/B judging in the Judge tab, two candidates' cuts of the same shot played side by side
(`primary = "judged"`), with the score checked against people's choices like a model judge.

## OCR: which cleanup and which reader on hard book pages
`ocr-pages` (`evals/specs/ocr-pages.toml`, built by `scripts/evals/ocr_pages.py` from strata-scan or a folder of PNGs) is ten hard pages of
two public-domain books (title pages, display-type adverts, small capitals, italics, footnotes, a grey 1828 scan) with their text typed by
hand from the page (`scripts/evals/ocr_pages/<item>.txt`). The item's only input is `image` (the dpi is in the PNG), so Tesseract and a
vision-language model read the same thing; `cer` decides, `wer` is context, and `tags.hard` breaks the report down by what makes a page hard.
- **Before OCR**: Tesseract candidates differ only in `clean` (steps of `runtimes/pageclean.py`: flatten, contrast, sharpen, denoise, otsu,
  adaptive, upscale) and `psm`.
- **Instead of OCR**: a vision-language model (`ocr-read-v1`) transcribes the page; it gives text but no boxes, so it is a candidate only.
- **After OCR**: a pipeline whose first stage is Tesseract (`into = "prompt"`) and whose candidate is a vision-language model with
  `ocr-correct-v1`, which corrects the OCR text against the page.
- **Judging OCR without a reference**: three vision judges (qwen, gemma, Gemini through agy) see the page and two transcriptions with the
  reference hidden (`reference = false`), each candidate against the baseline (`pairs = "baseline"`). The analysis scores each judge
  against `cer` (the right answer on a pair is the transcription with fewer errors), so the judge that agrees best can rank settings on the
  pages nobody has typed.
Ten items are few: expect "no clear difference" on small effects, and read the per-tag rows before changing strata-scan's defaults.

## Page rebuild: how close a rebuilt PDF gets to the cleaned page
The `page-build` role (`runtimes/pagebuild.py`, CPU) draws a page description (strata-python's `strata_python.pages`, the renderer
strata-scan prints with) as a PDF with real text, plus its raster at the scan's dpi; it only draws, so every candidate of a page-rebuild
eval differs in what fills the description (Tesseract's boxes, a layout model, a vision model's JSON, `font-match`'s fitted faces). The
baseline is the plain overlay: `ocr` given instead of `page`, one block per Tesseract block, each line one roman span in Times.
The reference is the cleaned page as a description (`reference.page`, boxes in scan pixels). Layout is scored by `block_f1` and
`figure_f1` (parts matched at IoU 0.5, greedy, each once; a block split into two or more pieces, or blocks merged into one, match nothing whatever the IoU), `role_acc` (right role among the
blocks found) and `order_tau` (Kendall's tau of the reading order of the blocks found); text by `cer_exact` on the text read back from the
PDF. Separate evals per stage (layout, text, style, figures, whole page): see the tracking issue jburnhams/strata-models#67.

`page-layout` (`evals/specs/page-layout.toml`, built by `scripts/evals/page_layout.py`) is the first of these evals: ten whole pages of the
two books (Ward 1874's title page, dialogue and two pages of advertisements; Collier 1828's chapter openings, prose, footnotes and a sonnet)
with their layout labelled by eye in `scripts/evals/page_layout/pages.toml` (86 blocks, 12 ornaments and rules; the conventions are at the top
of that file and in preset `page-layout-v1`, so a reader is told the same rules it is scored by). Each item's input is the scan (`image`)
and Tesseract's reading of it (`ocr`, psm 3, as strata-scan reads pages). The baseline draws the `ocr`; each vision-model candidate is a
pipeline whose stage answers preset `page-layout-v1` (schema-checked JSON: blocks in reading order with boxes in thousandths, roles and
lines; figures) `into = "layout"`, which page-build draws in the picture's pixels. `page_ink_f1` compares page-build's raster with the scan
inside the reference text blocks (borders, stamps and show-through count for nobody).

The `*-fix` candidates do for layout what `tesseract-*-fix` does for OCR text: a page-build stage numbers Tesseract's lines (`list_lines`: number,
Tesseract's block, box in thousandths, text; a line is split where its words stand more than 1.5 line heights apart, so a running head and its
folio part) `into = "context"`, a vision model sees them beside the page (`also = ["image"]`, preset `page-layout-fix-v1`, which pastes a
text `context` in as it is) and answers which lines make each block, with roles and figures, `into = "groups"`, and page-build draws
Tesseract's own lines in those blocks. Grouping the lines by the labels themselves scores block_f1 97.6 % (the ceiling of the approach: the
rest is lines Tesseract never read), against 84 % before the split. psm 3 stays: the settings best for OCR text find fewer blocks (block_f1
psm 4 67 %, flatten 64 %, flatten + psm 4 63 %, against 77 %).

The default (2026-11-23): the `page-build` role runs model `pagebuild-opus-layout`, whose option `group = "claude-opus-5-5-api"` makes a job
that sends a plain `ocr` answer and the picture do the opus-fix-v2 steps itself (list the lines, ask Opus with page-layout-fix-v2, draw in its
blocks): block_f1 96.5 % on the 20 pages against Tesseract's 64 %, about 3 cents a page. A failed call draws Tesseract's blocks and says why in
the answer's `grouper`. The eval candidates name model `pagebuild` (draws only), so their baselines and stages are unchanged.

`page-rebuild` (`evals/specs/page-rebuild.toml`, the page-layout dataset) puts the stage winners together and asks what each is worth to the
whole page, as a ladder: `tesseract`; `layout` (Sonnet regroups Tesseract's lines, page-layout-fix-v2); `layout-text` (+ Haiku corrects
each numbered line against the page, preset page-text-fix-v1, answering only the lines it changes, `{lines: [{n, text}]}`, given to
page-build as `texts`); `layout-text-type` (+ a font-match stage over the page's Tesseract words with Sonnet's shortlist, `answer = "json"`
into `font_fit`: each word set in the fitted face and style where it was measured, a corrected line with other words keeping the fit's face
and size, spaced to its box); `full` (+ `snap`: sizes within 10 % of their block's median take it, each word's width kept; `styles`: words
fitted bold or small capitals, which a word-by-word fit to worn print calls now and then, set in roman; `crop`: figures cut from the scan,
paper whitened; `rules`: the scan's long thin straight lines near the print, the border and the rules page-layout-fix-v2 leaves out, cut
from the scan as figures); and `full-opus` on Opus's layout. The primary metric `page_ink_all` reads ink over the whole printed area (the
labelled blocks and figures, grown by 6 % of the page, so the border counts and the book's edge beside the leaf does not); `page_ink_f1`
the text blocks alone. Text metrics (`cer_exact`, `wer_bag`, on the text read back from the PDF) need a reading of each page:
`scripts/evals/ocr_reference.py --pages <data>/evals/page-layout` has Opus read the pictures (about $0.03 a page, kept beside them in
`text/`), and page_layout.py adds them as `reference.text`. Tried here on three pages with layout from the labels, fonts from a three-face
shortlist (2026-11-23): page_ink_all 42 % to 72 % (Collier verse), 43 % to 85 % (Ward dialogue, its border drawn) and 30 % to 59 %
(Ward advertisement in display faces, which no single face fits).

## Files in candidate params
A candidate's params may name a file under the service's data folder as `"{data}/..."` (the runner's voice for the cloning TTS
candidates, for one), so a spec in this public repository names no machine's paths. `{data}` is filled in when each job is made, so the
result key, and with it every cached output, is the same as with the full path.

## Made from other candidates
A candidate may name `from = ["a", "b"]`: its job gets those candidates' output paths for the same item as `sources`, so a refinement of
an upscale (back-projection, a blend) is judged like any candidate without running the upscalers again. It runs after them in the same run,
waits (is not run, is not a failure) while any source is missing or failed, and its key includes theirs, so a new source output makes it
anew. One level only: a source may not itself be made from others.

## Fonts: which face a scanned block was printed in (`font-match`)
The role (`runtimes/fontfit.py`, CPU) sets a block's words, from Tesseract's boxes, in a candidate face and searches size, width (hscale),
tracking and weight to make the rendering's ink agree with the scan's (`evals/ink.py`: F1 with a pixel or two of tolerance); each word then
keeps the style that fits it, and the face that agrees best wins. Faces are open period fonts fetched from Fontsource (every Google Fonts family and more; a few, such as Century
Schoolbook and Latin Modern, from CTAN), listed in `runtimes/fonts.toml` (74 in 10 classes after Vox-ATypI with the 19th-century display kinds apart: modern, fat face, Clarendon, slab, grotesque, decorated...).
- **Dataset** (`scripts/evals/font_match.py`, from archive.org, no strata-scan needed): 38 real blocks of Ward 1874 and Collier 1828
  (`font_match/regions.toml`, class and style labelled by eye) and 30 synthetic blocks, the books' text set in a known catalogue face and
  printed badly (ink spread or starved, paper tone, noise, blur, tilt). Period type mostly exists in no font file, so real blocks are scored
  by ink and by class (`font_close`: 1 the same class, 0.5 a neighbour), and the synthetic ones by exact face (`font_top1`, `font_top3`) and size.
- **Candidates**: `naive` (Tesseract's boxes in Tinos, each word stretched to its box: what a rebuild does without the role), `exhaustive`
  (every face: a quick screen, then the full fit of the best 6), a vision model's shortlist then the fit (`qwen-fit`, `gemma-fit`,
  `flash-fit`, `haiku-fit`, `sonnet-fit`, `opus-fit`; preset `font-propose-v1` lists the catalogue), and the shortlist alone (`*-only`, `mode = "measure"`:
  its first face sized from the measured x-height), which shows what the fit adds.
- `ink_f1` decides against the baseline, but the fit maximises it, so compare the fitted candidates with each other on it and read
  `font_top3` and `font_close` beside it: a shortlist that keeps both against `exhaustive` in less time is the one to adopt.
- Model judges see the block and two renderings and pick the one whose type looks more like the print; the analysis scores them against
  `ink_f1`.
- Chosen (2026-11-23, Jonathan, from the 2026-11-15 trial): Claude Sonnet's shortlist, then the fit. The role does both itself: the
  `fontfit` model's `propose` option names the proposer, asked whenever a job gives no candidates (`propose = false` turns it off, as
  `exhaustive` does); if the call fails, every face is screened instead.
- Before the role is deployed, `scripts/evals/font_match_trial.py` runs the same candidates with the stages through the running service and
  the fit in its own process.

## Rules of thumb
- Compare on the same items, always; never compare numbers from two datasets or two dataset versions.
- One primary metric. Others are guards or context; they do not decide.
- A difference whose interval includes zero is "no clear difference", not "slightly better".
- Change the dataset, not the method, when results look wrong; then the version changes and the old report stays true for the old version.
- Keep failures in: a failed job is all errors for a rate and wrong for a share, never dropped.
