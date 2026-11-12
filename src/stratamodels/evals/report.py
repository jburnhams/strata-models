"""The write-up: every eval report has the same sections in the same order, written from the analysis, so two reports read alike and a reader knows
where to look. A person adds the parts no number gives (what the failures look like, what was decided and why) under the headings left for them.

Sections: Question and answer (the decision first), Setup (dataset and version, candidates with their full config, metrics and how they are
computed, the decision rule), Results (every metric with its 95 % interval), Comparisons with the baseline (paired differences, intervals, p
before and after Holm), Human judgments (if any), Cost (run time, load time), Failures (examples, by item id), Method notes (the standard
text on intervals and tests), Decision and follow-ups (for the person), Reproduce (the commands)."""

import datetime
import json
import math

from stratamodels.evals import metrics as M

METHOD = """- Each candidate is compared with the baseline on the items both finished, so comparisons are paired; a candidate run on a sample
  (`items` in the spec) is compared on its sample only.
- Intervals are 95 %. Shares use the Wilson score interval. Means and corpus rates (WER, CER, missed faces: all errors over all reference units)
  use a percentile bootstrap over items, with 4,000 resamples. Dataset-level metrics (verification AUC, TAR) use 200 resamples of items.
- Differences are candidate minus baseline on the same items. Their intervals come from the bootstrap of the paired differences, and p comes
  from a sign-flip (swap) permutation test. Right/wrong metrics use the exact McNemar test on the items where only one of the two was right.
- With several candidates, the primary metric's p values are also given after Holm's correction for multiple comparisons.
- "Better" means the whole interval of the difference is on the better side of zero. "Adopt" also needs the improvement to reach the
  experiment's minimum effect and every guard to hold.
- Text is normalised before WER and CER: lower case, punctuation removed, whitespace collapsed. Number words are not unified ("ten" against "10"
  counts as an error).
- Run time is the model's own time per item, with the model already loaded. Load time is given apart. The GPU ran nothing else meanwhile (one
  queue)."""


def _v(m, x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if m.unit == "%":
        return f"{x * 100:.2f}%"
    if m.unit == "s":
        return f"{x:.3f} s"
    return f"{x:.4f}"


def _d(m, x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 100:+.2f} pts" if m.unit == "%" else f"{x:+.3f}{' s' if m.unit == 's' else ''}"


def _p(p):
    return "n/a" if p is None or (isinstance(p, float) and math.isnan(p)) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def render(res, examples=None, date=None, pipelines=None):
    """Markdown for one analysis (core.analyse). `examples`: {candidate: [(item_id, text)]} of failures or worst items, shown under Failures.
    `pipelines` (from pipelines()): what each pipeline candidate's steps were sent and answered, shown under Pipelines."""
    e = res["experiment"]
    ds = res["dataset"]
    dec = res["decision"]
    names = [c["name"] for c in e["candidates"]]
    base = e["baseline"]
    prim = M.METRICS[e["primary"]]
    date = date or datetime.date.today().isoformat()
    L = []
    L += [
        f"# {e['name']}: {e['question']}",
        "",
        f"{date} · dataset `{ds['name']}` version `{ds['version']}` · {ds['compared']} of {ds['n']} items with the baseline's output · role `{e['role']}`",
        "",
    ]
    L += ["## Answer", ""]
    if dec["change"]:
        L += [f"**Change to `{dec['recommendation']}`.** {dec['per_candidate'][dec['recommendation']]['why']}.", ""]
    else:
        L += [
            f"**Keep `{base}`.** No candidate is clearly better on {e['primary']} by at least the minimum effect within its guards.",
            "",
        ]
    for n, v in dec["per_candidate"].items():
        L.append(f"- `{n}`: **{v['verdict']}**, {v['why']}.")
    L += [
        "",
        "## Setup",
        "",
        f"- **Dataset:** `{ds['name']}` ({ds['n']} items, version `{ds['version']}`). {ds.get('description') or ''}".rstrip(),
        f"- **Primary metric:** {e['primary']}, where {prim.better} is better. Minimum effect worth a change: {_d(prim, (1 if prim.better == 'higher' else -1) * e['min_effect']).lstrip('+-')}.",
    ]
    if e["guards"]:
        L.append("- **Guards:** " + ", ".join(f"{g} at most {v:g}x the baseline" for g, v in e["guards"].items()) + ".")
    L += ["- **Metrics:**"] + [f"  - `{m}`: {M.METRICS[m].description}." for m in e["metrics"]]
    L += ["- **Candidates** (baseline first):", ""]
    L += ["| Candidate | Model | Params | Load options |", "|---|---|---|---|"]
    for c in sorted(e["candidates"], key=lambda c: c["name"] != base):
        chain = "".join("`" + (st.get("model") or st["role"]) + "` → " for st in c.get("stages") or [])
        L.append(
            f"| `{c['name']}`{' (baseline)' if c['name'] == base else ''} | {chain}`{c['model']}` | `{json.dumps(c['params'], sort_keys=True)}` | `{json.dumps(c['options'], sort_keys=True)}` |"
        )
    staged = [c for c in e["candidates"] if c.get("stages")]
    if staged:
        L += ["", "Pipeline steps (each runs before the candidate's own job; its answer replaces the named param):", ""]
        for c in staged:
            for i, st in enumerate(c["stages"], 1):
                L.append(
                    f"- `{c['name']}` step {i}: role `{st['role']}`, model `{st.get('model') or '(the role default)'}`, params `{json.dumps(st['params'], sort_keys=True)}`, sent `{st['input']}` as `{st['send']}`, answer into `{st['into']}`"
                    + (f". {st['note']}" if st.get("note") else "")
                )
    L += [
        "",
        "## Results",
        "",
        "Each candidate is compared with the baseline on the items both have: a candidate run on a sample (`items` in its spec) has fewer, and"
        " intervals as wide as its sample earns.",
        "",
        "| Candidate | Items | " + " | ".join(e["metrics"]) + " |",
        "|---|---|" + "---|" * len(e["metrics"]),
    ]
    for n in names:
        mm = res["candidates"][n]["metrics"]
        L.append(
            f"| `{n}` | {res['candidates'][n].get('compared', res['dataset']['compared'])} | "
            + " | ".join(
                f"{_v(M.METRICS[m], mm[m]['value'])} [{_v(M.METRICS[m], mm[m]['low'])}, {_v(M.METRICS[m], mm[m]['high'])}]"
                if m in mm
                else "n/a"
                for m in e["metrics"]
            )
            + " |"
        )
    L += ["", "Each cell: value [95 % interval].", ""]
    if res["comparisons"]:
        L += [
            "## Compared with the baseline",
            "",
            f"Difference = candidate minus `{base}`, on the same items.",
            "",
            "| Candidate | Metric | Difference [95 % interval] | p | p (Holm) | Reading |",
            "|---|---|---|---|---|---|",
        ]
        for n, comp in res["comparisons"].items():
            for m, c in comp.items():
                mt = M.METRICS[m]
                read = {True: "better", False: "worse", None: "no clear difference"}[c["improves"]]
                extra = (
                    f" ({c['only_baseline']} items only the baseline got, {c['only_candidate']} only the candidate)"
                    if "only_baseline" in c
                    else ""
                )
                L.append(
                    f"| `{n}` | {m} | {_d(mt, c['diff'])} [{_d(mt, c['low'])}, {_d(mt, c['high'])}] | {_p(c['p'])} | {_p(c.get('p_holm')) if m == e['primary'] else ''} | {read}{extra} |"
                )
        L.append("")
    if res.get("judgments"):
        L += [
            "## Human judgments",
            "",
            f"{res['judgments']['n']} blind pairwise judgments. Bradley-Terry strength (1 = average) [95 % interval]:",
            "",
        ]
        L += [
            f"- `{n}`: {s[0]:.2f} [{s[1]:.2f}, {s[2]:.2f}]"
            for n, s in sorted(res["judgments"]["strength"].items(), key=lambda kv: -kv[1][0])
        ] + [""]
    L += [
        "## Cost",
        "",
        "| Candidate | Items run | Failed | Run time per item (mean, p95) | Load time |",
        "|---|---|---|---|---|",
    ]
    for n in names:
        c = res["candidates"][n]
        lat = c["metrics"].get("latency_s", {})
        load = "n/a" if c["load_s"] is None else f"{c['load_s']:.1f} s"
        L.append(
            f"| `{n}` | {c['done']} | {c['failed']} | {_v(M.METRICS['latency_s'], lat.get('value'))}, {_v(M.METRICS['latency_s'], lat.get('p95'))} | {load} |"
        )
    if pipelines:
        L += [
            "",
            "## Pipelines",
            "",
            "What each pipeline's steps were sent and answered (all of them: the Outputs tab; the judges saw only the end result).",
            "",
            "| Candidate | Words in, out (mean) | Step time per item (mean) | Failed steps |",
            "|---|---|---|---|",
        ]
        for n, p in pipelines.items():
            L.append(f"| `{n}` | {p['words_in']:.0f}, {p['words_out']:.0f} | {p['stage_s']:.1f} s | {p['failed']} |")
        for n, p in pipelines.items():
            L += ["", f"`{n}`:"]
            for item_id, steps in p["examples"]:
                L.append(f"- `{item_id}`:")
                for sent, answer in steps:
                    L += [f"  - sent: {sent}", f"  - answered: {answer}"]
    L += ["", "## Failures and worst items", ""]
    if examples:
        for n, ex in examples.items():
            if ex:
                L += [f"`{n}`:"] + [f"- `{i}`: {t}" for i, t in ex] + [""]
    else:
        L += ["_(none listed)_", ""]
    L += [
        "## What the numbers do not show",
        "",
        "_(Written by a person: what the failures look like, anything about the dataset that limits the answer.)_",
        "",
    ]
    L += [
        "## Decision and follow-ups",
        "",
        "_(Written by a person: what was decided, the registry change if any, and what to try next.)_",
        "",
    ]
    L += [
        "## Method",
        "",
        METHOD,
        "",
        "## Reproduce",
        "",
        "```",
        f"./strata-models eval run {e['name']}",
        f"./strata-models eval report {e['name']}",
        "```",
        "",
    ]
    return "\n".join(L)


def worst(spec, res_outputs, items, metric, k=5):
    """Up to `k` (item_id, line) per candidate: failed jobs first, then the items worst on `metric` (for ratio metrics, by errors per unit)."""
    m = M.METRICS[metric]
    out = {}
    for n, outs in res_outputs.items():
        rows = []
        for it, o in zip(items, outs):
            if o is None:
                continue
            if o[1]:
                rows.append((float("inf"), it["id"], f"failed: {o[1][:160]}"))
                continue
            s = m.score(dict(it, _run_s=None), *o) if m.score else None
            if s is None:
                continue
            v = s[0] / max(s[1], 1) if m.shape == "ratio" else float(s)
            bad = v if m.better == "lower" else -v
            if m.shape == "ratio" and s[0] == 0:
                continue
            text = M.transcript(o[0])[:160] if spec.role.startswith("transcribe") else ""
            rows.append((bad, it["id"], f"{metric} {_v(m, v)}" + (f': "{text}"' if text else "")))
        out[n] = [(i, t) for _, i, t in sorted(rows, key=lambda r: -r[0])[:k]]
    return out


def pipelines(spec, res_outputs, items, k=2):
    """{pipeline candidate: dict(words_in, words_out, stage_s, failed, examples=[(item_id, [(sent, answered) per step])])}: the first `k` items with
    every step's text, and means over all its outputs. Words in = the item's input to the first step, out = what the last step answered."""
    out = {}
    for c in spec.candidates:
        if not c.stages:
            continue
        win, wout, secs, ex, failed = [], [], [], [], 0
        for it, o in zip(items, res_outputs.get(c.name) or []):
            if o is None:
                continue
            trace = (o[0] or {}).get("pipeline") if isinstance(o[0], dict) else None
            if not trace:
                failed += 1 if o[1] and "stage" in o[1] else 0
                continue
            steps = trace[:-1]
            win.append(len(str(steps[0]["sent"]).split()))
            wout.append(len(str(steps[-1]["answer"]).split()))
            secs.append(sum(t.get("run_s") or 0 for t in steps))
            if len(ex) < k:
                ex.append((it["id"], [(t["sent"], t["answer"]) for t in steps]))
        mean = lambda v: sum(v) / len(v) if v else float("nan")
        out[c.name] = dict(words_in=mean(win), words_out=mean(wout), stage_s=mean(secs), failed=failed, examples=ex)
    return out
