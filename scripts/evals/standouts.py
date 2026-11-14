"""Pick an eval's stand-out items for the cloud models (Jonathan, 2026-11-14): the items where a metric scores the output, the local models
already do a reasonable job, and there is the most room left to improve. A cloud candidate run on only these says, for little money, how
much better it is and what one item costs (metric cost_usd), beside the local model's GPU time.

    uv run python scripts/evals/standouts.py ocr-pages --metric cer --local qwen-read gemma-read [--n 2] [--url http://127.0.0.1:8366]

Reads the experiment's outputs from the running service (GET /api/evals/NAME/outputs), scores each item with the metric, takes for each item
the best of the local candidates, drops the items they failed or did worst on (the bottom quarter: the cloud model would be rescuing a broken
item, not improving a reasonable one) and the top quarter (nothing left to gain), and prints the N with the most room, as a candidate's
`items = [...]` line."""

import argparse
import json
import sys
import urllib.request

from stratamodels.evals import metrics as M


def scores(items, metric, local):
    """[(item id, best local score)] of the items every named candidate has an output for and at least one scored."""
    m = M.METRICS[metric]
    best = max if m.better == "higher" else min
    out = []
    for it in items:
        vals = []
        for name in local:
            o = (it.get("outputs") or {}).get(name)
            if o is None:
                continue
            v = m.score(dict(it, _run_s=o.get("run_s")), o.get("output"), o.get("error"))
            if isinstance(v, tuple):  # a ratio metric scores (errors, count)
                v = v[0] / v[1] if v[1] else None
            if isinstance(v, bool):
                v = float(v)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            out.append((it["id"], best(vals)))
    return out


def pick(scored, better, n=2):
    """The n items in the middle half by score (not the worst quarter, not the best) with the most room: lowest first when higher is
    better, highest first when lower is better."""
    if not scored:
        return []
    good_first = sorted(scored, key=lambda s: -s[1] if better == "higher" else s[1])
    q = len(good_first) // 4
    middle = good_first[q : len(good_first) - q] if len(good_first) >= 4 else good_first
    return [i for i, _ in reversed(middle)][:n]


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("name")
    a.add_argument("--metric", required=True)
    a.add_argument("--local", nargs="+", required=True, help="the local candidates' names in the spec")
    a.add_argument("--n", type=int, default=2)
    a.add_argument("--url", default="http://127.0.0.1:8366")
    a = a.parse_args(argv)
    with urllib.request.urlopen(f"{a.url}/api/evals/{a.name}/outputs", timeout=120) as r:
        d = json.load(r)
    s = scores(d["items"], a.metric, a.local)
    if not s:
        sys.exit(f"no item has a {a.metric} score from {', '.join(a.local)}")
    ids = pick(s, M.METRICS[a.metric].better, a.n)
    for i, v in sorted(s, key=lambda x: x[1]):
        print(f"{'*' if i in ids else ' '} {i:40} {v:.4f}", file=sys.stderr)
    print(f"items = {json.dumps(ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
