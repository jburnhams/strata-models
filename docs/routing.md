# Routing: cloud or local

Every way a role can be served, local and cloud, mapped to power and cost, ranked for a job. Code: [`routing.py`](../src/stratamodels/routing.py); data: [`catalogue.toml`](../src/stratamodels/catalogue.toml) (the data folder's `catalogue.toml` is merged over it, field by field).

```bash
strata-models catalogue                         # every backend and cloud model: order, use, tier, price or quota a call
strata-models route vision-language --goal best # ask the running service: options ranked, the pick, and what it would pick if every backend could be called
curl 'http://omarchy:8366/api/route/text?goal=cheapest&private=true'
curl http://omarchy:8366/api/allowances         # what is left on agy, codex and the API keys (cached 5 min)
```

## What it compares

| Source | Gives |
|---|---|
| `catalogue.toml` `[backends]` | kind (local, subscription, api), `order` to spend in, `use` (Jonathan's words), the reserve (`min_left`), where the allowance is read, whether inputs leave the machine, plan price and electricity (for `usd_equiv`) |
| `catalogue.toml` `[offers]` | cloud models: roles, family, tier (1-5), price per Mtok or measured quota per call, which recorded models stand for it in evals, whether it takes pictures or audio |
| `catalogue.toml` `[local]` | tiers of the local LLMs (the rest comes from the registry) |
| registry (`models.toml`) | local models and their VRAM; a cloud model once it has a runtime here (matched to its offer by backend and `options.model`; `options.price` wins over the catalogue's) |
| job history (`model_stats`) | a local model's mean run seconds per call |
| eval summaries (`eval_summaries`) | per role and model: better (adopted over the baseline), worse (rejected, worse on the primary metric), same, or baseline. A recorded cloud model's verdict counts for its offer, and for the same model on another backend (`family`) |
| `evals.cloud.USAGE` / `stratamodels.spend.summary()` | what is left: agy and codex from their own CLIs (no model call), the API keys from their spend tracking |

Costs stay in their own units: GPU seconds, a share of a subscription window, requests (NIM), dollars. `usd_equiv` puts all of them in dollars (electricity; the plan's price spread over its calls) only to read side by side.

## The rules

Jonathan, 2026-11-14:

| Order | Backend | Use |
|---|---|---|
| 0 | local GPU and CPU | free (one card: VRAM and the queue) |
| 1 | agy (Google AI Pro), NVIDIA NIM, the free Gemini key | free but limited: as much as the limits allow; agy before a Gemini key for the same model |
| 2 | codex (ChatGPT Plus) | fairly sparingly: keeps 30 % in reserve |
| 3 | Gemini paid key ($10 a month, from the 1st) | only as a fallback after agy |
| 3 | Claude API ($100 until 2026-12-08, then monthly) | watched closely |

An option is left out when it cannot be called here, the job is `private` and the inputs would leave the machine, it takes no pictures (or audio) and the role needs them, an eval found it worse than the baseline, its tier is under the floor (by default the role's current model's tier; an eval winner passes anyway), its allowance is at or under the reserve (or one call would take it there), the paid key has less than a call costs, or it is a fallback whose primary has the same model.

- `goal=cheapest` (default): the first backend in the order; on it an eval winner, then the role's own model (an eval chose it), then the least spent.
- `goal=best`: eval verdict, then tier, then cost.
- `would_pick`: the option it would choose if every backend could be called (a subscription with no live runtime, a cloud model not yet registered): the case for the next runtime.

Model choices still come from evals with the current model as baseline: routing reads their verdicts and never overrides them with a tier.

## Not yet

- Jobs are not dispatched by it: `route` recommends. Next: a job with `model = "auto"` (or an app preference) resolved through `plan()` in the scheduler.
- The subscriptions have no live runtime (only `recorded` for evals, through `scripts/evals/record_cloud.py`); a live one goes in the CPU lane (`vram_gb = 0`).
- Quotas per call are measured for Gemini Flash, Argon and Sonnet on agy and pictures on both; the rest (Opus on agy, every codex text model) are guesses marked in the catalogue.
- The local options ignore how long the GPU queue is.
