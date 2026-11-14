"""Cloud or local: every way a role can be served here (the GPU and CPU lanes, the subscription CLIs, the API keys), each with its power, its
cost and what is left of its allowance, ranked so a job goes where it should.

The catalogue (catalogue.toml in this package, the data folder's merged over it) describes the backends and the cloud models; the registry
gives the local models (VRAM) and any cloud model already served (runtime, options.price); the job history gives a local model's GPU seconds
per call; the eval summaries say which models beat the role's current one on real data (Jonathan's rule: model choices come from evals, the
current model as baseline), and outrank the catalogue's tier guess.

Costs are kept in their own units, because they are not the same kind of thing: GPU seconds (free, but one card), a share of a
subscription's window (free until it runs out), requests (NIM), dollars (the paid keys). Each backend has an `order` (Jonathan, 2026-11-14:
the GPU, then the free but limited ones, then codex sparingly, then the paid keys), and `usd_equiv` prices every option in dollars
(electricity, the plan's monthly price spread over its calls) only so they can be read side by side.

goal "cheapest" (the default): among the options no worse than the role's current model, the one that spends first in that order.
goal "best": the strongest (eval verdict, then tier), then the cheapest of those.
Nothing here calls a model or handles a key: allowances are read from the CLIs' own usage commands (evals.cloud.USAGE) and the API keys'
spend tracking (stratamodels.spend, when it is deployed)."""

import dataclasses
import math
import os
import threading
import time
import tomllib

from stratamodels import settings

BUILTIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalogue.toml")
GOALS = ("cheapest", "best")
IMAGES = {"vision-language", "judge-vision", "ocr"}  # roles whose input is a picture: a cloud model must take one
AUDIO = {"transcribe", "transcribe-fast", "judge-audio"}
EVAL_RANK = {"better": 1, "same": 0, "baseline": 0, "worse": -1}


class CatalogueError(ValueError):
    """catalogue.toml is wrong."""


@dataclasses.dataclass(frozen=True)
class Backend:
    key: str
    kind: str  # local | subscription | api
    name: str = ""
    order: int = 9
    use: str = ""
    min_left: float = 0.10
    allowance: str = ""  # agy | codex | spend:<provider> | ""
    leaves: bool = True
    plan_usd_month: float | None = None
    watts: float | None = None
    usd_per_kwh: float | None = None
    fallback_for: str = ""
    note: str = ""

    def to_json(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Offer:
    key: str
    backend: str
    model: str
    roles: tuple
    family: str = ""
    tier: int | None = None
    images: bool = False
    audio: bool = False
    price: dict | None = None  # USD per million tokens {in, out}
    quota: dict | None = None  # {bucket, per_call, window?}
    evals: tuple = ()
    note: str = ""

    def to_json(self):
        d = dataclasses.asdict(self)
        d["roles"], d["evals"] = list(self.roles), list(self.evals)
        return d


def _read(path):
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as e:
        raise CatalogueError(f"{path}: {e}")


def _merge(base, extra):
    """Tables merged all the way down: the data folder's file changes one field of one backend (or of an offer's quota) without restating the
    rest. A value that is not a table replaces the one below it."""
    out = dict(base)
    for k, v in extra.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Catalogue:
    def __init__(self, backends, offers, local=None, defaults=None):
        self.backends, self.offers, self.local, self.defaults = (
            backends,
            offers,
            dict(local or {}),
            dict(defaults or {}),
        )
        for o in offers.values():
            if o.backend not in backends:
                raise CatalogueError(f"offer {o.key!r} names backend {o.backend!r}, which is not in the catalogue")
        for b in backends.values():
            if b.fallback_for and b.fallback_for not in backends:
                raise CatalogueError(
                    f"backend {b.key!r} is a fallback for {b.fallback_for!r}, which is not in the catalogue"
                )

    @classmethod
    def from_dict(cls, d):
        dflt = d.get("defaults") or {}
        backends = {}
        for k, b in (d.get("backends") or {}).items():
            try:
                backends[k] = Backend(
                    key=k,
                    kind=str(b["kind"]),
                    name=str(b.get("name", k)),
                    order=int(b.get("order", 9)),
                    use=str(b.get("use", "")),
                    min_left=float(b.get("min_left", dflt.get("min_left", 0.10))),
                    allowance=str(b.get("allowance", "")),
                    leaves=bool(b.get("leaves", b["kind"] != "local")),
                    plan_usd_month=None if b.get("plan_usd_month") is None else float(b["plan_usd_month"]),
                    watts=None if b.get("watts") is None else float(b["watts"]),
                    usd_per_kwh=None if b.get("usd_per_kwh") is None else float(b["usd_per_kwh"]),
                    fallback_for=str(b.get("fallback_for", "")),
                    note=str(b.get("note", "")),
                )
            except (KeyError, TypeError, ValueError) as e:
                raise CatalogueError(f"backend {k!r}: {e!r}")
            if backends[k].kind not in ("local", "subscription", "api"):
                raise CatalogueError(f"backend {k!r}: kind is local, subscription or api, not {backends[k].kind!r}")
        offers = {}
        for k, o in (d.get("offers") or {}).items():
            try:
                offers[k] = Offer(
                    key=k,
                    backend=str(o["backend"]),
                    model=str(o["model"]),
                    roles=tuple(o.get("roles") or ()),
                    family=str(o.get("family", o["model"])),
                    tier=None if o.get("tier") is None else int(o["tier"]),
                    images=bool(o.get("images", False)),
                    audio=bool(o.get("audio", False)),
                    price=dict(o["price"]) if o.get("price") is not None else None,
                    quota=dict(o["quota"]) if o.get("quota") is not None else None,
                    evals=tuple(o.get("evals") or ()),
                    note=str(o.get("note", "")),
                )
            except (KeyError, TypeError, ValueError) as e:
                raise CatalogueError(f"offer {k!r}: {e!r}")
        return cls(backends, offers, d.get("local"), dflt)

    @classmethod
    def load(cls, builtin=BUILTIN, override=None):
        """The built-in catalogue with the data folder's catalogue.toml merged over it."""
        return cls.from_dict(
            _merge(_read(builtin), _read(override or os.path.join(settings.data_dir(), "catalogue.toml")))
        )

    def tokens(self, role):
        return (self.defaults.get("tokens") or {}).get(role) or dict(
            self.defaults.get("default_tokens") or {"in": 2000, "out": 500}
        )

    def to_json(self):
        return dict(
            backends=[b.to_json() for b in sorted(self.backends.values(), key=lambda b: (b.order, b.key))],
            offers=[o.to_json() for o in self.offers.values()],
            local=self.local,
            defaults=self.defaults,
        )


# ---- eval evidence ----------------------------------------------------------------------------------------------------------------------------
def evidence_from(summaries):
    """{(role, model key): [verdict]} from experiments' analyses (evals.core.analyse: the kept eval_summaries rows). A candidate's verdict
    against its experiment's baseline becomes better (adopted), worse (rejected and worse on the primary metric), or same (unclear, or
    rejected for a small gain or a broken guard such as latency: not worse at the job); the baseline itself is "baseline"."""
    out = {}
    for res in summaries:
        exp = (res or {}).get("experiment") or {}
        role, base = exp.get("role"), exp.get("baseline")
        per = ((res or {}).get("decision") or {}).get("per_candidate") or {}
        base_model = next((c.get("model") for c in exp.get("candidates") or [] if c.get("name") == base), None)
        for c in exp.get("candidates") or []:
            if c.get("stages") or c.get("options"):
                continue  # a pipeline or a load variant: its verdict is that configuration's, not the model's as served
            name, model = c.get("name"), c.get("model")
            if not model or (name != base and model == base_model):
                continue  # no model, or the baseline's own model in another configuration (beam 1 against beam 5): tuning, not a model choice
            if name == base:
                v = dict(experiment=exp.get("name"), candidate=name, verdict="baseline", why="")
            elif name in per:
                d = per[name]
                gain = d.get("gain")
                worse = d.get("verdict") == "reject" and isinstance(gain, (int, float)) and gain == gain and gain < 0
                v = dict(
                    experiment=exp.get("name"),
                    candidate=name,
                    verdict="better" if d.get("verdict") == "adopt" else "worse" if worse else "same",
                    why=d.get("why", ""),
                )
            else:
                continue  # not run yet
            out.setdefault((role, model), []).append(v)
    return out


def evidence(engine):
    """evidence_from the kept analyses in the database (never worked out here: a stale one is still the best reading there is)."""
    from sqlalchemy import select

    from stratamodels.db import session
    from stratamodels.db.models import EvalSummary

    with session(engine) as s:
        return evidence_from([r.result for r in s.scalars(select(EvalSummary))])


def verdict(found):
    """One verdict from several experiments: better if one adopted it and none found it worse, worse the other way round, else same; None
    when no eval has it."""
    vs = {f["verdict"] for f in found}
    if not vs:
        return None
    if vs == {"baseline"}:
        return "baseline"
    if "better" in vs and "worse" not in vs:
        return "better"
    if "worse" in vs and "better" not in vs:
        return "worse"
    return "same"


# ---- allowances -----------------------------------------------------------------------------------------------------------------------------
class Allowances:
    """What is left of each backend's allowance, read at most every `ttl_s` (agy /usage takes seconds; none of these readings calls a model).
    left(offer) -> {left: 0-1 or None, detail, error?, usd_left?, unit?}."""

    def __init__(self, catalogue, usage=None, spend=None, ttl_s=300.0, clock=time.time):
        self.cat, self.usage, self.spend, self.ttl_s, self.clock = catalogue, usage, spend, ttl_s, clock
        self.memo, self.guard = {}, threading.Lock()

    def _usage(self):
        if self.usage is None:
            from stratamodels.evals import cloud

            self.usage = cloud.USAGE
        return self.usage

    def _spend(self):
        if self.spend is None:
            try:
                from stratamodels import (
                    spend,
                )  # the API keys' tracking (thread "Claude API client with budget monitoring")
            except ImportError:
                return None
            self.spend = spend.summary
        return self.spend

    def _cached(self, key, read):
        with self.guard:
            hit = self.memo.get(key)
            if hit and self.clock() - hit[0] < self.ttl_s:
                return hit[1]
        try:
            v = read()
        except Exception as e:  # a CLI that is signed out or slow: unknown, not empty
            v = dict(left=None, detail="", error=f"{type(e).__name__}: {e}")
        with self.guard:
            self.memo[key] = (self.clock(), v)
        return v

    def left(self, backend, model=None):
        b = self.cat.backends[backend]
        src = b.allowance
        if not src:
            return dict(left=None, detail="not tracked")
        if src in ("agy", "codex"):

            def read_cli():
                buckets = self._usage()[src](model)
                if not buckets:
                    return dict(left=None, detail="", error=f"{src} gave no readable allowance")
                low = min(buckets, key=lambda x: x[1])
                return dict(
                    left=float(low[1]),
                    detail=", ".join(f"{n} {r:.0%} left" for n, r, _ in buckets),
                    resets=low[2],
                )

            group = model
            if src == "agy":
                from stratamodels.evals import cloud

                group = cloud.agy_group(model)
            return self._cached((src, group), read_cli)
        if src.startswith("spend:"):
            provider = src.split(":", 1)[1]

            def read_spend():
                fn = self._spend()
                if fn is None:
                    return dict(left=None, detail="spend tracking not deployed yet")
                row = next((r for r in fn() if r.get("provider") == provider), None)
                if row is None:
                    return dict(left=None, detail=f"no budget set for {provider}")
                budget, left = row.get("budget"), row.get("left")
                frac = None if not budget or left is None else max(0.0, float(left) / float(budget))
                unit = row.get("unit") or "usd"
                return dict(
                    left=frac,
                    amount_left=left,
                    unit=unit,
                    detail=f"{left} of {budget} {unit} left"
                    + (f", rolls over {row['next_rollover']}" if row.get("next_rollover") else ""),
                )

            return self._cached(("spend", provider), read_spend)
        return dict(left=None, detail=f"unknown allowance source {src!r}")


# ---- the options for a role -------------------------------------------------------------------------------------------------------------
def _backend_of(cat, m):
    """A registry model's backend: an API runtime is its own backend (claude-api, nvidia-nim, gemini-api), or the one its options name
    (`backend`, e.g. the free Gemini key on the gemini-api runtime); everything else runs here, on the GPU or, with no VRAM, the CPU."""
    b = (m.options or {}).get("backend") or m.runtime
    if b in cat.backends and cat.backends[b].kind != "local":
        return b
    return "local-gpu" if m.vram_gb > 0 else "local-cpu"


def _usd(price, tokens):
    if not price or price.get("in") is None or price.get("out") is None:
        return None
    return (float(price["in"]) * tokens["in"] + float(price["out"]) * tokens["out"]) / 1e6


def options(cat, reg, role, stats=None, found=None, missing=None, card_gb=None):
    """Every way to serve `role`, unranked: dicts with key, backend, kind, model, tier, family, callable (and why not), evals, cost.
    `card_gb`: what the GPU can give a model (the card less the reserve); a local model that needs more cannot be called."""
    stats, found, missing = stats or {}, found or {}, missing or {}
    out = []
    matched = set()
    current = reg.roles[role].model if role in reg.roles else None
    for m in reg.models.values():
        if role not in m.roles or m.runtime == "recorded" or "~" in m.key:
            continue  # recorded outputs stand in evals for a cloud model (its offer's `evals`), they serve nothing
        bk = _backend_of(cat, m)
        b = cat.backends.get(bk)
        if b is None:
            continue
        offer = next(
            (o for o in cat.offers.values() if o.backend == bk and o.model == (m.options or {}).get("model")), None
        )
        if offer:
            matched.add(offer.key)
        why_not = (
            None
            if m.served and not missing.get(m.runtime)
            else f"runtime {m.runtime} is not served here"
            if not m.served
            else f"runtime {m.runtime} needs {', '.join(missing[m.runtime])}"
        )
        if why_not is None and b.kind == "local" and card_gb is not None and m.vram_gb > card_gb:
            why_not = f"needs {m.vram_gb:g} GB, the card gives {card_gb:.1f}"
        st = stats.get(m.key) or {}
        ev = list(found.get((role, m.key), []))
        if offer:
            for k in offer.evals:
                ev += found.get((role, k), [])
        o = dict(
            key=m.key,
            name=m.name,
            backend=bk,
            kind=b.kind,
            model=(m.options or {}).get("model") or m.name,
            family=(offer.family if offer else m.key),
            tier=(offer.tier if offer else (cat.local.get(m.key) or {}).get("tier")),
            current=m.key == current,
            registered=True,
            callable=why_not is None,
            why_not=why_not,
            images=True if b.kind == "local" else bool(offer and offer.images),
            audio=True if b.kind == "local" else bool(offer and offer.audio),
            evals=dict(verdict=verdict(ev), found=ev),
            note=(offer.note if offer else (cat.local.get(m.key) or {}).get("note", "")),
        )
        price = (m.options or {}).get("price") or (offer.price if offer else None)
        o["cost"] = _cost(cat, b, role, price, offer.quota if offer else None, st, m.vram_gb)
        out.append(o)
    for offer in cat.offers.values():
        if role not in offer.roles or offer.key in matched:
            continue
        b = cat.backends[offer.backend]
        ev = []
        for k in offer.evals:
            ev += found.get((role, k), [])
        out.append(
            dict(
                key=offer.key,
                name=offer.model,
                backend=offer.backend,
                kind=b.kind,
                model=offer.model,
                family=offer.family,
                tier=offer.tier,
                current=False,
                registered=False,
                callable=False,
                why_not=(
                    f"only recorded for evals: {offer.backend} has no live runtime here yet"
                    if b.kind == "subscription"
                    else f"not registered: no {offer.backend} model for {offer.model} in the registry yet"
                ),
                images=offer.images,
                audio=offer.audio,
                evals=dict(verdict=verdict(ev), found=ev),
                note=offer.note,
                cost=_cost(cat, b, role, offer.price, offer.quota, {}, 0.0),
            )
        )
    by_family = {}
    for o in out:  # the same model on another backend is as good: its evals count there too (agy's Gemini Flash for the Gemini keys)
        by_family.setdefault(o["family"], []).extend(o["evals"]["found"])
    for o in out:
        if not o["evals"]["found"] and by_family.get(o["family"]):
            o["evals"] = dict(verdict=verdict(by_family[o["family"]]), found=by_family[o["family"]], via="family")
    return out


def _cost(cat, b, role, price, quota, st, vram_gb):
    c = dict(order=b.order, use=b.use, vram_gb=vram_gb if b.kind == "local" else 0.0)
    if b.kind == "local":
        run_s = st.get("mean_run_s")
        c.update(lane="gpu" if b.key == "local-gpu" else "cpu", run_s=run_s, jobs=st.get("jobs", 0))
        if run_s is not None and b.watts and b.usd_per_kwh:
            c["usd_equiv"] = run_s * b.watts / 1000 / 3600 * b.usd_per_kwh
        return c
    c["lane"] = "cpu"  # a cloud call holds no VRAM (scheduler: vram_gb 0, the CPU lane)
    if quota:
        c.update(bucket=quota.get("bucket"), per_call=quota.get("per_call"), window=quota.get("window", "week"))
        if b.plan_usd_month and quota.get("per_call"):
            weeks = 4.35 if c["window"] == "week" else 30.4 * 24 / 5  # windows a month
            c["usd_equiv"] = b.plan_usd_month * float(quota["per_call"]) / weeks
    if b.kind == "api":
        tokens = cat.tokens(role)
        usd = _usd(price, tokens)
        c.update(tokens=tokens, usd_per_call=usd, price=price)
        if usd is not None:
            c["usd_equiv"] = usd
    return c


# ---- ranking ------------------------------------------------------------------------------------------------------------------------------
def route(cat, opts, role, goal="cheapest", private=False, min_tier=None, allowances=None):
    """Rank `opts` (options()) for a job: each gets `eligible` and `reasons`; the answer names the pick and why. Read allowances only for
    options that are otherwise eligible (each reading may run a CLI)."""
    if goal not in GOALS:
        raise ValueError(f"goal is one of {', '.join(GOALS)}, not {goal!r}")
    cur = next((o for o in opts if o["current"]), None)
    floor = min_tier if min_tier is not None else (cur or {}).get("tier")
    for o in opts:
        b = cat.backends[o["backend"]]
        r = []
        if not o["callable"]:
            r.append(o["why_not"])
        if private and b.leaves:
            r.append("private: the inputs would leave this machine")
        if role in IMAGES and not o["images"]:
            r.append("takes no pictures")
        if role in AUDIO and not o["audio"]:
            r.append("takes no audio")
        if o["evals"]["verdict"] == "worse":
            r.append(
                "worse than the baseline in "
                + ", ".join(e["experiment"] for e in o["evals"]["found"] if e["verdict"] == "worse")
            )
        if (
            (
                min_tier is not None or not o["current"]
            )  # the current model sets the default floor; an asked one holds it too
            and floor is not None
            and o["tier"] is not None
            and o["tier"] < floor
            and o["evals"]["verdict"] != "better"
        ):
            r.append(f"tier {o['tier']} under {floor} ({'asked' if min_tier is not None else 'the current model'})")
        o["reasons"] = r
        o["allowance"] = None
    for o in opts:
        if o["reasons"] or o["kind"] == "local" or allowances is None:
            continue
        b = cat.backends[o["backend"]]
        a = allowances.left(o["backend"], o["model"])
        o["allowance"] = a
        if a.get("left") is not None and a["left"] <= b.min_left:
            o["reasons"].append(f"{a['left']:.0%} of its allowance left, at or under the {b.min_left:.0%} reserve")
        per = o["cost"].get("per_call")
        if a.get("left") is not None and per and a["left"] - per < b.min_left and not o["reasons"]:
            o["reasons"].append(f"a call ({per:.1%}) would take it under the {b.min_left:.0%} reserve")
        usd = o["cost"].get("usd_per_call")
        if a.get("unit") == "usd" and a.get("amount_left") is not None and usd and float(a["amount_left"]) < usd:
            o["reasons"].append(f"${float(a['amount_left']):.2f} left, a call costs ~${usd:.4f}")
    for o in opts:  # a fallback backend only when its primary has nothing of the same family left
        b = cat.backends[o["backend"]]
        if (
            b.fallback_for
            and not o["reasons"]
            and any(p["backend"] == b.fallback_for and p["family"] == o["family"] and not p["reasons"] for p in opts)
        ):
            o["reasons"].append(f"a fallback: {b.fallback_for} has {o['family']}")
    for o in opts:
        o["eligible"] = not o["reasons"]

    def spend(o):
        c = o["cost"]
        amount = (
            c.get("run_s")
            if o["kind"] == "local"
            else c.get("per_call")
            if c.get("per_call") is not None
            else c.get("usd_per_call")
        )
        known = o["allowance"] is None or o["allowance"].get("left") is not None or o["kind"] == "local"
        return (
            c["order"],
            0 if known else 1,  # an allowance nobody could read comes after one that was read
            math.inf if amount is None else amount,
        )

    def strength(o):
        return (-EVAL_RANK.get(o["evals"]["verdict"] or "same", 0), -(o["tier"] or 0))

    def order(ok):
        if goal == "best":
            return lambda o: (
                0 if ok(o) else 1,
                *strength(o),
                0 if o["current"] else 1,
                o["cost"]["order"],
                1 if cat.backends[o["backend"]].fallback_for else 0,
                *spend(o)[1:],
                o["key"],
            )
        # cheapest: the first backend in the order; on it, an eval winner, then the role's own model (an eval chose it), then the least spent
        return (
            lambda o: (
                0 if ok(o) else 1,
                o["cost"]["order"],
                1 if cat.backends[o["backend"]].fallback_for else 0,  # agy before a Gemini key, for the same quality
                -EVAL_RANK.get(o["evals"]["verdict"] or "same", 0),
                0 if o["current"] else 1,
                *spend(o)[1:],
                -(o["tier"] or 0),
                o["key"],
            )
        )

    ranked = sorted(opts, key=order(lambda o: o["eligible"]))

    # What it would pick if every backend could be called here: the case for building the next runtime
    def only_uncallable(o):
        return o["eligible"] or (not o["callable"] and o["reasons"] == [o["why_not"]])

    ideal = next((o for o in sorted(opts, key=order(only_uncallable)) if only_uncallable(o)), None)
    pick = next((o for o in ranked if o["eligible"]), None)
    if pick is None:
        why = "nothing can serve it: " + "; ".join(f"{o['key']}: {', '.join(o['reasons'])}" for o in ranked[:5])
    else:
        b = cat.backends[pick["backend"]]
        bits = [f"{b.name}"]
        if pick["evals"]["verdict"]:
            bits.append(f"eval: {pick['evals']['verdict']}")
        if pick["tier"] is not None:
            bits.append(f"tier {pick['tier']}")
        bits.append(f"use {b.use}" if b.use else b.kind)
        why = f"{pick['key']} ({', '.join(bits)})"
    would = ideal["key"] if ideal is not None and ideal is not pick else None
    return dict(
        role=role,
        goal=goal,
        private=private,
        min_tier=floor,
        pick=pick["key"] if pick else None,
        why=why,
        would_pick=would,
        would_why=f"{would}: {ideal['why_not']}" if would and ideal is not None else None,
        options=ranked,
    )


def plan(cat, reg, role, store=None, engine=None, allowances=None, missing=None, card_gb=None, **policy):
    """route() for `role` with what the service knows: its job history (GPU seconds) and eval summaries."""
    if (
        role not in reg.roles
        and not any(role in m.roles for m in reg.models.values())
        and not any(role in o.roles for o in cat.offers.values())
    ):
        raise ValueError(f"no role {role!r}")
    stats = store.model_stats() if store is not None else {}
    found = evidence(engine) if engine is not None else {}
    return route(cat, options(cat, reg, role, stats, found, missing, card_gb), role, allowances=allowances, **policy)
