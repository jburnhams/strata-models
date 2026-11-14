"""Routing (stratamodels.routing): the catalogue, eval evidence, allowances and the ranking of local and cloud options for a role."""

import json

import pytest

from stratamodels import routing
from stratamodels.registry import Registry
from stratamodels.routing import Allowances, Catalogue, CatalogueError

BACKENDS = {
    "local-gpu": dict(kind="local", order=0, use="freely", watts=180, usd_per_kwh=0.36),
    "local-cpu": dict(kind="local", order=0, watts=60, usd_per_kwh=0.36),
    "agy": dict(kind="subscription", order=1, use="freely", allowance="agy", plan_usd_month=20.0),
    "nim": dict(kind="api", order=1, allowance="spend:nim"),
    "codex": dict(kind="subscription", order=2, use="sparingly", allowance="codex", min_left=0.3, plan_usd_month=20.0),
    "gem-paid": dict(kind="api", order=3, allowance="spend:gemini", fallback_for="agy"),
    "claude-api": dict(kind="api", order=3, use="budget", allowance="spend:claude"),
}
OFFERS = {
    "agy-flash": dict(
        backend="agy",
        model="flash-low",
        family="flash",
        roles=["vl", "text"],
        tier=4,
        images=True,
        quota=dict(bucket="Gemini", per_call=0.002),
        evals=["flash-rec"],
    ),
    "agy-argon": dict(
        backend="agy", model="argon", roles=["vl"], tier=5, images=True, quota=dict(bucket="Argon", per_call=0.003)
    ),
    "codex-sol": dict(
        backend="codex",
        model="sol",
        roles=["vl"],
        tier=5,
        images=True,
        quota=dict(bucket="codex", per_call=0.02, window="5h"),
    ),
    "gem-flash": dict(backend="gem-paid", model="flash", family="flash", roles=["vl", "text"], tier=4, images=True),
    "nim-kimi": dict(backend="nim", model="kimi", roles=["text"], tier=4, price={"in": 0.0, "out": 0.0}),
    "haiku": dict(
        backend="claude-api",
        model="claude-haiku-5-5",
        roles=["vl", "text"],
        tier=3,
        images=True,
        price={"in": 0.1, "out": 0.5},
    ),
}


def cat(**over):
    d = dict(
        defaults=dict(min_left=0.1, tokens={"vl": {"in": 2000, "out": 1000}}),
        backends={k: dict(v) for k, v in BACKENDS.items()},
        offers={k: dict(v) for k, v in OFFERS.items()},
        local={"qwen": {"tier": 2}, "gemma": {"tier": 2, "note": "bigger"}, "huge": {"tier": 3}},
    )
    d.update(over)
    return Catalogue.from_dict(d)


def reg(**models):
    m = dict(
        qwen=dict(runtime="fake", roles=["vl", "text"], vram_gb=7.5),
        gemma=dict(runtime="fake", roles=["vl", "text"], vram_gb=9),
        huge=dict(runtime="fake", roles=["vl"], vram_gb=14),
        tts=dict(runtime="fake", roles=["speak"], vram_gb=0),
        planned=dict(runtime="ultralytics", roles=["vl"], vram_gb=1),
        **{"flash-rec": dict(runtime="recorded", roles=["vl"], vram_gb=0)},
    )
    m.update(models)
    return Registry.from_dict(
        dict(roles=dict(vl=dict(model="qwen"), text=dict(model="qwen"), speak=dict(model="tts")), models=m)
    )


def by_key(r):
    return {o["key"]: o for o in r["options"]}


# ---- the catalogue ----------------------------------------------------------------------------------------------------------------------------
def test_builtin_catalogue_loads_and_every_offer_names_a_known_backend():
    c = Catalogue.load()
    assert {"local-gpu", "local-cpu", "agy", "codex", "claude-api", "nvidia-nim", "gemini-free", "gemini-api"} <= set(
        c.backends
    )
    assert c.backends["agy"].order < c.backends["codex"].order < c.backends["claude-api"].order
    assert c.backends["gemini-api"].fallback_for == "agy" and c.backends["codex"].min_left == 0.30
    assert c.offers["claude-sonnet-5-5"].price == {"in": 2.0, "out": 10.0} and "agy-gemini-3.8-flash" in c.offers
    j = c.to_json()
    assert [b["key"] for b in j["backends"]][:2] == ["local-cpu", "local-gpu"] and json.dumps(j)
    assert c.tokens("vision-language") == {"in": 2500, "out": 500} and c.tokens("unknown") == {"in": 2000, "out": 500}


def test_the_data_folders_catalogue_changes_one_field_without_restating_the_rest(tmp_path):
    builtin = tmp_path / "b.toml"
    builtin.write_text(
        '[backends.local-gpu]\nkind = "local"\nwatts = 170\nusd_per_kwh = 0.3\n[offers.x]\nbackend = "local-gpu"\nmodel = "m"\nroles = ["text"]\nquota = { bucket = "b", per_call = 0.1 }\n'
    )
    over = tmp_path / "o.toml"
    over.write_text("[backends.local-gpu]\nusd_per_kwh = 0.25\n[offers.x]\nquota = { per_call = 0.2 }\n")
    c = Catalogue.load(str(builtin), str(over))
    assert c.backends["local-gpu"].watts == 170 and c.backends["local-gpu"].usd_per_kwh == 0.25
    assert c.offers["x"].quota == {"bucket": "b", "per_call": 0.2} and c.offers["x"].family == "m"


@pytest.mark.parametrize(
    ("d", "match"),
    [
        (dict(backends=dict(a=dict(kind="cloud"))), "kind is local"),
        (dict(backends=dict(a=dict(order=1))), "backend 'a'"),
        (dict(backends=dict(a=dict(kind="api")), offers=dict(o=dict(backend="b", model="m"))), "names backend 'b'"),
        (dict(backends=dict(a=dict(kind="api", fallback_for="z"))), "fallback for 'z'"),
        (dict(backends=dict(a=dict(kind="api")), offers=dict(o=dict(backend="a"))), "offer 'o'"),
    ],
)
def test_a_broken_catalogue_is_refused(d, match):
    with pytest.raises(CatalogueError, match=match):
        Catalogue.from_dict(d)


def test_a_catalogue_that_is_not_toml_is_refused(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[backends\n")
    with pytest.raises(CatalogueError, match="bad.toml"):
        Catalogue.load(str(bad), str(tmp_path / "none.toml"))


# ---- eval evidence ----------------------------------------------------------------------------------------------------------------------------
def summary(role="vl", **per):
    cands = [dict(name="base", model="qwen")] + [dict(name=n, model=v.pop("model")) for n, v in per.items()]
    return dict(
        experiment=dict(
            name=f"exp-{role}",
            role=role,
            baseline="base",
            candidates=cands + [dict(name="pipe", model="gemma", stages=[{}]), dict(name="nomodel")],
        ),
        decision=dict(per_candidate=per),
    )


def test_eval_verdicts_become_better_worse_or_same():
    found = routing.evidence_from(
        [
            summary(
                flash=dict(model="flash-rec", verdict="adopt", gain=3.0, why="+3"),
                gemma=dict(model="gemma", verdict="reject", gain=-0.5, why="worse"),
                slow=dict(model="huge", verdict="reject", gain=1.0, why="latency"),
                nan=dict(model="planned", verdict="reject", gain=float("nan"), why="?"),
            ),
            None,
        ]
    )
    v = {k: [f["verdict"] for f in fs] for k, fs in found.items()}
    assert v == {
        ("vl", "qwen"): ["baseline"],
        ("vl", "flash-rec"): ["better"],
        ("vl", "gemma"): ["worse"],
        ("vl", "huge"): ["same"],
        ("vl", "planned"): ["same"],
    }
    assert routing.verdict([]) is None and routing.verdict([dict(verdict="baseline")]) == "baseline"
    assert routing.verdict([dict(verdict="better"), dict(verdict="worse")]) == "same"
    assert routing.verdict([dict(verdict="worse"), dict(verdict="same")]) == "worse"


def test_a_candidate_not_run_yet_has_no_verdict():
    s = summary(flash=dict(model="flash-rec", verdict="adopt", gain=1))
    s["experiment"]["candidates"].append(dict(name="later", model="huge"))
    assert ("vl", "huge") not in routing.evidence_from([s])


def test_evidence_reads_the_kept_summaries(scheduler):
    from stratamodels.db import session
    from stratamodels.db.models import EvalSummary

    with session(scheduler.store.engine) as s:
        s.add(
            EvalSummary(
                experiment="exp-vl",
                signature="x",
                result=summary(flash=dict(model="flash-rec", verdict="adopt", gain=1.0)),
                computed_at=1.0,
                seconds=0.1,
            )
        )
        s.commit()
    assert routing.evidence(scheduler.store.engine)[("vl", "flash-rec")][0]["verdict"] == "better"


# ---- options ----------------------------------------------------------------------------------------------------------------------------------
def test_options_list_local_models_cloud_offers_and_why_some_cannot_be_called():
    stats = dict(qwen=dict(jobs=10, mean_run_s=4.0))
    o = {x["key"]: x for x in routing.options(cat(), reg(), "vl", stats, {}, {"fake": []}, card_gb=12.0)}
    assert set(o) == {
        "qwen",
        "gemma",
        "huge",
        "planned",
        "agy-flash",
        "agy-argon",
        "codex-sol",
        "gem-flash",
        "haiku",
    }  # recorded and other roles left out
    q = o["qwen"]
    assert q["backend"] == "local-gpu" and q["current"] and q["callable"] and q["tier"] == 2
    assert (
        q["cost"]["run_s"] == 4.0
        and q["cost"]["lane"] == "gpu"
        and q["cost"]["usd_equiv"] == pytest.approx(4 * 180 / 1000 / 3600 * 0.36)
    )
    assert (
        o["gemma"]["note"] == "bigger" and o["gemma"]["cost"]["run_s"] is None and "usd_equiv" not in o["gemma"]["cost"]
    )
    assert (
        o["huge"]["why_not"] == "needs 14 GB, the card gives 12.0"
        and o["planned"]["why_not"] == "runtime ultralytics is not served here"
    )
    assert o["agy-flash"]["why_not"].startswith("only recorded for evals") and o["haiku"]["why_not"].startswith(
        "not registered"
    )
    assert o["agy-flash"]["cost"]["usd_equiv"] == pytest.approx(20 * 0.002 / 4.35) and o["codex-sol"]["cost"][
        "usd_equiv"
    ] == pytest.approx(20 * 0.02 / (30.4 * 24 / 5))
    assert (
        o["haiku"]["cost"]["usd_per_call"] == pytest.approx((0.1 * 2000 + 0.5 * 1000) / 1e6)
        and o["gem-flash"]["cost"]["usd_per_call"] is None
    )
    o2 = {x["key"]: x for x in routing.options(cat(), reg(), "vl", missing={"fake": ["torch"]})}
    assert o2["qwen"]["why_not"] == "runtime fake needs torch" and o2["huge"]["why_not"] == "runtime fake needs torch"


def test_a_registered_cloud_model_takes_its_offer_and_the_registrys_price():
    r = reg(
        **{
            "claude-haiku-5-5-api": dict(
                runtime="fake",
                roles=["vl"],
                vram_gb=0,
                options=dict(backend="claude-api", model="claude-haiku-5-5", price={"in": 1.0, "out": 5.0}),
            )
        }
    )
    o = {x["key"]: x for x in routing.options(cat(), r, "vl", missing={})}
    assert "haiku" not in o  # matched: listed once, under the registry's key
    h = o["claude-haiku-5-5-api"]
    assert h["backend"] == "claude-api" and h["kind"] == "api" and h["callable"] and h["tier"] == 3 and h["images"]
    assert h["cost"]["usd_per_call"] == pytest.approx((1.0 * 2000 + 5.0 * 1000) / 1e6) and h["cost"]["lane"] == "cpu"
    r2 = reg(free=dict(runtime="nim", roles=["text"], vram_gb=0, options=dict(model="kimi")))
    assert {x["key"]: x for x in routing.options(cat(), r2, "text", missing={})}["free"]["backend"] == "nim"
    r3 = reg(odd=dict(runtime="fake", roles=["speak"], vram_gb=0))
    assert {x["key"]: x["backend"] for x in routing.options(cat(), r3, "speak", missing={})} == {
        "tts": "local-cpu",
        "odd": "local-cpu",
    }


def test_evals_of_the_same_model_on_another_backend_count_for_it():
    found = routing.evidence_from([summary(flash=dict(model="flash-rec", verdict="adopt", gain=3.0))])
    o = {x["key"]: x for x in routing.options(cat(), reg(), "vl", found=found, missing={})}
    assert o["agy-flash"]["evals"]["verdict"] == "better" and "via" not in o["agy-flash"]["evals"]
    assert o["gem-flash"]["evals"]["verdict"] == "better" and o["gem-flash"]["evals"]["via"] == "family"
    assert o["qwen"]["evals"]["verdict"] == "baseline" and o["gemma"]["evals"]["verdict"] is None


# ---- ranking ----------------------------------------------------------------------------------------------------------------------------------
def test_cheapest_keeps_the_eval_chosen_local_model_and_names_the_cloud_one_it_would_use():
    found = routing.evidence_from([summary(flash=dict(model="flash-rec", verdict="adopt", gain=3.0))])
    r = routing.plan(cat(), reg(), "vl", missing={}, card_gb=12.0)
    assert r["pick"] == "qwen" and r["goal"] == "cheapest" and r["min_tier"] == 2 and "use freely" in r["why"]
    r = routing.route(
        cat(), routing.options(cat(), reg(), "vl", found=found, missing={}, card_gb=12.0), "vl", goal="best"
    )
    assert (
        r["pick"] == "qwen" and r["would_pick"] == "agy-flash" and r["would_why"].startswith("agy-flash: only recorded")
    )
    with pytest.raises(ValueError, match="goal is one of"):
        routing.route(cat(), [], "vl", goal="fast")


def live(**extra):
    """The fake registry with live cloud models: agy flash and argon, the paid Gemini flash, codex Sol, Claude Haiku."""
    m = {
        "agy-flash-live": dict(
            runtime="fake", roles=["vl", "text"], vram_gb=0, options=dict(backend="agy", model="flash-low")
        ),
        "agy-argon-live": dict(runtime="fake", roles=["vl"], vram_gb=0, options=dict(backend="agy", model="argon")),
        "gem-flash-live": dict(
            runtime="fake",
            roles=["vl", "text"],
            vram_gb=0,
            options=dict(backend="gem-paid", model="flash", price={"in": 0.3, "out": 2.5}),
        ),
        "sol-live": dict(runtime="fake", roles=["vl"], vram_gb=0, options=dict(backend="codex", model="sol")),
        "haiku-live": dict(
            runtime="fake",
            roles=["vl", "text"],
            vram_gb=0,
            options=dict(backend="claude-api", model="claude-haiku-5-5"),
        ),
    }
    m.update(extra)
    return reg(**m)


class FakeAllowances:
    def __init__(self, **left):
        self.left_by, self.asked = left, []

    def left(self, backend, model=None):
        self.asked.append((backend, model))
        return self.left_by.get(backend, dict(left=0.9, detail="plenty"))


def ranked(goal="cheapest", allowances=None, registry=None, found=None, **policy):
    c = cat()
    return routing.route(
        c,
        routing.options(c, registry or live(), "vl", found=found, missing={}, card_gb=12.0),
        "vl",
        goal=goal,
        allowances=allowances,
        **policy,
    )


def test_best_takes_the_eval_winner_then_the_strongest_tier():
    found = routing.evidence_from([summary(flash=dict(model="flash-rec", verdict="adopt", gain=3.0))])
    r = ranked("best", FakeAllowances(), found=found)
    assert r["pick"] == "agy-flash-live" and r["would_pick"] is None
    o = by_key(r)
    assert o["gem-flash-live"]["reasons"] == ["a fallback: agy has flash"]  # agy first for the same model
    r = ranked("best", FakeAllowances())
    assert r["pick"] == "agy-argon-live"  # no evals: tier 5, and agy spends before codex's Sol
    assert [x["key"] for x in r["options"][:2]] == ["agy-argon-live", "sol-live"]


def test_a_backend_at_its_reserve_is_skipped_and_the_paid_fallback_steps_in():
    a = FakeAllowances(agy=dict(left=0.08, detail="Gemini 8 % left"), codex=dict(left=0.31))
    r = ranked("best", a, min_tier=4)
    o = by_key(r)
    assert o["agy-argon-live"]["reasons"] == ["8% of its allowance left, at or under the 10% reserve"]
    assert o["sol-live"]["reasons"] == ["a call (2.0%) would take it under the 30% reserve"]
    assert (
        r["pick"] == "gem-flash-live"
        and o["gem-flash-live"]["eligible"]
        and o["haiku-live"]["reasons"] == ["tier 3 under 4 (asked)"]
    )
    assert ("local-gpu", None) not in a.asked and all(
        b != "claude-api" for b, _ in a.asked
    )  # no reading for what is already ruled out


def test_a_paid_key_without_the_money_for_a_call_is_skipped():
    a = FakeAllowances(**{"claude-api": dict(left=0.0001, amount_left=0.0001, unit="usd")})
    o = by_key(
        ranked(
            "best",
            a,
            min_tier=3,
            registry=live(
                **{
                    "haiku-live": dict(
                        runtime="fake",
                        roles=["vl"],
                        vram_gb=0,
                        options=dict(backend="claude-api", model="claude-haiku-5-5", price={"in": 1, "out": 5}),
                    )
                }
            ),
        )
    )
    assert o["haiku-live"]["reasons"] == [
        "0% of its allowance left, at or under the 10% reserve",
        "$0.00 left, a call costs ~$0.0070",
    ]


def test_private_jobs_stay_here_and_worse_or_weaker_models_are_left_out():
    found = routing.evidence_from([summary(gemma=dict(model="gemma", verdict="reject", gain=-1.0))])
    r = ranked("best", FakeAllowances(), found=found, private=True)
    o = by_key(r)
    assert r["pick"] == "qwen" and o["agy-argon-live"]["reasons"] == ["private: the inputs would leave this machine"]
    assert o["gemma"]["reasons"] == ["worse than the baseline in exp-vl"] and not o["gemma"]["eligible"]


def test_cheapest_spends_on_the_first_backend_in_the_order():
    r = ranked(
        "cheapest",
        FakeAllowances(),
        registry=live(
            qwen=dict(runtime="ultralytics", roles=["vl", "text"], vram_gb=7.5),
            gemma=dict(runtime="ultralytics", roles=["vl", "text"], vram_gb=9),
        ),
    )
    assert (
        r["pick"] == "agy-flash-live"
    )  # nothing local can run: agy (free, limited) before codex and the paid keys; per call, flash before argon
    assert [x["key"] for x in r["options"] if x["eligible"]][:3] == ["agy-flash-live", "agy-argon-live", "sol-live"]


def test_an_allowance_nobody_could_read_comes_after_one_that_was_read():
    a = FakeAllowances(agy=dict(left=None, detail="", error="signed out"))
    r = ranked(
        "cheapest",
        a,
        registry=live(
            qwen=dict(runtime="ultralytics", roles=["vl", "text"], vram_gb=7.5),
            gemma=dict(runtime="ultralytics", roles=["vl", "text"], vram_gb=9),
        ),
    )
    o = by_key(r)
    assert o["agy-flash-live"]["eligible"] and o["agy-flash-live"]["allowance"]["error"] == "signed out"


def test_nothing_eligible_says_why():
    r = ranked(
        "cheapest",
        None,
        private=True,
        registry=Registry.from_dict(
            dict(roles=dict(vl=dict(model="m")), models=dict(m=dict(runtime="ultralytics", roles=["vl"])))
        ),
    )
    assert r["pick"] is None and r["why"].startswith("nothing can serve it: ")


def test_cloud_models_that_cannot_take_the_input_are_left_out():
    c = Catalogue.from_dict(
        dict(
            backends=dict(BACKENDS),
            offers=dict(t=dict(backend="nim", model="t", roles=["transcribe", "vision-language"])),
        )
    )
    r = Registry.from_dict(
        dict(
            roles={},
            models=dict(
                live=dict(
                    runtime="fake",
                    roles=["transcribe", "vision-language"],
                    vram_gb=0,
                    options=dict(model="t", backend="nim"),
                )
            ),
        )
    )
    assert by_key(routing.route(c, routing.options(c, r, "transcribe", missing={}), "transcribe"))["live"][
        "reasons"
    ] == ["takes no audio"]
    assert by_key(routing.route(c, routing.options(c, r, "vision-language", missing={}), "vision-language"))["live"][
        "reasons"
    ] == ["takes no pictures"]


def test_plan_uses_the_job_history_and_refuses_an_unknown_role(scheduler):
    r = routing.plan(cat(), reg(), "vl", store=scheduler.store, engine=scheduler.store.engine, missing={})
    assert r["pick"] == "qwen" and by_key(r)["qwen"]["cost"]["jobs"] == 0
    with pytest.raises(ValueError, match="no role 'nope'"):
        routing.plan(cat(), reg(), "nope")


# ---- allowances -------------------------------------------------------------------------------------------------------------------------------
def test_allowances_read_the_clis_once_per_ttl_and_by_agy_group(clock):
    calls = []

    def agy(model):
        calls.append(("agy", model))
        return [("Gemini Models 5h", 0.7, "soon"), ("Gemini Models week", 0.4, "monday")]

    def codex(model):
        calls.append(("codex", model))
        return []

    a = Allowances(cat(), usage=dict(agy=agy, codex=codex), spend=lambda: [], ttl_s=60, clock=clock)
    got = a.left("agy", "gemini-3.8-flash-low")
    assert got == dict(left=0.4, detail="Gemini Models 5h 70% left, Gemini Models week 40% left", resets="monday")
    assert a.left("agy", "gemini-3.1-pro") is got and len(calls) == 1  # same group: one reading
    a.left("agy", "gemini-4-argon-medium")
    assert len(calls) == 2
    clock.t += 61
    a.left("agy", "gemini-3.8-flash-low")
    assert len(calls) == 3
    assert a.left("codex", "sol") == dict(left=None, detail="", error="codex gave no readable allowance")
    assert a.left("local-gpu") == dict(left=None, detail="not tracked")


def test_allowances_from_the_spend_tracking(clock):
    rows = [
        dict(provider="claude", budget=100.0, left=25.0, unit="usd", next_rollover="2026-12-08"),
        dict(provider="nim", budget=0, left=None, unit="requests"),
    ]
    a = Allowances(cat(), usage={}, spend=lambda: rows, clock=clock)
    assert a.left("claude-api") == dict(
        left=0.25, amount_left=25.0, unit="usd", detail="25.0 of 100.0 usd left, rolls over 2026-12-08"
    )
    assert a.left("nim")["left"] is None and a.left("gem-paid") == dict(left=None, detail="no budget set for gemini")


def test_an_allowance_that_cannot_be_read_is_unknown_not_empty(clock, monkeypatch):
    def broken(model):
        raise RuntimeError("agy is signed out")

    a = Allowances(cat(), usage=dict(agy=broken), clock=clock)
    assert a.left("agy", "x") == dict(left=None, detail="", error="RuntimeError: agy is signed out")
    c = Catalogue.from_dict(dict(backends=dict(odd=dict(kind="api", allowance="ledger"))))
    assert Allowances(c).left("odd") == dict(left=None, detail="unknown allowance source 'ledger'")
    import builtins

    real = builtins.__import__

    def no_spend(name, *args, **kw):
        if name == "stratamodels" and args and args[2] and "spend" in args[2]:
            raise ImportError("no spend yet")
        return real(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", no_spend)
    assert Allowances(cat(), clock=clock).left("claude-api") == dict(
        left=None, detail="spend tracking not deployed yet"
    )


def test_allowances_default_to_the_cli_usage_readers(monkeypatch):
    from stratamodels.evals import cloud

    monkeypatch.setitem(cloud.USAGE, "codex", lambda model: [("codex 5h", 0.5, None)])
    assert Allowances(cat()).left("codex", "sol")["left"] == 0.5


def test_an_asked_tier_floor_holds_the_current_model_too():
    o = by_key(ranked("cheapest", FakeAllowances(), min_tier=4))
    assert o["qwen"]["reasons"] == ["tier 2 under 4 (asked)"] and o["qwen"]["current"]
    assert by_key(ranked("cheapest", FakeAllowances()))["qwen"]["eligible"]


def test_another_configuration_of_the_baselines_model_or_a_load_variant_is_not_its_verdict():
    s = summary(
        beam1=dict(model="qwen", verdict="reject", gain=-0.2, why="worse"),
        int8=dict(model="gemma", verdict="reject", gain=-1.0),
    )
    s["experiment"]["candidates"][-3]["options"] = {"quant": "int8"}  # int8: gemma loaded another way
    found = routing.evidence_from([s])
    assert [f["verdict"] for f in found[("vl", "qwen")]] == ["baseline"] and ("vl", "gemma") not in found


def test_a_gemini_model_spends_as_its_first_key_and_measured_costs_beat_the_estimate():
    c = Catalogue.load()
    r = Registry.from_dict(
        dict(
            roles={},
            models={
                "flash-api": dict(
                    runtime="gemini-api", roles=["text"], vram_gb=0, options=dict(model="gemini-3.8-flash")
                ),
                "argon-api": dict(
                    runtime="gemini-api",
                    roles=["text"],
                    vram_gb=0,
                    options=dict(model="gemini-4-argon", keys=["gemini"]),
                ),
                "haiku-api": dict(
                    runtime="claude-api", roles=["text"], vram_gb=0, options=dict(model="claude-haiku-5-5")
                ),
            },
        )
    )
    measured = {("text", "haiku-api"): dict(n=12, usd=0.0031, tokens_in=900, tokens_out=200, failed=0)}
    o = {x["key"]: x for x in routing.options(c, r, "text", missing={}, measured=measured)}
    assert o["flash-api"]["backend"] == "gemini-free" and o["argon-api"]["backend"] == "gemini-api"
    assert (
        "gemini-free-3.8-flash" not in o and "gemini-api-3.8-flash" in o
    )  # the free offer is the registered model; the paid one stays listed
    assert o["haiku-api"]["cost"]["usd_per_call"] == 0.0031 and o["haiku-api"]["cost"]["measured"]["n"] == 12
    assert "measured" not in o["flash-api"]["cost"]


def test_a_free_key_has_no_budget_to_run_out_of(clock):
    a = Allowances(
        cat(),
        usage={},
        spend=lambda: [dict(provider="nim", budgeted=False, budget=None, left=None, unit="usd")],
        clock=clock,
    )
    assert a.left("nim") == dict(left=None, free=True, detail="nim: free, rate limited")


def test_per_task_reads_the_spend_ledger_when_there_is_one(monkeypatch):
    import sys
    import types

    mod = types.ModuleType("stratamodels.spend")
    mod.per_task = lambda: {("text", "m"): dict(n=1, usd=0.01)}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stratamodels.spend", mod)
    import stratamodels

    monkeypatch.setattr(stratamodels, "spend", mod, raising=False)
    assert routing.per_task() == {("text", "m"): dict(n=1, usd=0.01)}

    def broken():
        raise RuntimeError("no table")

    mod.per_task = broken  # type: ignore[attr-defined]
    assert routing.per_task() == {}
    monkeypatch.setitem(sys.modules, "stratamodels.spend", None)
    monkeypatch.delattr(stratamodels, "spend")
    assert routing.per_task() == {}
