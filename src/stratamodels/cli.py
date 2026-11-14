"""Command line: `doctor` checks the environment and the GPU, `serve` starts the service, `status` and `jobs` read a running one; `registry`,
`role`, `prefer`, `seed`, `set` and `eval` read and change the service's tables (schema stratamodels of the shared database); `catalogue` lists
every backend and cloud model routing knows, `route ROLE` asks the running service where a job for ROLE should go."""

import argparse
import importlib
import json
import sys
import urllib.error
import urllib.request

from stratamodels import __version__, gpu, runtimes, settings

CHECKS = (
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("sqlalchemy", "sqlalchemy"),
    ("omarchy-db", "omarchy_db"),
)


def doctor(args):
    ok = True
    print(f"strata-models {__version__}, python {sys.version.split()[0]}")
    for label, mod in CHECKS:
        try:
            m = importlib.import_module(mod)
            print(f"  ok       {label} {getattr(m, '__version__', '')}".rstrip())
        except ImportError:
            ok = False
            print(f"  MISSING  {label}: pip install -r requirements.txt")
    for name, missing in runtimes.available().items():
        print(
            f"  ok       runtime {name}"
            if not missing
            else f"  --       runtime {name}: needs {', '.join(missing)} (pip install -r requirements-gpu.txt)"
        )
    g = gpu.query()
    print(
        f"  ok       GPU {g['name']}, {g['total_mb'] / 1024:.1f} GB, {g['used_mb'] / 1024:.1f} GB in use"
        if g
        else "  --       no NVIDIA GPU found (nvidia-smi): models run on the CPU"
    )
    print(f"  models folder {settings.models_dir()}, VRAM reserve {settings.vram_reserve_gb():.1f} GB")
    return 0 if ok else 1


def _get(args, path):
    with urllib.request.urlopen(args.url.rstrip("/") + path, timeout=10) as r:
        return json.load(r)


def cmd_status(args):
    d = _get(args, "/api/status")
    if args.json:
        print(json.dumps(d, indent=1))
        return 0
    g = d["gpu"]
    print(
        f"{g['name']}: {g['used_mb'] / 1024:.1f} of {g['total_mb'] / 1024:.1f} GB, {g['util_pct']:.0f}% busy"
        if g
        else "no GPU"
    )
    for r in d["resident"]:
        print(f"  loaded   {r['model']:28} {r['vram_mb'] / 1024:5.1f} GB  {r['jobs']} job(s)")
    for j in d.get("running") or ([d["current"]] if d["current"] else []):
        where = ", ".join(str(x) for x in (j.get("model"), j.get("lane") and f"{j['lane']} lane") if x)
        print(f"  running  #{j['id']} {j['role']} for {j['app']}" + (f" ({where})" if where else ""))
    for j in d["queued"]:
        print(f"  queued   #{j['id']} {j['kind']} {j['role'] or j['note'] or ''} for {j['app']} ({j['priority']})")
    return 0


def _engine(need=False):
    """The service's database, with the settings bound to it; None when it cannot be reached or is at another schema revision (unless `need`).
    Only an empty database is migrated here: the running service migrates the live one when it is deployed."""
    try:
        from stratamodels.store import Store

        e = Store.connect(migrate="new").engine
    except Exception as ex:
        if need:
            sys.exit(f"the database cannot be reached: {ex}")
        print(f"(no database: {ex}; reading the files)", file=sys.stderr)
        return None
    settings.bind(e)
    return e


def _registry(engine):
    from stratamodels import registry as reg

    if engine is None:
        return reg.Registry.load()
    reg.seed(engine)
    return reg.Registry.from_db(engine)


def _poke(args):
    """Tell a running service to read the registry again (best effort: it also does at its next start)."""
    try:
        req = urllib.request.Request(args.url.rstrip("/") + "/api/registry/seed", method="POST")
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception:
        print("(the service is not running, or did not answer: it takes the change at its next start)", file=sys.stderr)


def cmd_role(args):
    from stratamodels import registry as reg

    try:
        r = reg.set_role(_engine(need=True), args.role, args.model, args.description)
    except reg.RegistryError as e:
        sys.exit(str(e))
    print(f"{args.role} -> {r.roles[args.role].model}")
    _poke(args)
    return 0


def cmd_prefer(args):
    from stratamodels import registry as reg

    try:
        params = json.loads(args.params) if args.params else None
    except ValueError as e:
        sys.exit(f"--params is JSON: {e}")
    try:
        r = reg.set_preference(
            _engine(need=True),
            args.app,
            args.role,
            None if args.clear else args.model,
            None if args.clear else params,
            args.note,
        )
    except reg.RegistryError as e:
        sys.exit(str(e))
    p = r.preference(args.app, args.role)
    print(f"{args.app} {args.role}: " + (json.dumps(p.to_json()) if p else "no preference (the role's model)"))
    _poke(args)
    return 0


def cmd_seed(args):
    from stratamodels import registry as reg

    e = _engine(need=True)
    for f in args.follow or []:
        kind, _, key = f.partition(":")
        try:
            reg.follow_seed(e, kind, key)
        except reg.RegistryError as ex:
            sys.exit(str(ex))
    print(json.dumps(reg.seed(e), indent=1))
    _poke(args)
    return 0


def cmd_registry(args):
    r = _registry(_engine())
    if args.json:
        print(json.dumps(r.to_json(), indent=1))
        return 0
    for role in r.roles.values():
        m = r.models[role.model]
        print(
            f"  {role.key:18} {m.key:24} {m.runtime:15} {'served' if m.served else 'not served yet: ' + ', '.join(m.used_by)}"
        )
    return 0


def cmd_jobs(args):
    d = _get(args, f"/api/jobs?limit={args.limit}")
    if args.json:
        print(json.dumps(d, indent=1))
        return 0
    for j in d["items"]:
        print(
            f"  #{j['id']:<6} {j['status']:9} {j['kind']:5} {(j['role'] or j['note'] or ''):20} {j['app']:16} {j.get('run_s') or ''}"
        )
    return 0


def cmd_eval(args):
    """Experiments: list, run (through the running service's queue), report (the standard write-up), forget (cached outputs)."""
    from stratamodels.evals import core as C
    from stratamodels.evals import report as R

    engine = _engine(need=args.action != "list")
    if args.action == "list":
        for name, s in C.list_specs(engine).items():
            print(f"  {name:28} {'ERROR ' + str(s) if isinstance(s, C.EvalError) else s.question}")
        for d in C.list_datasets(engine=engine):
            print(f"  dataset {d['name']:20} {d.get('n', '?')} items  {d.get('version') or d.get('error')}")
        return 0
    if not args.name:
        sys.exit("name the experiment")
    try:
        spec = C.load_spec(args.name, engine)
    except C.EvalError as e:
        sys.exit(str(e))
    if args.action == "run":
        import os

        sys.path.insert(
            0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "client")
        )
        import strata_models_client as sm

        if not sm.available():
            sys.exit(f"strata-models is not running at {sm.url()}: eval jobs go through its queue")
        r = C.run(spec, engine, C.ClientExecutor(sm), candidates=args.candidate or None, limit=args.limit)
        print(json.dumps(r))
        return 0
    if args.action == "judge":
        return _eval_judge(spec, engine, args)
    if args.action == "forget":
        print(f"{C.forget(spec, engine, args.candidate[0] if args.candidate else None)} outputs forgotten")
        return 0
    res = C.analyse(spec, engine)
    if args.json:
        print(json.dumps(res, indent=1, default=str))
        return 0
    ds, outs, _ = C.outputs_for(spec, engine)
    text = R.render(res, R.worst(spec, outs, ds.items, spec.primary), pipelines=R.pipelines(spec, outs, ds.items))
    if args.out:
        with open(args.out, "w", encoding="utf8") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def _eval_judge(spec, engine, args):
    """Model judges judge what they have still to judge: a local one through the running service's queue, a cloud one by calling its CLI here
    (on the plan: this machine's codex or agy must be signed in). Exit 75 when a plan's allowance ran out (run again later to go on)."""
    import os

    from stratamodels.evals import core as C
    from stratamodels.evals import judges as J

    names = args.judge or [j["name"] for j in spec.judges]
    if not names:
        sys.exit(f"{spec.name} names no judges ([[judges]] in its spec)")
    executor = None
    try:
        local = [J.find(spec, n) for n in names if not J.find(spec, n)["cli"]]
    except C.EvalError as e:
        sys.exit(str(e))
    if local:
        sys.path.insert(
            0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "client")
        )
        import strata_models_client as sm

        if not sm.available():
            sys.exit(f"strata-models is not running at {sm.url()}: a local judge's jobs go through its queue")
        executor = C.ClientExecutor(sm)
    out, code = {}, 0
    for n in names:
        r = out[n] = J.run_judge(spec, engine, n, J.asker_for(J.find(spec, n), executor), limit=args.limit)
        if r["stopped"]:
            code = 75
    print(json.dumps(out))
    return code


def cmd_set(args):
    _engine()
    if args.value is None:
        print(f"{args.key} = {settings.all_settings().get(args.key)}")
        return 0
    try:
        settings.save({args.key: None if args.value == "default" else args.value})
    except (KeyError, ValueError) as e:
        sys.exit(str(e))
    print(f"{args.key} = {settings.all_settings()[args.key]}")
    return 0


def cmd_catalogue(args):
    from stratamodels import routing

    try:
        c = routing.Catalogue.load()
    except routing.CatalogueError as e:
        sys.exit(str(e))
    if args.json:
        print(json.dumps(c.to_json(), indent=1))
        return 0
    for b in sorted(c.backends.values(), key=lambda b: (b.order, b.key)):
        print(f"{b.key} ({b.kind}, order {b.order}, use {b.use or '-'}): {b.name}")
        for o in c.offers.values():
            if o.backend != b.key:
                continue
            cost = (
                f"${o.price['in']:g}/${o.price['out']:g} per Mtok"
                if o.price
                else f"{o.quota['per_call']:.1%} of {o.quota['bucket']} a call"
                if o.quota
                else "unpriced"
            )
            print(f"    {o.key:30} tier {o.tier if o.tier is not None else '-'}  {cost:34} {', '.join(o.roles)}")
    return 0


def cmd_route(args):
    q = f"/api/route/{args.role}?goal={args.goal}&private={str(args.private).lower()}&allowance={str(not args.no_allowance).lower()}"
    if args.min_tier is not None:
        q += f"&min_tier={args.min_tier}"
    try:
        d = _get(args, q)
    except urllib.error.HTTPError as e:
        sys.exit(f"{e.code}: {json.load(e).get('detail')}")
    if args.json:
        print(json.dumps(d, indent=1))
        return 0
    print(f"{d['role']} ({d['goal']}{', private' if d['private'] else ''}): {d['why']}")
    if d.get("would_pick"):
        print(f"  if it could be called here: {d['would_why']}")
    for o in d["options"]:
        c = o["cost"]
        cost = (
            f"{c['run_s']:.1f} s on the {c['lane'].upper()}"
            if o["kind"] == "local" and c.get("run_s") is not None
            else f"{c['lane'].upper()}, not run yet"
            if o["kind"] == "local"
            else f"{c['per_call']:.1%} of {c.get('bucket')}"
            if c.get("per_call") is not None
            else f"${c['usd_per_call']:.4f}"
            if c.get("usd_per_call") is not None
            else "unpriced"
        )
        left = o.get("allowance") or {}
        mark = "*" if o["key"] == d["pick"] else " " if o["eligible"] else "-"
        print(
            f" {mark} {o['key']:32} {o['backend']:12} tier {o['tier'] if o['tier'] is not None else '-'}  eval {o['evals']['verdict'] or '-':8} {cost:28}"
            + (f" {left['left']:.0%} left" if left.get("left") is not None else "")
            + (f"  ({'; '.join(o['reasons'])})" if o["reasons"] else "")
        )
    return 0


def cmd_serve(args):
    from stratamodels.server import app as server_app

    server_app.main(["--host", args.host, "--port", str(args.port)] + (["--memory"] if args.memory else []))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="strata-models")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check the environment, the runtimes and the GPU").set_defaults(fn=doctor)
    for name, fn, h in (
        ("status", cmd_status, "what the running service is doing"),
        ("jobs", cmd_jobs, "recent jobs of the running service"),
    ):
        c = sub.add_parser(name, help=h)
        c.add_argument("--url", default="http://127.0.0.1:8366")
        c.add_argument("--json", action="store_true")
        c.set_defaults(fn=fn)
        if name == "jobs":
            c.add_argument("--limit", type=int, default=20)
    r = sub.add_parser("registry", help="roles and the model chosen for each")
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_registry)
    e = sub.add_parser(
        "eval", help="experiments: list | run NAME | judge NAME [--judge J] | report NAME [--out FILE] | forget NAME"
    )
    e.add_argument("action", choices=["list", "run", "judge", "report", "forget"])
    e.add_argument("name", nargs="?")
    e.add_argument("--candidate", action="append", help="only this candidate (repeatable)")
    e.add_argument("--limit", type=int, help="only the first N items (a trial run); judge: at most N pairs per judge")
    e.add_argument(
        "--judge", action="append", help="judge: only this model judge (repeatable; default every judge in the spec)"
    )
    e.add_argument("--out", help="write the report here")
    e.add_argument("--json", action="store_true")
    e.set_defaults(fn=cmd_eval)
    s = sub.add_parser(
        "set",
        help='show or change a setting (data_dir, models_dir, vram_reserve_gb, gpu_jobs, cpu_jobs, whole_card_gb); value "default" resets it',
    )
    s.add_argument("key")
    s.add_argument("value", nargs="?")
    s.set_defaults(fn=cmd_set)
    for name, fn, h in (
        ("role", cmd_role, "choose the model for a role: role ROLE MODEL"),
        (
            "prefer",
            cmd_prefer,
            "an app's own model or default params for a role: prefer APP ROLE [MODEL] [--params JSON] | --clear",
        ),
        ("seed", cmd_seed, "import models.toml into the database again"),
    ):
        c = sub.add_parser(name, help=h)
        c.add_argument("--url", default="http://127.0.0.1:8366")
        c.set_defaults(fn=fn)
        if name == "role":
            c.add_argument("role")
            c.add_argument("model", nargs="?")
            c.add_argument("--description")
        if name == "prefer":
            c.add_argument("app")
            c.add_argument("role")
            c.add_argument("model", nargs="?")
            c.add_argument("--params")
            c.add_argument("--note")
            c.add_argument("--clear", action="store_true")
        if name == "seed":
            c.add_argument(
                "--follow",
                action="append",
                help="model:KEY or role:KEY: drop its edits here and follow models.toml again",
            )
    c = sub.add_parser("catalogue", help="every backend and cloud model routing knows: roles, tier, cost")
    c.add_argument("--json", action="store_true")
    c.set_defaults(fn=cmd_catalogue)
    c = sub.add_parser(
        "route", help="where a job for ROLE should go, local or cloud, ranked (asks the running service)"
    )
    c.add_argument("role")
    c.add_argument("--goal", choices=["cheapest", "best"], default="cheapest")
    c.add_argument("--private", action="store_true", help="the inputs must not leave this machine")
    c.add_argument("--min-tier", type=int, help="at least this tier (default: the role's current model's)")
    c.add_argument(
        "--no-allowance", action="store_true", help="do not read what is left of each allowance (it runs agy and codex)"
    )
    c.add_argument("--url", default="http://127.0.0.1:8366")
    c.add_argument("--json", action="store_true")
    c.set_defaults(fn=cmd_route)
    v = sub.add_parser("serve", help="start the service")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8366)
    v.add_argument("--memory", action="store_true", help="keep jobs in memory instead of the database")
    v.set_defaults(fn=cmd_serve)
    a = p.parse_args(argv)
    return a.fn(a)
