"""Command line: `doctor` checks the environment and the GPU, `serve` starts the service, `status`, `registry` and `jobs` read a running one."""
import argparse, importlib, json, sys, urllib.request

from stratamodels import __version__, gpu, runtimes, settings

CHECKS = (('fastapi', 'fastapi'), ('uvicorn', 'uvicorn'), ('pydantic', 'pydantic'), ('sqlalchemy', 'sqlalchemy'), ('omarchy-db', 'omarchy_db'))


def doctor(args):
    ok = True; print(f'strata-models {__version__}, python {sys.version.split()[0]}')
    for label, mod in CHECKS:
        try: m = importlib.import_module(mod); print(f'  ok       {label} {getattr(m, "__version__", "")}'.rstrip())
        except ImportError: ok = False; print(f'  MISSING  {label}: pip install -r requirements.txt')
    for name, missing in runtimes.available().items():
        print(f'  ok       runtime {name}' if not missing else f'  --       runtime {name}: needs {", ".join(missing)} (pip install -r requirements-gpu.txt)')
    g = gpu.query()
    print(f'  ok       GPU {g["name"]}, {g["total_mb"] / 1024:.1f} GB, {g["used_mb"] / 1024:.1f} GB in use' if g else '  --       no NVIDIA GPU found (nvidia-smi): models run on the CPU')
    print(f'  models folder {settings.models_dir()}, VRAM reserve {settings.vram_reserve_gb():.1f} GB')
    return 0 if ok else 1


def _get(args, path):
    with urllib.request.urlopen(args.url.rstrip('/') + path, timeout=10) as r: return json.load(r)


def cmd_status(args):
    d = _get(args, '/api/status')
    if args.json: print(json.dumps(d, indent=1)); return 0
    g = d['gpu']
    print(f'{g["name"]}: {g["used_mb"] / 1024:.1f} of {g["total_mb"] / 1024:.1f} GB, {g["util_pct"]:.0f}% busy' if g else 'no GPU')
    for r in d['resident']: print(f'  loaded   {r["model"]:28} {r["vram_mb"] / 1024:5.1f} GB  {r["jobs"]} job(s)')
    if d['current']: print(f'  running  #{d["current"]["id"]} {d["current"]["role"]} for {d["current"]["app"]}')
    for j in d['queued']: print(f'  queued   #{j["id"]} {j["kind"]} {j["role"] or j["note"] or ""} for {j["app"]} ({j["priority"]})')
    return 0


def cmd_registry(args):
    from stratamodels.registry import Registry
    r = Registry.load()
    if args.json: print(json.dumps(r.to_json(), indent=1)); return 0
    for role in r.roles.values():
        m = r.models[role.model]; print(f'  {role.key:18} {m.key:24} {m.runtime:15} {"served" if m.served else "not served yet: " + ", ".join(m.used_by)}')
    return 0


def cmd_jobs(args):
    d = _get(args, f'/api/jobs?limit={args.limit}')
    if args.json: print(json.dumps(d, indent=1)); return 0
    for j in d['items']: print(f'  #{j["id"]:<6} {j["status"]:9} {j["kind"]:5} {(j["role"] or j["note"] or ""):20} {j["app"]:16} {j.get("run_s") or ""}')
    return 0


def cmd_eval(args):
    """Experiments: list, run (through the running service's queue), report (the standard write-up), forget (cached outputs)."""
    from stratamodels.evals import core as C, report as R
    if args.action == 'list':
        for name, s in C.list_specs().items(): print(f'  {name:28} {"ERROR " + str(s) if isinstance(s, C.EvalError) else s.question}')
        for d in C.list_datasets(): print(f'  dataset {d["name"]:20} {d.get("n", "?")} items  {d.get("version") or d.get("error")}')
        return 0
    if not args.name: sys.exit('name the experiment')
    try: spec = C.load_spec(args.name)
    except C.EvalError as e: sys.exit(str(e))
    from stratamodels.store import Store
    engine = Store.connect(migrate=args.action == 'run').engine
    if args.action == 'run':
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'client'))
        import strata_models_client as sm
        if not sm.available(): sys.exit(f'strata-models is not running at {sm.url()}: eval jobs go through its queue')
        r = C.run(spec, engine, C.ClientExecutor(sm), candidates=args.candidate or None, limit=args.limit); print(json.dumps(r)); return 0
    if args.action == 'forget': print(f'{C.forget(spec, engine, args.candidate[0] if args.candidate else None)} outputs forgotten'); return 0
    res = C.analyse(spec, engine)
    if args.json: print(json.dumps(res, indent=1, default=str)); return 0
    ds, outs, _ = C.outputs_for(spec, engine); text = R.render(res, R.worst(spec, outs, ds.items, spec.primary))
    if args.out:
        with open(args.out, 'w', encoding='utf8') as f: f.write(text)
        print(f'wrote {args.out}')
    else: print(text)
    return 0


def cmd_set(args):
    if args.value is None: print(f'{args.key} = {settings.all_settings().get(args.key)}'); return 0
    try: settings.save({args.key: None if args.value == 'default' else args.value})
    except (KeyError, ValueError) as e: sys.exit(str(e))
    print(f'{args.key} = {settings.all_settings()[args.key]}'); return 0


def cmd_serve(args):
    from stratamodels.server import app as server_app
    server_app.main(['--host', args.host, '--port', str(args.port)] + (['--memory'] if args.memory else [])); return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog='strata-models'); sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('doctor', help='check the environment, the runtimes and the GPU').set_defaults(fn=doctor)
    for name, fn, h in (('status', cmd_status, 'what the running service is doing'), ('jobs', cmd_jobs, 'recent jobs of the running service')):
        c = sub.add_parser(name, help=h); c.add_argument('--url', default='http://127.0.0.1:8366'); c.add_argument('--json', action='store_true'); c.set_defaults(fn=fn)
        if name == 'jobs': c.add_argument('--limit', type=int, default=20)
    r = sub.add_parser('registry', help='roles and the model chosen for each'); r.add_argument('--json', action='store_true'); r.set_defaults(fn=cmd_registry)
    e = sub.add_parser('eval', help='experiments: list | run NAME | report NAME [--out FILE] | forget NAME'); e.add_argument('action', choices=['list', 'run', 'report', 'forget']); e.add_argument('name', nargs='?')
    e.add_argument('--candidate', action='append', help='only this candidate (repeatable)'); e.add_argument('--limit', type=int, help='only the first N items (a trial run)'); e.add_argument('--out', help='write the report here'); e.add_argument('--json', action='store_true'); e.set_defaults(fn=cmd_eval)
    s = sub.add_parser('set', help='show or change a setting (data_dir, models_dir, vram_reserve_gb); value "default" resets it'); s.add_argument('key'); s.add_argument('value', nargs='?'); s.set_defaults(fn=cmd_set)
    v = sub.add_parser('serve', help='start the service'); v.add_argument('--host', default='127.0.0.1'); v.add_argument('--port', type=int, default=8366); v.add_argument('--memory', action='store_true', help='keep jobs in memory instead of the database'); v.set_defaults(fn=cmd_serve)
    a = p.parse_args(argv); return a.fn(a)
