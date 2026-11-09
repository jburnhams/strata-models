"""Command line: `doctor` checks the environment, `hello` prints or records a greeting, `serve` starts the web server."""
import argparse, importlib, json, sys

from app import __version__, greet, settings

CHECKS = (('fastapi', 'fastapi'), ('uvicorn', 'uvicorn'), ('pydantic', 'pydantic'))


def doctor(args):
    ok = True; print(f'app {__version__}, python {sys.version.split()[0]}')
    for label, mod in CHECKS:
        try: m = importlib.import_module(mod); print(f'  ok       {label} {getattr(m, "__version__", "")}'.rstrip())
        except ImportError: ok = False; print(f'  MISSING  {label}: pip install -r requirements.txt')
    return 0 if ok else 1


def cmd_hello(args):
    try:
        rec = greet.add_message(args.name, args.greeting) if args.save else greet.format_greeting(args.name, args.greeting)
    except greet.GreetingError as e: sys.exit(str(e))
    if args.json: print(json.dumps(rec)); return 0
    print(rec['message']); return 0


def cmd_messages(args):
    if args.clear:
        res = greet.clear_messages()
        print(json.dumps(res) if args.json else f"cleared {res['cleared']} message(s)")
        return 0
    items = greet.list_messages()
    if args.json: print(json.dumps(items, indent=1)); return 0
    if not items: print('  (no messages yet: ./app hello World --save)'); return 0
    for m in items: print(f"  #{m['id']:<4} {m['message']:30} ({m['created']})")
    return 0


def cmd_set(args):
    if args.value is None: print(f'{args.key} = {settings.all_settings().get(args.key)}'); return 0
    try: settings.save({args.key: None if args.value == 'default' else args.value})
    except KeyError as e: sys.exit(str(e))
    print(f'{args.key} = {settings.all_settings()[args.key]}'); return 0


def cmd_serve(args):
    from app.server import app as server_app
    server_app.main(['--host', args.host, '--port', str(args.port)]); return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog='app'); sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('doctor', help='check the environment').set_defaults(fn=doctor)
    h = sub.add_parser('hello', help='print a greeting (and optionally record it)'); h.add_argument('name', nargs='?', default='World'); h.add_argument('--greeting', help='override the default greeting word'); h.add_argument('--save', action='store_true', help='record the greeting in the message log'); h.add_argument('--json', action='store_true'); h.set_defaults(fn=cmd_hello)
    m = sub.add_parser('messages', help='list or clear recorded greetings'); m.add_argument('--clear', action='store_true'); m.add_argument('--json', action='store_true'); m.set_defaults(fn=cmd_messages)
    s = sub.add_parser('set', help='show or change a setting (data_dir, greeting); value "default" resets it'); s.add_argument('key'); s.add_argument('value', nargs='?'); s.set_defaults(fn=cmd_set)
    v = sub.add_parser('serve', help='start the web server'); v.add_argument('--host', default='127.0.0.1'); v.add_argument('--port', type=int, default=8360); v.set_defaults(fn=cmd_serve)
    a = p.parse_args(argv); return a.fn(a)
