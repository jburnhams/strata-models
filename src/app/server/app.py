"""The application server: `./app serve` starts the FastAPI backend and serves the built React app.

    ./app serve [--host 127.0.0.1] [--port 8360]

API (JSON): GET /api/health; GET /api/settings; PUT /api/settings {data_dir?, greeting?}; GET /api/hello?name=&greeting=;
GET /api/messages; POST /api/messages {name, greeting?}; DELETE /api/messages.
The page at / serves `src/app/server/static/index.html` when built (`cd web && npm run build`), else the plain stand-in `landing.html`."""
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import __version__, greet, settings

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
LANDING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'landing.html')


class SettingsIn(BaseModel):
    data_dir: str | None = None
    greeting: str | None = None


class MessageIn(BaseModel):
    name: str = 'World'
    greeting: str | None = None


def create_app():
    app = FastAPI(title='python-app-template', docs_url='/api/docs', openapi_url='/api/openapi.json')

    def guard(fn, *a, **kw):
        try: return fn(*a, **kw)
        except greet.GreetingError as e: raise HTTPException(400, str(e))

    @app.get('/', response_class=HTMLResponse)
    def index():
        built = os.path.join(STATIC, 'index.html')          # the React app when it has been built (cd web && npm run build), else the plain stand-in page
        with open(built if os.path.exists(built) else LANDING, encoding='utf8') as f: return f.read()

    @app.get('/api/health')
    def get_health(): return dict(ok=True, version=__version__)

    @app.get('/api/settings')
    def get_settings(): return settings.all_settings()

    @app.put('/api/settings')
    def put_settings(s: SettingsIn):
        if s.data_dir is not None and not os.path.isabs(os.path.expanduser(s.data_dir)): raise HTTPException(400, 'the data folder must be an absolute path')
        changes = {}
        if s.data_dir is not None: changes['data_dir'] = s.data_dir
        if s.greeting is not None: changes['greeting'] = s.greeting
        if changes: settings.save(changes)
        return settings.all_settings()

    @app.get('/api/hello')
    def get_hello(name: str = 'World', greeting: str | None = None):
        return guard(greet.format_greeting, name, greeting)

    @app.get('/api/messages')
    def get_messages(): return greet.list_messages()

    @app.post('/api/messages')
    def post_message(b: MessageIn): return guard(greet.add_message, b.name, b.greeting)

    @app.delete('/api/messages')
    def delete_messages(): return greet.clear_messages()

    if os.path.isdir(os.path.join(STATIC, 'assets')): app.mount('/assets', StaticFiles(directory=os.path.join(STATIC, 'assets')), name='assets')
    return app


def main(argv=None):
    import argparse, uvicorn
    a = argparse.ArgumentParser(); a.add_argument('--host', default='127.0.0.1'); a.add_argument('--port', type=int, default=8360); a = a.parse_args(argv)
    print(f'data folder: {settings.data_dir()}\nhttp://{a.host}:{a.port}/'); uvicorn.run(create_app(), host=a.host, port=a.port, log_level='warning')
