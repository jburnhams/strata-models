"""Vision-language models with llama.cpp's `llama-server` (GGUF weights and an mmproj vision projector): Qwen3.5, Qwen3.8, Gemma 4, and
whatever else llama.cpp runs.

A model's files are named in models.toml as Hugging Face references, "repo:path/in/repo" (or absolute paths), under options:
  model       the GGUF language model (required)
  mmproj      the vision projector (required for pictures)
  ctx         context size in tokens (default 8192)
  args        extra llama-server arguments, e.g. ["--n-cpu-moe", "20"] or ["-ngl", "40"] for a model larger than the card
Files are fetched once into <models>/llama/<repo>/. The binary is STRATAMODELS_LLAMA_BIN, else the newest
<models>/bin/llama/llama-*/llama-server; the CUDA runtime libraries in <models>/bin/llama/cudart-*/ go on its library path.

Loading starts one llama-server on a free local port and waits for it to answer; unloading stops it. While loaded, the card holds the model.

params:
  image        path of a picture (or images: a list of paths)
  prompt       the question or instruction (or preset: a prompt kept in this package, see presets/; its {context} is filled from `context`)
  context      a dict of facts about the picture (capture metadata), given to a preset prompt
  schema       a JSON schema the answer must follow (grammar-constrained); a preset brings its own. Off with schema=false.
  system, max_tokens (default 2048), temperature (default 0.2), seed (default 0), think (default false: thinking models answer directly)
Result: {text, json (when a schema was used and the answer parses), tokens_in, tokens_out, seconds, peak_vram_mb, model}."""
import atexit, base64, glob, json, mimetypes, os, socket, subprocess, threading, time, urllib.error, urllib.request

from stratamodels import gpu, settings
from stratamodels.runtimes import RunError

PRESETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presets')
LIVE = set()          # running llama-server processes: stopped at exit, so a restarted service never finds the card held by an orphan
PARAMS = {'image', 'images', 'prompt', 'preset', 'context', 'schema', 'system', 'max_tokens', 'temperature', 'seed', 'think'}


def binary(models_dir):
    if os.environ.get('STRATAMODELS_LLAMA_BIN'): return os.environ['STRATAMODELS_LLAMA_BIN']
    found = sorted(glob.glob(os.path.join(models_dir, 'bin', 'llama', 'llama-*', 'llama-server')))
    return found[-1] if found else os.path.join(models_dir, 'bin', 'llama', 'llama-server')


def library_path(exe):
    """The binary's own folder and any cudart-* folder beside it (llama.cpp's CUDA builds ship the CUDA runtime apart)."""
    here = os.path.dirname(exe); return [here] + sorted(glob.glob(os.path.join(os.path.dirname(here), 'cudart-*')))


def fetch(ref, models_dir):
    """A local path for "repo:path" (downloaded once into <models>/llama/<repo>/) or for an absolute path."""
    if os.path.isabs(ref):
        if not os.path.isfile(ref): raise RunError(f'no such model file: {ref}')
        return ref
    repo, _, path = ref.partition(':')
    if not path: raise RunError(f'a model file is "repo:path" or an absolute path, not {ref!r}')
    local = os.path.join(models_dir, 'llama', repo.replace('/', '--'))
    if os.path.isfile(os.path.join(local, path)): return os.path.join(local, path)
    from huggingface_hub import hf_hub_download
    try: return hf_hub_download(repo, path, local_dir=local)
    except Exception as e: raise RunError(f'could not fetch {ref}: {type(e).__name__}: {str(e)[:300]}')


def _free_port():
    with socket.socket() as s: s.bind(('127.0.0.1', 0)); return s.getsockname()[1]


def command(exe, files, port, ctx, args):
    cmd = [exe, '-m', files['model'], '--host', '127.0.0.1', '--port', str(port), '-c', str(ctx), '-np', '1', '--no-webui', '-fa', 'on']
    if files.get('mmproj'): cmd += ['--mmproj', files['mmproj']]
    return cmd + list(args)


def _get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r: return r.status, r.read()


def load(model, models_dir, popen=subprocess.Popen, wait_s=600):
    o = model.options; exe = binary(models_dir)
    if not os.path.isfile(exe): raise RunError(f'llama.cpp is not installed: no {exe} (set STRATAMODELS_LLAMA_BIN)')
    if not o.get('model'): raise RunError(f'model {model.key} names no model file in models.toml')
    files = {k: fetch(o[k], models_dir) for k in ('model', 'mmproj') if o.get(k)}
    port = _free_port(); cmd = command(exe, files, port, int(o.get('ctx', 8192)), o.get('args') or [])
    env = dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join(library_path(exe) + [os.environ.get('LD_LIBRARY_PATH', '')]))
    log = open(os.path.join(settings.data_dir(), f'llama-{model.key}.log'), 'w')
    p = popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env); LIVE.add(p); url = f'http://127.0.0.1:{port}'; t0 = time.time()
    while True:
        if p.poll() is not None:
            LIVE.discard(p); log.close(); tail = open(log.name).read()[-800:]
            raise RunError(f'llama-server exited while loading (exit {p.returncode}): {tail}')
        try:
            if _get(url + '/health')[0] == 200: break
        except (OSError, urllib.error.URLError): pass
        if time.time() - t0 > wait_s: _stop(p); log.close(); raise RunError(f'llama-server did not answer within {wait_s} s')
        time.sleep(0.5)
    return dict(key=model.key, proc=p, url=url, log=log, vision=bool(files.get('mmproj')))


def preset(name):
    """(prompt template, schema or None) of a preset: presets/<name>.md and, when there, presets/<name>.schema.json."""
    if not name or '/' in name or name.startswith('.'): raise RunError(f'no preset {name!r}')
    p = os.path.join(PRESETS, f'{name}.md')
    if not os.path.isfile(p): raise RunError(f'no preset {name!r} (have: {", ".join(sorted(f[:-3] for f in os.listdir(PRESETS) if f.endswith(".md")))})')
    s = os.path.join(PRESETS, f'{name}.schema.json')
    return open(p, encoding='utf8').read(), (json.load(open(s, encoding='utf8')) if os.path.isfile(s) else None)


def _image_part(path):
    if not isinstance(path, str) or not os.path.isfile(path): raise RunError(f'no such image: {path!r}')
    mime = mimetypes.guess_type(path)[0] or 'image/jpeg'
    return {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,' + base64.b64encode(open(path, 'rb').read()).decode()}}


def request(params):
    """The chat request body for a job (and the schema it asked for, if any)."""
    unknown = set(params) - PARAMS
    if unknown: raise RunError(f'unknown params {sorted(unknown)}')
    schema = params.get('schema')
    if params.get('preset'):
        if params.get('prompt'): raise RunError('give a prompt or a preset, not both')
        text, pschema = preset(params['preset']); schema = pschema if schema is None else schema
        text = text.replace('{context}', json.dumps(params.get('context') or {}, ensure_ascii=False))
    else:
        text = params.get('prompt')
        if not isinstance(text, str) or not text.strip(): raise RunError('give a prompt (or a preset)')
    imgs = params.get('images') or ([params['image']] if params.get('image') else [])
    if not isinstance(imgs, list): raise RunError('images is a list of paths')
    msgs = ([{'role': 'system', 'content': str(params['system'])}] if params.get('system') else []) + [{'role': 'user', 'content': [_image_part(i) for i in imgs] + [{'type': 'text', 'text': text}]}]
    body = dict(messages=msgs, max_tokens=int(params.get('max_tokens') or 2048), temperature=float(params.get('temperature', 0.2)), seed=int(params.get('seed', 0)),
                chat_template_kwargs=dict(enable_thinking=bool(params.get('think', False))))
    if schema: body['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'answer', 'schema': schema, 'strict': True}}
    return body, schema


def run(handle, params, probe=gpu.query, post=None):
    body, schema = request(params)
    if any(isinstance(c, dict) and c.get('type') == 'image_url' for m in body['messages'] for c in (m['content'] if isinstance(m['content'], list) else [])) and not handle.get('vision'):
        raise RunError(f'model {handle["key"]} has no vision projector (mmproj) in models.toml')
    peak = [0.0]; stop = threading.Event(); pid = handle['proc'].pid

    def watch():
        while not stop.wait(0.5):
            for pr in (probe() or {}).get('processes', []):
                if pr['pid'] == pid: peak[0] = max(peak[0], pr['used_mb'])
    threading.Thread(target=watch, daemon=True).start(); t0 = time.perf_counter()
    try: r = (post or _post)(handle['url'] + '/v1/chat/completions', body)
    finally: stop.set()
    msg = r['choices'][0]['message']; text = msg.get('content') or ''; u = r.get('usage') or {}
    out = dict(text=text, tokens_in=u.get('prompt_tokens'), tokens_out=u.get('completion_tokens'), seconds=round(time.perf_counter() - t0, 2), peak_vram_mb=round(peak[0]), model=handle['key'],
               finish=r['choices'][0].get('finish_reason'))
    if schema:
        try: out['json'] = json.loads(text)
        except ValueError: out['json'] = None
    return out


def _post(url, body, timeout=1800):
    req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
    except urllib.error.HTTPError as e: raise RunError(f'llama-server refused the request ({e.code}): {e.read().decode(errors="replace")[:500]}')


def _stop(p, wait_s=30):
    if p.poll() is None:
        p.terminate()
        try: p.wait(wait_s)
        except subprocess.TimeoutExpired: p.kill(); p.wait()
    LIVE.discard(p)


def unload(handle):
    p = handle.get('proc')
    if p is not None: _stop(p)
    if handle.get('log'): handle['log'].close()
    handle.clear()


@atexit.register
def stop_all():
    """Stop every llama-server still running (the service exiting, also on SIGTERM: uvicorn shuts down and Python runs its exit hooks)."""
    for p in list(LIVE):
        try: _stop(p, 10)
        except Exception: pass
