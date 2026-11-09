"""Outputs made elsewhere and recorded, served as if a model made them: a baseline that cannot run here (a cloud model, a person's labels) can
then stand in an eval beside the local candidates, on the same items, in the blind judging and the write-up.

options:
  file   a JSON lines file of {"key": ..., "output": {...}} (absolute, or relative to the data folder)
  by     the job param the key is matched on (default "image")
A job whose key was not recorded fails, so a missing record counts against the baseline rather than vanishing. Uses no GPU."""
import json, os

from stratamodels import settings
from stratamodels.runtimes import RunError


def load(model, models_dir):
    f = model.options.get('file')
    if not f: raise RunError(f'model {model.key} names no file of recorded outputs')
    path = f if os.path.isabs(f) else os.path.join(settings.data_dir(), f)
    try: lines = open(path, encoding='utf8').read().splitlines()
    except OSError as e: raise RunError(f'cannot read recorded outputs {path}: {e}')
    rec = {}
    for line in lines:
        if line.strip(): r = json.loads(line); rec[str(r['key'])] = r['output']
    return dict(key=model.key, by=model.options.get('by', 'image'), outputs=rec)


def run(handle, params):
    k = params.get(handle['by'])
    if k is None: raise RunError(f'give {handle["by"]}: recorded outputs are found by it')
    if str(k) not in handle['outputs']: raise RunError(f'nothing recorded for {handle["by"]}={k!r}')
    return dict(handle['outputs'][str(k)], model=handle['key'])


def unload(handle): handle.clear()
