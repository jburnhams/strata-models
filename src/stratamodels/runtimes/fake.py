"""A stand-in model for tests and demos: loads instantly, 'runs' by sleeping `seconds` and echoing its params (or, given `say`, answering like a
transcriber). Failing on purpose: `fail` = a message."""
import time

from stratamodels.runtimes import RunError

LOADED = []          # handles currently loaded, so tests can see loads and unloads


def load(model, models_dir):
    h = dict(model=model.key, options=dict(model.options)); LOADED.append(h); return h


def run(handle, params):
    if params.get('fail'): raise RunError(str(params['fail']))
    s = float(params.get('seconds', 0) or 0)
    if s: time.sleep(min(s, 30))
    if 'say' in params:                                       # a fake transcriber for eval tests: says the text, dropping every `drop`-th word
        words = str(params['say']).split(); n = int(params.get('drop') or 0)
        return dict(segments=[dict(text=' '.join(w for i, w in enumerate(words, 1) if not n or i % n))])
    return dict(model=handle['model'], echo=params)


def unload(handle):
    if handle in LOADED: LOADED.remove(handle)
