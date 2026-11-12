"""A stand-in model for tests and demos: loads instantly, 'runs' by sleeping `seconds` and echoing its params (or, given `say`, answering like a
transcriber, given `expand`, like a text model, given `favour` or `always`, like a model judge, or given `speak`, like a TTS model whose audio a fake transcriber hears back from a `path`). Failing on purpose: `fail` = a message."""

import json
import os
import time

from stratamodels.runtimes import RunError

LOADED = []  # handles currently loaded, so tests can see loads and unloads


def load(model, models_dir):
    h = dict(model=model.key, options=dict(model.options))
    LOADED.append(h)
    return h


def run(handle, params):
    if params.get("fail"):
        raise RunError(str(params["fail"]))
    s = float(params.get("seconds", 0) or 0)
    if s:
        time.sleep(min(s, 30))
    if "speak" in params:  # a fake speaker for TTS evals: the "audio" is a path that spells the words
        text = str(params["speak"])
        return dict(path="/fake/" + "_".join(text.split()) + ".wav", duration_s=len(text.split()) * 0.4, seconds=0.1)
    if "path" in params and str(params["path"]).startswith(
        "/fake/"
    ):  # ...and a fake listener that hears them back (dropping every `drop`-th)
        params = dict(params, say=os.path.basename(params["path"])[:-4].replace("_", " "))
    if (
        "favour" in params or "always" in params
    ):  # a fake model judge (evals.judges): picks the answer that says `favour`, or always one side
        p = str(params.get("prompt", ""))
        x, _, y = p.partition("Answer X:")[2].partition("Answer Y:")
        w = params.get("always") or (
            "X"
            if params["favour"] in x and params["favour"] not in y
            else "Y"
            if params["favour"] in y and params["favour"] not in x
            else "tie"
        )
        d = dict(reason=f"fake judge, favouring {params.get('favour')!r}", winner=w)
        return dict(text=json.dumps(d), json=d, model=handle["model"], images=params.get("images") or [])
    if "expand" in params:  # a fake text model for pipeline evals: "expands" a prompt (a style param says how)
        return dict(
            text=f'"{params.get("style", "detailed")}: {params["expand"]}"',
            tokens_out=len(str(params["expand"]).split()) + 1,
            model=handle["model"],
        )
    if "say" in params:  # a fake transcriber for eval tests: says the text, dropping every `drop`-th word
        words = str(params["say"]).split()
        n = int(params.get("drop") or 0)
        return dict(segments=[dict(text=" ".join(w for i, w in enumerate(words, 1) if not n or i % n))])
    return dict(model=handle["model"], echo=params)


def unload(handle):
    if handle in LOADED:
        LOADED.remove(handle)
