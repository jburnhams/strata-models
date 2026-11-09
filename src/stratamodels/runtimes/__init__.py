"""Runtimes: how a kind of model is loaded, run and freed. Each is a module with three functions:

    load(model, models_dir) -> handle        model is a registry.Model; weights go under models_dir/<runtime>/
    run(handle, params) -> result            params and result are JSON (inputs are file paths: everything is on this machine)
    unload(handle)                           free its GPU memory

Heavy libraries are imported inside `load`, so the service starts (and its tests run) without them; `available()` says which are installed."""
import importlib

RUNTIMES = {'fake': 'stratamodels.runtimes.fake', 'insightface': 'stratamodels.runtimes.faces', 'faster-whisper': 'stratamodels.runtimes.whisper', 'sd-cpp': 'stratamodels.runtimes.sdcpp', 'open-clip': 'stratamodels.runtimes.clip', 'face-kind': 'stratamodels.runtimes.facekind'}
NEEDS = {'fake': (), 'insightface': ('insightface', 'onnxruntime', 'cv2'), 'faster-whisper': ('faster_whisper', 'ctranslate2'), 'sd-cpp': ('huggingface_hub',), 'open-clip': ('open_clip', 'torch'), 'face-kind': ('open_clip', 'torch')}


class RunError(ValueError):
    """The job's params are wrong for this runtime (a missing file, an unknown option): the job fails, the model stays loaded."""


def get(name):
    if name not in RUNTIMES: raise KeyError(f'no runtime {name!r}')
    return importlib.import_module(RUNTIMES[name])


def available():
    """{runtime: list of missing Python modules} (an empty list means it can run here)."""
    out = {}
    for name, mods in NEEDS.items():
        missing = []
        for m in mods:
            try: importlib.import_module(m)
            except Exception: missing.append(m)
        out[name] = missing
    import os
    from stratamodels import settings
    from stratamodels.runtimes import sdcpp
    if not os.path.isfile(sdcpp.binary(settings.models_dir())): out['sd-cpp'] = out['sd-cpp'] + ['the sd-cli binary (stable-diffusion.cpp)']
    return out
