"""The real runtimes on the real GPU: run on the GPU machine with the GPU requirements installed. Not collected by CI (testpaths is tests/unit).

    STRATAMODELS_TEST_IMAGE=/path/photo-with-faces.jpg STRATAMODELS_TEST_AUDIO=/path/speech.flac pytest tests/gpu -s
Weights come from the real models folder (STRATAMODELS_MODELS or ~/models), so the isolated home of the other suites does not apply here."""
import os, time

import pytest

from stratamodels import gpu, runtimes
from stratamodels.registry import Registry

IMAGE, AUDIO = os.environ.get('STRATAMODELS_TEST_IMAGE'), os.environ.get('STRATAMODELS_TEST_AUDIO')
MODELS = os.path.expanduser(os.environ.get('STRATAMODELS_MODELS', '~/models'))


def measure(model, params):
    rt = runtimes.get(model.runtime); before = gpu.own_used_mb(gpu.query())
    t0 = time.perf_counter(); h = rt.load(model, MODELS); load_s = time.perf_counter() - t0
    vram = gpu.own_used_mb(gpu.query()) - before
    t0 = time.perf_counter(); out = rt.run(h, params); run_s = time.perf_counter() - t0
    t0 = time.perf_counter(); out2 = rt.run(h, params); warm_s = time.perf_counter() - t0
    rt.unload(h)
    print(f'\n{model.key}: load {load_s:.1f} s, VRAM {vram:.0f} MB, first run {run_s:.2f} s, warm run {warm_s:.2f} s')
    return out, out2


@pytest.mark.skipif(not IMAGE or runtimes.available()['insightface'], reason='needs STRATAMODELS_TEST_IMAGE and insightface')
def test_faces_on_the_gpu():
    out, _ = measure(Registry.load().resolve('faces'), dict(path=IMAGE, fields=['bbox', 'det_score', 'embedding', 'landmark_3d_68', 'pose']))
    f = out['faces'][0]
    assert out['faces'] and len(f['embedding']) == 512 and abs(sum(v * v for v in f['embedding']) - 1) < 1e-3 and len(f['landmark_3d_68']) == 68
    small, _ = measure(Registry.load().resolve('faces'), dict(path=IMAGE, max_side=640))
    assert abs(small['faces'][0]['bbox'][0] - f['bbox'][0]) < 0.05 * out['width']          # boxes come back in the original's pixels


@pytest.mark.skipif(not AUDIO or runtimes.available()['faster-whisper'], reason='needs STRATAMODELS_TEST_AUDIO and faster-whisper')
@pytest.mark.parametrize('role', ['transcribe', 'transcribe-fast'])
def test_whisper_on_the_gpu(role):
    out, _ = measure(Registry.load().resolve(role), dict(path=AUDIO, language='en'))
    text = ' '.join(s['text'] for s in out['segments']).lower()
    assert 'ask not what your country' in text and out['segments'][0]['words']
