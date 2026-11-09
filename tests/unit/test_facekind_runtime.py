"""The face-kind runtime without a model: cutting the faces out, the head, refusals. The model itself: tests/gpu."""
import numpy as np
import pytest

from stratamodels.runtimes import RunError, facekind


def test_crops_take_the_margin_and_stay_inside():
    rgb = np.zeros((100, 200, 3), np.uint8)
    c = facekind.crops(rgb, [[50, 20, 90, 60]], margin=0.25)[0]
    assert c.shape[:2] == (60, 60)                                         # 40 px box + 10 px each side
    edge = facekind.crops(rgb, [[0, 0, 40, 40]], margin=0.5)[0]
    assert edge.shape[:2] == (60, 60)                                      # cut at the picture's edge
    with pytest.raises(RunError, match='empty box'): facekind.crops(rgb, [[5, 5, 5, 9]])
    with pytest.raises(RunError, match='outside'): facekind.crops(rgb, [[300, 5, 310, 9]])
    with pytest.raises(RunError, match='a box is'): facekind.crops(rgb, [['a', 1, 2]])


def test_a_trained_head_gives_probabilities_over_its_classes():
    W = np.array([[1, 0], [0, 1]], np.float32); h = dict(head=dict(W=W, b=np.zeros(2, np.float32), classes=['real', 'statue'], file='h.npz'))
    p, names = facekind.classify(h, np.array([[5, 0], [0, 5]], np.float32))
    assert names == ['real', 'statue'] and p[0, 0] > 0.99 and p[1, 1] > 0.99 and np.allclose(p.sum(1), 1)


def test_zero_shot_uses_the_prompt_vectors():
    h = dict(text=np.eye(len(facekind.KINDS), 4, dtype=np.float32))
    p, names = facekind.classify(h, np.array([[0, 1, 0, 0]], np.float32))
    assert names[int(p.argmax())] == 'statue'


def test_params_are_checked():
    for prm, msg in ((dict(colour=1), 'unknown params'), (dict(), 'give path'), (dict(path='/x.jpg', boxes='a'), 'boxes is a list')):
        with pytest.raises(RunError, match=msg): facekind.run({}, prm)


def _fake_clip(monkeypatch, tmp_path):
    from stratamodels.runtimes import clip
    calls = []
    monkeypatch.setattr(clip, 'load', lambda model, models_dir: dict(model='clip'))
    monkeypatch.setattr(clip, 'unload', calls.append)
    monkeypatch.setattr(clip, '_read_rgb', lambda path: np.zeros((100, 200, 3), np.uint8))
    monkeypatch.setattr(clip, '_encode_images', lambda h, rgbs, resize: np.array([[1.0, 0.0] if rgbs[0].shape[0] > 40 else [0.0, 1.0]], np.float32))
    monkeypatch.setattr(clip, '_array', lambda spec: np.zeros((2, 32, 32, 3), np.uint8))
    return calls


def test_a_head_file_is_loaded_and_faces_are_classified_from_a_path_or_an_array(monkeypatch, tmp_path):
    from stratamodels.registry import Model
    calls = _fake_clip(monkeypatch, tmp_path)
    (tmp_path / 'face-kind').mkdir(); np.savez(tmp_path / 'face-kind' / 'h.npz', W=np.eye(2), b=np.zeros(2), classes=np.array(['real', 'statue']))
    h = facekind.load(Model(key='fk', runtime='face-kind', name='ViT-B-32', roles=('face-kind',), vram_gb=1, options=dict(head='h.npz', margin=0.25)), str(tmp_path))
    assert h['head']['classes'] == ['real', 'statue'] and h['margin'] == 0.25 and h['head']['W'].dtype == np.float32
    out = facekind.run(h, dict(path='/x.jpg', boxes=[[50, 20, 90, 60], [0, 0, 10, 10]]))
    assert out['head'] == 'trained h.npz' and [f['kind'] for f in out['faces']] == ['real', 'statue'] and out['faces'][0]['real'] > 0.5
    out = facekind.run(h, dict(array=dict(path='/crops.npy')))
    assert len(out['faces']) == 2 and all(f['kind'] == 'statue' for f in out['faces'])
    assert facekind.run(h, dict(path='/x.jpg', boxes=[])) == dict(faces=[], head='trained h.npz')
    facekind.unload(h); assert calls == [h]
