"""The CLIP runtime's input handling (stratamodels.runtimes.clip) without open_clip: arrays, refusals. The model itself: tests/gpu."""
import numpy as np
import pytest

from stratamodels.runtimes import RunError, clip


def test_arrays_are_read_and_checked(tmp_path):
    a = np.zeros((2, 8, 8, 3), np.uint8); a[..., 0] = 10
    np.save(tmp_path / 'a.npy', a); np.savez(tmp_path / 'b.npz', cam=a, other=a)
    assert clip._array(dict(path=str(tmp_path / 'a.npy'))).shape == (2, 8, 8, 3)
    assert clip._array(dict(path=str(tmp_path / 'b.npz'), key='cam', bgr=True))[0, 0, 0, 2] == 10
    assert clip._array(dict(path=str(tmp_path / 'b.npz'))).shape[0] == 2
    with pytest.raises(RunError, match='no array'): clip._array(dict(path=str(tmp_path / 'b.npz'), key='x'))
    with pytest.raises(RunError, match='no such array'): clip._array(dict(path='/nope.npz'))
    with pytest.raises(RunError, match='no such array'): clip._array('x')
    np.save(tmp_path / 'f.npy', np.zeros((2, 8, 8), np.float32))
    with pytest.raises(RunError, match='uint8'): clip._array(dict(path=str(tmp_path / 'f.npy')))
    with pytest.raises(RunError, match='no such image'): clip._read_rgb('/nope.jpg')


def test_params_are_checked_before_any_work(monkeypatch):
    import sys, types
    monkeypatch.setitem(sys.modules, 'torch', types.ModuleType('torch'))
    for p, msg in ((dict(colour=1), 'unknown params'), (dict(resize='stretch'), "'crop' or 'square'"), (dict(paths='a.jpg'), 'list of image files'), (dict(texts=[1]), 'list of strings'), (dict(), 'give paths')):
        with pytest.raises(RunError, match=msg): clip.run({}, p)
