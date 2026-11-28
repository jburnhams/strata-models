"""The nsfw runtime without a model: params, preprocessing, unreadable files, batching. The model itself: tests/gpu."""

import numpy as np
import pytest
from PIL import Image

from stratamodels.runtimes import RunError, nsfw


def _handle(batch=2, resize="square"):
    return dict(
        size=8,
        mean=np.full(3, 0.5, np.float32),
        std=np.full(3, 0.5, np.float32),
        resize=resize,
        transform=None,
        batch=batch,
        repo="r",
        dev="cpu",
    )


def _png(tmp_path, name, colour):
    p = tmp_path / name
    Image.new("RGB", (20, 10), colour).save(p)
    return str(p)


def test_prepare_squeezes_and_normalises(tmp_path):
    a = nsfw._prepare(_handle(), _png(tmp_path, "w.png", (255, 255, 255)))
    assert a.shape == (3, 8, 8) and np.allclose(a, 1.0)


def test_prepare_refuses_a_missing_file():
    with pytest.raises(RunError, match="no such image"):
        nsfw._prepare(_handle(), "/nonexistent.jpg")


def test_run_batches_and_keeps_going_past_a_corrupt_file(tmp_path, monkeypatch):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a picture")
    paths = [_png(tmp_path, f"{i}.png", (i * 40, 0, 0)) for i in range(3)]
    seen = []

    def fake(h, batch):
        seen.append(len(batch))
        return batch[:, 0, 0, 0] * 0 + 0.25

    monkeypatch.setattr(nsfw, "score", fake)
    r = nsfw.run(_handle(), {"paths": [paths[0], str(bad), paths[1], paths[2]]})
    assert seen == [2, 1] and r["model"] == "r"
    assert [i.get("sensitive") for i in r["items"]] == [0.25, None, 0.25, 0.25]
    assert r["items"][1]["error"].startswith("UnidentifiedImageError")


def test_single_path_answers_flat(tmp_path, monkeypatch):
    monkeypatch.setattr(nsfw, "score", lambda h, b: np.array([0.9]))
    r = nsfw.run(_handle(), {"path": _png(tmp_path, "a.png", (1, 2, 3))})
    assert r["sensitive"] == 0.9 and "items" not in r


def test_params_are_checked():
    for prm, msg in ((dict(colour=1), "unknown params"), (dict(), "give path"), (dict(paths=[]), "give path")):
        with pytest.raises(RunError, match=msg):
            nsfw.run(_handle(), prm)
    with pytest.raises(RunError, match="no such image"):
        nsfw.run(_handle(), {"paths": ["/nonexistent.png"]})
