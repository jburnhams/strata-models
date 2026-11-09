"""The view geometry of the faces runtime (no model needed): a point found in a view maps back to the original picture."""

import numpy as np, pytest

from stratamodels import runtimes
from stratamodels.runtimes import faces


def _img(w=200, h=100):
    return np.zeros((h, w, 3), np.uint8)


@pytest.mark.parametrize(
    "view",
    [
        dict(),
        dict(pad=2),
        dict(pad=3, rotate=90),
        dict(crop=[20, 10, 160, 80], pad=2),
        dict(crop=[20, 10, 160, 80], pad=1.5, rotate=270),
    ],
)
def test_a_point_maps_back_to_the_original(view):
    img = _img()
    img[40, 150] = 255  # one white pixel at x 150, y 40 of the original
    vimg, back = faces._view(img, view)
    ys, xs = np.nonzero(vimg[..., 0] == 255)
    assert len(xs) == 1
    assert back(float(xs[0]), float(ys[0])) == pytest.approx((150, 40))


def test_pad_grows_the_canvas_and_keeps_the_picture_in_the_middle():
    vimg, _ = faces._view(_img(200, 100), dict(pad=2))
    assert vimg.shape[:2] == (200, 400) and vimg[0, 0, 0] == 128 and vimg[100, 200, 0] == 0


@pytest.mark.parametrize("pad", ["x", 0.5, 5])
def test_pad_out_of_range_is_refused(pad):
    with pytest.raises(runtimes.RunError, match="pad must"):
        faces._view(_img(), dict(pad=pad))
