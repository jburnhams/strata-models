"""The view geometry of the faces runtime (no model needed): a point found in a view maps back to the original picture."""

import numpy as np
import pytest

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


@pytest.mark.parametrize("ext", [".jpg", ".heic", ".avif", ".jxl", ".png", ".webp"])
def test_the_runtime_reads_what_phones_and_cameras_write(tmp_path, ext):
    """HEIC from iPhones, AVIF, JPEG XL and the rest, as OpenCV's BGR array (red stays red), through strata-python's reader."""
    from PIL import Image
    from strata_python import images

    images.extensions()  # teaches Pillow to write HEIC and JPEG XL too, for this test's picture
    p = str(tmp_path / f"p{ext}")
    Image.new("RGB", (64, 48), (220, 20, 20)).save(p)
    img = faces._read(p)
    assert img.shape == (48, 64, 3)
    b, _, r = (int(v) for v in img[24, 32])
    assert r > 180 > b


def test_the_runtime_reads_a_camera_raw_file_and_turns_photos_by_exif(tmp_path):
    from PIL import Image
    from strata_python.testing import tiny_dng

    assert faces._read(tiny_dng(tmp_path / "a.dng", (64, 48), orientation=6)).shape == (64, 48, 3)  # held on its side
    p = str(tmp_path / "turned.jpg")
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (80, 40)).save(p, exif=exif.tobytes())
    assert faces._read(p).shape == (80, 40, 3)  # stored 80 x 40, shown 40 wide


def test_a_file_that_is_not_a_picture_is_a_runerror(tmp_path):
    p = tmp_path / "x.heic"
    p.write_bytes(b"not a picture")
    with pytest.raises(runtimes.RunError, match="not a picture this service can read"):
        faces._read(str(p))
    with pytest.raises(runtimes.RunError, match="no such image"):
        faces._read(str(tmp_path / "gone.jpg"))
