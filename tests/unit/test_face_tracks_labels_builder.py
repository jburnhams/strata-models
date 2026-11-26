"""scripts/evals/face_tracks_labels.py: proposing tracks to label (a fake client, no service or video) and building the dataset from labels."""

import importlib.util
import json
import os

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "face_tracks_labels",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "evals", "face_tracks_labels.py"),
)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def test_clip_arguments():
    V = os.path.abspath("/v/a.mp4")  # a drive letter and backslashes on Windows
    assert B.parse_clip("/v/a.mp4") == (V, None)
    assert B.parse_clip("/v/a.mp4@60-150.5") == (V, [60.0, 150.5])


def test_labels_name_people_and_switches():
    assert B.parse_label("") == [] and B.parse_label("ana") == [("ana", 0.0)]
    label = B.parse_label("ana, ben@14.2")
    assert label == [("ana", 0.0), ("ben", 14.2)]
    assert B.name_at(label, 14.0) == "ana" and B.name_at(label, 14.2) == "ben"
    assert B.name_at(B.parse_label("none"), 3) is None
    with pytest.raises(SystemExit, match="first name"):
        B.parse_label("ana@3")
    with pytest.raises(SystemExit, match="not name or name@seconds"):
        B.parse_label("ana, ben@soon")


class FakeClient:
    def __init__(self, tmp):
        self.tmp, self.calls = tmp, []

    def shots(self, path, app, **params):
        self.calls.append(("shots", params))
        return dict(cuts=[{"time": 2.0}])

    def face_tracks(self, path, app, **params):
        self.calls.append(("face_tracks", params))
        os.makedirs(params["out_dir"], exist_ok=True)
        still = os.path.join(params["out_dir"], "s.jpg")
        Image.new("RGB", (320, 180), "gray").save(still)
        box = [10.0, 10.0, 60.0, 60.0]
        tr = lambda i, a, b, x: dict(
            id=i,
            shot=0,
            start=a,
            end=b,
            faces=3,
            boxes=[[t, x, 10.0, x + 50, 60.0, 0.9] for t in (a, (a + b) / 2, b - 0.1)],
            stills=[dict(bbox=box, image=still)],
        )
        return dict(
            fps=10,
            width=640,
            height=360,
            duration=4.0,
            tracks=[tr(1, 0.0, 2.0, 10.0), tr(2, 0.0, 4.0, 300.0), tr(3, 2.0, 4.0, 10.0)],
        )


def test_propose_then_build(tmp_path):
    client = FakeClient(tmp_path)
    B.propose(str(tmp_path), ["/v/a.mp4@10-14"], client=client)
    assert client.calls[0] == ("shots", {"clip": [10.0, 14.0]})
    params = client.calls[1][1]
    assert params["cuts"] == [2.0] and params["tracker"] == "byte" and params["clip"] == [10.0, 14.0]
    labels = (tmp_path / "labels.toml").read_text()
    assert "[clip-01.tracks]" in labels and '1 = ""' in labels and os.path.isfile(tmp_path / "clip-01.png")
    B.propose(str(tmp_path), ["/v/a.mp4"], client=client)
    assert (tmp_path / "labels.new.toml").exists()  # the first labels are not overwritten

    saved = json.loads((tmp_path / "clip-01.json").read_text())
    saved["clip"] = None  # build from time 0 for easy numbers
    (tmp_path / "clip-01.json").write_text(json.dumps(saved))
    (tmp_path / "labels.toml").write_text(
        '[clip-01]\nextra = [["cat", 3.4, 500, 10, 550, 60]]\n[clip-01.tracks]\n1 = "ana"\n2 = "ben, none@1.0"\n3 = "ana"\n'
    )
    ds = B.build(str(tmp_path), name="ft-test", every=0.5)
    it = ds.items[0]
    assert it["input"] == {"path": os.path.abspath("/v/a.mp4"), "cuts": [2.0]}
    ref = it["reference"]
    assert ref["times"] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    assert [o[0] for o in ref["people"]["ana"]] == ref["times"]  # track 1 to 1.9 and half a frame, then track 3
    assert [o[0] for o in ref["people"]["ben"]] == [0.0, 0.5]
    assert ref["people"]["cat"] == [[3.5, 500, 10, 550, 60]]
    assert "3 people" in ds.description


def test_build_with_nothing_labelled_refuses(tmp_path):
    (tmp_path / "labels.toml").write_text("[clip-01]\n")
    (tmp_path / "clip-01.json").write_text(
        json.dumps(dict(path="/v", clip=None, cuts=[], answer=dict(fps=5, duration=1, tracks=[])))
    )
    with pytest.raises(SystemExit, match="no labelled clips"):
        B.build(str(tmp_path))
