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


def test_a_label_names_who_a_track_is_over_time():
    label = [("p1", 0.0), ("p2", 14.2)]
    assert B.name_at(label, 14.0) == "p1" and B.name_at(label, 14.2) == "p2"
    assert B.name_at([("none", 0.0)], 3) is None and B.name_at([], 3) is None


ANA, BEN = [1.0] + [0.0] * 511, [0.0, 1.0] + [0.0] * 510


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

        def tr(i, a, b, x, who):
            return dict(
                id=i,
                shot=0,
                start=a,
                end=b,
                faces=3,
                embedding=who,
                boxes=[[t, x, 10.0, x + 50, 60.0, 0.9] for t in (a, (a + b) / 2, b - 0.1)],
                stills=[dict(bbox=box, image=still, quality=0.5)],
            )

        return dict(
            fps=10,
            width=640,
            height=360,
            duration=4.0,
            tracks=[tr(1, 0.0, 2.0, 10.0, ANA), tr(2, 0.0, 4.0, 300.0, BEN), tr(3, 2.0, 4.0, 10.0, ANA)],
            together=[[1, 2], [2, 3]],
        )


class FakeStore:
    """The face store's two calls the script makes: pictures posted, and pictures listed by app a page at a time (2 a page here)."""

    def __init__(self):
        self.images, self.decided = [], {}

    def _call(self, method, path, body=None):
        if method == "POST":
            assert path == "/api/faces/images"
            self.images.append(body)
            return {}
        assert path.startswith("/api/faces/images?app=face-tracks-eval")
        before = int(path.split("before=")[1]) if "before=" in path else len(self.images)
        page = list(range(before - 1, max(before - 3, -1), -1))
        items = [
            dict(
                id=i,
                ref=self.images[i]["ref"],
                faces=[{"identity_id": None, "status": None, **self.decided.get(self.images[i]["ref"], {})}],
            )
            for i in page
        ]
        return dict(items=items, next=page[-1] if len(page) == 2 and page[-1] > 0 else None)


def _still(t, x, image, **kw):
    return {
        "time": t,
        "bbox": [x, 10.0, x + 50, 60.0],
        "det_score": 0.9,
        "embedding": ANA,
        "image": image,
        "quality": 0.5,
        **kw,
    }


def test_propose_store_then_build(tmp_path):
    client = FakeClient(tmp_path)
    B.propose(str(tmp_path), ["/v/a.mp4@10-14"], client=client)
    assert client.calls[0] == ("shots", {"clip": [10.0, 14.0]})
    params = client.calls[1][1]
    assert params["cuts"] == [2.0] and params["tracker"] == "byte" and params["clip"] == [10.0, 14.0]

    saved = json.loads((tmp_path / "clip-01.json").read_text())
    saved["clip"] = None  # build from time 0 for easy numbers
    img = saved["answer"]["tracks"][0]["stills"][0]["image"]
    t1, t2, t3 = saved["answer"]["tracks"]
    t1["stills"] = [_still(0.5, 10.0, img), _still(1.5, 10.0, img)]
    t2["stills"] = [
        _still(0.5, 300.0, img),
        _still(1.5, 300.0, img),
        _still(3.5, 300.0, img),
        _still(3.6, 300.0, img, det_score=0.2),
    ]
    t3["stills"] = [
        _still(3.0, 10.0, img),
        dict(_still(3.5, 10.0, img), bbox=[10.0, 10.0, 20.0, 20.0]),
    ]  # too small to store
    (tmp_path / "clip-01.json").write_text(json.dumps(saved))

    store = FakeStore()
    assert B.store(str(tmp_path), client=store) == 6
    first = store.images[0]
    assert first["app"] == "face-tracks-eval" and first["ref"] == "clip-01/t1/0.500" and first["model"] == "buffalo_l"
    assert first["tags"] == ["face-tracks-eval", "face-tracks-eval:clip-01"] and (first["width"], first["height"]) == (
        640,
        360,
    )
    assert first["faces"][0]["embedding"] == ANA and first["faces"][0]["found_by"] == "face-tracks"
    yes = lambda p: dict(identity_id=p, status="confirmed")
    store.decided = {
        "clip-01/t1/0.500": yes(7),
        "clip-01/t1/1.500": dict(identity_id=8, status="suggested"),  # only suggested: not a decision
        "clip-01/t2/0.500": yes(8),
        "clip-01/t2/3.500": yes(7),  # the generous setting mixed ana into ben's track: switches at 2.0 s
        "clip-01/t3/3.000": dict(identity_id=None, status="rejected"),  # not a face
    }
    ds = B.build(str(tmp_path), name="ft-test", every=0.5, client=store)
    it = ds.items[0]
    assert it["input"] == {"path": os.path.abspath("/v/a.mp4"), "cuts": [2.0]}
    ref = it["reference"]
    assert ref["times"] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    assert [o[0] for o in ref["people"]["p7"]] == [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
    ]  # track 1, then track 2 from 2.0 s
    assert [o[0] for o in ref["people"]["p8"]] == [0.0, 0.5, 1.0, 1.5]
    assert ref["ignore"] == [] and "2 people" in ds.description  # track 3 is `none`: answered faces there are false


def test_a_track_nobody_decided_about_is_ignored_not_false(tmp_path):
    B.propose(str(tmp_path), ["/v/a.mp4"], client=FakeClient(tmp_path))
    saved = json.loads((tmp_path / "clip-01.json").read_text())
    img = saved["answer"]["tracks"][0]["stills"][0]["image"]
    for tr in saved["answer"]["tracks"]:
        tr["stills"] = [_still(tr["start"] + 0.5, 10.0, img)]
    (tmp_path / "clip-01.json").write_text(json.dumps(saved))
    store = FakeStore()
    B.store(str(tmp_path), client=store)
    store.decided = {"clip-01/t1/0.500": dict(identity_id=7, status="confirmed")}
    ref = B.build(str(tmp_path), name="ft-test", client=store).items[0]["reference"]
    assert list(ref["people"]) == ["p7"] and ref["ignore"] and {o[1] for o in ref["ignore"]} == {300.0, 10.0}


def test_store_leaves_out_skipped_clips(tmp_path):
    B.propose(str(tmp_path), ["/v/a.mp4"], client=FakeClient(tmp_path))
    saved = json.loads((tmp_path / "clip-01.json").read_text())
    saved["answer"]["tracks"][0]["stills"] = [_still(0.5, 10.0, "s.jpg")]
    (tmp_path / "clip-01.json").write_text(json.dumps(saved))
    store = FakeStore()
    assert B.store(str(tmp_path), client=store, skip={"clip-01"}) == 0 and B.store(str(tmp_path), client=store) == 1


def test_build_with_nothing_decided_refuses(tmp_path):
    (tmp_path / "clip-01.json").write_text(
        json.dumps(dict(path="/v", clip=None, cuts=[], answer=dict(fps=5, duration=1, width=1, height=1, tracks=[])))
    )
    with pytest.raises(SystemExit, match="no clip has a face decided"):
        B.build(str(tmp_path), client=FakeStore())
