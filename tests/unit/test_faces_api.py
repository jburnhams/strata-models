"""The face store's HTTP API (stratamodels.faces.api) over the fake scheduler: indexing a real picture through a `faces` job (the fake model
answers made-up faces), the kept medium copy and crops cut from it, search by face, embedding or picture, clustering, people, decisions and
their undo, merges, tags and the error codes. The store itself has its own tests (test_faces_store.py)."""

import io
import os
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from stratamodels import gpu
from stratamodels.registry import Registry
from stratamodels.runtimes import fake
from stratamodels.server.app import create_app

DIM = 512


def person(k, noise=0.01, seed=0):
    v = np.zeros(DIM)
    v[k] = 1.0
    v = v + np.random.default_rng(seed).normal(0, noise, DIM)
    return [float(x) for x in v / np.linalg.norm(v)]


@pytest.fixture
def registry():
    """The shared fake registry plus a `faces` role on the fake runtime."""
    return Registry.from_dict(
        dict(
            roles=dict(echo=dict(model="small"), faces=dict(model="fakeface")),
            models=dict(
                small=dict(runtime="fake", roles=["echo"], vram_gb=2),
                fakeface=dict(runtime="fake", roles=["faces"], vram_gb=1),
            ),
        )
    )


@pytest.fixture
def client(scheduler, monkeypatch):
    """The fake model answers a faces job with two faces: person 1 on the left, person 2 on the right (the seed is the picture's name length,
    so pictures differ a little)."""
    real = fake.run

    def run(handle, params):
        if "embedding" not in (params.get("fields") or []):
            return real(handle, params)
        if "nobody" in params["path"]:
            return dict(faces=[], width=200, height=100)
        n = len(params["path"])
        return dict(
            width=200,
            height=100,
            faces=[
                dict(bbox=[10, 10, 60, 70], det_score=0.9, kps=[[30, 30]], pose=[0, 0, 0], embedding=person(1, seed=n)),
                dict(
                    bbox=[120, 20, 180, 90], det_score=0.8, kps=[[150, 40]], pose=[0, 0, 0], embedding=person(2, seed=n)
                ),
            ],
        )

    monkeypatch.setattr(fake, "run", run)
    h = gpu.History(scheduler.probe, every_s=5)
    return TestClient(create_app(scheduler, h))


def worker(scheduler, n=1):
    """Step the scheduler on a thread until `n` jobs ran, so a request that waits for its job sees it finish (as in test_server.py)."""

    def go():
        end, done = time.monotonic() + 30, 0
        while done < n and time.monotonic() < end:
            time.sleep(0.02)
            if scheduler.step() is not None:
                done += 1

    t = threading.Thread(target=go)
    t.start()
    return t


def picture(tmp_path, name="p.jpg", w=200, h=100):
    p = tmp_path / name
    Image.new("RGB", (w, h), (120, 80, 40)).save(p)
    return str(p)


def size_of(r):
    return Image.open(io.BytesIO(r.content)).size


def index(client, scheduler, path, ref, app="bright-horizons", **kw):
    t = worker(scheduler)
    r = client.post("/api/faces/images", json=dict(app=app, ref=ref, path=path, **kw))
    t.join()
    assert r.status_code == 200, r.text
    return r.json()


def test_index_a_picture_keep_its_copy_and_cut_crops(client, scheduler, tmp_path):
    p = picture(tmp_path)
    img = index(client, scheduler, p, "r1", tags=["nursery"])
    assert (img["model"], img["width"], len(img["faces"]), img["tags"]) == (
        "fakeface",
        200,
        2,
        ["bright-horizons", "nursery"],
    )
    assert img["medium"].endswith(".webp")
    again = client.post(
        "/api/faces/images", json=dict(app="bright-horizons", ref="r1", path=p)
    ).json()  # no job: nothing steps the scheduler
    assert again["id"] == img["id"]
    assert client.get(f"/api/faces/images/{img['id']}").json()["faces"][1]["box"] == [120.0, 20.0, 180.0, 90.0]
    assert (
        client.get("/api/faces/images", params=dict(app="bright-horizons", ref="r1")).json()["items"][0]["id"]
        == img["id"]
    )
    assert [i["id"] for i in client.get("/api/faces/images", params=dict(include="nursery")).json()["items"]] == [
        img["id"]
    ]
    r = client.get(f"/api/faces/images/{img['id']}/picture")
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    f1 = img["faces"][0]["id"]
    assert client.get(f"/api/faces/{f1}").json()["id"] == f1
    crop = client.get(f"/api/faces/{f1}/crop", params=dict(px=64))
    assert crop.status_code == 200 and max(size_of(crop)) == 64  # square, less what falls off the picture's edge
    os.remove(p)  # the original went away: the crop comes from the kept copy instead
    crop = client.get(f"/api/faces/{img['faces'][1]['id']}/crop", params=dict(px=48, pad=9))
    assert crop.status_code == 200 and max(size_of(crop)) == 48


def test_faces_made_elsewhere_and_bad_requests(client, tmp_path):
    made = dict(bbox=[1, 2, 30, 40], det_score=0.7, embedding=person(3))
    r = client.post(
        "/api/faces/images",
        json=dict(app="strata-360", ref="x", faces=[made], model="buffalo_l", path=str(tmp_path / "gone.jpg")),
    )
    assert r.status_code == 200 and r.json()["faces"][0]["found_by"] == "import" and r.json()["medium"] is None
    big = client.post(
        "/api/faces/images",
        json=dict(
            app="strata-360", ref="big", faces=[made], model="buffalo_l", path=picture(tmp_path, "big.jpg", 2400, 1200)
        ),
    ).json()
    assert (big["width"], big["height"]) == (2400, 1200) and big[
        "medium"
    ]  # the original's size, so a crop from the 1600 px copy scales its box
    given = client.post(
        "/api/faces/images",
        json=dict(app="strata-360", ref="w", faces=[made], model="buffalo_l", width=640, height=480),
    ).json()
    assert (given["width"], given["height"]) == (640, 480)
    assert client.get(f"/api/faces/images/{r.json()['id']}/picture").status_code == 404
    assert (
        client.get(f"/api/faces/{r.json()['faces'][0]['id']}/crop").status_code == 404
    )  # neither the picture nor a copy
    assert client.post("/api/faces/images", json=dict(app="a", ref="y", faces=[made])).status_code == 400  # no model
    assert (
        client.post("/api/faces/images", json=dict(app="a", ref="y", path=str(tmp_path / "none.jpg"))).status_code
        == 400
    )
    assert client.get("/api/faces/images/999").status_code == 404
    assert client.get("/api/faces/999").status_code == 404
    assert client.post("/api/faces/search", json=dict(app="a")).status_code == 400
    assert client.post("/api/faces/search", json=dict(app="a", path=str(tmp_path / "none.jpg"))).status_code == 400


def test_a_failed_job_is_a_422(client, scheduler, tmp_path, monkeypatch):
    def broken(handle, params):
        raise fake.RunError("cannot read that picture")

    monkeypatch.setattr(fake, "run", broken)
    t = worker(scheduler)
    r = client.post("/api/faces/images", json=dict(app="a", ref="r", path=picture(tmp_path)))
    t.join()
    assert r.status_code == 422 and "cannot read" in r.json()["detail"]


def test_search_cluster_people_decisions_and_merges(client, scheduler, tmp_path):
    imgs = [index(client, scheduler, picture(tmp_path, "p" * i + ".jpg"), f"r{i}") for i in range(1, 4)]
    lefts = [i["faces"][0]["id"] for i in imgs]
    hits = client.post("/api/faces/search", json=dict(app="a", face_id=lefts[0], k=5)).json()
    assert hits["model"] == "fakeface" and {h["face"]["id"] for h in hits["hits"][:2]} == set(lefts[1:])
    by_vec = client.post("/api/faces/search", json=dict(app="a", embedding=person(2), k=3)).json()["hits"]
    assert {h["face"]["id"] for h in by_vec} == {i["faces"][1]["id"] for i in imgs}
    t = worker(scheduler)
    by_pic = client.post("/api/faces/search", json=dict(app="a", path=picture(tmp_path, "q.jpg"), k=1)).json()["hits"]
    t.join()
    assert by_pic[0]["face"]["id"] in {
        i["faces"][1]["id"] for i in imgs
    }  # the biggest face in the picture is person 2's
    t = worker(scheduler)
    r = client.post("/api/faces/search", json=dict(app="a", path=picture(tmp_path, "nobody.jpg")))
    t.join()
    assert r.status_code == 400 and "no face" in r.json()["detail"]

    groups = client.post("/api/faces/cluster", json=dict(app="a")).json()["clusters"]
    assert sorted(g["size"] for g in groups) == [3, 3]

    ada = client.post(
        "/api/identities", json=dict(name="Ada", app="bright-horizons", app_ref="c1", app_fields=dict(room="Robins"))
    )
    assert ada.status_code == 201
    a = ada.json()["id"]
    assert client.get(f"/api/identities/{a}", params=dict(app="bright-horizons")).json()["fields"]["room"] == "Robins"
    assert client.put(f"/api/identities/{a}/apps/strata-360", json=dict(fields=dict(wearer=True))).status_code == 200
    assert client.patch(f"/api/identities/{a}", json=dict(fields=dict(born="2020"))).json()["fields"]["born"] == "2020"
    assert [i["id"] for i in client.get("/api/identities", params=dict(q="ad")).json()["items"]] == [a]
    batch = client.post("/api/faces/assign", json=dict(face_ids=lefts, identity_id=a, app="bright-horizons")).json()[
        "batch"
    ]
    assert len(client.get(f"/api/identities/{a}/faces").json()["items"]) == 3
    assert (
        client.post("/api/faces/assign", json=dict(face_ids=lefts, identity_id=a, app="x", status="maybe")).status_code
        == 400
    )
    client.post(f"/api/faces/batches/{batch}/undo")
    assert client.get(f"/api/identities/{a}/faces").json()["items"] == []
    client.post("/api/faces/assign", json=dict(face_ids=lefts, identity_id=a, app="bright-horizons"))

    b = client.post("/api/identities", json=dict(name="Ada L")).json()["id"]
    client.post(
        "/api/faces/assign", json=dict(face_ids=[imgs[0]["faces"][0]["id"]], identity_id=b, app="strata-stills")
    )
    client.post("/api/faces/assign", json=dict(face_ids=lefts[1:], identity_id=a, app="bright-horizons"))
    sugg = client.get("/api/identities/merge-suggestions").json()
    assert sugg["model"] == "fakeface"
    assert client.post("/api/identities/not-same", json=dict(a=a, b=a)).status_code == 400
    m = client.post(f"/api/identities/{b}/merge", json=dict(into=a, app="strata-models"))
    assert m.status_code == 200 and m.json()["merged"] == [b]
    assert client.post(f"/api/identities/{b}/merge", json=dict(into=a, app="strata-models")).status_code == 400
    assert client.post(f"/api/identities/{b}/unmerge").status_code == 200
    assert client.post("/api/identities/not-same", json=dict(a=a, b=b)).status_code == 200
    assert client.get("/api/identities/merge-suggestions").json()["items"] == []
    assert client.get("/api/identities/999").status_code == 404


def test_tags(client):
    assert client.put("/api/faces/tags/nursery", json=dict(opt_in=True, note="children")).json()["opt_in"] is True
    assert [t["tag"] for t in client.get("/api/faces/tags").json()["items"]] == ["nursery"]
    assert client.put("/api/faces/tags/%20", json=dict(opt_in=True)).status_code == 400


def test_any_picture_format_and_a_photo_taken_on_its_side(client, tmp_path):
    """A HEIC (an iPhone's) and a camera raw file get a kept copy and crops; a JPEG stored on its side reports its upright size, the
    pixels its faces' boxes are in."""
    from strata_python.testing import tiny_dng, tiny_picture

    made = dict(bbox=[10, 10, 40, 40], det_score=0.9, embedding=person(4))
    heic = tiny_picture(
        tmp_path / "phone.heic", (300, 200), (90, 60, 30)
    )  # in a child: x265 lowers the encoding process's priority (macOS)
    raw = tiny_dng(tmp_path / "camera.dng", (120, 80))
    turned = tmp_path / "turned.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (300, 200)).save(turned, exif=exif.tobytes())
    for ref, path, size in (("h", str(heic), (300, 200)), ("r", raw, (120, 80)), ("t", str(turned), (200, 300))):
        img = client.post(
            "/api/faces/images", json=dict(app="strata-stills", ref=ref, path=path, faces=[made], model="buffalo_l")
        ).json()
        assert (img["width"], img["height"]) == size
        assert img["medium"].endswith(".webp")
        crop = client.get(f"/api/faces/{img['faces'][0]['id']}/crop", params=dict(px=32))
        assert crop.status_code == 200
        assert max(size_of(crop)) == 32


def test_hints_and_propose_over_http(client, scheduler, tmp_path):
    imgs = [index(client, scheduler, picture(tmp_path, f"s{i}.jpg"), f"s{i}") for i in range(10)]
    hints = [dict(image_id=i["id"], app_ref="ann", name="Ann", face_id=i["faces"][0]["id"]) for i in imgs]
    r = client.post("/api/faces/hints", json=dict(app="takeout", hints=hints, note="names"))
    assert r.status_code == 201 and r.json()["added"] == 10
    assert len(client.get("/api/faces/hints", params=dict(app="takeout", app_ref="ann")).json()["items"]) == 10
    assert (
        client.get("/api/faces/hints", params=dict(app="takeout", image_id=imgs[0]["id"])).json()["items"][0]["name"]
        == "Ann"
    )
    assert (
        client.post("/api/faces/hints", json=dict(app="takeout", hints=[dict(image_id=imgs[0]["id"])])).status_code
        == 400
    )
    dry = client.post("/api/faces/propose", json=dict(app="takeout")).json()
    assert (dry["model"], dry["dry_run"], dry["qualifying"], dry["suggested"]) == ("fakeface", True, 1, 10)
    assert client.post("/api/faces/propose", json=dict(app="takeout", min_seeds=11)).json()["qualifying"] == 0
    assert (
        client.post("/api/faces/propose", json=dict(app="takeout", bogus_param=1, max_distance="x")).status_code == 422
    )
    done = client.post("/api/faces/propose", json=dict(app="takeout", dry_run=False, groups=True, limit=100)).json()
    assert done["created"] == 1 and done["batch"] is not None
    (ann,) = client.get("/api/identities", params=dict(app="takeout", app_ref="ann")).json()["items"]
    assert len(client.get(f"/api/identities/{ann['id']}/faces", params=dict(status="suggested")).json()["items"]) == 10
    assert client.post(f"/api/faces/batches/{done['batch']}/undo").status_code == 200


def test_suggested_faces_paged_counted_and_left_out_of_groups_over_http(client, scheduler, tmp_path):
    imgs = [index(client, scheduler, picture(tmp_path, f"q{i}.jpg"), f"q{i}") for i in range(3)]
    ids = [i["faces"][0]["id"] for i in imgs]
    who = client.post("/api/identities", json=dict(name="Ann")).json()
    assert (
        client.post(
            "/api/faces/assign", json=dict(face_ids=ids, identity_id=who["id"], app="t", status="suggested")
        ).status_code
        == 200
    )
    assert client.get(f"/api/identities/{who['id']}").json()["suggested"] == 3
    page = client.get(f"/api/identities/{who['id']}/faces", params=dict(status="suggested", limit=2, offset=2)).json()[
        "items"
    ]
    assert len(page) == 1
    with_sug = client.post("/api/faces/cluster", json=dict(app="strata-models")).json()["clusters"]
    without = client.post("/api/faces/cluster", json=dict(app="strata-models", suggested=False)).json()["clusters"]
    assert set(ids) <= {f for g in with_sug for f in g["faces"]}
    assert not set(ids) & {f for g in without for f in g["faces"]}


def test_review_round_over_http(client, scheduler, tmp_path):
    imgs = [index(client, scheduler, picture(tmp_path, f"r{i}.jpg"), f"r{i}") for i in range(4)]
    who = client.post("/api/identities", json=dict(name="Ann")).json()
    ids = [i["faces"][0]["id"] for i in imgs]
    client.post("/api/faces/assign", json=dict(face_ids=ids, identity_id=who["id"], app="t", status="suggested"))
    r = client.get(f"/api/identities/{who['id']}/review", params=dict(limit=2)).json()
    assert r["basis"] == "group" and len(r["items"]) == 2 and "distance" in r["items"][0]
    assert client.get("/api/identities/999/review").status_code == 404
