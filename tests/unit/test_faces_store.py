"""The face store (stratamodels.faces.store) on the real migrated SQLite schema: pictures and faces, re-indexing with another model, tags and
opt-in scopes, search, decisions and their undo, people with per-app fields, merges and their suggestions, clustering. Embeddings are
synthetic: a few directions in 512 dimensions, one per made-up person, with a little noise."""

import numpy as np
import pytest

from stratamodels.faces.store import FaceStore, NotFound
from stratamodels.store import memory_store

DIM = 512


def person(k, noise=0.0, seed=0):
    """An embedding near direction `k` (one made-up person per k)."""
    v = np.zeros(DIM)
    v[k] = 1.0
    if noise:
        v = v + np.random.default_rng(seed).normal(0, noise, DIM)
    return [float(x) for x in v / np.linalg.norm(v)]


def face(k, x=0, seed=0, score=0.9):
    return dict(
        bbox=[x, 10, x + 40, 60], det_score=score, kps=[[1, 2]], pose=[0, 0, 0], embedding=person(k, 0.01, seed)
    )


@pytest.fixture
def fs(clock):
    return FaceStore(memory_store(clock).engine, clock=clock)


def index(fs, app, ref, faces, tags=(), model="buffalo_l"):
    return fs.save_image(
        app=app,
        ref=ref,
        faces=faces,
        model=model,
        settings_hash="h1",
        path=f"/pics/{ref}.jpg",
        tags=tags,
        width=800,
        height=600,
    )


def test_a_picture_and_its_faces(fs):
    img = index(fs, "bright-horizons", "p1", [face(1), face(2, x=100)], tags=["nursery"])
    assert (img["app"], img["tags"], img["width"], len(img["faces"])) == (
        "bright-horizons",
        ["bright-horizons", "nursery"],
        800,
        2,
    )
    assert img["faces"][0]["box"] == [0.0, 10.0, 40.0, 60.0]
    assert img["faces"][0]["identity_id"] is None
    assert fs.find_image("bright-horizons", "p1")["id"] == img["id"]
    assert fs.find_image("bright-horizons", "nope") is None
    assert fs.image(img["id"])["faces"][1]["kps"] == [[1, 2]]
    assert fs.is_current("bright-horizons", "p1", "buffalo_l", "h1")
    assert not fs.is_current("bright-horizons", "p1", "buffalo_l", "h2")
    with pytest.raises(NotFound, match="no such picture"):
        fs.image(999)
    with pytest.raises(ValueError, match="app and ref"):
        fs.save_image(app="", ref="x", faces=[], model="m")


def test_reindexing_with_another_model_keeps_faces_people_decided_about(fs):
    img = index(fs, "a", "p1", [face(1), face(2, x=100)])
    f1, f2 = (f["id"] for f in img["faces"])
    pid = fs.create_identity(name="One")["id"]
    fs.assign([f1], pid, "a")
    again = fs.save_image(
        app="a", ref="p1", faces=[dict(face(1), bbox=[2, 11, 41, 61])], model="better", settings_hash="z"
    )
    assert [f["id"] for f in again["faces"]] == [
        f1
    ]  # the same face, found again (IoU >= 0.5): the decision stands; f2 was not found and nobody decided
    assert again["faces"][0]["identity_id"] == pid
    assert fs.embedding(f1, "better") == pytest.approx(person(1, 0.01, 0))
    assert fs.embedding(f1, "buffalo_l") == pytest.approx(
        person(1, 0.01, 0)
    )  # the old model's embedding is kept beside it
    with pytest.raises(NotFound):
        fs.face(f2)
    with pytest.raises(ValueError, match="no unknown embedding"):
        fs.embedding(f1, "unknown")


def test_search_by_tag_and_opt_in(fs):
    index(fs, "bright-horizons", "n1", [face(1, seed=1)], tags=["nursery"])
    index(fs, "strata-stills", "s1", [face(1, seed=2)])
    index(fs, "strata-stills", "s2", [face(5, seed=3)])
    q = person(1)
    assert [h["app"] for h in fs.search(q, "buffalo_l", k=2)] in (
        ["bright-horizons", "strata-stills"],
        ["strata-stills", "bright-horizons"],
    )
    fs.set_tag("nursery", True, note="other people's children")
    assert [h["ref"] for h in fs.search(q, "buffalo_l", k=2)] == ["s1", "s2"]  # opt-in: left out by default
    assert {h["ref"] for h in fs.search(q, "buffalo_l", k=2, include=["nursery"])} == {"n1", "s1"}
    assert [h["ref"] for h in fs.search(q, "buffalo_l", k=5, tags=["nursery"])] == [
        "n1"
    ]  # naming it in tags asks for it
    assert [h["ref"] for h in fs.search(q, "buffalo_l", k=5, tags=["strata-stills"], max_distance=0.5)] == ["s1"]
    assert fs.search(q, "other-model") == []
    tags = {t["tag"]: t for t in fs.tags()}
    assert (tags["nursery"]["opt_in"], tags["nursery"]["images"], tags["strata-stills"]["images"]) == (True, 1, 2)
    with pytest.raises(ValueError, match="needs a name"):
        fs.set_tag(" ", True)


def test_search_named_or_not(fs):
    img = index(fs, "a", "p1", [face(1, seed=1), face(1, x=100, seed=2), face(1, x=200, seed=3)])
    ids = [f["id"] for f in img["faces"]]
    pid = fs.create_identity(name="One")["id"]
    fs.assign(ids[:1], pid, "a")
    hits = fs.search(person(1), "buffalo_l", k=5, named=True)
    assert [(h["face"]["id"], h["face"]["identity_id"]) for h in hits] == [(ids[0], pid)]
    assert sorted(h["face"]["id"] for h in fs.search(person(1), "buffalo_l", k=5, named=False)) == ids[1:]
    assert ids[0] not in [h["face"]["id"] for h in fs.search(person(1), "buffalo_l", k=5, exclude=[ids[0]])]


def test_decisions_are_append_only_and_undone_by_batch(fs):
    ids = [f["id"] for f in index(fs, "a", "p1", [face(1), face(2, x=100)])["faces"]]
    one, two = fs.create_identity(name="One")["id"], fs.create_identity(name="Two")["id"]
    b1 = fs.assign(ids, one, "a")
    b2 = fs.assign(ids[1:], two, "a", source="auto", confidence=0.97)
    assert [f["identity_id"] for f in fs.image(1)["faces"]] == [one, two]
    fs.undo(b2)
    assert [f["identity_id"] for f in fs.image(1)["faces"]] == [one, one]
    assert fs.undo(b2)["undone_at"] is not None  # twice is harmless
    fs.assign(ids[:1], None, "a", status="rejected", note="not a face")
    assert fs.face(ids[0])["status"] == "rejected"
    assert fs.identity(one)["faces"] == 1
    assert [f["id"] for f in fs.faces_of(one)] == [ids[1]]
    assert [f["id"] for f in fs.faces_of(one, status="rejected")] == []
    with pytest.raises(ValueError, match="status is one of"):
        fs.assign(ids, one, "a", status="maybe")
    with pytest.raises(ValueError, match="only a rejection"):
        fs.assign(ids, None, "a")
    with pytest.raises(NotFound, match="no such face"):
        fs.assign([999], one, "a")
    with pytest.raises(NotFound, match="no such person"):
        fs.assign(ids, 999, "a")
    with pytest.raises(NotFound, match="no such batch"):
        fs.undo(999)
    with pytest.raises(ValueError, match="no faces"):
        fs.assign([], one, "a")
    assert b1 != b2


def test_one_person_with_each_apps_own_fields(fs):
    p = fs.create_identity(
        name="Ada",
        fields={"born": 2022},
        app="bright-horizons",
        app_fields={"name": "Ada (2-3s)", "kind": "child"},
        app_ref="c7",
    )
    assert (p["name"], p["fields"]["kind"]) == (
        "Ada (2-3s)",
        "child",
    )  # created by bright-horizons: it sees its own fields
    plain = fs.identity(p["id"])
    assert (plain["name"], plain["fields"], plain["global_fields"]) == (
        "Ada",
        {"born": 2022, "name": "Ada"},
        {"born": 2022},
    )
    mine = fs.set_app_fields(p["id"], "strata-360", {"role": "runner"}, app_ref="friend-1")
    assert (mine["name"], mine["fields"]["role"], sorted(mine["apps"])) == (
        "Ada",
        "runner",
        ["bright-horizons", "strata-360"],
    )
    assert fs.identity(p["id"], "bright-horizons")["name"] == "Ada (2-3s)"
    upd = fs.update_identity(p["id"], name="Ada L", fields={"born": None, "eyes": "brown"})
    assert (upd["name"], upd["global_fields"]) == ("Ada L", {"eyes": "brown"})
    assert [i["id"] for i in fs.identities(app="bright-horizons", app_ref="c7")] == [p["id"]]
    assert fs.identities(app="bright-horizons", app_ref="nope") == []
    assert [i["id"] for i in fs.identities(q="2-3s", app="bright-horizons")] == [p["id"]]
    assert [i["id"] for i in fs.identities(q="friend-1")] == [p["id"]]  # an app's key is searchable too
    other = fs.create_identity(name="B")["id"]
    with pytest.raises(ValueError, match="already calls person"):
        fs.set_app_fields(other, "bright-horizons", {}, app_ref="c7")
    with pytest.raises(NotFound):
        fs.identity(999)
    with pytest.raises(NotFound):
        fs.update_identity(999, name="x")
    with pytest.raises(NotFound):
        fs.set_app_fields(999, "a", {})


def test_merge_reads_faces_through_and_unmerges(fs):
    ids = [f["id"] for f in index(fs, "a", "p1", [face(1), face(1, x=100, seed=1)])["faces"]]
    x = fs.create_identity(app="bright-horizons", app_fields={"name": "Ada"}, app_ref="c7")["id"]
    y = fs.create_identity(app="strata-stills", app_fields={"name": "A."})["id"]
    fs.assign(ids[:1], x, "bright-horizons")
    fs.assign(ids[1:], y, "strata-stills")
    merged = fs.merge(x, y, "strata-models")
    assert (merged["id"], merged["faces"], merged["merged"]) == (y, 2, [x])
    assert [f["identity_id"] for f in fs.image(1)["faces"]] == [y, y]
    assert fs.identity(x)["id"] == y  # asking for the merged one answers the person it lives in
    assert [i["id"] for i in fs.identities()] == [y]
    # the merged person's app fields and id are read through the survivor, and stay on the merged row for an unmerge
    assert fs.identity(y, "bright-horizons")["fields"]["name"] == "Ada"
    assert fs.identity(y)["apps"]["bright-horizons"]["app_ref"] == "c7"
    assert [i["id"] for i in fs.identities("bright-horizons", app_ref="c7")] == [y]
    fs.set_app_fields(
        y, "bright-horizons", {"room": "Robins"}, app_ref="c7"
    )  # the same app id, already the survivor's through x
    assert fs.identity(y, "bright-horizons")["fields"] == {"name": "Ada", "room": "Robins"}
    b = fs.assign(ids[:1], x, "a")  # deciding for a merged person decides for the one it lives in
    assert fs.face(ids[0])["identity_id"] == y
    with pytest.raises(ValueError, match="a merge"):
        fs.undo(b - 1)  # the batch before this assign is the merge's
    with pytest.raises(ValueError, match="already the same"):
        fs.merge(x, y, "a")
    fs.unmerge(x)
    assert sorted(i["id"] for i in fs.identities()) == [x, y]
    assert [i["id"] for i in fs.identities("bright-horizons", app_ref="c7")] == [x]
    with pytest.raises(NotFound):
        fs.merge(x, 999, "a")
    with pytest.raises(NotFound):
        fs.unmerge(999)


def test_merge_suggestions_rank_close_people_and_respect_not_same(fs):
    faces = [face(1, seed=1), face(1, x=100, seed=2), face(1, x=200, seed=3), face(9, x=300, seed=4)]
    ids = [f["id"] for f in index(fs, "a", "p1", faces)["faces"]]
    a, b, c = (
        fs.create_identity(name=n, app=app, app_fields={}, app_ref=n)["id"]
        for n, app in (("a", "bright-horizons"), ("b", "strata-360"), ("c", "x"))
    )
    fs.assign(ids[:2], a, "bright-horizons")
    fs.assign(ids[2:3], b, "strata-360")
    fs.assign(ids[3:], c, "x")
    s = fs.merge_suggestions("buffalo_l")
    assert [(p["a"], p["b"]) for p in s] == [(a, b)]  # c is far from both
    assert (s[0]["apps_a"], s[0]["apps_b"]) == (["bright-horizons"], ["strata-360"])
    assert s[0]["distance"] < 0.1
    fs.not_same(b, a)
    assert fs.merge_suggestions("buffalo_l") == []
    assert fs.merge_suggestions("other") == []
    with pytest.raises(ValueError, match="same as themself"):
        fs.not_same(a, a)
    with pytest.raises(NotFound):
        fs.not_same(a, 999)


def test_cluster_groups_undecided_faces(fs):
    faces = (
        [face(1, x=i * 50, seed=i) for i in range(4)]
        + [face(2, x=300 + i * 50, seed=10 + i) for i in range(3)]
        + [face(7, x=600, seed=20)]
    )
    ids = [f["id"] for f in index(fs, "a", "p1", faces, tags=["nursery"])["faces"]]
    groups = fs.cluster("buffalo_l")
    assert [g["faces"] for g in groups] == [ids[:4], ids[4:7]]  # the lone face is no group
    fs.set_tag("nursery", True)
    assert fs.cluster("buffalo_l") == []
    assert len(fs.cluster("buffalo_l", include=["nursery"])) == 2
    pid = fs.create_identity(name="One")["id"]
    b = fs.assign(ids[:3], pid, "a")
    assert [g["faces"] for g in fs.cluster("buffalo_l", tags=["nursery"])] == [
        ids[4:7]
    ]  # only one of person 1's faces is left
    fs.undo(b)
    assert [g["size"] for g in fs.cluster("buffalo_l", tags=["nursery"], min_size=4)] == [4]  # undone: free again
    assert fs.cluster("buffalo_l", tags=["nursery"], min_score=0.95) == []
    assert [g["size"] for g in fs.cluster("buffalo_l", tags=["nursery"], threshold=1.5, min_size=8)] == [8]


def test_a_text_search_looks_past_the_limit(fs):
    for n in range(5):
        fs.create_identity(name=f"Person {n}")
    fs.create_identity(name="Grace")
    assert [i["name"] for i in fs.identities(q="grace", limit=2)] == ["Grace"]
    assert len(fs.identities(q="person", limit=2)) == 2
    assert len(fs.identities(limit=2)) == 2
