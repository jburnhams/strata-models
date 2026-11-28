"""The face store (stratamodels.faces.store) on the real migrated SQLite schema: pictures and faces, re-indexing with another model, tags and
opt-in scopes, search, decisions and their undo, people with per-app fields, merges and their suggestions, clustering. Embeddings are
synthetic: a few directions in 512 dimensions, one per made-up person, with a little noise."""

import numpy as np
import pytest

from stratamodels.faces.store import FaceStore, NotFound, group_apart
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


def test_an_opt_in_tag_can_share_its_named_faces(fs):
    tape = index(fs, "strata-reels", "t1", [face(1, seed=1), face(1, x=100, seed=4)], tags=["strata-reels"])
    index(fs, "strata-stills", "s1", [face(5, seed=3)])
    named, unnamed = (f["id"] for f in tape["faces"])
    fs.set_tag("strata-reels", True)
    pid = fs.create_identity(name="Ann")["id"]
    fs.assign([named], pid, "strata-reels")
    q = person(1)
    assert [h["ref"] for h in fs.search(q, "buffalo_l", k=3)] == ["s1"]  # opt-in, not shared: none of it
    assert fs.set_tag("strata-reels", True, share_named=True) == dict(tag="strata-reels", opt_in=True, share_named=True)
    # shared: the named face counts in every search, the unnamed one never does
    assert [h["ref"] for h in fs.search(q, "buffalo_l", k=3)] == ["t1", "s1"]
    assert [h["face"]["id"] for h in fs.search(q, "buffalo_l", k=3, named=True)] == [named]
    assert [h["ref"] for h in fs.search(q, "buffalo_l", k=3, named=False)] == ["s1"]
    # asking for the tag still gives all of it; leaving share_named out keeps it as it was
    assert {h["face"]["id"] for h in fs.search(q, "buffalo_l", k=3, include=["strata-reels"])} >= {named, unnamed}
    fs.set_tag("strata-reels", True, note="family tapes")
    assert {t["tag"]: t for t in fs.tags()}["strata-reels"]["share_named"] is True
    assert fs.set_tag("strata-reels", True, share_named=False)["share_named"] is False
    assert [h["ref"] for h in fs.search(q, "buffalo_l", k=3)] == ["s1"]
    # many unnamed tape faces nearer than the public one never crowd it out of a short answer, with nothing named to lend
    fs.set_tag("strata-reels", True, share_named=True)
    index(fs, "strata-reels", "t2", [face(1, x=10 * i, seed=10 + i) for i in range(60)], tags=["strata-reels"])
    fs.assign([named], None, "strata-reels", status="rejected")
    assert [h["ref"] for h in fs.search(q, "buffalo_l", k=1)] == ["s1"]


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


def test_hints_are_stored_once_listed_and_undone_by_batch(fs):
    a = index(fs, "immich", "a", [face(1)])
    b = index(fs, "immich", "b", [face(1), face(2, x=100)])
    r = fs.add_hints(
        "takeout",
        [
            dict(image_id=a["id"], app_ref="Ann", name="Ann Example"),
            dict(image_id=b["id"], app_ref="Ann"),
            dict(image_id=b["id"], app_ref="Bob", face_id=b["faces"][1]["id"], status="confirmed"),
            dict(image_id=b["id"], app_ref="Ann"),  # the same hint twice in one call
        ],
        note="first",
    )
    assert (r["added"], r["skipped"]) == (3, 1)
    assert fs.add_hints("takeout", [dict(image_id=a["id"], app_ref="Ann")])["added"] == 0
    got = fs.hints("takeout")
    assert [(h["app_ref"], h["status"], h["name"]) for h in got] == [
        ("Ann", "possible", "Ann Example"),
        ("Ann", "possible", None),
        ("Bob", "confirmed", None),
    ]
    assert [h["app_ref"] for h in fs.hints("takeout", image_id=b["id"])] == ["Ann", "Bob"]
    assert len(fs.hints("takeout", app_ref="Bob")) == 1 and fs.hints("other") == []
    fs.undo(r["batch"])
    assert fs.hints("takeout") == []
    assert fs.add_hints("takeout", [dict(image_id=a["id"], app_ref="Ann")])["added"] == 1
    for bad, match in [
        (dict(image_id=a["id"]), "image_id and app_ref"),
        (dict(image_id=a["id"], app_ref="x", status="sure"), "status"),
        (dict(image_id=a["id"], app_ref="x", face_id=b["faces"][0]["id"]), "is not in picture"),
    ]:
        with pytest.raises(ValueError, match=match):
            fs.add_hints("takeout", [bad])
    with pytest.raises(NotFound, match="no such picture"):
        fs.add_hints("takeout", [dict(image_id=999, app_ref="x")])
    with pytest.raises(ValueError, match="need an app"):
        fs.add_hints("", [dict(image_id=a["id"], app_ref="x")])
    with pytest.raises(ValueError, match="no hints"):
        fs.add_hints("takeout", [])


def test_propose_counts_then_writes_suggestions_for_a_new_or_mapped_person(fs):
    solo = [index(fs, "immich", f"s{i}", [face(1, seed=i)]) for i in range(10)]
    group = index(fs, "immich", "g", [face(2, seed=50), face(1, x=100, seed=51)])
    hints = [dict(image_id=i["id"], app_ref="Ann", name="Ann E") for i in solo]
    hints += [dict(image_id=group["id"], app_ref="Ann"), dict(image_id=group["id"], app_ref="Bob")]
    fs.add_hints("takeout", hints)
    dry = fs.propose("takeout", "buffalo_l")
    assert (dry["dry_run"], dry["qualifying"], dry["suggested"], dry["batch"], dry["hinted_pictures"]) == (
        True,
        1,
        11,
        None,
        11,
    )
    assert dry["people"][0] | {} == dict(
        app_ref="Ann",
        name="Ann E",
        pictures=11,
        seeds=10,
        purity=1.0,
        qualifies=True,
        suggested=11,
        by={"picture": 11},
        identity_id=None,
    )
    assert fs.identities() == []  # a dry run writes nothing
    done = fs.propose("takeout", "buffalo_l", dry_run=False)
    assert (done["created"], done["suggested"]) == (1, 11) and done["batch"] is not None
    (ann,) = fs.identities(app="takeout")
    assert (ann["name"], ann["apps"]["takeout"]["app_ref"], ann["faces"]) == (
        "Ann E",
        "Ann",
        0,
    )  # suggestions are not confirmed faces
    sug = fs.faces_of(ann["id"], status="suggested")
    assert group["faces"][1]["id"] in [f["id"] for f in sug] and len(sug) == 11
    assert fs.propose("takeout", "buffalo_l", dry_run=False)["suggested"] == 0  # nothing new: already suggested
    fs.undo(done["batch"])
    assert fs.faces_of(ann["id"], status="suggested") == []
    again = fs.propose("takeout", "buffalo_l", dry_run=False, min_seeds=5)  # now Ann is mapped: no new person
    assert (again["created"], again["people"][0]["identity_id"], again["params"]["min_seeds"]) == (0, ann["id"], 5)


def test_propose_uses_the_mapped_persons_confirmed_faces_and_groups(fs):
    imgs = [index(fs, "immich", f"g{i}", [face(1, seed=i), face(2, x=100, seed=100 + i)]) for i in range(4)]
    ann = fs.create_identity("Ann", app="takeout", app_ref="Ann")
    conf = [index(fs, "immich", f"c{i}", [face(1, seed=200 + i)]) for i in range(10)]
    fs.assign([c["faces"][0]["id"] for c in conf], ann["id"], "immich")
    fs.add_hints("takeout", [dict(image_id=i["id"], app_ref="Ann") for i in imgs])
    r = fs.propose("takeout", "buffalo_l")
    assert r["people"][0]["seeds"] == 10 and r["people"][0]["suggested"] == 4
    extra = [index(fs, "immich", f"x{i}", [face(1, seed=300 + i)]) for i in range(2)]
    r = fs.propose("takeout", "buffalo_l", groups=True, group_min=2, group_share=0.5)
    assert r["groups"] >= 1 and r["people"][0]["by"].get("group", 0) == 2 and r["people"][0]["suggested"] == 6
    w = fs.propose("takeout", "buffalo_l", groups=True, group_min=2, group_share=0.5, limit=5000, dry_run=False)
    assert w["created"] == 0 and {f["id"] for f in fs.faces_of(ann["id"], status="suggested")} >= {
        x["faces"][0]["id"] for x in extra
    }


def test_propose_with_no_hints_and_merge_suggestions_from_suggested_faces(fs):
    assert fs.propose("nobody", "buffalo_l", dry_run=False)["batch"] is None
    imgs = [index(fs, "immich", f"s{i}", [face(1, seed=i)]) for i in range(10)]
    fs.add_hints("takeout", [dict(image_id=i["id"], app_ref="Ann") for i in imgs])
    fs.propose("takeout", "buffalo_l", dry_run=False)
    cat = fs.create_identity("Cat")
    other = index(fs, "immich", "o", [face(1, seed=77)])
    fs.assign([other["faces"][0]["id"]], cat["id"], "bright-horizons")
    (pair,) = fs.merge_suggestions("buffalo_l")
    assert {pair["basis_a"], pair["basis_b"]} == {"confirmed", "suggested"}


def test_suggested_faces_are_counted_paged_and_kept_out_of_unnamed_groups_on_request(fs):
    imgs = [index(fs, "immich", f"p{i}", [face(1, seed=i, score=0.9 - i / 100)]) for i in range(6)]
    ids = [i["faces"][0]["id"] for i in imgs]
    ann = fs.create_identity("Ann")
    fs.assign(ids[:4], ann["id"], "takeout", status="suggested", source="hint")
    fs.assign(ids[:1], ann["id"], "strata-models")
    got = fs.identity(ann["id"])
    assert (got["faces"], got["suggested"]) == (1, 3)
    assert [p["suggested"] for p in fs.identities()] == [3]
    page1 = [f["id"] for f in fs.faces_of(ann["id"], status="suggested", limit=2)]
    page2 = [f["id"] for f in fs.faces_of(ann["id"], status="suggested", limit=2, offset=2)]
    assert page1 == ids[1:3] and page2 == ids[3:4]
    every = sorted(f for g in fs.cluster("buffalo_l") for f in g["faces"])
    assert every == ids[1:]  # suggested faces still count as undecided by default
    assert sorted(f for g in fs.cluster("buffalo_l", suggested=False) for f in g["faces"]) == ids[4:]


def test_review_offers_the_tightest_group_first_then_ranks_by_confirmed_faces(fs):
    # 8 faces of person 1 suggested with low confidence, 4 of person 2 (wrongly) with high confidence: the largest tight group wins
    ones = [index(fs, "immich", f"a{i}", [face(1, seed=i)])["faces"][0]["id"] for i in range(8)]
    twos = [index(fs, "immich", f"b{i}", [face(2, seed=50 + i)])["faces"][0]["id"] for i in range(4)]
    ann = fs.create_identity("Ann")
    fs.assign(ones, ann["id"], "t", status="suggested", source="hint", confidence=0.6)
    fs.assign(twos, ann["id"], "t", status="suggested", source="hint", confidence=0.9)
    r = fs.review(ann["id"], "buffalo_l", limit=5)
    assert (r["basis"], r["basis_faces"], r["suggested"], r["confirmed"]) == ("group", 8, 12, 0)
    assert {f["id"] for f in r["items"]} <= set(ones) and len(r["items"]) == 5
    assert r["items"][0]["distance"] <= r["items"][-1]["distance"] and r["items"][0]["status"] == "suggested"
    assert r["near"]["0.3"] == 8 and r["left_out"] == 0
    assert (
        r["counts"]["0.50"] == 8
        and r["counts"]["0.95"] <= 8
        and next(iter(r["counts"])) == "0.50"
        and len(r["counts"]) == 10
    )
    every = fs.review(ann["id"], "buffalo_l", limit=10000, min_likeness=0.5)
    assert len(every["items"]) == 8 and all(1 - f["distance"] >= 0.5 for f in every["items"])
    assert fs.review(ann["id"], "buffalo_l", limit=10000, min_likeness=0.9999)["items"] == []
    fs.assign(
        twos[:1], ann["id"], "strata-models", status="rejected"
    )  # one of the strays is not Ann: its lookalikes leave the round
    fs.assign(ones[:3], ann["id"], "strata-models")  # three confirmed: the person's own faces now rank the rest
    r = fs.review(ann["id"], "buffalo_l", limit=50)
    assert (r["basis"], r["basis_faces"], r["suggested"], r["confirmed"]) == ("confirmed", 3, 8, 3)
    got = [f["id"] for f in r["items"]]
    assert set(got) == set(ones[3:]) and r["left_out"] == 3
    single = fs.create_identity("Solo")
    one = index(fs, "immich", "s", [face(3)])["faces"][0]["id"]
    fs.assign([one], single["id"], "t", status="suggested")
    assert [f["id"] for f in fs.review(single["id"], "buffalo_l")["items"]] == [one]
    assert fs.review(fs.create_identity("Nobody")["id"], "buffalo_l")["items"] == []
    other = index(fs, "immich", "x", [face(4)], model="other")["faces"][0]["id"]
    nobody_here = fs.create_identity("Elsewhere")
    fs.assign([other], nobody_here["id"], "t", status="suggested")
    assert fs.review(nobody_here["id"], "buffalo_l")["items"] == []  # no embedding of this model
    with pytest.raises(NotFound):
        fs.review(999, "buffalo_l")


def test_faces_under_the_bar_are_kept_but_never_searched_grouped_or_reviewed(clock):
    bar = [0.7, 40.0]
    fs = FaceStore(memory_store(clock).engine, clock=clock, bar=lambda: tuple(bar))
    good = [face(1, x=i * 50, seed=i) for i in range(3)]
    tiny = [dict(face(1, x=200 + i * 50, seed=10 + i), bbox=[200 + i * 50, 10, 220 + i * 50, 30]) for i in range(3)]
    doubtful = [face(1, x=400 + i * 50, seed=20 + i, score=0.6) for i in range(3)]
    ids = [f["id"] for f in index(fs, "a", "p1", good + tiny + doubtful)["faces"]]
    assert len(ids) == 9  # every face is stored
    assert [g["faces"] for g in fs.cluster("buffalo_l")] == [ids[:3]]
    assert sorted(h["face"]["id"] for h in fs.search(person(1), "buffalo_l", k=20)) == ids[:3]
    assert fs.cluster("buffalo_l", min_score=0.5)[0]["faces"] == ids[:3]  # a caller can raise the bar, never lower it
    pid = fs.create_identity(name="One")["id"]
    fs.assign(ids, pid, "a", status="suggested", source="hint", confidence=0.9)
    r = fs.review(pid, "buffalo_l")
    assert sorted(i["id"] for i in r["items"]) == ids[:3] and r["suggested"] == 3
    bar[:] = [0.0, 0.0]  # the setting lowered: the same faces come back at once
    assert sorted(h["face"]["id"] for h in fs.search(person(1), "buffalo_l", k=20)) == sorted(ids)
    assert fs.review(pid, "buffalo_l")["suggested"] == 9


def test_delete_images_counts_first_then_removes_everything_on_them(fs):
    a = index(fs, "eval", "c1", [face(1), face(2, x=100)], tags=["face-tracks-eval"])
    b = index(fs, "eval", "c2", [face(1, seed=3)], tags=["face-tracks-eval"])
    keep = index(fs, "photos", "p1", [face(1, seed=4)])
    pid = fs.create_identity(name="One")["id"]
    fs.assign([a["faces"][0]["id"], keep["faces"][0]["id"]], pid, "photos")
    fs.add_hints("eval", [dict(image_id=b["id"], app_ref="x", name="One")])
    with pytest.raises(ValueError, match="say which pictures"):
        fs.delete_images()
    plan = fs.delete_images(tag="face-tracks-eval")
    assert plan == dict(images=2, faces=3, decisions=1, confirmed=1, hints=1, deleted=False, media=[])
    assert fs.image(a["id"])["faces"]  # a dry run changes nothing
    assert fs.delete_images(tag="face-tracks-eval", app="photos")["images"] == 0  # every selector must match
    assert (
        fs.delete_images(ids=[], app="eval", dry_run=False)["images"] == 0
    )  # an empty id list matches nothing, never the whole app
    done = fs.delete_images(tag="face-tracks-eval", dry_run=False)
    assert done["deleted"] and done["images"] == 2
    for img in (a, b):
        with pytest.raises(NotFound):
            fs.image(img["id"])
    assert [f["id"] for f in fs.faces_of(pid)] == [keep["faces"][0]["id"]]  # the person keeps their other faces
    assert fs.hints("eval") == [] and fs.image(keep["id"])["faces"]
    assert fs.delete_images(ids=[keep["id"]], dry_run=False)["faces"] == 1


def test_delete_images_names_only_the_kept_copies_no_other_picture_uses(fs):
    ids = [index(fs, app, ref, [face(1)])["id"] for app, ref in (("a", "one"), ("a", "two"), ("b", "three"))]
    with fs.session() as s:
        from stratamodels.db.models import FaceImage

        for img in s.query(FaceImage):
            img.medium = "faces/medium/shared.webp" if img.ref != "two" else "faces/medium/own.webp"
        s.commit()
    assert fs.delete_images(app="a")["media"] == ["faces/medium/own.webp"]  # "three" still shows the shared copy
    assert sorted(fs.delete_images(app="a", dry_run=True)["media"]) == ["faces/medium/own.webp"]
    assert fs.delete_images(app="b")["media"] == []  # "one" still uses it
    assert fs.delete_images(ids=ids)["media"] == ["faces/medium/own.webp", "faces/medium/shared.webp"]


def test_one_source_people_and_review_round(fs):
    photos = index(fs, "immich", "p1", [face(1, x=i * 50, seed=i) for i in range(3)])
    clip = index(fs, "eval", "c1", [face(1, x=i * 50, seed=10 + i) for i in range(2)], tags=["video"])
    other = index(fs, "immich", "p2", [face(2, seed=30)])
    one, two = fs.create_identity(name="One")["id"], fs.create_identity(name="Two")["id"]
    fs.assign([f["id"] for f in photos["faces"]], one, "a")
    fs.assign([f["id"] for f in clip["faces"]], one, "a", status="suggested", source="hint", confidence=0.9)
    fs.assign([other["faces"][0]["id"]], two, "a", status="suggested", source="hint", confidence=0.9)
    everyone = {p["id"]: (p["faces"], p["suggested"]) for p in fs.identities()}
    assert everyone == {one: (3, 2), two: (0, 1)}
    video = fs.identities(tag="video")
    assert [(p["id"], p["faces"], p["suggested"]) for p in video] == [
        (one, 0, 2)
    ]  # only people seen there, counting those faces
    assert fs.identities(tag="nowhere") == []
    r = fs.review(one, "buffalo_l", tag="video")
    assert r["suggested"] == 2 and sorted(i["id"] for i in r["items"]) == sorted(f["id"] for f in clip["faces"])
    assert r["basis"] == "confirmed"  # the person's model still comes from all their confirmed faces
    assert fs.review(two, "buffalo_l", tag="video")["items"] == []


def track(key, k, seed=0, noise=0.01):
    return dict(key=key, embedding=person(k, noise, seed))


def test_group_apart_never_joins_tracks_seen_together():
    E = np.array([person(1, 0.01, 1), person(1, 0.01, 2), person(1, 0.01, 3), person(5, 0.01, 4)])
    assert group_apart(E, [], 0.55) == [[0, 1, 2], [3]]
    assert group_apart(E, [(0, 2)], 0.55) in (
        [[0, 1], [2], [3]],
        [[0], [1, 2], [3]],
    )  # one of them on its own, never 0 with 2
    assert group_apart(E[:1], [], 0.55) == [[0]]
    assert group_apart(E, [], 0.0) == [[0], [1], [2], [3]]


def test_group_tracks_reads_the_named_faces_once_and_answers_as_search_does(fs):
    """group_tracks' suggestions come from one read of the confirmed faces (a search per group was an exact scan each): for any query,
    the same faces at the same distances, in scope, under the bar, through undone decisions, as search(named=True)."""
    faces = [face(k % 4, x=i * 50, seed=i) for i, k in enumerate(range(12))]
    ids = [f["id"] for f in index(fs, "stills", "p1", faces)["faces"]]
    hidden = [
        f["id"] for f in index(fs, "kids", "k1", [face(1, seed=40), face(2, x=50, seed=41)], tags=["nursery"])["faces"]
    ]
    tiny = index(fs, "stills", "p2", [face(1, score=0.3, seed=42)])["faces"][0]["id"]
    ann, bob = (fs.create_identity(name=n)["id"] for n in ("Ann", "Bob"))
    fs.assign(ids[0::2] + hidden[:1] + [tiny], ann, "stills")
    undone = fs.assign(ids[1::4], bob, "stills")
    fs.assign(ids[3::4], bob, "stills")
    fs.undo(undone)
    fs.set_tag("nursery", True)
    queries = np.array([person(k, 0.05, 100 + k) for k in range(4)], dtype=np.float32)
    for include in ((), ("nursery",)):
        got = fs._named_nearest("buffalo_l", queries, k=3, max_distance=0.5, include=include)
        for q, hits in zip(queries, got, strict=True):
            want = fs.search(
                [float(x) for x in q], "buffalo_l", k=3, named=True, max_distance=0.5, include=list(include)
            )
            assert [(h["face"]["id"], h["face"]["identity_id"]) for h in hits] == [
                (h["face"]["id"], h["face"]["identity_id"]) for h in want
            ]
            assert [h["distance"] for h in hits] == pytest.approx([h["distance"] for h in want], abs=1e-5)
    assert fs._named_nearest("buffalo_l", queries[:0], k=3) == []
    assert fs._named_nearest("other-model", queries, k=3) == [[], [], [], []]


def test_group_tracks_suggests_a_known_person_only_when_most_nearest_faces_agree(fs):
    ids = [
        f["id"]
        for f in index(fs, "stills", "p1", [face(1, x=i * 50, seed=i) for i in range(4)] + [face(2, x=400, seed=9)])[
            "faces"
        ]
    ]
    ann = fs.create_identity(name="Ann")["id"]
    fs.assign(ids[:4], ann, "stills")
    tracks = [track("t1", 1, 11), track("t2", 1, 12), track("t3", 3, 13), track("t4", 1, 14)]
    groups = fs.group_tracks("buffalo_l", tracks, apart=[("t1", "t4"), ("t1", "nope")])["groups"]
    assert sorted(len(g["tracks"]) for g in groups) == [1, 1, 2] and ["t3"] in [g["tracks"] for g in groups]
    assert not any({"t1", "t4"} <= set(g["tracks"]) for g in groups)
    big = groups[0]
    assert big["suggested"] == dict(identity_id=ann, name="Ann", share=1.0)
    assert next(g for g in groups if g["tracks"] == ["t3"])["suggested"] is None  # nobody near person 3
    fs.set_tag("stills", True)
    assert fs.group_tracks("buffalo_l", tracks)["groups"][0]["suggested"] is None  # Ann's faces are opt-in now
    assert fs.group_tracks("buffalo_l", tracks, include=["stills"])["groups"][0]["suggested"]["identity_id"] == ann
    bob = fs.create_identity(name="Bob")["id"]
    fs.merge(ann, bob, "stills")
    merged = fs.group_tracks("buffalo_l", tracks, include=["stills"])["groups"][0]
    assert merged["suggested"]["identity_id"] == bob  # read through the merge
    assert fs.group_tracks("buffalo_l", []) == dict(groups=[], hidden=[])
    tiny = index(fs, "reels", "still", [face(1, score=0.3)])["faces"][0]["id"]  # under the store's bar
    got = fs.group_tracks("buffalo_l", [dict(track("t5", 1), faces=[tiny]), dict(track("t6", 1), faces=[ids[0]])])
    assert (got["hidden"], [g["tracks"] for g in got["groups"]]) == (["t5"], [["t6"]])
    with pytest.raises(ValueError, match="key of its own"):
        fs.group_tracks("buffalo_l", [track("a", 1), track("a", 2)])
