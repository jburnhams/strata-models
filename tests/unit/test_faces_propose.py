"""Planning suggestions from an app's hints (stratamodels.faces.propose), on synthetic unit embeddings: one direction per made-up person."""

import numpy as np

from stratamodels.faces.propose import Hint, Params, centroid, plan, unit

DIM = 64


def emb(k, seed=0, noise=0.05):
    v = np.zeros(DIM)
    v[k] = 1.0
    return unit(v + np.random.default_rng(seed).normal(0, noise, DIM))


def solo(n, k, ref, start=0, name=None):
    """n pictures each with one face of person k, hinted as `ref` only: n seeds."""
    faces = {start + i: [(1000 + start + i, emb(k, start + i), 0.9)] for i in range(n)}
    hints = [Hint(start + i, ref, name=name) for i in range(n)]
    return faces, hints


def test_seeds_make_a_centroid_and_enough_agreeing_seeds_qualify():
    faces, hints = solo(12, 1, "ann", name="Ann")
    f2, h2 = solo(4, 2, "bob", start=100)
    faces |= f2
    people = plan(hints + h2, faces, decided=set())
    ann, bob = people["ann"], people["bob"]
    assert (ann.name, ann.pictures, ann.seeds, ann.qualifies, ann.purity) == ("Ann", 12, 12, True, 1.0)
    assert (bob.seeds, bob.qualifies) == (4, False)  # too few
    assert list(people) == ["ann", "bob"]  # most pictures first
    assert ann.summary()["by"] == {"picture": 12} and ann.summary()["suggested"] == 12
    assert bob.suggestions == []


def test_mixed_up_seeds_do_not_qualify():
    faces, hints = solo(6, 1, "x")
    f2, h2 = solo(6, 2, "x", start=50)  # half the photos tagged x are someone else
    people = plan(hints + h2, faces | f2, decided=set())
    assert people["x"].seeds == 12 and not people["x"].qualifies and people["x"].purity <= 0.5


def test_a_group_photo_names_the_clearly_closest_face_one_each():
    faces, hints = solo(10, 1, "ann")
    f2, h2 = solo(10, 2, "bob", start=100)
    faces |= f2
    faces[500] = [(5001, emb(2, 7), 0.9), (5002, emb(1, 8), 0.9), (5003, emb(3, 9), 0.9)]
    hints += h2 + [Hint(500, "ann"), Hint(500, "bob")]
    people = plan(hints, faces, decided=set())
    assert (5002, "picture") in [(f, h) for f, _, h in people["ann"].suggestions]
    assert (5001, "picture") in [(f, h) for f, _, h in people["bob"].suggestions]
    assert all(f != 5003 for w in people.values() for f, _, _ in w.suggestions)


def test_two_lookalike_faces_are_too_close_to_call_and_decided_faces_stay():
    faces, hints = solo(10, 1, "ann")
    faces[600] = [(6001, emb(1, 1), 0.9), (6002, emb(1, 2), 0.9)]  # twins: neither is clearly closest
    faces[601] = [(6011, emb(1, 3), 0.9)]
    hints += [Hint(600, "ann"), Hint(601, "ann")]
    people = plan(hints, faces, decided={6011, 1000})
    got = {f for f, _, _ in people["ann"].suggestions}
    assert 6001 not in got and 6002 not in got and 6011 not in got and 1000 not in got
    assert people["ann"].seeds == 11  # a decided face (6011, alone in its picture) still counts as a seed


def test_low_scored_faces_are_neither_seeds_nor_suggested():
    faces, hints = solo(10, 1, "ann")
    faces[0] = [(1000, emb(1, 0), 0.3)]
    people = plan(hints, faces, decided=set(), params=Params(min_score=0.5))
    assert people["ann"].seeds == 9 and not people["ann"].qualifies and people["ann"].suggestions == []


def test_face_hints_seed_and_are_suggested_when_near_or_confirmed_by_the_app():
    faces = {i: [(i * 10, emb(1, i), 0.9), (i * 10 + 1, emb(2, i), 0.9)] for i in range(1, 11)}
    hints = [Hint(i, "dee", face_id=i * 10, status="confirmed") for i in range(1, 11)]
    hints.append(
        Hint(1, "kid", face_id=11, status="confirmed")
    )  # one face of someone else: too few to qualify, still suggested
    hints.append(Hint(2, "kid", face_id=21))  # possible, not qualifying: not suggested
    hints.append(Hint(3, "dee", face_id=999))  # a face with no embedding is ignored
    people = plan(hints, faces, decided=set())
    dee = people["dee"]
    assert dee.qualifies and dee.seeds == 10
    assert sorted(f for f, _, _ in dee.suggestions) == [i * 10 for i in range(1, 11)]
    assert [(f, h) for f, _, h in people["kid"].suggestions] == [(11, "face")]
    assert people["kid"].summary()["by"] == {"face": 1}


def test_confirmed_faces_of_the_mapped_person_count_as_seeds():
    faces = {i: [(i, emb(1, i), 0.9), (100 + i, emb(2, i), 0.9)] for i in range(5)}
    hints = [Hint(i, "ann") for i in range(5)]  # never alone in a picture: no seeds of their own
    assert not plan(hints, faces, decided=set())["ann"].qualifies
    people = plan(
        hints, faces, decided=set(), confirmed={"ann": [emb(1, 50 + i) for i in range(10)], "other": [emb(3)]}
    )
    assert people["ann"].qualifies and sorted(f for f, _, _ in people["ann"].suggestions) == list(range(5))


def test_groups_mostly_in_ones_pictures_are_named_whole():
    faces, hints = solo(10, 1, "ann")
    for i in range(3):  # pictures nobody hinted, with ann's face
        faces[700 + i] = [(7000 + i, emb(1, 70 + i), 0.9)]
    faces[800] = [(8000, emb(2, 1), 0.9)]
    groups = [
        [1000 + i for i in range(6)] + [7000, 7001, 7002],  # 6 of 6 hinted members are ann's
        [8000, 1009],  # too few hinted
        [7000],  # no hinted member at all
    ]
    decided = {1000 + i for i in range(10)}  # seeds already decided, so only the group can add
    people = plan(hints, faces, decided=decided, groups=groups)
    assert sorted((f, h) for f, _, h in people["ann"].suggestions) == [
        (7000, "group"),
        (7001, "group"),
        (7002, "group"),
    ]
    far = {**faces, 900: [(9000, emb(3), 0.9)]}  # a group whose faces are far from ann's centroid is not named
    people = plan(hints, far, decided=decided, groups=[[1000 + i for i in range(6)] + [9000]])
    assert [f for f, _, h in people["ann"].suggestions if h == "group"] == []
    people = plan(
        hints, faces, decided=decided, groups=[[1000 + i for i in range(6)] + [7000]], params=Params(group_min=7)
    )
    assert people["ann"].suggestions == []
    people = plan(
        hints, faces, decided=decided | {1000}, groups=[[1000, 4242]]
    )  # members with no embedding are skipped
    assert people["ann"].suggestions == []


def test_centroid_drops_outliers_and_never_empties():
    E = unit(np.array([emb(1, i) for i in range(8)] + [emb(5)]))
    c = centroid(E)
    assert float(c @ unit(np.array([emb(1, i) for i in range(8)]).mean(0))) > 0.999  # the stray row is left out
    assert float(c @ centroid(E, sample=4)) > 0.99  # a sample picks the same start
    assert np.isfinite(centroid(E, radius=0.0)).all()  # nobody near: the start row itself
