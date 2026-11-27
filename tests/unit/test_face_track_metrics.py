"""The face-tracks eval metrics (evals/metrics.py): tracks looked up at labelled moments, identity F1, mixed and split tracks, missed faces."""

import pytest

from stratamodels.evals import metrics as M

A, B = [0, 0, 100, 100], [300, 0, 400, 100]


def item(people, times, cuts=()):
    return dict(input=dict(path="v.mp4", cuts=list(cuts)), reference=dict(people=people, times=times))


def track(tid, spans):
    """spans: (time, box) pairs."""
    return dict(id=tid, boxes=[[t, *box, 0.9] for t, box in spans])


def out(*tracks, fps=5):
    return dict(fps=fps, tracks=list(tracks))


def score(name, it, o, error=None):
    return M.METRICS[name].score(it, o, error)


TIMES = [0.0, 0.5, 1.0, 1.5]
TWO = item({"ana": [[t, *A] for t in TIMES], "ben": [[t, *B] for t in TIMES]}, TIMES)


def test_a_track_is_looked_up_between_and_just_beyond_its_boxes():
    boxes = [[0.0, 0, 0, 10, 10], [1.0, 10, 0, 20, 10]]
    assert M.track_box_at(boxes, 0.5, 5) == [5, 0, 15, 10]
    assert M.track_box_at(boxes, 1.05, 5) == [10, 0, 20, 10]  # within half a frame of the last
    assert M.track_box_at(boxes, 1.2, 5) is None and M.track_box_at([], 0, 5) is None
    assert M.track_box_at([[0.0, 1, 2, 3, 4]], 0.0, 0) == [1, 2, 3, 4]


def test_perfect_tracks_score_one():
    o = out(track(1, [(t, A) for t in TIMES]), track(2, [(t, B) for t in TIMES]))
    assert score("id_f1", TWO, o) == 1
    assert score("track_mixed", TWO, o) == (0, 2)
    assert score("track_split", TWO, o) == (0, 2)
    assert score("track_face_miss", TWO, o) == (0, 8)


def test_a_track_that_switches_person_is_mixed_and_costs_identity():
    # track 1 follows ana then jumps to ben; track 2 the other way round
    o = out(
        track(1, [(0.0, A), (0.5, A), (1.0, B), (1.5, B)]),
        track(2, [(0.0, B), (0.5, B), (1.0, A), (1.5, A)]),
    )
    assert score("id_f1", TWO, o) == pytest.approx(0.5)
    assert score("track_mixed", TWO, o) == (2, 2)


def test_one_person_split_into_two_tracks_in_a_shot():
    o = out(track(1, [(0.0, A), (0.5, A)]), track(2, [(1.0, A), (1.5, A)]), track(3, [(t, B) for t in TIMES]))
    assert score("track_split", TWO, o) == (1, 2)
    assert score("id_f1", TWO, o) == pytest.approx(2 * 6 / 16)


def test_a_split_at_a_cut_is_not_a_split():
    it = item(TWO["reference"]["people"], TIMES, cuts=[1.0])
    o = out(
        track(1, [(0.0, A), (0.5, A)]),
        track(2, [(1.0, A), (1.5, A)]),
        track(3, [(0.0, B), (0.5, B)]),
        track(4, [(1.0, B), (1.5, B)]),
    )
    assert score("track_split", it, o) == (0, 4)


def test_missed_and_false_faces():
    o = out(track(1, [(t, A) for t in TIMES]), track(9, [(t, [600, 0, 700, 100]) for t in TIMES]))
    assert score("track_face_miss", TWO, o) == (4, 8)
    assert score("id_f1", TWO, o) == pytest.approx(2 * 4 / 16)


def test_failed_jobs_and_unlabelled_items():
    assert score("id_f1", TWO, None, error="boom") == 0
    assert score("track_face_miss", TWO, None, error="boom") == (8, 8)
    assert (
        score("track_mixed", TWO, None, error="boom") is None and score("track_split", TWO, None, error="boom") is None
    )
    plain = dict(input={}, reference={})
    for m in ("id_f1", "track_mixed", "track_split", "track_face_miss"):
        assert score(m, plain, out()) is None
    assert score("id_f1", item({}, [0.0]), out()) == 1  # nobody there and nobody answered


def test_a_face_where_nobody_was_named_is_not_counted():
    it = item({"ana": [[t, *A] for t in TIMES]}, TIMES)
    it["reference"]["ignore"] = [[t, *B] for t in TIMES]
    o = out(track(1, [(t, A) for t in TIMES]), track(2, [(t, B) for t in TIMES]))
    assert score("id_f1", it, o) == 1
    it["reference"]["ignore"] = []
    assert score("id_f1", it, o) == pytest.approx(2 * 4 / 12)


def test_a_person_in_two_shots_scores_both_tracks_per_shot_not_over_the_clip():
    it = item(TWO["reference"]["people"], TIMES, cuts=[1.0])
    o = out(  # one track per person per shot, as face-tracks answers: every face in its person's track
        track(1, [(0.0, A), (0.5, A)]),
        track(2, [(1.0, A), (1.5, A)]),
        track(3, [(0.0, B), (0.5, B)]),
        track(4, [(1.0, B), (1.5, B)]),
    )
    assert score("id_f1", it, o) == 1
    assert score("id_f1_clip", it, o) == pytest.approx(
        2 * 4 / 16
    )  # over the clip, each person can count one of their two tracks


def test_without_cuts_the_two_scores_agree_and_a_switch_within_a_shot_still_costs():
    switch = out(
        track(1, [(0.0, A), (0.5, A), (1.0, B), (1.5, B)]),
        track(2, [(0.0, B), (0.5, B), (1.0, A), (1.5, A)]),
    )
    assert score("id_f1", TWO, switch) == score("id_f1_clip", TWO, switch) == pytest.approx(0.5)
    it = item(
        TWO["reference"]["people"], TIMES, cuts=[1.0]
    )  # the same switch exactly at a cut: per shot it is two clean pairs
    assert score("id_f1", it, switch) == 1 and score("track_mixed", it, switch) == (
        2,
        2,
    )  # but the mixed track is still reported
    assert (
        score("id_f1_clip", TWO, None, error="boom") == 0
        and score("id_f1_clip", dict(input={}, reference={}), out()) is None
    )
