"""The strata-360 speech dataset builder's reference (scripts/evals/strata360_speech.py): user corrections all applied, over Gemini's."""

import importlib.util, json, os

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
spec = importlib.util.spec_from_file_location(
    "s360speech", os.path.join(ROOT, "scripts", "evals", "strata360_speech.py")
)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def clip(tmp_path, words, fixes, edits):
    d = tmp_path / "c"
    d.mkdir(exist_ok=True)
    tr = dict(segments=[dict(t0=0, t1=1, words=[dict(w=w) for w in ws]) for ws in words])
    (d / "transcribe35.json").write_text(json.dumps(dict(fixes=fixes)))
    if edits is not None:
        (d / "transcript_edits.json").write_text(json.dumps(dict(edits=edits)))
    return str(d), tr


def test_user_corrections_all_apply_and_win_over_gemini(tmp_path):
    d, tr = clip(
        tmp_path,
        [["a", "b", "c", "d"], ["C", "'est", "bon"], ["x", "y"]],
        [dict(seg=0, word=1, through=2, to="BEE SEA"), dict(seg=2, word=0, to="ex")],
        {
            "0:0": dict(user=dict(text="one")),
            "0:2": dict(user=dict(text="three")),
            "1:2": dict(orig="bon", gemini=dict(text="bonne")),
        },
    )
    ref = B.reference(d, tr)
    assert ref[0] == ("one b three d", "user")  # Gemini's fix overlapped a user word: dropped; both user words applied
    assert ref[1] == ("C'est bon", "none") and ref[2] == ("ex y", "gemini")
    d2, tr2 = clip(tmp_path, [["a", "b", "c"]], [dict(seg=0, word=2, to="see")], {"0:0": dict(user=dict(text="A"))})
    assert B.reference(d2, tr2)[0] == ("A b see", "user+gemini")


def test_no_gemini_pass_no_reference(tmp_path):
    (tmp_path / "e").mkdir()
    assert B.reference(str(tmp_path / "e"), dict(segments=[])) is None
    d, tr = clip(tmp_path, [["a"]], [], None)
    assert B.reference(d, tr) == [("a", "none")]
