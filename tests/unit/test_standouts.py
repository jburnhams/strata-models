import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("standouts", os.path.join(ROOT, "scripts", "evals", "standouts.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)  # type: ignore[union-attr]


def item(i, ref, *texts):
    return dict(
        id=i,
        reference=dict(text=ref),
        outputs={
            f"m{k}": (None if t is None else dict(output=dict(text=t), error=None if t != "!" else "boom", run_s=1.0))
            for k, t in enumerate(texts)
        },
    )


ITEMS = [
    item("perfect", "abcdefghij", "abcdefghij", "abcdefghiX"),
    item("good", "abcdefghij", "abcdefghXX", "abcdefgXXX"),
    item("fair", "abcdefghij", "abcdeXXXXX", "abcdefXXXX"),
    item("poor", "abcdefghij", "abXXXXXXXX", "abcXXXXXXX"),
    item("broken", "abcdefghij", "!", None),
    item("none", "abcdefghij", None, None),
]


def test_scores_take_the_best_local_candidate():
    got = dict(S.scores(ITEMS, "cer", ["m0", "m1"]))
    assert (
        got["perfect"] == 0.0
        and got["good"] == pytest.approx(0.2)
        and got["fair"] == pytest.approx(0.4)
        and got["broken"] == 1.0
        and "none" not in got
    )


def test_pick_keeps_the_middle_with_most_room():
    s = S.scores(ITEMS, "cer", ["m0", "m1"])
    assert S.pick(s, "lower", 2) == [
        "poor",
        "fair",
    ]  # perfect (best quarter) and broken (worst quarter: failed) dropped
    assert S.pick([("a", 0.9), ("b", 0.5), ("c", 0.7), ("d", 0.1)], "higher", 1) == ["b"]
    assert S.pick([("a", 0.3)], "lower") == ["a"] and S.pick([], "lower") == []


def test_main_prints_an_items_line(monkeypatch, capsys):
    class R:
        def __init__(self, d):
            self.d = d

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            return json.dumps(self.d).encode()

    monkeypatch.setattr(S.urllib.request, "urlopen", lambda url, timeout: R(dict(items=ITEMS)))
    assert S.main(["ocr-pages", "--metric", "cer", "--local", "m0", "m1"]) == 0
    assert capsys.readouterr().out.strip() == 'items = ["poor", "fair"]'
    with pytest.raises(SystemExit, match="no item has a cer score"):
        S.main(["ocr-pages", "--metric", "cer", "--local", "nobody"])
