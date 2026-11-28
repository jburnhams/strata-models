"""scripts/evals/image_kind_hard.py: Haiku's misses (and a few it got right) as the dataset Sonnet and Opus run on."""

import importlib.util
import os
import sys
import time

from stratamodels import settings
from stratamodels.db.models import EvalResult
from stratamodels.evals import core as C

spec = importlib.util.spec_from_file_location(
    "image_kind_hard", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "evals", "image_kind_hard.py")
)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def test_the_misses_and_a_few_right_ones_are_kept(scheduler, monkeypatch, capsys):
    engine = scheduler.store.engine
    monkeypatch.setattr(settings, "engine", lambda: engine)
    items = [
        dict(id=f"p{i}", input=dict(image=f"/x/{i}.png"), reference=dict(kind="engraving"), tags={}) for i in range(6)
    ]
    C.write_dataset("image-kind", items, engine=engine)
    s = C.load_spec("image-kind", engine)
    haiku = s.candidate("haiku")
    answers = [
        "engraving",
        "photograph",
        "woodcut",
        "engraving",
        None,
        "engraving",
    ]  # p4 failed; p5 has no output at all
    rows = []
    for it, kind in zip(items[:5], answers, strict=False):
        out = None if kind is None else dict(kind=kind, group="photo" if kind == "photograph" else "illustration")
        rows.append(
            EvalResult(
                key=C.result_key(s, haiku, it),
                dataset="image-kind",
                item_id=it["id"],
                role=s.role,
                model=haiku.model,
                created_at=time.time(),
                output=out,
                error=None if out else "boom",
            )
        )
    C.save(engine, rows)
    wrong, right = B.split(s, engine, "haiku")
    assert [it["id"] for it in wrong] == ["p1", "p2", "p4"] and [it["id"] for it in right] == ["p0", "p3"]
    monkeypatch.setattr(sys, "argv", ["b", "--right", "1"])
    B.main()
    hard = C.load_dataset("image-kind-hard", engine=engine).items
    assert len(hard) == 4 and {it["id"] for it in hard} >= {"p1", "p2", "p4"}
    assert "4 items (3 haiku got wrong)" in capsys.readouterr().out
