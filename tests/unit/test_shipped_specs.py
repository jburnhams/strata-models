"""Every eval spec shipped in evals/specs parses (known metrics, similarity kinds, judges), so a typo is found by the gate and not when someone opens the page."""

import pathlib
import tomllib

import pytest

from stratamodels.evals import core as C

SPECS = sorted((pathlib.Path(__file__).resolve().parents[2] / "evals" / "specs").glob("*.toml"))


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_spec_parses(path):
    spec = C.parse_spec(tomllib.loads(path.read_text(encoding="utf-8")), source=str(path))
    assert spec.name == path.stem and spec.candidates
