"""The text-embedding runtime (stratamodels.runtimes.llamaembed) against a fake llama-server, and the retrieval metrics."""

import math

import numpy as np
import pytest
from test_llama import FakeProc, setup  # noqa: F401  (the fake server process and the fixture that fakes the binary and Hugging Face)

from stratamodels.evals import metrics as M
from stratamodels.registry import Model
from stratamodels.runtimes import RunError, llamaembed


def model(**o):
    return Model(
        key="emb",
        runtime="llama-embed",
        name="emb",
        roles=("text-embed",),
        vram_gb=1,
        options=dict(model="org/m-GGUF:m.gguf", **o),
    )


def fake_post(url, body, timeout=0):
    """An embedding server whose vectors are one-hot on the first letter of the text (after any prefix is read off by the test)."""
    assert url.endswith("/v1/embeddings")
    out = []
    for i, t in enumerate(body["input"]):
        v = [0.0] * 4
        v["abcd".index(t.strip()[-1])] = 3.0 if t.endswith(("a", "b", "c", "d")) else 0.0
        out.append(dict(index=i, embedding=v + [0.0]))
    return dict(
        data=out[::-1], usage=dict(prompt_tokens=len(body["input"]))
    )  # reversed: the runtime must sort by index


def loaded(tmp, **opts):
    procs = []
    h = llamaembed.load(model(**opts), str(tmp), popen=lambda cmd, **kw: procs.append(FakeProc(cmd, **kw)) or procs[-1])
    return h, procs[0].cmd


def test_load_adds_embedding_flags_and_keeps_the_prefixes(setup):  # noqa: F811
    tmp, _ = setup
    h, cmd = loaded(tmp, pooling="last", query_prefix="q: ", doc_prefix="d: ", ctx=1024, dim=3)
    assert "--embedding" in cmd and cmd[cmd.index("--pooling") + 1] == "last"
    assert (
        cmd[cmd.index("-b") + 1] == "1024"
        and cmd[cmd.index("-ub") + 1] == "1024"
        and cmd[cmd.index("-c") + 1] == "1024"
    )
    assert (h["query_prefix"], h["doc_prefix"], h["dim"]) == ("q: ", "d: ", 3)
    llamaembed.unload(h)


def test_query_and_docs_are_scored_by_cosine_in_the_docs_order(setup):  # noqa: F811
    tmp, _ = setup
    h, _ = loaded(tmp)
    r = llamaembed.run(h, dict(query="a", docs=["b", "c", "a", "d"]), probe=lambda: {}, post=fake_post)
    assert (
        r["scores"] == [0.0, 0.0, 1.0, 0.0] and r["dim"] == 5 and r["tokens_in"] == 4
    )  # "a" is embedded once: the query's vector serves the document too
    llamaembed.unload(h)


def test_texts_come_back_normalised_or_in_a_file(setup, tmp_path):  # noqa: F811
    tmp, _ = setup
    h, _ = loaded(tmp, dim=2)
    r = llamaembed.run(h, dict(texts=["a", "b"]), probe=lambda: {}, post=fake_post)
    assert r["vectors"] == [[1.0, 0.0], [0.0, 1.0]] and r["dim"] == 2
    f = str(tmp_path / "v.npz")
    r = llamaembed.run(h, dict(texts=["a", "c", "d"], out=f), probe=lambda: {}, post=fake_post)
    assert "vectors" not in r and np.load(f)["vectors"].shape == (3, 2)
    llamaembed.unload(h)


def test_bad_params_are_refused(setup):  # noqa: F811
    tmp, _ = setup
    h, _ = loaded(tmp)
    for bad in (
        dict(),
        dict(texts=["a"], query="a", docs=["b"]),
        dict(query="a", docs=[]),
        dict(texts="a"),
        dict(nope=1),
    ):
        with pytest.raises(RunError):
            llamaembed.run(h, bad, probe=lambda: {}, post=fake_post)
    llamaembed.unload(h)


def test_registry_entries_name_real_files():
    from stratamodels import registry

    reg = registry.Registry.load()
    for key in ("qwen3-embedding-0.6b", "qwen3-embedding-4b", "bge-m3", "embeddinggemma-300m", "nomic-embed-v1_5"):
        m = reg.models[key]
        assert isinstance(m, Model) and m.runtime == "llama-embed" and ":" in m.options["model"]
    assert reg.roles["text-embed"].model in reg.models


def test_retrieval_metrics_rank_the_answer_in_its_pool():
    def item(a):
        return dict(reference=dict(answer=a))

    out = dict(scores=[0.1, 0.9, 0.5, 0.7])
    assert (
        M.METRICS["hit_at_1"].score(item(1), out, None) is True
        and M.METRICS["hit_at_1"].score(item(3), out, None) is False
    )
    assert M.METRICS["hit_at_5"].score(item(0), out, None) is True and M.METRICS["mrr"].score(item(3), out, None) == 0.5
    assert M.METRICS["mrr"].score(item(0), out, None) == 0.25
    tie = dict(scores=[0.5, 0.5, 0.5])
    assert M.METRICS["mrr"].score(item(1), tie, None) == pytest.approx(1 / 3)  # ties count against the answer
    assert (
        M.METRICS["hit_at_1"].score(item(0), None, "boom") is False
        and M.METRICS["mrr"].score(item(0), None, "boom") == 0.0
    )
    assert M.METRICS["mrr"].score(dict(reference={}), out, None) is None
    assert math.isinf(M._rank_of_answer(item(9), out, None))
