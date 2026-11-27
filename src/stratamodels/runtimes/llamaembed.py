"""Text embeddings with llama.cpp's `llama-server --embedding` (GGUF: Qwen3-Embedding, bge-m3, EmbeddingGemma, nomic-embed, ...): the same binary, files and
loading as the vision-language models (runtimes/llama.py), answering /v1/embeddings.

options (models.toml), besides llama.py's `model`, `ctx` and `args`:
  pooling       mean | cls | last | none: how the model pools tokens (llama-server reads it from the GGUF when not given)
  query_prefix  put before every query (Qwen3-Embedding: "Instruct: <task>\nQuery: ", nomic: "search_query: ", e5: "query: ")
  doc_prefix    put before every document / text ("search_document: ", "passage: ")
  dim           keep only the first N dimensions of each vector (Matryoshka models), then renormalise

params (one of):
  texts   a list of strings: one L2-normalised vector each. `out` names an .npz file to write {vectors} into instead of answering with the numbers.
  query + docs   a string and a list of strings: cosine similarity of the query to each document, in the documents' order ("scores"); the
                 query gets query_prefix, the documents doc_prefix. This is what a retrieval eval item is: a question and a pool to rank.
  query + corpus + pick   the same, with the documents named by index: `corpus` is a JSON file holding a list of strings, `pick` the indices of
                 this item's pool in it. A dataset of hundreds of pools over one shared corpus stays small, and the file is read once per model.
Answer: {scores, dim, tokens_in, cut, seconds, peak_vram_mb, model} or {vectors | out, n, dim, cut, ...}; `cut` counts the texts that were longer than the
model's context and were cut at the end to fit (a text is cut the same way every time)."""

import dataclasses
import json
import math
import os
import threading
import time

import numpy as np

from stratamodels import gpu
from stratamodels.runtimes import RunError, llama

PARAMS = {"texts", "query", "docs", "corpus", "pick", "out"}
BATCH = 16
MAX_BODY_CHARS = 200_000
CACHE_MAX = 20_000  # vectors kept per loaded model


def load(model, models_dir, **kw):
    o = dict(model.options)
    args = list(o.get("args") or []) + ["--embedding"]
    if o.get("pooling"):
        args += ["--pooling", o["pooling"]]
    ctx = int(o.get("ctx", 2048))
    args += [
        "-b",
        str(ctx),
        "-ub",
        str(ctx),
    ]  # the whole input must fit one micro-batch for a non-causal embedding model
    h = llama.load(dataclasses.replace(model, options={**o, "args": args, "ctx": ctx}), models_dir, **kw)
    h.update(
        query_prefix=o.get("query_prefix", ""),
        doc_prefix=o.get("doc_prefix", ""),
        dim=int(o["dim"]) if o.get("dim") else None,
    )
    return h


def _normalise(v, dim):
    a = np.asarray(v, dtype=np.float32)
    if dim:
        a = a[:dim]
    n = float(np.linalg.norm(a))
    return a / n if n else a


def _post_batch(handle, chunk, post):
    r = (post or llama._post)(handle["url"] + "/v1/embeddings", {"input": chunk, "model": handle["key"]})
    data = sorted(r["data"], key=lambda d: d["index"])
    if len(data) != len(chunk):
        raise RunError(f"llama-server returned {len(data)} embeddings for {len(chunk)} texts")
    return data, int((r.get("usage") or {}).get("prompt_tokens") or 0)


def _too_long(e):
    m = str(e)
    return "exceeds the available context" in m or "too large to process" in m


def _embed_one(handle, text, post):
    """(embedding, tokens, truncated?) of one text. One longer than the model's context is cut to 60 % at a time until it fits: a single long
    document must not fail the whole pool it sits in (what is cut is the end, where a chunk's least distinctive words are)."""
    cut = False
    for _ in range(8):
        try:
            data, tok = _post_batch(handle, [text], post)
            return data[0]["embedding"], tok, cut
        except RunError as e:
            if not _too_long(e) or len(text) < 50:
                raise
            text, cut = text[: int(len(text) * 0.6)], True
    raise RunError("a text is too long for the model's context even cut to a fraction")


def embed(handle, texts, post=None):
    """(vectors (n, dim) float32, prompt tokens, how many texts were cut to fit) of `texts`, in order. A text embedded before (the pools of a
    retrieval eval share most of their documents) is answered from the handle's cache, so a pool of 100 costs its new documents only; the cache
    is dropped with the model."""
    cache = handle.setdefault("cache", {})
    texts = [t[:MAX_BODY_CHARS] or " " for t in texts]
    todo = list(dict.fromkeys(t for t in texts if t not in cache))
    tokens = cuts = 0
    for i in range(0, len(todo), BATCH):
        chunk = todo[i : i + BATCH]
        try:
            data, tok = _post_batch(handle, chunk, post)
            got = [(d["embedding"], False) for d in data]
        except RunError as e:
            if not _too_long(e):
                raise
            got, tok = [], 0  # one text in the batch is too long: embed them one at a time, cutting the long one
            for t in chunk:
                emb, k, cut = _embed_one(handle, t, post)
                got.append((emb, cut))
                tok += k
        for t, (emb, cut) in zip(chunk, got, strict=True):
            cache[t] = _normalise(emb, handle.get("dim"))
            cuts += cut
        tokens += tok
    out = [cache[t] for t in texts]
    if len(cache) > CACHE_MAX:  # keep the newest half
        for k in list(cache)[: len(cache) // 2]:
            del cache[k]
    return (np.stack(out) if out else np.zeros((0, 0), dtype=np.float32)), tokens, cuts


def corpus_docs(handle, path, pick):
    """The documents `pick` names in the JSON list at `path`, read once per loaded model (and again when the file changes)."""
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
        cache = handle.setdefault("corpora", {})
        if key not in cache:
            with open(path, encoding="utf-8") as f:
                docs = json.load(f)
            if not isinstance(docs, list) or not all(isinstance(d, str) for d in docs):
                raise RunError(f"{path}: a corpus is a JSON list of strings")
            cache.clear()  # one corpus at a time
            cache[key] = docs
        docs = cache[key]
    except OSError as e:
        raise RunError(f"corpus file {path}: {e}")
    except ValueError as e:
        raise RunError(f"corpus file {path} is not JSON: {e}")
    if not isinstance(pick, list) or not pick or not all(isinstance(i, int) and 0 <= i < len(docs) for i in pick):
        raise RunError(f"pick is a non-empty list of indices into the {len(docs)} documents of the corpus")
    return [docs[i] for i in pick]


def run(handle, params, probe=gpu.query, post=None):
    unknown = set(params) - PARAMS
    if unknown:
        raise RunError(f"unknown params for a text-embedding model: {sorted(unknown)}")
    pairs = "query" in params or "docs" in params or "corpus" in params
    if pairs == ("texts" in params):
        raise RunError("give either texts, or query with docs (or corpus and pick)")
    if "corpus" in params:
        if "docs" in params:
            raise RunError("give docs, or corpus and pick, not both")
        params = dict(params, docs=corpus_docs(handle, params["corpus"], params.get("pick")))
    peak = [0.0]
    stop = threading.Event()
    pid = handle["proc"].pid

    def watch():
        while not stop.wait(0.5):
            for pr in (probe() or {}).get("processes", []):
                if pr["pid"] == pid:
                    peak[0] = max(peak[0], pr["used_mb"])

    threading.Thread(target=watch, daemon=True).start()
    t0 = time.perf_counter()
    try:
        if pairs:
            q, docs = params.get("query"), params.get("docs")
            if (
                not isinstance(q, str)
                or not isinstance(docs, list)
                or not docs
                or not all(isinstance(d, str) for d in docs)
            ):
                raise RunError("query is a string and docs a non-empty list of strings")
            vq, tq, cq = embed(handle, [handle["query_prefix"] + q], post)
            vd, td, cd = embed(handle, [handle["doc_prefix"] + d for d in docs], post)
            scores = [round(float(x), 6) for x in vd @ vq[0]]
            if any(math.isnan(x) for x in scores):
                raise RunError("the model returned NaN similarities (a too-long input overflowing its context?)")
            res: dict = dict(scores=scores, tokens_in=tq + td, dim=int(vq.shape[1]), cut=cq + cd)
        else:
            texts = params["texts"]
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                raise RunError("texts is a list of strings")
            v, tok, cut = embed(handle, [handle["doc_prefix"] + t for t in texts], post)
            res = dict(n=len(texts), dim=int(v.shape[1]) if len(texts) else 0, tokens_in=tok, cut=cut)
            if params.get("out"):
                np.savez_compressed(params["out"], vectors=v)
                res["out"] = params["out"]
            else:
                res["vectors"] = [[round(float(x), 6) for x in row] for row in v]
    finally:
        stop.set()
    return dict(res, seconds=round(time.perf_counter() - t0, 2), peak_vram_mb=round(peak[0]), model=handle["key"])


unload = llama.unload
