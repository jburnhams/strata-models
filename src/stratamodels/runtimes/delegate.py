"""A runtime that asks another model of the registry (image-kind's vision-language reader, vectorise's drawing model): loaded once in this
process, as page-build loads its layout grouper (a cloud model holds no GPU; its spend is counted by its own runtime), or with `via =
"service"` sent as a job to the running service's queue (role vision-language), the preset expanded here so the service needs nothing new.
`eval run --local` (an eval from a branch, outside the service) sends every delegate that way (STRATAMODELS_DELEGATE_VIA): a local model
then runs on the service's card in its turn, and a cloud model's call is counted in the service's ledger against its budget. A model with a
batch API (Claude) always goes that way, inside the service too (a job on itself, low priority, so batched at half price: Jonathan,
2026-11-28, every Claude call batched but a single high-priority one); a batch job is never in a lane, so the job waiting on it cannot block
it, though it holds the CPU lane while it waits. Never set `via` in models.toml."""

import os
import sys
from collections.abc import Callable
from typing import Any

from stratamodels.runtimes import RunError


def _registry():
    from stratamodels import registry, settings

    eng = settings.engine()
    return registry.Registry.from_db(eng) if eng is not None else registry.Registry.load()


def _batched(key: str) -> bool:
    """Whether registry model `key` runs on a runtime with a batch API (claude-api)."""
    from stratamodels import runtimes

    m = _registry().models.get(key)
    return m is not None and callable(getattr(runtimes.get(m.runtime), "batch_submit", None))


def caller(
    key: str, models_dir: str, via: str | None = None, options: dict[str, Any] | None = None
) -> tuple[Callable[[dict[str, Any]], Any], Callable[[], None]]:
    """(run(params) -> answer, unload()) for registry model `key`; `options`: its load options changed (a variant, as an eval candidate's:
    Qwen with a longer context), only through the service."""
    via = via or os.environ.get("STRATAMODELS_DELEGATE_VIA") or None
    if via is None and _batched(key):
        via = "service"  # a model with a batch API is asked as a job, so the batcher sends it at half price (batches.py), never called from here
    if via == "service":
        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "client"
            ),
        )
        import strata_models_client as sm  # pyright: ignore[reportMissingImports]

        from stratamodels.runtimes import llama

        def via_service(p: dict[str, Any]) -> Any:
            q = dict(p)
            if "preset" in q:
                text, schema = llama.preset(str(q.pop("preset")))
                q["prompt"] = llama.fill_context(text, q.pop("context", None))
                if schema is not None and "schema" not in q:
                    q["schema"] = schema
            if options:
                (jid,) = sm.submit_many("vision-language", [q], "evals", "low", key, dict(options))
                return sm.wait(jid)
            return sm.run("vision-language", q, "evals", "low", key)

        return via_service, lambda: None
    if via:
        raise RunError(f"via is 'service' or not given, not {via!r}")
    if options:
        raise RunError(
            f"load options for {key!r} ({sorted(options)}) are a variant the service loads: ask it with via = 'service'"
        )
    from stratamodels import runtimes

    m = _registry().models.get(key)
    if m is None:
        raise RunError(f"the model {key!r} is not in the registry")
    if (m.vram_gb or 0) > 0:
        # loaded here it would hold the card outside the scheduler's budget (a second llama-server beside OCR's, or one during an
        # exclusive lease): a GPU model is asked only as its own job, through the queue
        raise RunError(
            f"the model {key!r} needs the GPU ({m.vram_gb} GB) and is asked only as a job on the service: run the eval with --local, "
            "which sends it there"
        )
    rt = runtimes.get(m.runtime)
    h = rt.load(m, models_dir)
    return (lambda p: rt.run(h, p)), (lambda: rt.unload(h))
