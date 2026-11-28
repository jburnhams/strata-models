"""A runtime that asks another model of the registry (image-kind's vision-language reader, vectorise's drawing model): loaded once in this
process, as page-build loads its layout grouper (a cloud model holds no GPU; its spend is counted by its own runtime), or with `via =
"service"` sent as a job to the running service's queue (role vision-language), the preset expanded here so the service needs nothing new.
`eval run --local` (an eval from a branch, outside the service) sends every delegate that way (STRATAMODELS_DELEGATE_VIA): a local model
then runs on the service's card in its turn, and a cloud model's call is counted in the service's ledger against its budget. Never set
`via` in models.toml: the service would queue a job on itself."""

import os
import sys
from collections.abc import Callable
from typing import Any

from stratamodels.runtimes import RunError


def caller(
    key: str, models_dir: str, via: str | None = None
) -> tuple[Callable[[dict[str, Any]], Any], Callable[[], None]]:
    """(run(params) -> answer, unload()) for registry model `key`."""
    via = via or os.environ.get("STRATAMODELS_DELEGATE_VIA") or None
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
            return sm.run("vision-language", q, "evals", "low", key)

        return via_service, lambda: None
    if via:
        raise RunError(f"via is 'service' or not given, not {via!r}")
    from stratamodels import registry, runtimes, settings

    eng = settings.engine()
    reg = registry.Registry.from_db(eng) if eng is not None else registry.Registry.load()
    m = reg.models.get(key)
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
