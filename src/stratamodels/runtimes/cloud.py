"""What the cloud API runtimes (claude-api, gemini-api, nvidia-nim) share: one job's params read the same way as llama-server's, so a cloud
model can stand in the same evals with the same presets; the HTTP call; the request as the ledger logs it.

params:
  prompt | preset (+ context)   the instruction, or a prompt kept in presets/ (its {context} filled from `context`)
  system | system_preset        the system prompt
  image / images                picture paths, sent before the text
  schema                        a JSON schema the answer must follow (a preset brings its own; schema=false turns it off)
  max_tokens                    the most the answer may be (default 2048): the budget reserves this much output before the call
  effort                        low | medium | high (how hard a model thinks; the model's own default when not given)
  temperature, seed, think      accepted so a job written for llama-server runs unchanged; the cloud models ignore them
Result: {text, json (when a schema was used and the answer parses), tokens_in, tokens_out, cost_usd, seconds, model, provider, finish}."""

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request

from stratamodels import keys
from stratamodels.runtimes import RunError

PARAMS = {
    "image",
    "images",
    "prompt",
    "preset",
    "context",
    "schema",
    "system",
    "system_preset",
    "max_tokens",
    "effort",
    "temperature",
    "seed",
    "think",
}
EFFORTS = ("low", "medium", "high")


MAX_B64 = (
    5 * 1024 * 1024
)  # the largest picture sent as it is (base64): the Claude API refuses one over 10 MB, NIM's limits are lower still
LONG_EDGE = 2576  # a bigger picture is scaled to this on its long side: the APIs scale it down anyway, so nothing they would read is lost


def encode(path):
    """(mime, base64) of a picture to send: as it is when it fits under MAX_B64, else a copy no longer than LONG_EDGE (a strata-scan page at
    full size is about 16 MB), as PNG, or JPEG when the PNG is still too big."""
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    if os.path.getsize(path) * 4 / 3 < MAX_B64:
        with open(path, "rb") as f:
            return mime, base64.b64encode(f.read()).decode()
    import io

    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("L" if im.mode in ("1", "L", "LA", "I", "I;16") else "RGB")
        if max(im.size) > LONG_EDGE:
            k = LONG_EDGE / max(im.size)
            im = im.resize((round(im.width * k), round(im.height * k)), Image.Resampling.LANCZOS)
        for fmt, kw in (("PNG", dict(optimize=True)), ("JPEG", dict(quality=90))):
            buf = io.BytesIO()
            im.save(buf, fmt, **kw)
            if len(buf.getvalue()) * 4 / 3 < MAX_B64 or fmt == "JPEG":
                return f"image/{fmt.lower()}", base64.b64encode(buf.getvalue()).decode()
    raise AssertionError("unreachable")  # pragma: no cover


# JSON-schema keywords that only bound a value: the vendors' structured outputs refuse them (Claude: "For 'integer' type, property 'minimum'
# is not supported"; Gemini: "Request contains an invalid argument") while llama.cpp's grammar takes them. The shape (types, properties,
# required, enum, items, additionalProperties) is what they enforce; the bounds stay in the prompt, and the eval checks the answer after.
BOUNDS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "maxItems",
    "uniqueItems",
    "$schema",
)


LITERAL = ("enum", "const", "default", "examples")


def portable(schema):
    """`schema` without BOUNDS anywhere in it, and with minItems only where it is 0 or 1 (Claude takes no other), for a vendor's API."""
    if isinstance(schema, list):
        return [portable(x) for x in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k in BOUNDS or (k == "minItems" and v not in (0, 1)):
            continue
        if k in LITERAL:  # data, not a schema: {"enum": [{"format": "png"}]} keeps its "format"
            out[k] = v
            continue
        out[k] = (
            {n: portable(x) for n, x in v.items()}
            if k in ("properties", "$defs", "definitions") and isinstance(v, dict)
            else portable(v)
        )
    return out


def parse(params):
    """{text, system, images: [(path, mime, base64)], schema, max_tokens, effort} from a job's params."""
    from stratamodels.runtimes import llama

    unknown = set(params) - PARAMS
    if unknown:
        raise RunError(f"unknown params {sorted(unknown)}")
    schema = params.get("schema")
    if params.get("preset"):
        if params.get("prompt"):
            raise RunError("give a prompt or a preset, not both")
        text, pschema = llama.preset(params["preset"])
        schema = pschema if schema is None else schema
        text = text.replace("{context}", json.dumps(params.get("context") or {}, ensure_ascii=False))
    else:
        text = params.get("prompt")
        if not isinstance(text, str) or not text.strip():
            raise RunError("give a prompt (or a preset)")
    system = params.get("system")
    if params.get("system_preset"):
        if system:
            raise RunError("give a system or a system_preset, not both")
        system = llama.preset(params["system_preset"])[0].strip()
    imgs = params.get("images") or ([params["image"]] if params.get("image") else [])
    if not isinstance(imgs, list):
        raise RunError("images is a list of paths")
    images = []
    for p in imgs:
        if not isinstance(p, str) or not os.path.isfile(p):
            raise RunError(f"no such image: {p!r}")
        images.append((p, *encode(p)))
    effort = params.get("effort")
    if effort is not None and effort not in EFFORTS:
        raise RunError(f"effort is one of {', '.join(EFFORTS)}")
    try:
        max_tokens = int(params.get("max_tokens") or 2048)
    except (TypeError, ValueError):
        raise RunError("max_tokens is a whole number")
    if not 1 <= max_tokens <= 128000:
        raise RunError("max_tokens must be between 1 and 128000")
    return dict(
        text=text,
        system=str(system) if system else None,
        images=images,
        schema=schema or None,
        max_tokens=max_tokens,
        effort=effort,
    )


def logged(job):
    """The request as the ledger keeps it: the text in full, pictures as their paths (never the bytes)."""
    return dict(
        system=job["system"],
        prompt=job["text"],
        images=[p for p, _, _ in job["images"]],
        schema=job["schema"],
        max_tokens=job["max_tokens"],
        effort=job["effort"],
    )


def answer(job, text):
    """{text, json?}: the JSON parsed when a schema was asked for."""
    out = dict(text=text)
    if job["schema"]:
        try:
            out["json"] = json.loads(text)
        except ValueError:
            pass
    return out


class HttpError(Exception):
    def __init__(self, status, body, retry_after=None):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status, self.body, self.retry_after = status, body, retry_after


def post(url, body, headers, timeout=600):
    """POST JSON, return the parsed reply; HttpError on a non-2xx answer (its text redacted of every key)."""
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"content-type": "application/json", **headers}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        text = keys.redact(e.read().decode("utf8", "replace"))
        ra = e.headers.get("retry-after") if e.headers else None
        raise HttpError(e.code, text, float(ra) if ra and ra.replace(".", "", 1).isdigit() else None)
    except (urllib.error.URLError, OSError) as e:
        raise RunError(keys.redact(f"could not reach {url.split('?')[0]}: {e}"))
