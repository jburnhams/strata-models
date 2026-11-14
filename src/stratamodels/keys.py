"""API keys for the cloud backends (Claude API, NVIDIA NIM, Google AI Studio), kept out of the repository, the database and the logs.

A key comes from its environment variable, else from the secrets file: `~/.config/strata-models/secrets.env` (STRATAMODELS_SECRETS moves it),
lines of NAME=value, `#` comments. On Linux and macOS the file must be readable by its owner only (chmod 600): a key in a file anyone on the
machine can read is refused, not used. Nothing here prints a key; `redact` takes every known key out of a text before it is shown.

  ANTHROPIC_API_KEY   claude-api
  NVIDIA_API_KEY      nvidia-nim
  GEMINI_FREE_API_KEY gemini-api, the free tier (tried first)
  GEMINI_API_KEY      gemini-api, the paid key (after the free tier)"""

import os
import stat

NAMES = {
    "claude": "ANTHROPIC_API_KEY",
    "nim": "NVIDIA_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "gemini-free": "GEMINI_FREE_API_KEY",
}


class MissingKey(LookupError):
    """No usable key for a provider (none set, or the secrets file is open to others)."""


def path():
    return os.environ.get("STRATAMODELS_SECRETS") or os.path.join(
        os.path.expanduser("~"), ".config", "strata-models", "secrets.env"
    )


def _file():
    p = path()
    try:
        st = os.stat(p)
    except OSError:
        return {}
    if os.name != "nt" and st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise MissingKey(f"{p} can be read by other users: chmod 600 it")
    out = {}
    with open(p, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            out[k.strip().removeprefix("export ").strip()] = v
    return out


def get(provider):
    """The key for a provider ("claude", "nim", "gemini", "gemini-free"); raises MissingKey naming where to put it."""
    name = NAMES[provider]
    v = os.environ.get(name) or _file().get(name)
    if not v:
        raise MissingKey(f"no {name}: set it in the environment or in {path()} (chmod 600)")
    return v


def have(provider):
    """Whether a key is set (never the key)."""
    try:
        get(provider)
        return True
    except MissingKey:
        return False


def redact(text):
    """`text` with every key we know replaced by [REDACTED] (error messages from the vendors' SDKs can echo a request)."""
    s = str(text)
    vals = [os.environ.get(n) for n in NAMES.values()]
    try:
        vals += list(_file().values())
    except MissingKey:
        pass
    for v in vals:
        if v and len(v) >= 8:
            s = s.replace(v, "[REDACTED]")
    return s
