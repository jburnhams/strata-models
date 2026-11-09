"""Hello-world domain logic and persistent message log."""
import json, os
from datetime import datetime, timezone

from app import oslib, settings


class GreetingError(ValueError):
    """Raised when a greeting request is invalid."""


def format_greeting(name: str = 'World', greeting: str | None = None) -> dict:
    """Format a greeting for `name` using `greeting` (or the configured default)."""
    clean_name = (name or '').strip() or 'World'
    clean_greeting = (greeting or '').strip() or settings.greeting()
    if len(clean_name) > 120: raise GreetingError('name is too long (at most 120 characters)')
    if len(clean_greeting) > 60: raise GreetingError('greeting is too long (at most 60 characters)')
    return dict(name=clean_name, greeting=clean_greeting, message=f'{clean_greeting}, {clean_name}!')


def _store_path() -> str:
    return os.path.join(settings.data_dir(), 'messages.json')


def list_messages() -> list[dict]:
    """Return the recorded greetings from `<data_dir>/messages.json` (newest first)."""
    p = _store_path()
    if not os.path.exists(p): return []
    try:
        with open(p, encoding='utf8') as f: data = json.load(f)
    except (OSError, ValueError): return []
    return data if isinstance(data, list) else []


def add_message(name: str = 'World', greeting: str | None = None) -> dict:
    """Create and persist a greeting entry; returns the created record."""
    rec = format_greeting(name, greeting)
    lock_path = _store_path() + '.lock'
    os.makedirs(settings.data_dir(), exist_ok=True)
    with open(lock_path, 'a+', encoding='utf8') as lf, oslib.file_lock(lf):
        items = list_messages()
        entry = dict(id=len(items) + 1, **rec, created=datetime.now(timezone.utc).isoformat())
        items.insert(0, entry)
        tmp = _store_path() + '.tmp'
        with open(tmp, 'w', encoding='utf8') as f: json.dump(items, f, indent=1)
        os.replace(tmp, _store_path())
    return entry


def clear_messages() -> dict:
    """Remove all recorded greetings; returns `{cleared: count}`."""
    lock_path = _store_path() + '.lock'
    os.makedirs(settings.data_dir(), exist_ok=True)
    with open(lock_path, 'a+', encoding='utf8') as lf, oslib.file_lock(lf):
        n = len(list_messages())
        p = _store_path()
        if os.path.exists(p): os.remove(p)
    return dict(cleared=n)
