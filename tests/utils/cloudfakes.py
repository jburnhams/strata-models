"""Doubles for the cloud API runtimes: a ledger bound to an in-memory store, a key file, a scripted HTTP post, a fake anthropic client."""

import os
from types import SimpleNamespace

import pytest

from stratamodels import settings, spend
from stratamodels.store import memory_store


@pytest.fixture
def ledger(monkeypatch):
    """spend bound to a fresh in-memory store (settings in the same database), reset after the test."""
    st = memory_store(clock=lambda: 1_800_000_000.0)
    settings.bind(st.engine)
    spend.bind(st)
    monkeypatch.setattr(spend, "_reserved", {})
    yield st
    spend.bind(None)
    settings.bind(None)


def key_file(tmp_path, monkeypatch, **names):
    """A chmod 600 secrets file holding NAME=value lines, pointed to by STRATAMODELS_SECRETS."""
    p = tmp_path / "secrets.env"
    p.write_text("".join(f"{k}={v}\n" for k, v in names.items()))
    os.chmod(p, 0o600)
    monkeypatch.setenv("STRATAMODELS_SECRETS", str(p))
    return p


class Post:
    """A scripted cloud.post: each call pops the next reply (a dict, or an exception to raise) and is recorded as (url, body, headers)."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, url, body, headers, timeout=600):
        self.calls.append((url, body, headers))
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


class FakeMessages:
    def __init__(self, count=100, reply=None, error=None):
        self.count, self.reply, self.error, self.created = count, reply, error, []

    def count_tokens(self, **kw):
        if isinstance(self.count, BaseException):
            raise self.count
        return SimpleNamespace(input_tokens=self.count)

    def create(self, **kw):
        self.created.append(kw)
        if self.error is not None:
            raise self.error
        return self.reply


def claude_reply(
    text="ok",
    stop="end_turn",
    tokens_in=100,
    tokens_out=10,
    cache_read=0,
    cache_write=0,
    model="claude-haiku-5-5",
    category=None,
):
    return SimpleNamespace(
        model=model,
        stop_reason=stop,
        stop_details=SimpleNamespace(category=category) if category else None,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ),
    )


class FakeAnthropic:
    def __init__(self, **kw):
        self.messages = FakeMessages(**kw)
        self.closed = False

    def close(self):
        self.closed = True
