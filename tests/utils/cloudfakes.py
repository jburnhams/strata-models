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


class FakeBatches:
    """messages.batches: `create` records each batch's requests (or raises `error`); `status` is what retrieve says; `outcomes` (custom_id ->
    outcome, see succeeded/errored/ended) is what results yields once asked, or `broken` raised after the first."""

    def __init__(self):
        self.created, self.status, self.outcomes, self.error, self.broken = [], "in_progress", {}, None, None

    def create(self, requests):
        if self.error is not None:
            raise self.error
        self.created.append(requests)
        return SimpleNamespace(id=f"msgbatch_{len(self.created)}")

    def retrieve(self, batch_id):
        if isinstance(self.status, BaseException):
            raise self.status
        n = len(self.created[-1]) if self.created else 0
        done = self.status == "ended"
        return SimpleNamespace(
            id=batch_id,
            processing_status=self.status,
            request_counts=SimpleNamespace(
                processing=0 if done else n, succeeded=n if done else 0, errored=0, canceled=0, expired=0
            ),
        )

    def results(self, batch_id):
        for i, (cid, o) in enumerate(self.outcomes.items()):
            if self.broken is not None and i == 1:
                raise self.broken
            yield SimpleNamespace(custom_id=cid, result=o)


def succeeded(reply):
    return SimpleNamespace(type="succeeded", message=reply)


def errored(kind="invalid_request_error", message="bad image"):
    return SimpleNamespace(
        type="errored", error=SimpleNamespace(type="error", error=SimpleNamespace(type=kind, message=message))
    )


def ended(kind="expired"):
    return SimpleNamespace(type=kind)


class FakeMessages:
    def __init__(self, count=100, reply=None, error=None):
        self.count, self.reply, self.error, self.created = count, reply, error, []
        self.batches = FakeBatches()

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
