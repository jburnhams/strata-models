"""Giving memory back (stratamodels.memory): the allocator calls made, and nothing where there is no glibc."""

import types

from stratamodels import memory


class Libc:
    def __init__(self, ok=1):
        self.ok, self.calls = ok, []

    def mallopt(self, k, v):
        self.calls.append((k, v))
        return self.ok

    def malloc_trim(self, pad):
        self.calls.append("trim")

    def mallinfo2(self):
        return types.SimpleNamespace(uordblks=5 << 20, fordblks=7 << 20, hblkhd=2 << 20)


def test_tune_limits_the_arenas_and_maps_big_arrays():
    c = Libc()
    assert memory.tune(c) is True
    assert c.calls == [(memory.M_ARENA_MAX, memory.ARENAS), (memory.M_MMAP_THRESHOLD, memory.MMAP_BYTES)]
    assert memory.tune(Libc(ok=0)) is False


def test_trim_collects_first_only_when_asked(monkeypatch):
    seen = []
    monkeypatch.setattr(memory.gc, "collect", lambda: seen.append("gc"))
    c = Libc()
    assert memory.trim(libc=c) is True and seen == [] and c.calls == ["trim"]
    assert memory.trim(collect=True, libc=c) is True and seen == ["gc"]


def test_snapshot_reads_the_allocator():
    d = memory.snapshot(Libc())
    assert (d["malloc_used_mb"], d["malloc_free_mb"], d["malloc_mapped_mb"]) == (5, 7, 2)


def test_without_glibc_nothing_happens(monkeypatch):
    monkeypatch.setattr(memory.sys, "platform", "darwin")
    assert memory.tune() is False and memory.trim() is False and memory.snapshot()["malloc_used_mb"] is None


def test_a_libc_without_the_calls_is_not_glibc(monkeypatch):
    monkeypatch.setattr(memory.sys, "platform", "linux")
    monkeypatch.setattr(memory.ctypes, "CDLL", lambda name: types.SimpleNamespace())
    assert memory._glibc() is None
