"""Giving this process's free memory back to the system.

The service runs each job on a thread of its own, and glibc keeps an arena per thread (up to 8 per core) whose freed memory stays with the
process: on omarchy 7.6 GB sat in such arenas after a night of small jobs (2026-11-25). `tune` (at start, before the threads) keeps the arenas few
and makes big arrays (page scans, masks) mappings of their own, which go back when freed; `trim` returns the free pages of the rest. glibc only:
elsewhere both do nothing."""

import ctypes
import gc
import sys

ARENAS = 4
MMAP_BYTES = (
    1 << 20
)  # an allocation this big is mapped on its own; naming it also stops glibc raising it to 32 MB as big ones are freed
M_MMAP_THRESHOLD, M_ARENA_MAX = -3, -8  # malloc.h


class _Mallinfo2(ctypes.Structure):
    _fields_ = [
        (n, ctypes.c_size_t)
        for n in [
            "arena",
            "ordblks",
            "smblks",
            "hblks",
            "hblkhd",
            "usmblks",
            "fsmblks",
            "uordblks",
            "fordblks",
            "keepcost",
        ]
    ]


def _glibc():
    if not sys.platform.startswith("linux"):
        return None
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.gnu_get_libc_version  # noqa: B018 (musl has none, and its mallopt does nothing)
        libc.mallinfo2.restype = _Mallinfo2
        return libc
    except (OSError, AttributeError):
        return None


def tune(libc=None):
    """Set the allocator's limits for this process; whether it could."""
    libc = libc or _glibc()
    if libc is None:
        return False
    return bool(libc.mallopt(M_ARENA_MAX, ARENAS)) and bool(libc.mallopt(M_MMAP_THRESHOLD, MMAP_BYTES))


def trim(collect=False, libc=None):
    """Give free pages back to the system (tens of ms); `collect` first frees what only cycles hold (an unloaded model)."""
    if collect:
        gc.collect()
    libc = libc or _glibc()
    if libc is None:
        return False
    libc.malloc_trim(ctypes.c_size_t(0))
    return True


def snapshot(libc=None):
    """MB of this process: resident and swapped, and of the allocator's: in use, free (pages already given
    back by a trim included), mapped apart (None where unknown)."""
    d = dict(rss_mb=None, swap_mb=None, malloc_used_mb=None, malloc_free_mb=None, malloc_mapped_mb=None)
    try:
        with open("/proc/self/status", encoding="utf8") as f:
            kb = {k: v.split()[0] for k, _, v in (line.partition(":") for line in f) if k in ("VmRSS", "VmSwap")}
        d.update(rss_mb=int(kb["VmRSS"]) // 1024, swap_mb=int(kb["VmSwap"]) // 1024)
    except (OSError, KeyError):
        pass
    libc = libc or _glibc()
    if libc is not None:
        m = libc.mallinfo2()
        d.update(malloc_used_mb=m.uordblks >> 20, malloc_free_mb=m.fordblks >> 20, malloc_mapped_mb=m.hblkhd >> 20)
    return d
