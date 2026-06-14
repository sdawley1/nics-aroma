#!/usr/bin/env python3

"""
Tests for the generic parallel-map helper.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import os

# ----- local modules -----
from aroma.parallel import parallel_map, resolve_jobs, resolve_threads

# ============================================================
# MODULE-LEVEL WORKER (must be picklable for the spawn start method)
# ============================================================


def _square(x: int) -> int:
    """Return x squared."""
    return x * x


# ============================================================
# TESTS
# ============================================================


def test_resolve_jobs() -> None:
    """Non-positive jobs means all cores; positive is taken as-is."""
    assert resolve_jobs(0) == (os.cpu_count() or 1)
    assert resolve_jobs(-1) == (os.cpu_count() or 1)
    assert resolve_jobs(3) == 3


def test_resolve_threads_autosplit() -> None:
    """Threads auto-split across workers but never drop below one."""
    cores = os.cpu_count() or 1
    assert resolve_threads(0, 1) == cores
    assert resolve_threads(0, cores) == 1
    assert resolve_threads(2 * cores, 4) == 2 * cores  # explicit override wins


def test_parallel_map_serial_matches_parallel() -> None:
    """jobs=1 (in-process) and jobs=2 (pool) give identical, ordered results."""
    items = list(range(8))
    expected = [x * x for x in items]
    assert parallel_map(_square, items, jobs=1) == expected
    assert parallel_map(_square, items, jobs=2) == expected


def test_parallel_map_empty() -> None:
    """An empty work list yields an empty result with no workers spawned."""
    assert parallel_map(_square, [], jobs=4) == []
