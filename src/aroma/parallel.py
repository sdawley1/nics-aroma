#!/usr/bin/env python3

"""
Portable, single-node parallel execution for independent scans.

Independent ``(geometry, ring)`` shielding calculations are embarrassingly
parallel: each is a self-contained SCF. This module maps such work across worker
processes with :class:`concurrent.futures.ProcessPoolExecutor`, capping each
worker's PySCF/BLAS thread count so N processes do not oversubscribe the cores.
Pure standard-library multiprocessing — no cluster scheduler is assumed.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Sequence, TypeVar

# ============================================================
# TYPES
# ============================================================

_T = TypeVar("_T")
_R = TypeVar("_R")

# ============================================================
# JOB / THREAD RESOLUTION
# ============================================================


def resolve_jobs(jobs: int) -> int:
    """Resolve a ``--jobs`` value to a concrete worker count (>= 1).

    ``jobs <= 0`` means "use every core"; any positive value is taken as-is.
    """
    if jobs <= 0:
        return max(1, os.cpu_count() or 1)
    return jobs


def resolve_threads(threads: int, n_workers: int) -> int:
    """Resolve per-worker thread count, auto-splitting cores when unset.

    Parameters
    ----------
    threads : int
        Explicit per-worker thread count; ``<= 0`` auto-derives it.
    n_workers : int
        Number of worker processes that will run concurrently.

    Returns
    -------
    int
        Threads per worker (>= 1). Auto value is ``cores // n_workers`` so the
        workers together do not exceed the physical core count.
    """
    assert n_workers >= 1, "n_workers must be at least 1"
    if threads > 0:
        return threads
    return max(1, (os.cpu_count() or 1) // n_workers)


def _init_worker(threads: int) -> None:
    """Pin a worker's thread pools so concurrent workers do not oversubscribe."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    # PySCF reads its thread count at runtime, so this is effective regardless of
    # import order. Skipped cleanly when PySCF is not installed (e.g. mock tests).
    try:
        from pyscf import lib

        lib.num_threads(threads)
    except ImportError:
        pass


# ============================================================
# PARALLEL MAP
# ============================================================


def parallel_map(
    func: Callable[[_T], _R],
    items: Sequence[_T],
    jobs: int = 1,
    threads: int = 0,
) -> List[_R]:
    """Apply ``func`` to each item, optionally across worker processes.

    Parameters
    ----------
    func : callable
        A top-level (picklable) function taking one item and returning a result.
    items : sequence
        Work items; each must be picklable when ``jobs`` selects > 1 worker.
    jobs : int
        Worker count (``<= 0`` = all cores). ``1`` runs in-process with no pool.
    threads : int
        Per-worker thread count (``<= 0`` auto-splits the cores).

    Returns
    -------
    list
        Results in the same order as ``items``.
    """
    assert items is not None, "items must not be None"
    if not items:
        return []

    n_workers = min(resolve_jobs(jobs), len(items))
    if n_workers == 1:
        return [func(item) for item in items]

    per_worker = resolve_threads(threads, n_workers)
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker, initargs=(per_worker,)
    ) as executor:
        return list(executor.map(func, items))
