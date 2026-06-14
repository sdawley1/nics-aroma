#!/usr/bin/env python3

"""
Batch ("super") runner: NICS scans over many geometries in one process.

Replaces the legacy .suarm shell-driven multi-job runner. Rings are auto-
perceived per geometry and each is scanned in-process; no subprocesses are
spawned. Ring selection lives here so the CLI can reuse it without a cycle.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from pathlib import Path
from typing import Iterator, List, Tuple

# ----- local modules -----
from aroma.backend.base import ShieldingBackend
from aroma.connectivity import adjacency_list, bond_matrix
from aroma.constants import DEFAULT_BQ_RANGE, DEFAULT_BQ_STEP
from aroma.io import load_geometry, log_has_bq_shielding, read_log_nics
from aroma.molecule import Molecule
from aroma.nics import (
    NicsResult,
    XyNicsResult,
    match_probe_ring,
    nics_from_precomputed,
    run_nics_scan,
    run_xy_scan,
)
from aroma.parallel import parallel_map
from aroma.rings import find_rings, is_planar, order_ring

# ============================================================
# JOB TYPES
# ============================================================

# A unit of axial-scan work: (path, molecule, ring, backend, start, stop, step).
AxialJob = Tuple[Path, Molecule, List[int], ShieldingBackend, float, float, float]
# A unit of XY-scan work: (path, molecule, ring, backend, half_extent, step, height).
XyJob = Tuple[Path, Molecule, List[int], ShieldingBackend, float, float, float]

# ============================================================
# RING SELECTION
# ============================================================


def select_rings(
    mol: Molecule, spec: str = "auto", planar_only: bool = False
) -> List[List[int]]:
    """Return ordered rings, auto-perceived or parsed from a 1-based spec.

    Parameters
    ----------
    mol : Molecule
        Geometry to analyze.
    spec : str
        ``"auto"`` to perceive all rings, or comma-separated 1-based atom
        indices naming a single ring.
    planar_only : bool
        When ``True`` (auto mode only), keep only planar rings, approximating
        an aromatic-ring filter.

    Returns
    -------
    rings : list of ordered node lists
    """
    adj = adjacency_list(bond_matrix(mol))
    if spec == "auto":
        rings = find_rings(adj)
        assert rings, "no rings detected; specify the ring explicitly"
        if planar_only:
            rings = [r for r in rings if is_planar(mol.coords, r)]
            assert rings, "no planar rings detected; rerun without --planar-only"
        return rings
    atoms = [int(tok) - 1 for tok in spec.split(",")]
    assert all(0 <= a < mol.n_atoms for a in atoms), "ring atom index out of range"
    return [order_ring(adj, atoms)]


# ============================================================
# PARALLEL WORKERS
# ============================================================
#
# These run in worker processes, so they must be importable module-level
# functions (the spawn start method pickles them by qualified name).


def _axial_job(job: AxialJob) -> Tuple[Path, List[int], NicsResult]:
    """Run one axial NICS scan; return (path, ring, result)."""
    path, mol, ring, backend, start, stop, step = job
    return path, ring, run_nics_scan(mol, ring, backend, start, stop, step)


def _xy_job(job: XyJob) -> Tuple[Path, List[int], XyNicsResult]:
    """Run one in-plane (XY) NICS scan; return (path, ring, result)."""
    path, mol, ring, backend, half_extent, step, height = job
    return path, ring, run_xy_scan(mol, ring, backend, half_extent, step, height)


# ============================================================
# BATCH SCANNING
# ============================================================


def scan_paths(
    paths: List[Path],
    backend: ShieldingBackend,
    start: float = DEFAULT_BQ_RANGE[0],
    stop: float = DEFAULT_BQ_RANGE[1],
    step: float = DEFAULT_BQ_STEP,
    planar_only: bool = False,
    jobs: int = 1,
    threads: int = 0,
) -> Iterator[Tuple[Path, List[int], NicsResult]]:
    """Yield (path, ring, result) for every perceived ring of every geometry.

    Parameters
    ----------
    paths : list of Path
        Geometry files to process in order.
    backend : ShieldingBackend
        Shared shielding backend.
    start, stop, step : float
        Axial scan grid parameters (angstrom).
    planar_only : bool
        Restrict each geometry's rings to the planar ones.
    jobs : int
        Worker processes for the independent scans (``<= 0`` = all cores;
        ``1`` runs serially in-process).
    threads : int
        Per-worker PySCF/BLAS thread count (``<= 0`` auto-splits the cores).

    Yields
    ------
    (path, ring, result) : Tuple[Path, List[int], NicsResult]
        One tuple per ring, in file then ring order.
    """
    assert paths, "no geometries to process"
    # Ring perception is cheap and stays in the parent; only the SCF-bound scans
    # are dispatched to workers. Logs that already carry Bq shielding are
    # reported inline (no recompute) and skip the backend entirely.
    worklist: List[AxialJob] = []
    for path in paths:
        if path.suffix.lower() in (".log", ".out") and log_has_bq_shielding(path):
            data = read_log_nics(path)
            rings = select_rings(data.mol, "auto", planar_only)
            ring, _ = match_probe_ring(data.mol, rings, data.bq_coords)
            yield path, ring, nics_from_precomputed(
                data.mol, ring, data.bq_coords, data.bq_tensors
            )
            continue
        mol = load_geometry(path)
        for ring in select_rings(mol, "auto", planar_only):
            worklist.append((path, mol, ring, backend, start, stop, step))
    yield from parallel_map(_axial_job, worklist, jobs=jobs, threads=threads)
