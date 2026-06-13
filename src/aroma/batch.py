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
from aroma.io import load_geometry
from aroma.molecule import Molecule
from aroma.nics import NicsResult, run_nics_scan
from aroma.rings import find_rings, is_planar, order_ring

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
# BATCH SCANNING
# ============================================================


def scan_paths(
    paths: List[Path],
    backend: ShieldingBackend,
    start: float = DEFAULT_BQ_RANGE[0],
    stop: float = DEFAULT_BQ_RANGE[1],
    step: float = DEFAULT_BQ_STEP,
    planar_only: bool = False,
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

    Yields
    ------
    (path, ring, result) : Tuple[Path, List[int], NicsResult]
        One tuple per ring, in file then ring order.
    """
    assert paths, "no geometries to process"
    for path in paths:
        mol = load_geometry(path)
        for ring in select_rings(mol, "auto", planar_only):
            yield path, ring, run_nics_scan(mol, ring, backend, start, stop, step)
