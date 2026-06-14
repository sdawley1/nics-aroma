#!/usr/bin/env python3

"""
Tests for the in-process batch runner using a mock backend.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from pathlib import Path
from typing import List, Tuple

# ----- third party -----
import numpy as np

# ----- local modules -----
from aroma.batch import scan_paths, select_rings
from aroma.io import load_geometry
from aroma.rings import is_planar

# ============================================================
# MOCK BACKEND
# ============================================================


class _UnitBackend:
    """Returns an identity shielding tensor at every probe."""

    def shielding(self, mol: object, ghosts: np.ndarray) -> np.ndarray:
        return np.broadcast_to(np.eye(3), (ghosts.shape[0], 3, 3)).copy()


# ============================================================
# TESTS
# ============================================================


def test_scan_paths_covers_every_ring(data_dir: Path) -> None:
    """Batch yields one result per ring across multiple geometries."""
    paths = [
        data_dir / "benzene/benzene.in",   # 1 ring
        data_dir / "phenalene/phenalene.in",  # 3 rings
    ]
    seen: List[Tuple[str, int]] = []
    for path, ring, result in scan_paths(
        paths, _UnitBackend(), start=0.0, stop=1.0, step=0.5
    ):
        seen.append((path.name, len(ring)))
        assert result.distances.shape == (3,)
        assert np.allclose(result.nics_iso, -1.0)  # -trace(I)/3

    assert seen.count(("benzene.in", 6)) == 1
    assert seen.count(("phenalene.in", 6)) == 3


def test_select_rings_planar_only(data_dir: Path) -> None:
    """`planar_only` keeps phenalene's three (planar) rings and they stay planar."""
    mol = load_geometry(data_dir / "phenalene/phenalene.in")
    rings = select_rings(mol, "auto", planar_only=True)
    assert len(rings) == 3
    assert all(is_planar(mol.coords, ring) for ring in rings)


def test_scan_paths_parallel_matches_serial(data_dir: Path) -> None:
    """Parallel batch (jobs=2) reproduces the serial results and their order."""
    paths = [
        data_dir / "benzene/benzene.in",      # 1 ring
        data_dir / "phenalene/phenalene.in",  # 3 rings
    ]
    serial = list(scan_paths(paths, _UnitBackend(), start=0.0, stop=1.0, step=0.5))
    parallel = list(
        scan_paths(paths, _UnitBackend(), start=0.0, stop=1.0, step=0.5, jobs=2)
    )

    assert [(p.name, ring) for p, ring, _ in parallel] == [
        (p.name, ring) for p, ring, _ in serial
    ]
    for (_, _, s), (_, _, par) in zip(serial, parallel):
        assert np.allclose(s.distances, par.distances)
        assert np.allclose(s.nics_iso, par.nics_iso)
        assert np.allclose(s.nics_zz, par.nics_zz)
