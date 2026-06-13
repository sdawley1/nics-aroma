#!/usr/bin/env python3

"""
Fast NICS-driver tests using a synthetic (mock) shielding backend.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from pathlib import Path

# ----- third party -----
import numpy as np

# ----- local modules -----
from aroma.connectivity import adjacency_list, bond_matrix
from aroma.io import load_geometry
from aroma.nics import run_nics_scan, run_xy_scan
from aroma.rings import find_rings

# ============================================================
# MOCK BACKEND
# ============================================================


class _ConstantBackend:
    """Returns a fixed shielding tensor at every probe point."""

    def __init__(self, in_plane: float, out_of_plane: float) -> None:
        self._tensor = np.diag([in_plane, in_plane, out_of_plane]).astype(np.float64)

    def shielding(
        self, mol: object, ghosts: np.ndarray
    ) -> np.ndarray:
        return np.broadcast_to(self._tensor, (ghosts.shape[0], 3, 3)).copy()


# ============================================================
# TESTS
# ============================================================


def test_scan_sign_conventions(data_dir: Path) -> None:
    """NICS components are the negated shielding principal values."""
    mol = load_geometry(data_dir / "benzene/benzene.in")
    adj = adjacency_list(bond_matrix(mol))
    ring = find_rings(adj)[0]

    backend = _ConstantBackend(in_plane=10.0, out_of_plane=40.0)
    result = run_nics_scan(mol, ring, backend, start=0.0, stop=1.0, step=0.5)

    assert result.distances.shape == (3,)
    assert np.allclose(result.distances, [0.0, 0.5, 1.0])
    assert np.allclose(result.nics_iso, -(10.0 + 10.0 + 40.0) / 3.0)
    assert np.allclose(result.nics_zz, -40.0)
    assert np.allclose(result.nics_oop, -40.0)
    assert np.allclose(result.nics_inp, -10.0)


def test_xy_scan_grid_and_signs(data_dir: Path) -> None:
    """The in-plane scan covers the full lattice and negates the shielding."""
    mol = load_geometry(data_dir / "benzene/benzene.in")
    adj = adjacency_list(bond_matrix(mol))
    ring = find_rings(adj)[0]

    backend = _ConstantBackend(in_plane=10.0, out_of_plane=40.0)
    result = run_xy_scan(mol, ring, backend, half_extent=1.0, step=0.5, height=1.7)

    assert result.xs.shape == (25,)  # 5 x 5 lattice
    assert result.ys.shape == (25,)
    assert np.allclose(result.nics_zz, -40.0)
    assert np.allclose(result.nics_inp, -10.0)
