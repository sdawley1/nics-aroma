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
from aroma.geometry import rodrigues_matrix
from aroma.io import load_geometry
from aroma.molecule import Molecule
from aroma.nics import nics_from_precomputed, run_nics_scan, run_xy_scan
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


# ============================================================
# PRE-COMPUTED PATH
# ============================================================


def _tilted_ring(theta_deg: float, dists: np.ndarray) -> tuple:
    """A planar hexagon tilted by theta about x, with probes on its normal."""
    ang = np.deg2rad(np.arange(6) * 60.0)
    hexagon = np.stack(
        [1.4 * np.cos(ang), 1.4 * np.sin(ang), np.zeros(6)], axis=1
    )
    tilt = rodrigues_matrix(np.array([1.0, 0.0, 0.0]), np.deg2rad(theta_deg))
    coords = hexagon @ tilt.T
    normal = tilt @ np.array([0.0, 0.0, 1.0])
    bq = np.outer(dists, normal)
    mol = Molecule(numbers=np.full(6, 6, dtype=np.int_), coords=coords)
    return mol, [0, 1, 2, 3, 4, 5], bq, normal


def test_precomputed_tensor_rotation() -> None:
    """zz is the shielding projected onto the ring normal, not the global z.

    The tensor is diagonal in the global frame (diag(a, a, b)); reusing it for a
    tilted ring must rotate it so NICS_zz = -(a*sin^2 + b*cos^2) about the normal.
    A missing rotation would wrongly give the global -b.
    """
    a, b = 10.0, 40.0
    sigma = np.diag([a, a, b]).astype(np.float64)
    dists = np.array([0.5, 1.0, 1.5])
    for theta in (0.0, 30.0, 70.0):
        mol, ring, bq, normal = _tilted_ring(theta, dists)
        tensors = np.broadcast_to(sigma, (dists.size, 3, 3)).copy()
        result = nics_from_precomputed(mol, ring, bq, tensors)
        nz2 = float(normal[2]) ** 2
        assert np.allclose(result.nics_iso, -(2 * a + b) / 3.0)
        assert np.allclose(result.nics_zz, -(a * (1.0 - nz2) + b * nz2))
        assert np.allclose(np.sort(np.abs(result.distances)), np.sort(dists))


def test_precomputed_isotropic_is_negative_trace() -> None:
    """NICS_iso equals -trace/3 for arbitrary tensors, regardless of frame."""
    mol, ring, bq, _ = _tilted_ring(25.0, np.array([0.0, 0.7, 1.3, 1.9]))
    rng = np.random.default_rng(1)
    base = rng.standard_normal((4, 3, 3))
    tensors = base + base.transpose(0, 2, 1)
    result = nics_from_precomputed(mol, ring, bq, tensors)
    expected = -np.trace(tensors, axis1=1, axis2=2) / 3.0
    assert np.allclose(np.sort(result.nics_iso), np.sort(expected))
