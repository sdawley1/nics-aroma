#!/usr/bin/env python3

"""
Tests for geometric primitives, reorientation, and Bq grids.

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
from aroma.geometry import polygon_area, rodrigues_matrix
from aroma.grid import axial_grid, xy_scan_grid
from aroma.io import load_geometry
from aroma.molecule import Molecule
from aroma.reorient import reorient_ring_to_xy
from aroma.rings import find_rings, is_planar

# ============================================================
# GEOMETRY PRIMITIVES
# ============================================================


def test_rodrigues_is_rotation() -> None:
    """A Rodrigues matrix is orthogonal with determinant +1."""
    r = rodrigues_matrix(np.array([1.0, 2.0, 3.0]), 0.7)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_benzene_ring_is_planar(data_dir: Path) -> None:
    """The benzene ring passes the planarity test; the area is ~5.2 A^2."""
    mol = load_geometry(data_dir / "benzene/benzene.in")
    adj = adjacency_list(bond_matrix(mol))
    ring = find_rings(adj)[0]
    assert is_planar(mol.coords, ring)
    assert 4.5 < polygon_area(mol.coords, ring) < 6.0


def test_bond_matrix_handles_heavy_elements() -> None:
    """Elements heavier than the radius table (e.g. Br, Z=35) must not overflow."""
    mol = Molecule(
        numbers=np.array([35, 35], dtype=np.int_),
        coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]], dtype=np.float64),
    )
    bonds = bond_matrix(mol)
    assert bonds.shape == (2, 2)
    assert bool(bonds[0, 1]) and bool(bonds[1, 0])


# ============================================================
# REORIENTATION
# ============================================================


def test_reorient_places_ring_in_xy(data_dir: Path) -> None:
    """After reorientation the ring is centered at the origin in the XY plane."""
    mol = load_geometry(data_dir / "phenalene/phenalene.in")
    adj = adjacency_list(bond_matrix(mol))
    ring = find_rings(adj)[0]
    reoriented, rotation = reorient_ring_to_xy(mol, ring)

    ring_pts = reoriented.coords[ring]
    assert np.allclose(ring_pts[:, 2], 0.0, atol=1e-6)
    assert np.allclose(ring_pts.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_reorient_tolerates_nonplanar_input(data_dir: Path) -> None:
    """A slightly non-planar ring (pyrrole) reorients without error."""
    mol = load_geometry(data_dir / "pyrrole/pyrrole.in")
    adj = adjacency_list(bond_matrix(mol))
    ring = find_rings(adj)[0]
    reoriented, _ = reorient_ring_to_xy(mol, ring)
    # Residual out-of-plane spread stays tiny (input noise ~1e-4 A).
    assert np.max(np.abs(reoriented.coords[ring][:, 2])) < 1e-2


def test_reorient_preserves_bond_lengths(data_dir: Path) -> None:
    """Reorientation is rigid: pairwise distances are unchanged."""
    mol = load_geometry(data_dir / "benzene/benzene.in")
    adj = adjacency_list(bond_matrix(mol))
    ring = find_rings(adj)[0]
    reoriented, _ = reorient_ring_to_xy(mol, ring)
    before = np.linalg.norm(mol.coords[0] - mol.coords[1])
    after = np.linalg.norm(reoriented.coords[0] - reoriented.coords[1])
    assert np.isclose(before, after)


# ============================================================
# GRIDS
# ============================================================


def test_axial_grid_default_count() -> None:
    """The default 0-4 A grid at 0.1 A has 41 inclusive points along +z."""
    grid = axial_grid()
    assert grid.shape == (41, 3)
    assert np.allclose(grid[:, :2], 0.0)
    assert np.isclose(grid[0, 2], 0.0) and np.isclose(grid[-1, 2], 4.0)


def test_xy_scan_grid_shape() -> None:
    """The XY grid is a square lattice at the requested height."""
    grid = xy_scan_grid(half_extent=1.0, step=0.5, height=1.7)
    assert grid.shape == (25, 3)
    assert np.allclose(grid[:, 2], 1.7)
