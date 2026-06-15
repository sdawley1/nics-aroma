#!/usr/bin/env python3

"""
Fast sigma-only NICS_pizz tests using a synthetic (mock) shielding backend.

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
import pytest

# ----- local modules -----
from aroma.connectivity import adjacency_list, bond_matrix
from aroma.io import load_geometry
from aroma.molecule import Molecule
from aroma.reorient import reorient_ring_to_xy
from aroma.rings import find_rings
from aroma.sigma_only import (
    SigmaOnlyResult,
    fit_pi_zz,
    perceive_pi_centers,
    run_sigma_only_scan,
    sigma_only_model,
)

# ============================================================
# MOCK BACKEND
# ============================================================


class _AtomCountBackend:
    """Returns ``diag(n, n, 2n)`` (n = atom count) so real and model differ."""

    def shielding(self, mol: Molecule, ghosts: np.ndarray) -> np.ndarray:
        n = float(mol.n_atoms)
        tensor = np.diag([n, n, 2.0 * n]).astype(np.float64)
        return np.broadcast_to(tensor, (ghosts.shape[0], 3, 3)).copy()


# ============================================================
# HELPERS
# ============================================================


def _benzene(data_dir: Path):
    """Load benzene and its single ring."""
    mol = load_geometry(data_dir / "benzene/benzene.in")
    ring = find_rings(adjacency_list(bond_matrix(mol)))[0]
    return mol, ring


# ============================================================
# PI-CENTER / MODEL TESTS
# ============================================================


def test_perceive_pi_centers_benzene(data_dir: Path) -> None:
    """All six benzene carbons are pi-centers."""
    mol, _ = _benzene(data_dir)
    centers = perceive_pi_centers(mol)
    assert len(centers) == 6
    assert all(int(mol.numbers[c]) == 6 for c in centers)


def test_sigma_only_model_geometry(data_dir: Path) -> None:
    """One H is added per pi-center, 1 A below its carbon along the ring normal."""
    mol, ring = _benzene(data_dir)
    reoriented, _ = reorient_ring_to_xy(mol, ring)
    centers = perceive_pi_centers(mol)

    model = sigma_only_model(reoriented, centers, h_distance=1.0)
    assert model.n_atoms == reoriented.n_atoms + len(centers)
    assert np.all(model.numbers[-len(centers):] == 1)  # appended atoms are H
    for k, c in enumerate(centers):
        h = model.coords[reoriented.n_atoms + k]
        assert np.allclose(h[:2], reoriented.coords[c][:2])          # same x, y
        assert np.isclose(h[2], reoriented.coords[c][2] - 1.0)        # z - h_distance


def test_sigma_only_opposite_face(data_dir: Path) -> None:
    """Localizing H's sit on -z, opposite the +z probe grid."""
    mol, ring = _benzene(data_dir)
    reoriented, _ = reorient_ring_to_xy(mol, ring)
    model = sigma_only_model(reoriented, perceive_pi_centers(mol))
    added = model.coords[reoriented.n_atoms:]
    assert np.all(added[:, 2] < 0.0)  # ring centroid is at z=0 after reorientation


def test_sigma_only_charge_parity(data_dir: Path) -> None:
    """Charge bumps by one only for an odd number of added hydrogens."""
    mol, ring = _benzene(data_dir)
    reoriented, _ = reorient_ring_to_xy(mol, ring)
    centers = perceive_pi_centers(mol)  # 6 (even)

    even = sigma_only_model(reoriented, centers)
    assert even.charge == reoriented.charge
    assert even.mult == 1

    odd = sigma_only_model(reoriented, centers[:5])  # 5 (odd)
    assert odd.charge == reoriented.charge + 1
    assert odd.mult == 1


# ============================================================
# DRIVER TESTS
# ============================================================


def test_sigma_only_driver_subtracts(data_dir: Path) -> None:
    """NICS_pizz is the delocalized-minus-model zz; the deviation is consistent."""
    mol, ring = _benzene(data_dir)
    backend = _AtomCountBackend()
    result = run_sigma_only_scan(mol, ring, backend, start=0.0, stop=1.0, step=0.5)

    n_real = mol.n_atoms
    n_model = n_real + len(result.pi_centers)
    # zz = -tensor[2,2] = -2n; confirms real and model were not swapped.
    assert np.allclose(result.nics_zz_real, -2.0 * n_real)
    assert np.allclose(result.nics_zz_model, -2.0 * n_model)
    assert np.allclose(result.nics_pi_zz, result.nics_zz_real - result.nics_zz_model)
    assert np.allclose(
        result.three_delta_iso, 3.0 * (result.nics_iso_real - result.nics_iso_model)
    )
    assert np.allclose(
        result.som_deviation, result.nics_pi_zz - result.three_delta_iso
    )


def test_fit_pi_zz_recovers_cubic() -> None:
    """fit_pi_zz returns the underlying cubic's value at 1 A."""
    coeffs = [0.5, -2.0, 1.0, 3.0]
    dist = np.linspace(0.0, 4.0, 41)
    pi_zz = np.polyval(coeffs, dist)
    zeros = np.zeros_like(dist)
    result = SigmaOnlyResult(
        distances=dist, nics_pi_zz=pi_zz, nics_zz_real=pi_zz, nics_zz_model=zeros,
        nics_iso_real=pi_zz, nics_iso_model=zeros, three_delta_iso=pi_zz,
        som_deviation=zeros, pi_centers=(),
    )
    assert fit_pi_zz(result, dist_start=1.1, deg=3) == pytest.approx(
        float(np.polyval(coeffs, 1.0))
    )
