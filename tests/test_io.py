#!/usr/bin/env python3

"""
Tests for geometry IO across sample formats.

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
from aroma.io import load_geometry, log_has_bq_shielding, read_log_nics

# ============================================================
# TESTS
# ============================================================

# (relative path, expected real-atom count, charge, multiplicity)
_CASES = [
    ("pyrrole/pyrrole.in", 10, 0, 1),
    ("benzene/benzene.in", 12, 0, 1),
    ("benzene/benzene-opt.log", 12, 0, 1),
    ("phenalene/phenalene.in", 22, 1, 1),
    ("indene/indene-center1.in", 16, 0, 1),
]


@pytest.mark.parametrize("rel,n_atoms,charge,mult", _CASES)
def test_load_geometry(
    data_dir: Path, rel: str, n_atoms: int, charge: int, mult: int
) -> None:
    """Each sample loads with the expected size, charge, and multiplicity."""
    mol = load_geometry(data_dir / rel)
    assert mol.n_atoms == n_atoms
    assert mol.charge == charge
    assert mol.mult == mult
    assert mol.coords.shape == (n_atoms, 3)
    assert np.isfinite(mol.coords).all()


def test_input_skips_ghost_atoms(data_dir: Path) -> None:
    """The benzene deck carries Bq probes that must be excluded."""
    mol = load_geometry(data_dir / "benzene/benzene.in")
    assert np.all(mol.numbers != 0)
    assert set(np.unique(mol.numbers).tolist()) == {1, 6}


def test_read_log_nics(data_dir: Path) -> None:
    """The nmr=giao log yields real atoms plus Bq probe coords and tensors."""
    data = read_log_nics(data_dir / "benzene/benzene-nics.log")
    assert data.mol.n_atoms == 12
    assert set(np.unique(data.mol.numbers).tolist()) == {1, 6}
    assert (data.mol.charge, data.mol.mult) == (0, 1)
    assert data.bq_coords.shape == (8, 3)
    assert data.bq_tensors.shape == (8, 3, 3)
    assert np.isfinite(data.bq_tensors).all()
    # The probes are written z-ascending, so the first sits at the ring center.
    assert np.allclose(data.bq_coords[0], [0.0, 0.0, 0.0], atol=1e-6)
    iso_center = float(np.trace(data.bq_tensors[0]) / 3.0)
    assert iso_center == pytest.approx(11.5, abs=0.6)


def test_log_has_bq_shielding(data_dir: Path) -> None:
    """Only the nmr=giao log (not the opt-only log) reports Bq shielding."""
    assert log_has_bq_shielding(data_dir / "benzene/benzene-nics.log")
    assert not log_has_bq_shielding(data_dir / "benzene/benzene-opt.log")
