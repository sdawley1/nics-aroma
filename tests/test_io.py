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
from aroma.io import load_geometry

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
