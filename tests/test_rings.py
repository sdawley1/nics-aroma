#!/usr/bin/env python3

"""
Tests for connectivity and ring perception on sample molecules.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from pathlib import Path
from typing import List

# ----- third party -----
import pytest

# ----- local modules -----
from aroma.connectivity import adjacency_list, bond_matrix
from aroma.io import load_geometry
from aroma.rings import find_rings, order_ring

# ============================================================
# HELPERS
# ============================================================


def _ring_sizes(data_dir: Path, rel: str) -> List[int]:
    """Return sorted ring sizes detected for a sample geometry."""
    mol = load_geometry(data_dir / rel)
    adj = adjacency_list(bond_matrix(mol))
    return sorted(len(r) for r in find_rings(adj))


# ============================================================
# TESTS
# ============================================================

# (relative path, expected sorted ring sizes)
_CASES = [
    ("benzene/benzene.in", [6]),
    ("pyrrole/pyrrole.in", [5]),
    ("indene/indene-center1.in", [5, 6]),
    ("phenalene/phenalene.in", [6, 6, 6]),
]


@pytest.mark.parametrize("rel,sizes", _CASES)
def test_ring_sizes(data_dir: Path, rel: str, sizes: List[int]) -> None:
    """SSSR ring count and sizes match the known structure of each sample."""
    assert _ring_sizes(data_dir, rel) == sizes


def test_benzene_connectivity(data_dir: Path) -> None:
    """Benzene has 12 bonds (6 C-C ring + 6 C-H) and degree-3 carbons."""
    mol = load_geometry(data_dir / "benzene/benzene.in")
    bonds = bond_matrix(mol)
    assert int(bonds.sum()) // 2 == 12
    carbons = mol.numbers == 6
    assert all(bonds[i].sum() == 3 for i in range(mol.n_atoms) if carbons[i])


def test_order_ring_is_connected(data_dir: Path) -> None:
    """order_ring returns a sequence where neighbors are bonded."""
    mol = load_geometry(data_dir / "benzene/benzene.in")
    adj = adjacency_list(bond_matrix(mol))
    ring = find_rings(adj)[0]
    ordered = order_ring(adj, list(reversed(ring)))
    for k, atom in enumerate(ordered):
        assert ordered[(k + 1) % len(ordered)] in adj[atom]
