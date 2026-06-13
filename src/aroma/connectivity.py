#!/usr/bin/env python3

"""
Bond perception from atomic coordinates.

Vectorized covalent-radius distance test producing a boolean adjacency matrix
and an adjacency list, replacing the legacy O(N^2) Python double loop.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import List, cast

# ----- numerics -----
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

# ----- local modules -----
from aroma.constants import COVALENT_BOND_TOLERANCE
from aroma.elements import covalent_radius_array
from aroma.molecule import Molecule

# ============================================================
# CONSTANTS
# ============================================================

# Minimum separation (angstrom) for two atoms to be considered distinct; guards
# against pathological zero-distance duplicates.
_MIN_SEPARATION = 0.1

# ============================================================
# BOND PERCEPTION
# ============================================================


def bond_matrix(
    mol: Molecule, tol: float = COVALENT_BOND_TOLERANCE
) -> npt.NDArray[np.bool_]:
    """Symmetric boolean adjacency from a covalent-radius distance test.

    Two atoms are bonded when their separation does not exceed the sum of their
    covalent radii plus ``tol``.

    Parameters
    ----------
    mol : Molecule
        Geometry to analyze (real atoms only).
    tol : float
        Slack added to the summed covalent radii (angstrom).

    Returns
    -------
    bonds : (N, N) bool array
        ``bonds[i, j]`` is True iff atoms i and j are bonded; diagonal is False.
    """
    assert mol.n_atoms >= 2, "need at least two atoms for bond perception"
    assert tol >= 0.0, "tolerance must be non-negative"

    radii = covalent_radius_array()[mol.numbers]
    cutoff = radii[:, None] + radii[None, :] + tol
    dist = cdist(mol.coords, mol.coords)

    bonds = (dist <= cutoff) & (dist > _MIN_SEPARATION)
    np.fill_diagonal(bonds, False)
    assert np.array_equal(bonds, bonds.T), "adjacency must be symmetric"
    return cast("npt.NDArray[np.bool_]", bonds)


def adjacency_list(bonds: npt.NDArray[np.bool_]) -> List[List[int]]:
    """Convert a boolean adjacency matrix to a per-atom neighbor list.

    Parameters
    ----------
    bonds : (N, N) bool array
        Symmetric adjacency matrix.

    Returns
    -------
    neighbors : list of lists
        ``neighbors[i]`` holds the sorted indices bonded to atom i.
    """
    assert bonds.ndim == 2 and bonds.shape[0] == bonds.shape[1], "bonds must be square"
    return [np.flatnonzero(row).tolist() for row in bonds]
