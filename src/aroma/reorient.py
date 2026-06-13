#!/usr/bin/env python3

"""
Reorient a molecule so a chosen ring lies in the XY plane.

Translates the ring centroid to the origin and rotates the ring's best-fit
normal onto +z, so that NICS probes placed along +z sample the out-of-plane
direction and the zz shielding component is the out-of-plane component.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import List, Tuple

# ----- numerics -----
import numpy as np
import numpy.typing as npt

# ----- local modules -----
from aroma.geometry import centroid, ring_normal, rodrigues_matrix
from aroma.molecule import Molecule

# ============================================================
# CONSTANTS
# ============================================================

_Z_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# Below this sine the normal is treated as already (anti)parallel to +z.
_PARALLEL_EPS = 1.0e-9

# ============================================================
# REORIENTATION
# ============================================================


def _alignment_rotation(normal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Rotation mapping a unit ``normal`` onto +z.

    Handles the degenerate cases where the normal is already parallel
    (identity) or antiparallel (180 deg about x) to +z.
    """
    axis = np.cross(normal, _Z_AXIS)
    sin_angle = float(np.linalg.norm(axis))
    cos_angle = float(np.dot(normal, _Z_AXIS))
    if sin_angle < _PARALLEL_EPS:
        if cos_angle > 0.0:
            return np.eye(3, dtype=np.float64)
        return rodrigues_matrix(np.array([1.0, 0.0, 0.0]), np.pi)
    return rodrigues_matrix(axis, float(np.arctan2(sin_angle, cos_angle)))


def reorient_ring_to_xy(
    mol: Molecule, ring: List[int]
) -> Tuple[Molecule, npt.NDArray[np.float64]]:
    """Translate and rotate ``mol`` so ``ring`` lies in the XY plane.

    Parameters
    ----------
    mol : Molecule
        Geometry to reorient.
    ring : list of int
        Indices of the ring atoms (any order).

    Returns
    -------
    reoriented : Molecule
        Copy with the ring centroid at the origin and its normal along +z.
    rotation : (3, 3) float array
        The applied rotation matrix.
    """
    assert len(ring) >= 3, "a ring needs at least three atoms"
    assert max(ring) < mol.n_atoms, "ring index out of range"

    center = centroid(mol.coords[ring])
    translated = mol.coords - center
    rotation = _alignment_rotation(ring_normal(translated, ring))
    new_coords = translated @ rotation.T

    # The best-fit ring plane must now be the XY plane (its normal along z) and
    # the ring centroid must sit at the origin. Per-atom z need not vanish: real
    # geometries are not perfectly planar.
    assert abs(abs(float(ring_normal(new_coords, ring)[2])) - 1.0) < 1e-6, (
        "ring plane is not aligned with XY after reorientation"
    )
    assert np.allclose(new_coords[ring].mean(axis=0), 0.0, atol=1e-6), (
        "ring centroid is not at the origin after reorientation"
    )
    reoriented = Molecule(
        numbers=mol.numbers, coords=new_coords, charge=mol.charge, mult=mol.mult
    )
    return reoriented, rotation
