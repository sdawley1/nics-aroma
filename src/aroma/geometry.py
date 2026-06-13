#!/usr/bin/env python3

"""
Vectorized geometric primitives.

Centroids, best-fit ring normals (via SVD), Rodrigues rotation matrices,
dihedral angles, and planar polygon areas. Replaces the legacy scalar helpers
and manual 3x3 matrix arithmetic with NumPy operations.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import List

# ----- numerics -----
import numpy as np
import numpy.typing as npt

# ============================================================
# CONSTANTS
# ============================================================

# Vectors shorter than this (angstrom) are treated as numerically zero.
_EPS = 1.0e-9

# ============================================================
# PRIMITIVES
# ============================================================


def centroid(coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Geometric center (unweighted mean) of an (N, 3) coordinate array."""
    assert coords.ndim == 2 and coords.shape[1] == 3, "coords must be (N, 3)"
    assert coords.shape[0] >= 1, "need at least one point"
    return np.asarray(coords.mean(axis=0), dtype=np.float64)


def ring_normal(
    coords: npt.NDArray[np.float64], ring: List[int]
) -> npt.NDArray[np.float64]:
    """Unit normal of the best-fit plane through a ring's atoms.

    The normal is the least-significant right-singular vector of the
    mean-centered ring coordinates (the total-least-squares plane fit), which is
    more robust than the legacy average of pairwise cross products.

    Parameters
    ----------
    coords : (N, 3) float array
        Coordinates of all atoms.
    ring : list of int
        Indices of the ring atoms.

    Returns
    -------
    normal : (3,) float array
        Unit normal vector.
    """
    assert len(ring) >= 3, "a ring plane needs at least three atoms"
    pts = coords[ring]
    centered = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    assert norm > _EPS, "degenerate ring plane"
    return np.asarray(normal / norm, dtype=np.float64)


def rodrigues_matrix(
    axis: npt.NDArray[np.float64], angle: float
) -> npt.NDArray[np.float64]:
    """Rotation matrix for ``angle`` radians about a (nonzero) ``axis``.

    Parameters
    ----------
    axis : (3,) float array
        Rotation axis; need not be normalized.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    R : (3, 3) float array
        Orthogonal rotation matrix with determinant +1.
    """
    norm = float(np.linalg.norm(axis))
    assert norm > _EPS, "rotation axis must be nonzero"
    kx, ky, kz = axis / norm
    k = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    rot = np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9), "R must be orthogonal"
    return np.asarray(rot, dtype=np.float64)


def dihedral(
    p0: npt.NDArray[np.float64],
    p1: npt.NDArray[np.float64],
    p2: npt.NDArray[np.float64],
    p3: npt.NDArray[np.float64],
) -> float:
    """Signed dihedral angle (degrees) about the p1-p2 bond.

    Parameters
    ----------
    p0, p1, p2, p3 : (3,) float arrays
        The four points defining the torsion.

    Returns
    -------
    angle : float
        Dihedral in degrees, in (-180, 180].
    """
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    assert np.linalg.norm(b1) > _EPS, "central bond is degenerate"
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return float(np.degrees(np.arctan2(y, x)))


def polygon_area(coords: npt.NDArray[np.float64], ring: List[int]) -> float:
    """Area of a (near-)planar polygon given its ordered ring vertices.

    Uses the magnitude of the summed vertex cross products, valid for any
    planar polygon embedded in 3-D.
    """
    assert len(ring) >= 3, "a polygon needs at least three vertices"
    pts = coords[ring]
    pts = pts - pts.mean(axis=0)
    cross_sum = np.zeros(3, dtype=np.float64)
    for k in range(len(ring)):
        cross_sum += np.cross(pts[k], pts[(k + 1) % len(ring)])
    return 0.5 * float(np.linalg.norm(cross_sum))
