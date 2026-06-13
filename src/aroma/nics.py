#!/usr/bin/env python3

"""
NICS-scan driver.

Reorients the molecule so the chosen ring lies in the XY plane, lays a Bq probe
grid along the ring normal, asks a shielding backend for the GIAO tensors, and
converts them to NICS components. NICS = -sigma; after reorientation +z is the
ring normal, so NICS_zz is the out-of-plane component directly.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from dataclasses import dataclass
from typing import List, Tuple

# ----- numerics -----
import numpy as np
import numpy.typing as npt

# ----- local modules -----
from aroma.backend.base import ShieldingBackend
from aroma.constants import DEFAULT_BQ_RANGE, DEFAULT_BQ_STEP
from aroma.grid import axial_grid
from aroma.molecule import Molecule
from aroma.reorient import reorient_ring_to_xy

# ============================================================
# RESULT CONTAINER
# ============================================================


@dataclass(frozen=True)
class NicsResult:
    """NICS components (ppm) along an axial scan.

    Each array is indexed by probe; ``distances`` holds the probe heights
    (angstrom) above the ring centroid.
    """

    distances: npt.NDArray[np.float64]
    nics_iso: npt.NDArray[np.float64]
    nics_zz: npt.NDArray[np.float64]
    nics_oop: npt.NDArray[np.float64]
    nics_inp: npt.NDArray[np.float64]


# ============================================================
# TENSOR DECOMPOSITION
# ============================================================


def _principal_components(
    tensor: npt.NDArray[np.float64],
) -> Tuple[float, float, float, float]:
    """Decompose one shielding tensor into (iso, zz, out-of-plane, in-plane).

    The out-of-plane principal value is the eigenvalue whose eigenvector aligns
    best with +z (the ring normal); the remaining two are averaged in-plane.
    """
    sym = 0.5 * (tensor + tensor.T)
    iso = float(np.trace(sym)) / 3.0
    zz = float(sym[2, 2])
    eigvals, eigvecs = np.linalg.eigh(sym)
    oop_idx = int(np.argmax(np.abs(eigvecs[2, :])))
    oop = float(eigvals[oop_idx])
    in_plane = [float(eigvals[i]) for i in range(3) if i != oop_idx]
    return iso, zz, oop, sum(in_plane) / 2.0


# ============================================================
# SCAN DRIVER
# ============================================================


def run_nics_scan(
    mol: Molecule,
    ring: List[int],
    backend: ShieldingBackend,
    start: float = DEFAULT_BQ_RANGE[0],
    stop: float = DEFAULT_BQ_RANGE[1],
    step: float = DEFAULT_BQ_STEP,
) -> NicsResult:
    """Run an axial NICS scan over one ring.

    Parameters
    ----------
    mol : Molecule
        Input geometry.
    ring : list of int
        Atom indices of the ring to probe (any order).
    backend : ShieldingBackend
        Provider of GIAO shielding tensors.
    start, stop, step : float
        Axial grid extent and spacing (angstrom).

    Returns
    -------
    NicsResult
        NICS components at each probe height.
    """
    assert len(ring) >= 3, "a ring needs at least three atoms"

    reoriented, _ = reorient_ring_to_xy(mol, ring)
    grid = axial_grid(start, stop, step)
    tensors = backend.shielding(reoriented, grid)
    assert tensors.shape[0] == grid.shape[0], "backend returned wrong probe count"

    n = grid.shape[0]
    iso = np.empty(n, dtype=np.float64)
    zz = np.empty(n, dtype=np.float64)
    oop = np.empty(n, dtype=np.float64)
    inp = np.empty(n, dtype=np.float64)
    for i in range(n):
        iso[i], zz[i], oop[i], inp[i] = _principal_components(tensors[i])

    # NICS is the negative of the shielding.
    return NicsResult(
        distances=grid[:, 2].copy(),
        nics_iso=-iso,
        nics_zz=-zz,
        nics_oop=-oop,
        nics_inp=-inp,
    )
