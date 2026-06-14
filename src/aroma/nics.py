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
from aroma.constants import DEFAULT_BQ_RANGE, DEFAULT_BQ_STEP, DEFAULT_XY_DISTANCE
from aroma.grid import axial_grid, xy_scan_grid
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


@dataclass(frozen=True)
class XyNicsResult:
    """NICS components (ppm) on an in-plane (XY) scan grid.

    Each array is indexed by probe; ``xs`` and ``ys`` hold the in-plane probe
    coordinates (angstrom) at the fixed scan ``height`` above the ring centroid.
    """

    xs: npt.NDArray[np.float64]
    ys: npt.NDArray[np.float64]
    nics_iso: npt.NDArray[np.float64]
    nics_zz: npt.NDArray[np.float64]
    nics_oop: npt.NDArray[np.float64]
    nics_inp: npt.NDArray[np.float64]


# ============================================================
# TENSOR DECOMPOSITION
# ============================================================


def _decompose(
    tensors: npt.NDArray[np.float64],
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Decompose a stack of shielding tensors into NICS component arrays.

    All probes are processed at once: ``np.linalg.eigh`` is batched over the
    leading axis, so there is no Python per-probe loop. For each tensor the
    out-of-plane principal value is the eigenvalue whose eigenvector aligns best
    with +z (the ring normal); the other two eigenvalues are averaged in-plane.

    Parameters
    ----------
    tensors : (M, 3, 3) float array
        One shielding tensor per probe.

    Returns
    -------
    (iso, zz, oop, inp) : tuple of (M,) float arrays
        Negated to NICS sign convention (NICS = -sigma).
    """
    assert tensors.ndim == 3 and tensors.shape[1:] == (3, 3), "tensors must be (M,3,3)"
    n = tensors.shape[0]
    sym = 0.5 * (tensors + tensors.transpose(0, 2, 1))
    iso = np.trace(sym, axis1=1, axis2=2) / 3.0
    zz = sym[:, 2, 2]

    eigvals, eigvecs = np.linalg.eigh(sym)  # eigvals (M,3) asc; eigvecs columns
    # z-component of each eigenvector: pick the one most aligned with +z.
    oop_idx = np.argmax(np.abs(eigvecs[:, 2, :]), axis=1)
    oop = eigvals[np.arange(n), oop_idx]
    inp = (eigvals.sum(axis=1) - oop) / 2.0

    # NICS is the negative of the shielding.
    return -iso, -zz, -oop, -inp


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

    iso, zz, oop, inp = _decompose(tensors)
    return NicsResult(
        distances=grid[:, 2].copy(),
        nics_iso=iso,
        nics_zz=zz,
        nics_oop=oop,
        nics_inp=inp,
    )


def run_xy_scan(
    mol: Molecule,
    ring: List[int],
    backend: ShieldingBackend,
    half_extent: float,
    step: float = DEFAULT_BQ_STEP,
    height: float = DEFAULT_XY_DISTANCE,
) -> XyNicsResult:
    """Run an in-plane (XY) NICS scan over a square lattice above one ring.

    Parameters
    ----------
    mol : Molecule
        Input geometry.
    ring : list of int
        Atom indices of the ring to probe (any order).
    backend : ShieldingBackend
        Provider of GIAO shielding tensors.
    half_extent : float
        Half-width of the square scan region (angstrom), centered on the origin.
    step : float
        Lattice spacing (angstrom).
    height : float
        Constant z offset of the scan plane (angstrom).

    Returns
    -------
    XyNicsResult
        NICS components at each in-plane probe.
    """
    assert len(ring) >= 3, "a ring needs at least three atoms"

    reoriented, _ = reorient_ring_to_xy(mol, ring)
    grid = xy_scan_grid(half_extent, step, height)
    tensors = backend.shielding(reoriented, grid)
    assert tensors.shape[0] == grid.shape[0], "backend returned wrong probe count"

    iso, zz, oop, inp = _decompose(tensors)
    return XyNicsResult(
        xs=grid[:, 0].copy(),
        ys=grid[:, 1].copy(),
        nics_iso=iso,
        nics_zz=zz,
        nics_oop=oop,
        nics_inp=inp,
    )
