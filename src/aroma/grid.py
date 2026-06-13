#!/usr/bin/env python3

"""
Bq (ghost-atom) probe grids for NICS scans.

The molecule is assumed already reoriented so the ring of interest lies in the
XY plane with its centroid at the origin. The axial grid walks the +z ring
normal (the standard NICS-scan direction); the XY grid tiles a plane at a fixed
height for in-plane scans.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- numerics -----
import numpy as np
import numpy.typing as npt

# ----- local modules -----
from aroma.constants import DEFAULT_BQ_RANGE, DEFAULT_BQ_STEP, DEFAULT_XY_DISTANCE

# ============================================================
# GRID GENERATION
# ============================================================


def axial_grid(
    start: float = DEFAULT_BQ_RANGE[0],
    stop: float = DEFAULT_BQ_RANGE[1],
    step: float = DEFAULT_BQ_STEP,
) -> npt.NDArray[np.float64]:
    """Probe points along +z, from ``start`` to ``stop`` inclusive.

    Parameters
    ----------
    start, stop : float
        First and last probe heights (angstrom); ``stop`` is included.
    step : float
        Spacing between probes (angstrom).

    Returns
    -------
    points : (M, 3) float array
        Probe coordinates ``(0, 0, z)`` for z in the requested range.
    """
    assert step > 0.0, "step must be positive"
    assert stop >= start, "stop must not precede start"
    count = int(round((stop - start) / step)) + 1
    z = np.linspace(start, stop, count, dtype=np.float64)
    points = np.zeros((count, 3), dtype=np.float64)
    points[:, 2] = z
    assert points.shape == (count, 3), "axial grid has wrong shape"
    return points


def xy_scan_grid(
    half_extent: float,
    step: float = DEFAULT_BQ_STEP,
    height: float = DEFAULT_XY_DISTANCE,
) -> npt.NDArray[np.float64]:
    """Probe points on a square XY lattice at a fixed height above the ring.

    Parameters
    ----------
    half_extent : float
        Half-width of the square scan region (angstrom), centered on the origin.
    step : float
        Lattice spacing (angstrom).
    height : float
        Constant z offset of the scan plane (angstrom).

    Returns
    -------
    points : (M, 3) float array
        Probe coordinates ``(x, y, height)`` over the lattice.
    """
    assert half_extent > 0.0, "half_extent must be positive"
    assert step > 0.0, "step must be positive"
    count = int(round(2.0 * half_extent / step)) + 1
    axis = np.linspace(-half_extent, half_extent, count, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack(
        [xx.ravel(), yy.ravel(), np.full(xx.size, height, dtype=np.float64)]
    )
    assert points.shape == (count * count, 3), "xy grid has wrong shape"
    return points
