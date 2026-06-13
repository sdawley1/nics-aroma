#!/usr/bin/env python3

"""
Polynomial analysis of NICS-vs-distance curves.

Fits low-order polynomials to each NICS component over the long-range portion of
an axial scan (probes at or beyond ``dist_start``), mirroring the legacy
cubic-fit analysis. The fits can be evaluated at any distance, e.g. NICS(1).

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Dict

# ----- numerics -----
import numpy as np

# ----- local modules -----
from aroma.constants import DEFAULT_FIT_START
from aroma.nics import NicsResult

# ============================================================
# CURVE FITTING
# ============================================================


def fit_nics_curve(
    result: NicsResult, dist_start: float = DEFAULT_FIT_START, deg: int = 3
) -> Dict[str, np.poly1d]:
    """Fit each NICS component versus distance with a polynomial.

    Parameters
    ----------
    result : NicsResult
        An axial NICS scan.
    dist_start : float
        Only probes at this distance (angstrom) or greater are fitted, avoiding
        the near-ring region where the curve is steep and basis-sensitive.
    deg : int
        Polynomial degree.

    Returns
    -------
    fits : dict of str to numpy.poly1d
        Fitted polynomials keyed by component: ``"iso"``, ``"zz"``, ``"oop"``,
        ``"inp"``.
    """
    assert deg >= 1, "polynomial degree must be at least 1"
    mask = result.distances >= dist_start
    assert int(mask.sum()) > deg, (
        f"need more than {deg} probes beyond {dist_start} A to fit degree {deg}"
    )

    x = result.distances[mask]
    components = {
        "iso": result.nics_iso,
        "zz": result.nics_zz,
        "oop": result.nics_oop,
        "inp": result.nics_inp,
    }
    return {
        name: np.poly1d(np.polyfit(x, values[mask], deg))
        for name, values in components.items()
    }
