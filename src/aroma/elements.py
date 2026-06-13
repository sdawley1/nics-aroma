#!/usr/bin/env python3

"""
Element data: symbol/atomic-number maps and covalent radii.

Provides bidirectional symbol <-> atomic-number lookup and a vectorized
covalent-radius array used by the connectivity detector. Ghost atoms ("Bq")
are represented by atomic number 0.

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
import numpy.typing as npt

# ============================================================
# ELEMENT TABLES
# ============================================================

# Atomic number keyed by upper-case element symbol. 0 = ghost/dummy ("X"/"Bq").
SYMBOL_TO_NUMBER: Dict[str, int] = {
    "X": 0, "BQ": 0,
    "H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "NE": 10, "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15,
    "S": 16, "CL": 17, "AR": 18, "TI": 22,
}

# Inverse map (first symbol wins; ghost resolves to "X").
NUMBER_TO_SYMBOL: Dict[int, str] = {
    0: "X", 1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N",
    8: "O", 9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si",
    15: "P", 16: "S", 17: "Cl", 18: "Ar", 22: "Ti",
}

# Covalent radii (angstrom) keyed by atomic number; used for bond detection.
COVALENT_RADII: Dict[int, float] = {
    0: 0.70, 1: 0.23, 2: 0.32, 3: 1.23, 4: 0.90, 5: 0.82, 6: 0.77,
    7: 0.75, 8: 0.73, 9: 0.71, 10: 0.69, 11: 1.54, 12: 1.36, 13: 1.18,
    14: 1.11, 15: 1.06, 16: 1.02, 17: 0.99, 18: 0.97, 22: 1.32,
}

# Highest atomic number present in the tables; bounds the radius array.
_MAX_Z = max(COVALENT_RADII)

# ============================================================
# LOOKUPS
# ============================================================


def symbol_to_number(symbol: str) -> int:
    """Return the atomic number for an element symbol (case-insensitive)."""
    key = symbol.strip().upper()
    assert key, "empty element symbol"
    assert key in SYMBOL_TO_NUMBER, f"unknown element symbol: {symbol!r}"
    return SYMBOL_TO_NUMBER[key]


def number_to_symbol(number: int) -> str:
    """Return the element symbol for an atomic number."""
    assert number >= 0, f"atomic number must be non-negative, got {number}"
    assert number in NUMBER_TO_SYMBOL, f"unknown atomic number: {number}"
    return NUMBER_TO_SYMBOL[number]


def covalent_radius_array() -> npt.NDArray[np.float64]:
    """Covalent radii indexed by atomic number.

    Returns
    -------
    radii : (Z_max + 1,) float array
        ``radii[z]`` is the covalent radius (angstrom) of element ``z``;
        atomic numbers absent from the table default to 0.70 angstrom.
    """
    radii = np.full(_MAX_Z + 1, 0.70, dtype=np.float64)
    for z, r in COVALENT_RADII.items():
        radii[z] = r
    assert radii.shape == (_MAX_Z + 1,), "radius array has wrong shape"
    assert np.all(radii > 0.0), "covalent radii must be positive"
    return radii
