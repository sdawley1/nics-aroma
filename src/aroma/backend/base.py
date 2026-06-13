#!/usr/bin/env python3

"""
Shielding-backend protocol.

A backend computes magnetic shielding tensors at a set of ghost (Bq) probe
points for a given molecule. Isolating this behind a Protocol keeps the NICS
driver independent of any particular quantum-chemistry package and allows
mock backends in tests.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Protocol, runtime_checkable

# ----- numerics -----
import numpy as np
import numpy.typing as npt

# ----- local modules -----
from aroma.molecule import Molecule

# ============================================================
# PROTOCOL
# ============================================================


@runtime_checkable
class ShieldingBackend(Protocol):
    """Computes shielding tensors at ghost probe points."""

    def shielding(
        self, mol: Molecule, ghosts: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Return the (M, 3, 3) shielding tensors at the M ghost points (ppm)."""
        ...
