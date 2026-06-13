#!/usr/bin/env python3

"""
Immutable molecular geometry container.

Replaces the legacy dict-of-lists geometry ({index: [Z, x, y, z]}, 1-based) with
a frozen dataclass holding parallel NumPy arrays and 0-based indexing.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from dataclasses import dataclass
from typing import List, cast

# ----- numerics -----
import numpy as np
import numpy.typing as npt

# ----- local modules -----
from aroma.elements import number_to_symbol

# ============================================================
# MOLECULE
# ============================================================


@dataclass(frozen=True)
class Molecule:
    """A molecular geometry as parallel arrays.

    Parameters
    ----------
    numbers : (N,) int array
        Atomic numbers; 0 denotes a ghost/dummy atom (Bq).
    coords : (N, 3) float array
        Cartesian coordinates in angstrom.
    charge : int
        Total molecular charge.
    mult : int
        Spin multiplicity (2S + 1).
    """

    numbers: npt.NDArray[np.int_]
    coords: npt.NDArray[np.float64]
    charge: int = 0
    mult: int = 1

    def __post_init__(self) -> None:
        assert self.numbers.ndim == 1, "numbers must be 1-D"
        assert self.coords.shape == (self.numbers.size, 3), (
            "coords must have shape (N, 3) matching numbers"
        )
        assert self.mult >= 1, "multiplicity must be >= 1"

    @property
    def n_atoms(self) -> int:
        """Number of atoms (including ghosts)."""
        return int(self.numbers.size)

    def real_mask(self) -> npt.NDArray[np.bool_]:
        """Boolean mask selecting real (non-ghost) atoms."""
        return cast("npt.NDArray[np.bool_]", self.numbers != 0)

    def symbols(self) -> List[str]:
        """Element symbols for every atom, in order."""
        return [number_to_symbol(int(z)) for z in self.numbers]
