#!/usr/bin/env python3

"""
Reader for Gaussian formatted-checkpoint files (.fchk).

Pure-Python parsing of the geometry, charge, and multiplicity. Unlike the legacy
code this does not invoke ``formchk``; supply an already-formatted ``.fchk``
(produce one on a login node with ``formchk file.chk``).

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from pathlib import Path
from typing import List

# ----- numerics -----
import numpy as np

# ----- local modules -----
from aroma.molecule import Molecule

# ============================================================
# CONSTANTS
# ============================================================

# Bohr -> angstrom (fchk stores coordinates in atomic units).
_BOHR_TO_ANGSTROM = 0.52917721067

# ============================================================
# PARSING
# ============================================================


def _scalar_int(lines: List[str], label: str) -> int:
    """Return the integer value of a scalar fchk field."""
    for line in lines:
        if line.startswith(label):
            return int(line.split()[-1])
    raise AssertionError(f"fchk field not found: {label!r}")


def _array_block(lines: List[str], label: str) -> List[float]:
    """Return the flattened numeric values of an ``N=`` fchk array field."""
    start = next((i for i, ln in enumerate(lines) if ln.startswith(label)), None)
    assert start is not None, f"fchk array field not found: {label!r}"
    count = int(lines[start].split("N=")[1].split()[0])

    values: List[float] = []
    i = start + 1
    while i < len(lines) and len(values) < count:
        values.extend(float(tok) for tok in lines[i].split())
        i += 1
    assert len(values) == count, f"{label!r}: expected {count}, got {len(values)}"
    return values


def read_fchk(path: Path) -> Molecule:
    """Read a formatted-checkpoint file into a :class:`Molecule`.

    Parameters
    ----------
    path : Path
        Path to a ``.fchk`` file.

    Returns
    -------
    Molecule
        Coordinates converted from bohr to angstrom.
    """
    lines = Path(path).read_text().splitlines()
    assert lines, f"empty file: {path}"

    charge = _scalar_int(lines, "Charge")
    mult = _scalar_int(lines, "Multiplicity")
    numbers = [int(z) for z in _array_block(lines, "Atomic numbers")]
    flat = _array_block(lines, "Current cartesian coordinates")
    assert len(flat) == 3 * len(numbers), "coordinate count mismatch"

    coords = np.asarray(flat, dtype=np.float64).reshape(-1, 3) * _BOHR_TO_ANGSTROM
    return Molecule(
        numbers=np.asarray(numbers, dtype=np.int_),
        coords=coords,
        charge=charge,
        mult=mult,
    )
