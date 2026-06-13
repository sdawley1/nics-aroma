#!/usr/bin/env python3

"""
Reader for plain XYZ geometry files.

Standard XYZ: an atom count, a comment line, then one "symbol x y z" row per
atom. Charge and multiplicity are not encoded in XYZ and default to 0 / 1.

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
from aroma.elements import symbol_to_number
from aroma.molecule import Molecule

# ============================================================
# PARSING
# ============================================================


def read_xyz(path: Path) -> Molecule:
    """Read a standard XYZ file into a :class:`Molecule`.

    Parameters
    ----------
    path : Path
        Path to a ``.xyz`` file.

    Returns
    -------
    Molecule
        Charge 0 and multiplicity 1 (XYZ encodes neither).
    """
    lines = Path(path).read_text().splitlines()
    assert len(lines) >= 3, f"XYZ file too short: {path}"

    count = int(lines[0].split()[0])
    assert count > 0, f"non-positive atom count in {path}"

    numbers: List[int] = []
    coords: List[List[float]] = []
    for line in lines[2 : 2 + count]:
        fields = line.split()
        assert len(fields) >= 4, f"malformed XYZ row: {line!r}"
        token = fields[0]
        numbers.append(
            int(token) if token.lstrip("-").isdigit() else symbol_to_number(token)
        )
        coords.append([float(v) for v in fields[1:4]])
    assert len(numbers) == count, "atom count does not match parsed rows"

    return Molecule(
        numbers=np.asarray(numbers, dtype=np.int_),
        coords=np.asarray(coords, dtype=np.float64),
    )
