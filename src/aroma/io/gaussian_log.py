#!/usr/bin/env python3

"""
Reader for Gaussian output files (.log / .out).

Extracts the final standard/input orientation geometry plus charge and
multiplicity. Reading the last orientation block yields the optimized geometry
when the log is from a geometry optimization.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import re
from pathlib import Path
from typing import List, Tuple

# ----- numerics -----
import numpy as np

# ----- local modules -----
from aroma.molecule import Molecule

# ============================================================
# CONSTANTS
# ============================================================

_CHARGE_MULT = re.compile(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)")
_SEPARATOR = re.compile(r"-{5,}")

# ============================================================
# PARSING
# ============================================================


def _read_charge_mult(lines: List[str]) -> Tuple[int, int]:
    """Return (charge, multiplicity) from the first matching line."""
    for line in lines:
        match = _CHARGE_MULT.search(line)
        if match is not None:
            return int(match.group(1)), int(match.group(2))
    raise AssertionError("no 'Charge = .. Multiplicity = ..' line found")


def _last_orientation_start(lines: List[str]) -> int:
    """Return the index of the last 'orientation' header line."""
    idx = next(
        (i for i in range(len(lines) - 1, -1, -1)
         if "orientation:" in lines[i].lower()),
        None,
    )
    assert idx is not None, "no orientation block found in log"
    return idx


def read_log(path: Path) -> Molecule:
    """Read the final geometry from a Gaussian output file.

    Parameters
    ----------
    path : Path
        Path to a ``.log`` / ``.out`` file.

    Returns
    -------
    Molecule
        The last-printed orientation, with parsed charge and multiplicity.
    """
    lines = Path(path).read_text().splitlines()
    assert lines, f"empty file: {path}"

    charge, mult = _read_charge_mult(lines)

    # The coordinate table begins four lines past the orientation header (after
    # two title rows and a separator) and ends at the next separator rule.
    start = _last_orientation_start(lines) + 5
    numbers: List[int] = []
    coords: List[List[float]] = []
    for line in lines[start:]:
        if _SEPARATOR.search(line):
            break
        fields = line.split()
        assert len(fields) >= 6, f"malformed coordinate row: {line!r}"
        numbers.append(int(fields[1]))
        coords.append([float(v) for v in fields[3:6]])
    assert numbers, "no atoms parsed from orientation block"

    return Molecule(
        numbers=np.asarray(numbers, dtype=np.int_),
        coords=np.asarray(coords, dtype=np.float64),
        charge=charge,
        mult=mult,
    )
