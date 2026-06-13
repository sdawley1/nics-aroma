#!/usr/bin/env python3

"""
Reader for Gaussian input files (.com / .gjf / .in).

Parses the molecular specification (charge, multiplicity, Cartesian atoms) from
a Gaussian input deck, skipping ghost ("Bq") placeholder lines. Z-matrix decks
are not supported (the legacy code converted them via a Gaussian guess=only
run, which this package no longer shells out to).

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
from aroma.elements import symbol_to_number
from aroma.molecule import Molecule

# ============================================================
# CONSTANTS
# ============================================================

# Matches a charge/multiplicity line, e.g. "0 1" or "-1, 2".
_CHARGE_MULT = re.compile(r"^\s*(-?\d+)[\s,]+(\d+)\s*$")

# ============================================================
# PARSING
# ============================================================


def _atom_token_to_number(token: str) -> int:
    """Return the atomic number for a numeric or symbolic atom token."""
    token = token.strip()
    assert token, "empty atom token"
    return int(token) if token.lstrip("-").isdigit() else symbol_to_number(token)


def _parse_atom_lines(lines: List[str]) -> Tuple[List[int], List[List[float]]]:
    """Parse consecutive atom lines into numbers and coordinates.

    Parsing stops at the first blank line. Ghost ("Bq") rows are skipped so the
    returned geometry contains only real atoms.
    """
    numbers: List[int] = []
    coords: List[List[float]] = []
    for line in lines:
        if not line.strip():
            break
        fields = re.split(r"[\s,]+", line.strip())
        assert len(fields) >= 4, f"malformed atom line: {line!r}"
        if fields[0].upper() in ("BQ", "X"):
            continue
        numbers.append(_atom_token_to_number(fields[0]))
        coords.append([float(v) for v in fields[1:4]])
    assert numbers, "no real atoms found in input deck"
    return numbers, coords


def read_com(path: Path) -> Molecule:
    """Read a Gaussian input deck into a :class:`Molecule`.

    Parameters
    ----------
    path : Path
        Path to a ``.com`` / ``.gjf`` / ``.in`` file with a Cartesian geometry.

    Returns
    -------
    Molecule
        Real atoms only, with parsed charge and multiplicity.
    """
    lines = Path(path).read_text().splitlines()
    assert lines, f"empty file: {path}"

    charge_idx = next(
        (i for i, ln in enumerate(lines) if _CHARGE_MULT.match(ln)), None
    )
    assert charge_idx is not None, f"no charge/multiplicity line in {path}"

    match = _CHARGE_MULT.match(lines[charge_idx])
    assert match is not None, "charge/multiplicity match lost"
    charge, mult = int(match.group(1)), int(match.group(2))

    numbers, coords = _parse_atom_lines(lines[charge_idx + 1 :])
    return Molecule(
        numbers=np.asarray(numbers, dtype=np.int_),
        coords=np.asarray(coords, dtype=np.float64),
        charge=charge,
        mult=mult,
    )
