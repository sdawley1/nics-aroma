#!/usr/bin/env python3

"""
Reader for Gaussian output files (.log / .out).

Extracts the final standard/input orientation geometry plus charge and
multiplicity. Reading the last orientation block yields the optimized geometry
when the log is from a geometry optimization.

When the job also ran ``nmr=giao`` with ``Bq`` ghost probes, the magnetic
shielding tensor printed at each ghost IS the NICS data (NICS = -sigma at the
probe point). :func:`read_log_nics` recovers those tensors so a NICS scan can be
reported directly, without recomputing the shielding.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----- numerics -----
import numpy as np
import numpy.typing as npt

# ----- local modules -----
from aroma.molecule import Molecule

# ============================================================
# CONSTANTS
# ============================================================

_CHARGE_MULT = re.compile(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)")
_SEPARATOR = re.compile(r"-{5,}")

# Header preceding the per-atom magnetic shielding tensors in an nmr=giao job.
_SHIELDING_HEADER = re.compile(r"[Mm]agnetic shielding")
# Start of one atom's shielding entry, e.g. "     13  Bq   Isotropic = ...".
_ATOM_HEADER = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,2})\s+Isotropic\s*=")
# A single labelled tensor component, e.g. "ZZ=  145.9456".
_COMPONENT = re.compile(r"\b([XYZ][XYZ])=\s*(-?\d+\.\d+)")
# True iff the log carries shielding for at least one Bq ghost (precise: the
# atom label and the Isotropic field only coincide inside the shielding block).
_BQ_SHIELDING = re.compile(r"(?im)^\s*\d+\s+Bq\s+Isotropic\s*=")

# Cartesian axis order used to place labelled components into a 3x3 tensor.
_AXIS = {"X": 0, "Y": 1, "Z": 2}

# ============================================================
# RESULT CONTAINER
# ============================================================


@dataclass(frozen=True)
class GaussianNics:
    """Pre-computed NICS data parsed from a Gaussian nmr=giao output.

    Attributes
    ----------
    mol : Molecule
        Real atoms only (ghosts removed), for ring perception/reorientation.
    bq_coords : (M, 3) float array
        Ghost (Bq) probe coordinates, in the same frame as ``bq_tensors``.
    bq_tensors : (M, 3, 3) float array
        GIAO magnetic shielding tensor (ppm) at each ghost probe.
    """

    mol: Molecule
    bq_coords: npt.NDArray[np.float64]
    bq_tensors: npt.NDArray[np.float64]

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


def _parse_last_geometry(lines: List[str]) -> Tuple[List[int], List[List[float]]]:
    """Parse the last orientation block into atomic numbers and coordinates.

    Ghost (Bq) centers appear here with atomic number 0; they are kept so the
    caller can align them with the shielding tensors by position.
    """
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
    return numbers, coords


def _assemble_tensor(components: Dict[str, float]) -> npt.NDArray[np.float64]:
    """Build a 3x3 shielding tensor from its nine labelled components."""
    assert len(components) == 9, f"expected 9 components, got {len(components)}"
    tensor = np.zeros((3, 3), dtype=np.float64)
    for label, value in components.items():
        tensor[_AXIS[label[0]], _AXIS[label[1]]] = value
    return tensor


def _parse_shielding(lines: List[str]) -> npt.NDArray[np.float64]:
    """Parse the per-atom magnetic shielding tensors from an nmr=giao job.

    Returns an ``(K, 3, 3)`` array in atom (center) order, or an empty
    ``(0, 3, 3)`` array when the log has no shielding block.
    """
    start = next(
        (i for i in range(len(lines) - 1, -1, -1)
         if _SHIELDING_HEADER.search(lines[i])),
        None,
    )
    if start is None:
        return np.empty((0, 3, 3), dtype=np.float64)

    tensors: List[npt.NDArray[np.float64]] = []
    current: Optional[Dict[str, float]] = None
    for line in lines[start + 1:]:
        if _ATOM_HEADER.match(line) is not None:
            current = {}
            continue
        if current is None:
            continue
        for label, value in _COMPONENT.findall(line):
            current[label] = float(value)
        if len(current) == 9:
            tensors.append(_assemble_tensor(current))
            current = None
    return np.asarray(tensors, dtype=np.float64) if tensors else np.empty(
        (0, 3, 3), dtype=np.float64
    )


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
    numbers, coords = _parse_last_geometry(lines)
    return Molecule(
        numbers=np.asarray(numbers, dtype=np.int_),
        coords=np.asarray(coords, dtype=np.float64),
        charge=charge,
        mult=mult,
    )


def read_log_nics(path: Path) -> GaussianNics:
    """Read pre-computed NICS data (Bq shielding) from a Gaussian output.

    Parameters
    ----------
    path : Path
        Path to a ``.log`` / ``.out`` file from an ``nmr=giao`` job that
        included ``Bq`` ghost probes.

    Returns
    -------
    GaussianNics
        Real-atom geometry plus the ghost coordinates and shielding tensors.
    """
    lines = Path(path).read_text().splitlines()
    assert lines, f"empty file: {path}"

    charge, mult = _read_charge_mult(lines)
    numbers_list, coords_list = _parse_last_geometry(lines)
    numbers = np.asarray(numbers_list, dtype=np.int_)
    coords = np.asarray(coords_list, dtype=np.float64)
    tensors = _parse_shielding(lines)
    assert tensors.shape[0] == numbers.size, (
        f"shielding count {tensors.shape[0]} != atom count {numbers.size}"
    )

    ghost = numbers == 0
    assert ghost.any(), "no Bq ghost atoms found; nothing to reuse"
    real = ~ghost
    mol = Molecule(
        numbers=numbers[real], coords=coords[real], charge=charge, mult=mult
    )
    return GaussianNics(
        mol=mol, bq_coords=coords[ghost], bq_tensors=tensors[ghost]
    )


def log_has_bq_shielding(path: Path) -> bool:
    """Return True if the log contains a magnetic shielding tensor for a Bq."""
    return _BQ_SHIELDING.search(Path(path).read_text()) is not None
