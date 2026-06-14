#!/usr/bin/env python3

"""
Geometry IO: format dispatch by file extension.

Exposes :func:`load_geometry`, which routes a path to the appropriate reader
(Gaussian input/output/fchk or plain XYZ) and returns a :class:`Molecule`.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from pathlib import Path
from typing import Callable, Dict

# ----- local modules -----
from aroma.io.fchk import read_fchk
from aroma.io.gaussian_com import read_com
from aroma.io.gaussian_log import log_has_bq_shielding, read_log, read_log_nics
from aroma.io.xyz import read_xyz
from aroma.molecule import Molecule

# ============================================================
# DISPATCH
# ============================================================

# Reader keyed by lower-case file extension (without the leading dot).
_READERS: Dict[str, Callable[[Path], Molecule]] = {
    "com": read_com,
    "gjf": read_com,
    "in": read_com,
    "log": read_log,
    "out": read_log,
    "fchk": read_fchk,
    "xyz": read_xyz,
}


def load_geometry(path: Path) -> Molecule:
    """Load a geometry, dispatching on file extension.

    Parameters
    ----------
    path : Path
        Geometry file; extension selects the reader.

    Returns
    -------
    Molecule
        The parsed geometry.
    """
    path = Path(path)
    ext = path.suffix.lstrip(".").lower()
    assert ext in _READERS, f"unsupported geometry extension: {path.suffix!r}"
    mol = _READERS[ext](path)
    assert mol.n_atoms > 0, f"no atoms parsed from {path}"
    return mol


__all__ = [
    "load_geometry",
    "log_has_bq_shielding",
    "read_com",
    "read_fchk",
    "read_log",
    "read_log_nics",
    "read_xyz",
]
