#!/usr/bin/env python3

"""
Tests for NICS curve fitting and the CLI plumbing (no PySCF required).

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from pathlib import Path

# ----- third party -----
import numpy as np

# ----- local modules -----
from aroma.analysis import fit_nics_curve
from aroma.batch import select_rings
from aroma.cli import _build_parser
from aroma.io import load_geometry
from aroma.nics import NicsResult

# ============================================================
# CURVE FITTING
# ============================================================


def test_fit_recovers_known_cubic() -> None:
    """A synthetic cubic NICS curve is recovered, including its value at d=1."""
    dist = np.linspace(0.0, 4.0, 41)
    true = np.poly1d([0.5, -2.0, 1.0, -8.0])
    iso = true(dist)
    result = NicsResult(
        distances=dist, nics_iso=iso, nics_zz=2.0 * iso,
        nics_oop=iso, nics_inp=iso,
    )
    fits = fit_nics_curve(result, dist_start=1.1, deg=3)
    assert np.isclose(fits["iso"](1.0), true(1.0), atol=1e-6)
    assert np.isclose(fits["zz"](2.0), 2.0 * true(2.0), atol=1e-6)


# ============================================================
# CLI PLUMBING
# ============================================================


def test_ring_spec_is_one_based(data_dir: Path) -> None:
    """An explicit 1-based ring spec maps onto the perceived ring."""
    mol = load_geometry(data_dir / "benzene/benzene.in")
    rings = select_rings(mol, "1,2,3,4,5,6")
    assert sorted(rings[0]) == [0, 1, 2, 3, 4, 5]


def test_parser_defaults(data_dir: Path) -> None:
    """The scan parser binds a handler and sensible grid defaults."""
    parser = _build_parser()
    args = parser.parse_args(["scan", str(data_dir / "benzene/benzene.in")])
    assert args.command == "scan"
    assert args.range == [0.0, 4.0]
    assert args.step == 0.1
    assert hasattr(args, "func")
