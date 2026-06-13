#!/usr/bin/env python3

"""
Command-line interface for NICS scans.

The ``aroma scan`` subcommand loads a geometry, perceives (or accepts) aromatic
rings, runs an axial NICS scan per ring via the PySCF backend, and prints a
NICS-vs-distance table plus the fitted NICS(1) value.

Note: a packaged console-script requires a callable entry point, so this module
exposes ``main`` (referenced by ``[project.scripts]``) in addition to the
``__main__`` block.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import sys
from pathlib import Path
from typing import List, Optional

# ----- local modules -----
from aroma.analysis import fit_nics_curve
from aroma.backend.base import ShieldingBackend
from aroma.batch import scan_paths, select_rings
from aroma.constants import (
    DEFAULT_BASIS,
    DEFAULT_BQ_RANGE,
    DEFAULT_BQ_STEP,
    DEFAULT_FIT_START,
    DEFAULT_METHOD,
)
from aroma.io import load_geometry
from aroma.nics import NicsResult, run_nics_scan

# ============================================================
# CONSTANTS
# ============================================================

# Degree of the NICS-vs-distance polynomial fit reported by the CLI.
_FIT_DEGREE = 3

# ============================================================
# OUTPUT
# ============================================================


def _print_scan(ring: List[int], result: NicsResult, fit_start: float) -> None:
    """Print one ring's NICS table and the fitted NICS(1) summary."""
    one_based = ",".join(str(a + 1) for a in ring)
    print(f"\n# Ring atoms (1-based): {one_based}", flush=True)
    print(f"# {'dist':>6} {'iso':>10} {'zz':>10} {'oop':>10} {'inp':>10}", flush=True)
    for d, iso, zz, oop, inp in zip(
        result.distances, result.nics_iso, result.nics_zz,
        result.nics_oop, result.nics_inp,
    ):
        print(f"  {d:6.2f} {iso:10.3f} {zz:10.3f} {oop:10.3f} {inp:10.3f}", flush=True)

    if int((result.distances >= fit_start).sum()) <= _FIT_DEGREE:
        print(
            f"# (skipping fit: need > {_FIT_DEGREE} probes beyond {fit_start} A)",
            flush=True,
        )
        return
    fits = fit_nics_curve(result, dist_start=fit_start, deg=_FIT_DEGREE)
    nics1 = {name: float(poly(1.0)) for name, poly in fits.items()}
    print(
        f"# Fitted NICS(1): iso={nics1['iso']:.3f} zz={nics1['zz']:.3f} "
        f"oop={nics1['oop']:.3f} inp={nics1['inp']:.3f}",
        flush=True,
    )


# ============================================================
# SCAN COMMAND
# ============================================================


def _load_backend(method: str, basis: str) -> Optional[ShieldingBackend]:
    """Instantiate the PySCF backend, or print guidance and return None."""
    try:
        from aroma.backend.pyscf_nmr import PyscfNmrBackend
    except ImportError:
        print(
            "error: the PySCF backend is required; install with "
            "'pip install aroma-nics[pyscf]'",
            file=sys.stderr, flush=True,
        )
        return None
    return PyscfNmrBackend(method=method, basis=basis)


def _run_scan(args: argparse.Namespace) -> int:
    """Execute the ``scan`` subcommand; return a process exit code."""
    backend = _load_backend(args.method, args.basis)
    if backend is None:
        return 2

    mol = load_geometry(args.geometry)
    rings = select_rings(mol, args.ring)
    start, stop = args.range
    print(
        f"# {args.geometry.name}: {len(rings)} ring(s), "
        f"{args.method}/{args.basis}",
        flush=True,
    )
    for ring in rings:
        result = run_nics_scan(mol, ring, backend, start, stop, args.step)
        _print_scan(ring, result, args.fit_start)
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    """Execute the ``batch`` subcommand over several geometries; return code."""
    backend = _load_backend(args.method, args.basis)
    if backend is None:
        return 2

    start, stop = args.range
    print(f"# batch: {len(args.geometries)} file(s), {args.method}/{args.basis}",
          flush=True)
    for path, ring, result in scan_paths(
        args.geometries, backend, start, stop, args.step
    ):
        print(f"\n## {path.name}", flush=True)
        _print_scan(ring, result, args.fit_start)
    return 0


# ============================================================
# PARSER
# ============================================================


def _add_common_options(p: argparse.ArgumentParser) -> None:
    """Add the method/basis/grid/analysis options shared by scan and batch."""
    # ----- method -----
    p.add_argument("--method", default=DEFAULT_METHOD, help="'hf' or a DFT functional")
    p.add_argument("--basis", default=DEFAULT_BASIS, help="orbital basis set")
    # ----- grid -----
    p.add_argument(
        "--range", nargs=2, type=float, default=list(DEFAULT_BQ_RANGE),
        metavar=("START", "STOP"), help="axial scan range (angstrom)",
    )
    p.add_argument(
        "--step", type=float, default=DEFAULT_BQ_STEP, help="probe spacing (angstrom)"
    )
    # ----- analysis -----
    p.add_argument(
        "--fit-start", type=float, default=DEFAULT_FIT_START,
        help="fit probes at/beyond this distance (angstrom)",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="aroma", description="NICS scans via PySCF GIAO NMR."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="run an axial NICS scan on one geometry")
    scan.add_argument("geometry", type=Path, help="geometry file (.com/.log/.xyz/...)")
    scan.add_argument(
        "--ring", default="auto",
        help="'auto' (perceive rings) or comma-separated 1-based atom indices",
    )
    _add_common_options(scan)
    scan.set_defaults(func=_run_scan)

    batch = sub.add_parser("batch", help="scan several geometries (rings auto)")
    batch.add_argument("geometries", type=Path, nargs="+", help="geometry files")
    _add_common_options(batch)
    batch.set_defaults(func=_run_batch)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    assert hasattr(args, "func"), "no subcommand handler bound"
    exit_code = int(args.func(args))
    return exit_code


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    sys.exit(main())
