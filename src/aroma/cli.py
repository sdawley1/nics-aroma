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
from aroma.batch import _axial_job, _xy_job, scan_paths, select_rings
from aroma.constants import (
    DEFAULT_BASIS,
    DEFAULT_BQ_RANGE,
    DEFAULT_BQ_STEP,
    DEFAULT_FIT_START,
    DEFAULT_METHOD,
    DEFAULT_XY_DISTANCE,
)
from aroma.io import load_geometry, log_has_bq_shielding, read_log_nics
from aroma.nics import (
    NicsResult,
    XyNicsResult,
    match_probe_ring,
    nics_from_precomputed,
)
from aroma.parallel import parallel_map

# ============================================================
# CONSTANTS
# ============================================================

# Degree of the NICS-vs-distance polynomial fit reported by the CLI.
_FIT_DEGREE = 3

# Warn when reused Bq probes sit this far (angstrom, RMS) off the ring axis.
_OFFAXIS_WARN_A = 0.25

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


def _print_xy_scan(ring: List[int], result: XyNicsResult) -> None:
    """Print one ring's in-plane NICS table over the XY scan grid."""
    one_based = ",".join(str(a + 1) for a in ring)
    print(f"\n# Ring atoms (1-based): {one_based}", flush=True)
    print(
        f"# {'x':>6} {'y':>6} {'iso':>10} {'zz':>10} {'oop':>10} {'inp':>10}",
        flush=True,
    )
    for x, y, iso, zz, oop, inp in zip(
        result.xs, result.ys, result.nics_iso,
        result.nics_zz, result.nics_oop, result.nics_inp,
    ):
        print(
            f"  {x:6.2f} {y:6.2f} {iso:10.3f} {zz:10.3f} {oop:10.3f} {inp:10.3f}",
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


def _is_reusable_log(path: Path) -> bool:
    """True if the geometry is a Gaussian log that already holds Bq shielding."""
    return path.suffix.lower() in (".log", ".out") and log_has_bq_shielding(path)


def _run_scan_precomputed(args: argparse.Namespace) -> int:
    """Report a NICS scan directly from Gaussian Bq shielding (no PySCF)."""
    data = read_log_nics(args.geometry)
    rings = select_rings(data.mol, args.ring, args.planar_only)
    ring, off_axis = match_probe_ring(data.mol, rings, data.bq_coords)
    print(
        f"# {args.geometry.name}: reusing GIAO shielding from log "
        f"({data.bq_coords.shape[0]} Bq probes, no recompute)",
        flush=True,
    )
    if off_axis > _OFFAXIS_WARN_A:
        print(
            f"warning: Bq probes sit {off_axis:.2f} A (RMS) off the ring axis",
            file=sys.stderr, flush=True,
        )
    result = nics_from_precomputed(
        data.mol, ring, data.bq_coords, data.bq_tensors
    )
    _print_scan(ring, result, args.fit_start)
    return 0


def _run_scan(args: argparse.Namespace) -> int:
    """Execute the ``scan`` subcommand; return a process exit code."""
    if _is_reusable_log(args.geometry):
        return _run_scan_precomputed(args)

    backend = _load_backend(args.method, args.basis)
    if backend is None:
        return 2

    mol = load_geometry(args.geometry)
    rings = select_rings(mol, args.ring, args.planar_only)
    start, stop = args.range
    print(
        f"# {args.geometry.name}: {len(rings)} ring(s), "
        f"{args.method}/{args.basis}",
        flush=True,
    )
    worklist = [
        (args.geometry, mol, ring, backend, start, stop, args.step) for ring in rings
    ]
    for _path, ring, result in parallel_map(
        _axial_job, worklist, jobs=args.jobs, threads=args.threads
    ):
        _print_scan(ring, result, args.fit_start)
    return 0


def _run_xyscan(args: argparse.Namespace) -> int:
    """Execute the ``xyscan`` subcommand (in-plane scan); return exit code."""
    backend = _load_backend(args.method, args.basis)
    if backend is None:
        return 2

    mol = load_geometry(args.geometry)
    rings = select_rings(mol, args.ring, args.planar_only)
    print(
        f"# {args.geometry.name}: {len(rings)} ring(s), "
        f"{args.method}/{args.basis}, xy half-extent {args.half_extent} A "
        f"at z={args.height} A",
        flush=True,
    )
    worklist = [
        (args.geometry, mol, ring, backend, args.half_extent, args.step, args.height)
        for ring in rings
    ]
    for _path, ring, result in parallel_map(
        _xy_job, worklist, jobs=args.jobs, threads=args.threads
    ):
        _print_xy_scan(ring, result)
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
        args.geometries, backend, start, stop, args.step, args.planar_only,
        args.jobs, args.threads,
    ):
        print(f"\n## {path.name}", flush=True)
        _print_scan(ring, result, args.fit_start)
    return 0


# ============================================================
# PARSER
# ============================================================


def _add_selection_options(p: argparse.ArgumentParser) -> None:
    """Add the method/basis/ring-selection options shared by all scan commands."""
    # ----- method -----
    p.add_argument("--method", default=DEFAULT_METHOD, help="'hf' or a DFT functional")
    p.add_argument("--basis", default=DEFAULT_BASIS, help="orbital basis set")
    # ----- ring selection -----
    p.add_argument(
        "--planar-only", action="store_true",
        help="keep only planar rings when perceiving rings automatically",
    )
    # ----- parallelism -----
    p.add_argument(
        "--jobs", type=int, default=1,
        help="worker processes for independent scans (0 = all cores)",
    )
    p.add_argument(
        "--threads", type=int, default=0,
        help="PySCF threads per worker (0 = auto-split across cores)",
    )


def _add_common_options(p: argparse.ArgumentParser) -> None:
    """Add the method/basis/grid/analysis options shared by scan and batch."""
    _add_selection_options(p)
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

    xyscan = sub.add_parser("xyscan", help="in-plane (XY) NICS scan above a ring")
    xyscan.add_argument(
        "geometry", type=Path, help="geometry file (.com/.log/.xyz/...)"
    )
    xyscan.add_argument(
        "--ring", default="auto",
        help="'auto' (perceive rings) or comma-separated 1-based atom indices",
    )
    _add_selection_options(xyscan)
    xyscan.add_argument(
        "--half-extent", type=float, default=2.0,
        help="half-width of the square scan region (angstrom)",
    )
    xyscan.add_argument(
        "--step", type=float, default=DEFAULT_BQ_STEP, help="lattice spacing (angstrom)"
    )
    xyscan.add_argument(
        "--height", type=float, default=DEFAULT_XY_DISTANCE,
        help="scan-plane height above the ring (angstrom)",
    )
    xyscan.set_defaults(func=_run_xyscan)
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
