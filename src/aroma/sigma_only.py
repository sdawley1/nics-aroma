#!/usr/bin/env python3

"""
Sigma-only model dissection of NICS_pizz.

Computes the pi contribution to the out-of-plane (zz) NICS by the sigma-only
model (Stanger, J. Org. Chem. 2010, 75, 2281): the delocalized molecule is
scanned as usual, then an artificial "sigma-only" model is built by freezing the
geometry and adding one hydrogen to every pi-center, perpendicular to the
molecular plane on the face opposite the probes. Each added H localizes a
carbon's pz electron into a C-H bond, quenching the pi ring current while
leaving the sigma framework intact. The pi contribution is the difference

    NICS_pizz(r) = NICS_zz(delocalized) - NICS_zz(model).

A good model satisfies the built-in check NICS_zz ~ 3 * dNICS_iso; the deviation
from this identity is reported as a quality indicator.

This reuses the ordinary shielding backend unchanged; no per-MO decomposition is
required. The whole pi-system is saturated (not just the probed ring) so that
neighboring-ring currents are removed as well.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ----- numerics -----
import numpy as np
import numpy.typing as npt

# ----- local modules -----
from aroma.backend.base import ShieldingBackend
from aroma.connectivity import adjacency_list, bond_matrix
from aroma.constants import (
    DEFAULT_BQ_RANGE,
    DEFAULT_BQ_STEP,
    DEFAULT_FIT_START,
    DEFAULT_SOM_H_DISTANCE,
)
from aroma.geometry import centroid, plane_normal
from aroma.grid import axial_grid
from aroma.molecule import Molecule
from aroma.reorient import reorient_ring_to_xy
from aroma.rings import find_rings

# ============================================================
# CONSTANTS
# ============================================================

# Max atom deviation (angstrom) from the best-fit plane for the molecule-level
# scan, whose shared molecular normal and one-face model assume a planar system.
_PLANARITY_TOL_A = 0.3

# ============================================================
# RESULT CONTAINER
# ============================================================


@dataclass(frozen=True)
class SigmaOnlyResult:
    """Sigma-only NICS_pizz components (ppm) along an axial scan.

    Each array is indexed by probe; ``distances`` holds the probe heights
    (angstrom) above the ring centroid. ``som_deviation`` is the residual of the
    model's built-in identity ``NICS_zz ~ 3 * dNICS_iso`` and should be near zero.
    """

    distances: npt.NDArray[np.float64]
    nics_pi_zz: npt.NDArray[np.float64]
    nics_zz_real: npt.NDArray[np.float64]
    nics_zz_model: npt.NDArray[np.float64]
    nics_iso_real: npt.NDArray[np.float64]
    nics_iso_model: npt.NDArray[np.float64]
    three_delta_iso: npt.NDArray[np.float64]
    som_deviation: npt.NDArray[np.float64]
    pi_centers: Tuple[int, ...]


# ============================================================
# PI-CENTER PERCEPTION
# ============================================================


def perceive_pi_centers(mol: Molecule) -> List[int]:
    """Carbon atoms of the conjugated pi-system to localize in the model.

    For an all-carbon aromatic this is every ring carbon (the union of all
    perceived rings). Heteroatom rings carry lone pairs whose electron count and
    model charge are not handled here and raise.

    Parameters
    ----------
    mol : Molecule
        Real-atom geometry.

    Returns
    -------
    centers : list of int
        Sorted 0-based indices of the pi-center carbons.
    """
    rings = find_rings(adjacency_list(bond_matrix(mol)))
    assert rings, "no rings detected; cannot identify pi-centers"
    ring_atoms = sorted({a for ring in rings for a in ring})
    carbons = [a for a in ring_atoms if int(mol.numbers[a]) == 6]
    assert len(carbons) == len(ring_atoms), (
        "sigma-only model currently supports all-carbon pi-systems; heteroatom "
        "rings need explicit pi-centers and a model charge"
    )
    return carbons


# ============================================================
# MODEL CONSTRUCTION
# ============================================================


def sigma_only_model(
    reoriented: Molecule,
    pi_centers: List[int],
    h_distance: float = DEFAULT_SOM_H_DISTANCE,
) -> Molecule:
    """Build the sigma-only model from a ring-frame (reoriented) geometry.

    Assumes ``reoriented`` already has the probed ring in the XY plane with the
    ring normal along +z (the probe direction), so each localizing H is placed at
    ``-z`` (the face opposite the probes), ``h_distance`` angstrom below its
    carbon. The charge is bumped by one only when an odd number of hydrogens are
    added, keeping the model an even-electron closed-shell singlet.

    Parameters
    ----------
    reoriented : Molecule
        Real-atom geometry already reoriented into the ring frame.
    pi_centers : list of int
        Indices of the carbons to saturate.
    h_distance : float
        Perpendicular C-H placement distance (angstrom).

    Returns
    -------
    Molecule
        The frozen geometry plus one H per pi-center, with a closed-shell charge.
    """
    assert bool(reoriented.real_mask().all()), "model must contain only real atoms"
    assert len(pi_centers) >= 1, "need at least one pi-center"
    assert h_distance > 0.0, "H placement distance must be positive"

    offset = np.array([0.0, 0.0, -h_distance], dtype=np.float64)
    h_coords = reoriented.coords[pi_centers] + offset
    numbers = np.concatenate(
        [reoriented.numbers, np.ones(len(pi_centers), dtype=np.int_)]
    )
    coords = np.vstack([reoriented.coords, h_coords])
    charge = reoriented.charge + (len(pi_centers) % 2)
    return Molecule(numbers=numbers, coords=coords, charge=charge, mult=1)


def sigma_only_model_global(
    mol: Molecule,
    pi_centers: List[int],
    normal: npt.NDArray[np.float64],
    h_distance: float = DEFAULT_SOM_H_DISTANCE,
) -> Molecule:
    """Build the sigma-only model in the molecule's own frame.

    Like :func:`sigma_only_model`, but places each localizing H along ``-normal``
    (the molecular plane normal) rather than ``-z``, so one model serves every
    ring of a molecule-level scan. The probes sit on the ``+normal`` face.

    Parameters
    ----------
    mol : Molecule
        Real-atom geometry (unreoriented).
    pi_centers : list of int
        Indices of the carbons to saturate.
    normal : (3,) float array
        Unit molecular-plane normal; H's go on the ``-normal`` face.
    h_distance : float
        Perpendicular C-H placement distance (angstrom).
    """
    assert bool(mol.real_mask().all()), "model must contain only real atoms"
    assert len(pi_centers) >= 1, "need at least one pi-center"
    assert h_distance > 0.0, "H placement distance must be positive"
    assert normal.shape == (3,), "normal must be a 3-vector"

    h_coords = mol.coords[pi_centers] - h_distance * normal
    numbers = np.concatenate([mol.numbers, np.ones(len(pi_centers), dtype=np.int_)])
    coords = np.vstack([mol.coords, h_coords])
    charge = mol.charge + (len(pi_centers) % 2)
    return Molecule(numbers=numbers, coords=coords, charge=charge, mult=1)


# ============================================================
# COMPONENT EXTRACTION
# ============================================================


def _iso_zz(
    tensors: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Isotropic and zz NICS (negated shielding) from a stack of tensors.

    The sigma-only model needs only the isotropic average and the zz (out-of-
    plane, +z is the ring normal) component, so the full eigen-decomposition used
    for ``oop``/``inp`` is unnecessary here.
    """
    assert tensors.ndim == 3 and tensors.shape[1:] == (3, 3), "tensors must be (M,3,3)"
    sym = 0.5 * (tensors + tensors.transpose(0, 2, 1))
    iso = -np.trace(sym, axis1=1, axis2=2) / 3.0
    zz = -sym[:, 2, 2]
    return iso, zz


def _iso_zz_along(
    tensors: npt.NDArray[np.float64], normal: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Isotropic and out-of-plane NICS for tensors in an arbitrary frame.

    Like :func:`_iso_zz`, but the out-of-plane component is taken along ``normal``
    (``zz = -n . sigma . n``) instead of the +z axis, so the molecule-level scan
    can keep tensors in the input frame and avoid a per-ring reorientation.
    """
    assert tensors.ndim == 3 and tensors.shape[1:] == (3, 3), "tensors must be (M,3,3)"
    assert normal.shape == (3,), "normal must be a 3-vector"
    sym = 0.5 * (tensors + tensors.transpose(0, 2, 1))
    iso = -np.trace(sym, axis1=1, axis2=2) / 3.0
    zz = -np.einsum("i,nij,j->n", normal, sym, normal)
    return iso, zz


# ============================================================
# SCAN DRIVER
# ============================================================


def run_sigma_only_scan(
    mol: Molecule,
    ring: List[int],
    backend: ShieldingBackend,
    pi_centers: Optional[List[int]] = None,
    start: float = DEFAULT_BQ_RANGE[0],
    stop: float = DEFAULT_BQ_RANGE[1],
    step: float = DEFAULT_BQ_STEP,
    h_distance: float = DEFAULT_SOM_H_DISTANCE,
) -> SigmaOnlyResult:
    """Run a sigma-only NICS_pizz scan over one ring.

    Reorients into the ring frame once, builds the sigma-only model in that same
    frame, and runs the shared probe grid against both the delocalized molecule
    and the model so the two scans subtract cleanly.

    Parameters
    ----------
    mol : Molecule
        Input geometry.
    ring : list of int
        Atom indices of the ring to probe (any order).
    backend : ShieldingBackend
        Provider of GIAO shielding tensors.
    pi_centers : list of int, optional
        Carbons to saturate; defaults to the whole conjugated pi-system.
    start, stop, step : float
        Axial grid extent and spacing (angstrom).
    h_distance : float
        Perpendicular C-H placement distance for the model (angstrom).

    Returns
    -------
    SigmaOnlyResult
        NICS_pizz and supporting components at each probe height.
    """
    assert len(ring) >= 3, "a ring needs at least three atoms"

    reoriented, _ = reorient_ring_to_xy(mol, ring)
    grid = axial_grid(start, stop, step)
    centers = list(pi_centers) if pi_centers is not None else perceive_pi_centers(mol)
    model = sigma_only_model(reoriented, centers, h_distance)

    real_tensors = backend.shielding(reoriented, grid)
    model_tensors = backend.shielding(model, grid)
    assert real_tensors.shape[0] == grid.shape[0], "backend returned wrong probe count"
    assert model_tensors.shape[0] == grid.shape[0], "model probe count mismatch"

    iso_real, zz_real = _iso_zz(real_tensors)
    iso_model, zz_model = _iso_zz(model_tensors)
    nics_pi_zz = zz_real - zz_model
    three_delta_iso = 3.0 * (iso_real - iso_model)
    return SigmaOnlyResult(
        distances=grid[:, 2].copy(),
        nics_pi_zz=nics_pi_zz,
        nics_zz_real=zz_real,
        nics_zz_model=zz_model,
        nics_iso_real=iso_real,
        nics_iso_model=iso_model,
        three_delta_iso=three_delta_iso,
        som_deviation=nics_pi_zz - three_delta_iso,
        pi_centers=tuple(centers),
    )


# ============================================================
# MOLECULE-LEVEL SCAN (one real + one model SCF for all rings)
# ============================================================


def run_sigma_only_molecule(
    mol: Molecule,
    rings: List[List[int]],
    backend: ShieldingBackend,
    pi_centers: Optional[List[int]] = None,
    start: float = DEFAULT_BQ_RANGE[0],
    stop: float = DEFAULT_BQ_RANGE[1],
    step: float = DEFAULT_BQ_STEP,
    h_distance: float = DEFAULT_SOM_H_DISTANCE,
) -> List[Tuple[List[int], SigmaOnlyResult]]:
    """Sigma-only NICS_pizz for every ring of a molecule in two SCFs total.

    The real-molecule and sigma-only-model GIAO shieldings do not depend on which
    ring is probed, so they are each computed once, with the axial probes of all
    rings stacked into a single ghost set. The out-of-plane component is read
    along the shared molecular normal, avoiding any per-ring reorientation. This
    replaces the per-ring driver's ``2 * len(rings)`` SCFs with exactly two.

    Parameters
    ----------
    mol : Molecule
        Input geometry (planar pi-system assumed).
    rings : list of list of int
        Atom indices of each ring to probe.
    backend : ShieldingBackend
        Provider of GIAO shielding tensors.
    pi_centers : list of int, optional
        Carbons to saturate; defaults to the whole conjugated pi-system.
    start, stop, step : float
        Axial grid extent and spacing (angstrom).
    h_distance : float
        Perpendicular C-H placement distance for the model (angstrom).

    Returns
    -------
    list of (ring, SigmaOnlyResult)
        One result per input ring, in ``rings`` order.
    """
    assert rings, "need at least one ring"
    normal = plane_normal(mol.coords)
    out_of_plane = np.abs((mol.coords - mol.coords.mean(axis=0)) @ normal)
    assert float(out_of_plane.max()) < _PLANARITY_TOL_A, (
        "molecule-level sigma-only scan assumes a planar pi-system; "
        f"max out-of-plane deviation is {float(out_of_plane.max()):.2f} A"
    )

    centers = list(pi_centers) if pi_centers is not None else perceive_pi_centers(mol)
    distances = axial_grid(start, stop, step)[:, 2]
    n_probe = distances.size

    # Stack every ring's axial probes (centroid + d * normal) into one ghost set.
    blocks = [
        centroid(mol.coords[ring]) + np.outer(distances, normal) for ring in rings
    ]
    probes = np.vstack(blocks)

    model = sigma_only_model_global(mol, centers, normal, h_distance)
    iso_real, zz_real = _iso_zz_along(backend.shielding(mol, probes), normal)
    iso_model, zz_model = _iso_zz_along(backend.shielding(model, probes), normal)

    results: List[Tuple[List[int], SigmaOnlyResult]] = []
    for i, ring in enumerate(rings):
        sl = slice(i * n_probe, (i + 1) * n_probe)
        pi_zz = zz_real[sl] - zz_model[sl]
        three_delta_iso = 3.0 * (iso_real[sl] - iso_model[sl])
        results.append((ring, SigmaOnlyResult(
            distances=distances.copy(),
            nics_pi_zz=pi_zz,
            nics_zz_real=zz_real[sl],
            nics_zz_model=zz_model[sl],
            nics_iso_real=iso_real[sl],
            nics_iso_model=iso_model[sl],
            three_delta_iso=three_delta_iso,
            som_deviation=pi_zz - three_delta_iso,
            pi_centers=tuple(centers),
        )))
    return results


# ============================================================
# CURVE FITTING
# ============================================================


def fit_pi_zz(
    result: SigmaOnlyResult, dist_start: float = DEFAULT_FIT_START, deg: int = 3
) -> float:
    """Fit NICS_pizz versus distance and evaluate it at 1 angstrom.

    Mirrors the package's NICS(1) convention: only probes at or beyond
    ``dist_start`` are fitted (avoiding the basis-sensitive near-ring region).

    Parameters
    ----------
    result : SigmaOnlyResult
        A sigma-only NICS_pizz scan.
    dist_start : float
        Minimum probe distance (angstrom) included in the fit.
    deg : int
        Polynomial degree.

    Returns
    -------
    float
        The fitted NICS_pizz at 1 angstrom (ppm).
    """
    assert deg >= 1, "polynomial degree must be at least 1"
    mask = result.distances >= dist_start
    assert int(mask.sum()) > deg, (
        f"need more than {deg} probes beyond {dist_start} A to fit degree {deg}"
    )
    poly = np.poly1d(np.polyfit(result.distances[mask], result.nics_pi_zz[mask], deg))
    return float(poly(1.0))
