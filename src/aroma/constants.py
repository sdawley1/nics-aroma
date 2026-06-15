#!/usr/bin/env python3

"""
Physical and method defaults for NICS scans.

Geometry tolerances for connectivity/planarity detection, default Bq grid
parameters, and default PySCF method/basis settings. All Gaussian executable
paths from the legacy code are intentionally dropped.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Tuple

# ============================================================
# GEOMETRY TOLERANCES
# ============================================================

# Two atoms are bonded if their separation is within this slack (angstrom) of
# the sum of their covalent radii.
COVALENT_BOND_TOLERANCE = 0.45

# Maximum dihedral deviation (degrees) tolerated when testing ring planarity.
PLANARITY_TOLERANCE_DEG = 15.0

# Largest ring size (atoms) considered during ring perception.
MAX_RING_SIZE = 12

# ============================================================
# BQ GRID DEFAULTS
# ============================================================

# Axial scan: probe points placed every BQ_STEP angstrom over BQ_RANGE along
# the ring normal, measured from the ring centroid.
DEFAULT_BQ_STEP = 0.1
DEFAULT_BQ_RANGE: Tuple[float, float] = (0.0, 4.0)

# Out-of-plane offset (angstrom) for an in-plane (XY) scan.
DEFAULT_XY_DISTANCE = 1.7

# Curve fitting ignores probes closer than this distance (angstrom) to the ring.
DEFAULT_FIT_START = 1.1

# ============================================================
# PYSCF METHOD DEFAULTS
# ============================================================

DEFAULT_METHOD = "b3lyp"
DEFAULT_BASIS = "6-311+g*"
DEFAULT_XC_GRID = "ultrafine"

# ============================================================
# SIGMA-ONLY MODEL DEFAULTS
# ============================================================

# Distance (angstrom) at which a localizing H is placed below each pi-center
# when building the sigma-only model. The review chapter (Gershoni-Poranne &
# Stanger) states "approximately 1 angstrom".
# TODO: confirm the exact value against Stanger, J. Org. Chem. 2010, 75, 2281.
DEFAULT_SOM_H_DISTANCE = 1.0

# Warn when the sigma-only model's built-in check |NICS_zz - 3*dNICS_iso|
# exceeds this many ppm, indicating a poor (inaccurate) model.
SOM_DEVIATION_WARN = 2.0
