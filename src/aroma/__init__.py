#!/usr/bin/env python3

"""
aroma: nucleus-independent chemical shift (NICS) scans for aromatic molecules.

Top-level re-exports for the common entry points.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- local modules -----
from aroma.io import load_geometry
from aroma.molecule import Molecule

# ============================================================
# PUBLIC API
# ============================================================

__version__ = "0.1.0"

__all__ = ["Molecule", "load_geometry", "__version__"]
