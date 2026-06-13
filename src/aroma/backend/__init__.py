#!/usr/bin/env python3

"""
Shielding backends.

Defines the :class:`ShieldingBackend` protocol and exposes the default PySCF
GIAO NMR implementation.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- local modules -----
from aroma.backend.base import ShieldingBackend

__all__ = ["ShieldingBackend"]
