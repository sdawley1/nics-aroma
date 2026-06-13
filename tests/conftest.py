#!/usr/bin/env python3

"""
Shared pytest fixtures: paths to bundled sample geometries.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from pathlib import Path

# ----- third party -----
import pytest

# ============================================================
# FIXTURES
# ============================================================

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def data_dir() -> Path:
    """Directory holding the bundled sample geometries."""
    return DATA_DIR
