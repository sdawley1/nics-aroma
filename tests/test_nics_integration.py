#!/usr/bin/env python3

"""
PySCF NICS integration regression (slow; requires the pyscf extra).

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
import pytest

# ----- local modules -----
from aroma.connectivity import adjacency_list, bond_matrix
from aroma.io import load_geometry
from aroma.nics import run_nics_scan
from aroma.rings import find_rings

pyscf = pytest.importorskip("pyscf", reason="pyscf extra not installed")
pytest.importorskip("pyscf.prop.nmr", reason="pyscf properties extension not installed")

# ============================================================
# TESTS
# ============================================================


@pytest.mark.slow
def test_benzene_nics_regression(data_dir: Path) -> None:
    """Benzene NICS(1) reproduces the locked HF/STO-3G baseline.

    Uses two probes (0 and 1 A) to keep the SCF small. The numbers are
    method/basis-dependent; these bands lock the first measured HF/STO-3G
    values (NICS(1)_iso ~ -10.2, NICS(1)_zz ~ -26.1 ppm). With empty-basis Bq
    probes this lands near the textbook B3LYP/6-311+G* value of ~ -10 ppm.
    """
    from aroma.backend.pyscf_nmr import PyscfNmrBackend

    mol = load_geometry(data_dir / "benzene/benzene.in")
    adj = adjacency_list(bond_matrix(mol))
    ring = find_rings(adj)[0]

    backend = PyscfNmrBackend(method="hf", basis="sto-3g")
    result = run_nics_scan(mol, ring, backend, start=0.0, stop=1.0, step=1.0)

    assert np.isfinite(result.nics_zz).all()
    nics_1_iso = float(result.nics_iso[-1])
    nics_1_zz = float(result.nics_zz[-1])
    assert nics_1_iso == pytest.approx(-10.2, abs=0.6), f"NICS(1)_iso={nics_1_iso:.3f}"
    assert nics_1_zz == pytest.approx(-26.1, abs=1.5), f"NICS(1)_zz={nics_1_zz:.3f}"
