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
from aroma.batch import select_rings
from aroma.connectivity import adjacency_list, bond_matrix
from aroma.io import load_geometry
from aroma.nics import run_nics_scan
from aroma.rings import find_rings
from aroma.sigma_only import run_sigma_only_molecule, run_sigma_only_scan

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


@pytest.mark.slow
def test_benzene_sigma_only_pizz(data_dir: Path) -> None:
    """Benzene sigma-only NICS_pizz is strongly diatropic and self-consistent.

    Two probes (0 and 1 A) keep the SCFs small. At 1 A the pi contribution to the
    out-of-plane NICS should be clearly negative (aromatic, diatropic), and the
    model's built-in identity NICS_zz ~ 3*dNICS_iso should nearly hold (small
    deviation). Bands are loose: the absolute value is method/basis-dependent and
    the exact H-placement distance is not yet locked to the literature.
    """
    from aroma.backend.pyscf_nmr import PyscfNmrBackend

    mol = load_geometry(data_dir / "benzene/benzene.in")
    ring = find_rings(adjacency_list(bond_matrix(mol)))[0]

    backend = PyscfNmrBackend(method="hf", basis="sto-3g")
    result = run_sigma_only_scan(mol, ring, backend, start=0.0, stop=1.0, step=1.0)

    assert np.isfinite(result.nics_pi_zz).all()
    pi_zz_1 = float(result.nics_pi_zz[-1])
    deviation_1 = float(result.som_deviation[-1])
    assert pi_zz_1 < -8.0, f"NICS_pizz(1)={pi_zz_1:.3f} (expected strongly diatropic)"
    assert abs(deviation_1) < 5.0, f"SOM deviation at 1 A = {deviation_1:.3f} ppm"

    # The molecule-level (two-SCF) driver must reproduce the per-ring driver.
    (mol_ring, mol_result), = run_sigma_only_molecule(
        mol, [ring], backend, start=0.0, stop=1.0, step=1.0
    )
    assert mol_ring == ring
    assert np.allclose(mol_result.nics_pi_zz, result.nics_pi_zz, atol=1e-4)


@pytest.mark.slow
def test_sigma_only_molecule_matches_per_ring(data_dir: Path) -> None:
    """The two-SCF molecule driver reproduces the per-ring driver on every ring.

    Phenalene has three rings, so the per-ring path runs 6 SCFs and the
    molecule path 2; both must give the same NICS_pizz curves.
    """
    from aroma.backend.pyscf_nmr import PyscfNmrBackend

    mol = load_geometry(data_dir / "phenalene/phenalene.in")
    rings = select_rings(mol, "auto", planar_only=False)
    backend = PyscfNmrBackend(method="hf", basis="sto-3g")

    molecule_results = run_sigma_only_molecule(
        mol, rings, backend, start=0.0, stop=1.0, step=0.5
    )
    for ring, mol_result in molecule_results:
        per_ring = run_sigma_only_scan(
            mol, ring, backend, start=0.0, stop=1.0, step=0.5
        )
        assert np.allclose(mol_result.nics_pi_zz, per_ring.nics_pi_zz, atol=1e-3)
        assert np.allclose(mol_result.nics_iso_real, per_ring.nics_iso_real, atol=1e-3)
