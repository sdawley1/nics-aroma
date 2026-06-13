#!/usr/bin/env python3

"""
PySCF GIAO NMR shielding backend.

Builds a PySCF ``Mole`` containing the real atoms plus the Bq probe points as
zero-charge ghost atoms, runs a single SCF, and evaluates GIAO NMR shielding
tensors for every center in one call. The whole probe grid is batched into one
calculation; with empty-basis ghosts the extra centers add negligible cost.

Requires the optional ``pyscf`` extra (``pip install aroma-nics[pyscf]``).

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import contextlib
import io
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# ----- numerics -----
import numpy as np
import numpy.typing as npt

# ----- quantum chemistry -----
from pyscf import dft, gto, scf
from pyscf.prop.nmr import rhf as nmr_rhf
from pyscf.prop.nmr import rks as nmr_rks

# ----- local modules -----
from aroma.constants import DEFAULT_BASIS, DEFAULT_METHOD, DEFAULT_XC_GRID
from aroma.elements import number_to_symbol
from aroma.molecule import Molecule

# ============================================================
# CONSTANTS
# ============================================================

# Map a Gaussian-style integration-grid name to a PySCF grid level.
_GRID_LEVEL: Dict[str, int] = {"coarse": 2, "fine": 3, "ultrafine": 5}

# Label for ghost (Bq) centers. A bare "X" with no basis-dict entry is treated
# by PySCF as a zero-charge dummy carrying no basis functions, reproducing
# Gaussian's bare Bq probe (no ghost-basis / BSSE-like contamination).
_GHOST_LABEL = "X"

# ============================================================
# PYSCF BACKEND
# ============================================================


@dataclass(frozen=True)
class PyscfNmrBackend:
    """GIAO NMR shielding via PySCF (restricted, closed-shell).

    Parameters
    ----------
    method : str
        ``"hf"`` for Hartree-Fock, otherwise a DFT exchange-correlation name
        (e.g. ``"b3lyp"``).
    basis : str
        Orbital basis set for the real atoms. The Bq probes carry no basis.
    xc_grid : str
        Integration-grid preset for DFT (see ``_GRID_LEVEL``).
    """

    method: str = DEFAULT_METHOD
    basis: str = DEFAULT_BASIS
    xc_grid: str = DEFAULT_XC_GRID

    def _build_mole(
        self, mol: Molecule, ghosts: npt.NDArray[np.float64]
    ) -> "gto.Mole":
        """Assemble a PySCF Mole with real atoms followed by ghost probes."""
        assert mol.mult == 1, "PySCF backend supports closed-shell (mult=1) only"
        real = mol.real_mask()
        atoms: List[Tuple[str, Tuple[float, float, float]]] = [
            (number_to_symbol(int(z)), tuple(xyz))
            for z, xyz in zip(mol.numbers[real], mol.coords[real])
        ]
        atoms += [(_GHOST_LABEL, tuple(pt)) for pt in ghosts]

        # Real elements only; the ghost label is deliberately absent so PySCF
        # places no basis functions on the probes.
        basis: Dict[str, Any] = {
            number_to_symbol(int(z)): self.basis for z in np.unique(mol.numbers[real])
        }
        # PySCF writes a harmless "Basis not found for atom X" note to stderr
        # for the basis-less ghosts; suppress it (exceptions still propagate).
        with contextlib.redirect_stderr(io.StringIO()):
            return gto.M(
                atom=atoms, basis=basis, charge=mol.charge, spin=0,
                unit="Angstrom", verbose=0,
            )

    def _run_scf(self, mole: "gto.Mole") -> Tuple[Any, Any]:
        """Converge the SCF and return (mean-field, NMR-property object)."""
        if self.method.lower() == "hf":
            mf = scf.RHF(mole).run()
            return mf, nmr_rhf.NMR(mf)
        mf = dft.RKS(mole)
        mf.xc = self.method
        mf.grids.level = _GRID_LEVEL.get(self.xc_grid, 3)
        mf.run()
        return mf, nmr_rks.NMR(mf)

    def shielding(
        self, mol: Molecule, ghosts: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Shielding tensors (ppm) at the ghost probe points.

        Parameters
        ----------
        mol : Molecule
            The real molecule (ghost atoms in ``mol`` are ignored).
        ghosts : (M, 3) float array
            Probe coordinates (angstrom).

        Returns
        -------
        tensors : (M, 3, 3) float array
            One GIAO shielding tensor per probe, in Mole order.
        """
        assert ghosts.ndim == 2 and ghosts.shape[1] == 3, "ghosts must be (M, 3)"
        assert ghosts.shape[0] >= 1, "need at least one probe point"

        mole = self._build_mole(mol, ghosts)
        n_real = int(mol.real_mask().sum())
        _, prop = self._run_scf(mole)
        msc = np.asarray(prop.kernel(), dtype=np.float64)

        tensors = msc[n_real:]
        assert tensors.shape == (ghosts.shape[0], 3, 3), "unexpected shielding shape"
        return tensors
