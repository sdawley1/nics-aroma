# aroma

Nucleus-independent chemical shift (NICS) scans for aromatic molecules, computed
in-process with [PySCF](https://pyscf.org) GIAO NMR shielding.

`aroma` perceives aromatic rings from a molecular geometry, reorients each ring
into the XY plane, places a grid of ghost (Bq) probes along the ring normal, and
evaluates the GIAO shielding tensor at every probe in a single SCF — reporting
the isotropic, out-of-plane (`zz`), and in-plane NICS components plus a fitted
NICS(1).

## Installation

```bash
pip install aroma-nics[pyscf]
```

The NMR shielding code lives in the separate PySCF `properties` extension, which
is pulled in by the `pyscf` extra. The core library (geometry, ring perception,
grids, analysis) installs without it:

```bash
pip install aroma-nics          # core only (no shielding backend)
```

## Command line

```bash
# Scan one geometry; rings are perceived automatically.
aroma scan benzene.com --method b3lyp --basis 6-311+g* --range 0 4 --step 0.1

# Restrict to a specific ring by 1-based atom indices.
aroma scan indene.xyz --ring 1,2,3,4,5

# Scan several geometries in one process.
aroma batch a.com b.com c.log --method hf --basis sto-3g
```

Accepted geometry formats: Gaussian input (`.com`/`.gjf`/`.in`), Gaussian output
(`.log`/`.out`), formatted checkpoint (`.fchk`), and XYZ (`.xyz`).

## Library

```python
from aroma import load_geometry
from aroma.connectivity import adjacency_list, bond_matrix
from aroma.rings import find_rings
from aroma.nics import run_nics_scan
from aroma.backend.pyscf_nmr import PyscfNmrBackend

mol = load_geometry("benzene.com")
ring = find_rings(adjacency_list(bond_matrix(mol)))[0]
result = run_nics_scan(mol, ring, PyscfNmrBackend(method="b3lyp", basis="6-311+g*"))
print(result.distances, result.nics_iso, result.nics_zz)
```

NICS probes are placed as bare ghost atoms with **no basis functions**,
reproducing Gaussian's `Bq` convention; benzene NICS(1) lands near the textbook
−10 ppm.

## Development

```bash
pip install -e .[dev,pyscf]
ruff check src tests
mypy src
pytest                  # add -m "not slow" to skip the PySCF integration test
```

## License

MIT
