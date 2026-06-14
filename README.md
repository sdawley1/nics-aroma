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

# Keep only planar (aromatic-like) rings when perceiving automatically.
aroma scan phenalene.com --planar-only

# In-plane (XY) NICS map on a plane above the ring.
aroma xyscan benzene.com --half-extent 2.0 --step 0.5 --height 1.7

# Scan several geometries in one process.
aroma batch a.com b.com c.log --method hf --basis sto-3g

# Run independent scans in parallel across 4 worker processes.
aroma batch *.log --jobs 4
```

### Parallelism

Independent `(geometry, ring)` scans are embarrassingly parallel — each is a
self-contained SCF. Pass `--jobs N` (on `scan`, `batch`, or `xyscan`) to run them
across `N` worker processes; `--jobs 0` uses every core. Each worker's PySCF
thread count is auto-split as `cores // jobs` to avoid oversubscription, with
`--threads M` to override. This is portable single-node multiprocessing (no
cluster scheduler required). The default `--jobs 1` runs serially in-process.

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
