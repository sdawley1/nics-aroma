#!/usr/bin/env python3

"""
End-to-end tests for the precomputed-Gaussian NICS reuse path.

A Gaussian nmr=giao log that already carries Bq shielding must be reported
directly, without invoking (or even importing) the PySCF backend.

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import subprocess
import sys
from pathlib import Path

# ----- third party -----
import pytest

# ----- local modules -----
from aroma.cli import main

# ============================================================
# TESTS
# ============================================================


def test_cli_scan_reuses_shielding(
    data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``aroma scan`` on the nmr=giao log prints NICS straight from the log."""
    fixture = data_dir / "benzene/benzene-nics.log"
    code = main(["scan", str(fixture)])
    assert code == 0

    out = capsys.readouterr().out
    assert "reusing GIAO shielding" in out
    center = [ln for ln in out.splitlines() if ln.strip().startswith("0.00")]
    assert center, out
    iso_center = float(center[0].split()[1])
    assert iso_center == pytest.approx(-11.5, abs=1.0)


def test_reuse_path_does_not_import_pyscf(data_dir: Path) -> None:
    """The reuse path must never import PySCF (core-only install support)."""
    fixture = data_dir / "benzene/benzene-nics.log"
    script = (
        "import sys; from aroma.cli import main; "
        f"rc = main(['scan', {str(fixture)!r}]); "
        "assert rc == 0, rc; "
        "assert 'pyscf' not in sys.modules, 'pyscf was imported on the reuse path'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
