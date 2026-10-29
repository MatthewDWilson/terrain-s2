"""ML-3b: the channel U-Net on a tiny synthetic chip dataset (CPU), run in a separate process.

PyTorch is not imported in the pytest process: on Windows its OpenMP runtime next to numpy's made a later
numpy.linalg call abort the whole test run (17 Oct 2026). The checks are in tests/unet_cases.py.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch not installed")
def test_unet_checks():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    if os.name == "nt":                       # two OpenMP runtimes (PyTorch's, numpy's) in the checks' process
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "unet_cases.py")], capture_output=True, text=True,
                       timeout=1800, env=env, cwd=ROOT)              # CPU PyTorch on Windows is slow
    assert r.returncode == 0, (r.stdout[-3000:] + r.stderr[-3000:])
    assert "unet checks ok" in r.stdout
