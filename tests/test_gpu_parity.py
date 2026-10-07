"""CPU vs GPU feature parity. Skipped unless CuPy and a CUDA device are available."""
import numpy as np
import pytest

from terrain_s2.backend import get_backend
from terrain_s2.features import feature_stack
from terrain_s2.synthetic import make_scene

try:
    GPU = get_backend("gpu")
except Exception:  # pragma: no cover
    GPU = None


@pytest.mark.skipif(GPU is None, reason="no CuPy/CUDA device")
def test_feature_parity():
    s = make_scene()
    cpu = feature_stack(s.z, 1.0, get_backend("cpu"))
    gpu = feature_stack(s.z, 1.0, GPU)
    for k in cpu:
        a, b = cpu[k], gpu[k]
        ok = np.isfinite(a) & np.isfinite(b)
        assert (np.isfinite(a) == np.isfinite(b)).all(), k
        # float32 reductions differ slightly between devices
        assert np.nanpercentile(np.abs(a[ok] - b[ok]), 99.9) < 1e-3 * (1 + np.nanmax(np.abs(a))), k
