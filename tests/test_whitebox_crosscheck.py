"""Cross-check our least-cost breaching against Whitebox Workflows NG (optional dependency)."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from terrain_s2 import hydro
from terrain_s2.synthetic import make_scene

wbw = pytest.importorskip("whitebox_workflows")


def test_breached_surfaces_agree(tmp_path):
    import rasterio
    s = make_scene()
    z = np.where(np.isfinite(s.z), s.z, -9999.0).astype(np.float32)
    p = tmp_path / "dem.tif"
    with rasterio.open(p, "w", driver="GTiff", height=z.shape[0], width=z.shape[1], count=1,
                       dtype="float32", crs=s.crs, transform=s.transform, nodata=-9999.0) as d:
        d.write(z, 1)
    env = wbw.WbEnvironment()
    dem = env.read_raster(str(p))
    wb = env.hydrology.depressions_storage.breach_depressions_least_cost(dem=dem, max_dist=100, fill_deps=True)
    zw = wb.to_numpy().astype(np.float64)
    zw[zw <= -9998] = np.nan
    ours, _ = hydro.fill(hydro.breach_least_cost(s.z, 1.0).z_breached)

    def components(zb):
        d = np.nan_to_num(s.z - zb, nan=0.0)
        lab, n = ndi.label(d > 0.3, structure=np.ones((3, 3), bool))
        return sorted((float(d[lab == i].max()), ndi.center_of_mass(lab == i)) for i in range(1, n + 1))

    a, b = components(ours), components(zw)
    # One-cell-wide paths may differ laterally, so compare breaches, not cells.
    assert len(a) == len(b) >= 1
    for (cut_a, c_a), (cut_b, c_b) in zip(a, b):
        assert abs(cut_a - cut_b) < 0.3
        assert np.hypot(c_a[0] - c_b[0], c_a[1] - c_b[1]) < 5
