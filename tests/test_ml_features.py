"""Step 4: candidate features and oriented patches from a synthetic run."""
import json

import numpy as np
import pandas as pd

from terrain_s2.ml import features as F
from terrain_s2.run import network
from test_run import _write


def test_features_and_patches(tmp_path):
    d = _write(tmp_path)
    res = network(d / "dem.tif", d / "dsm.tif", d / "aoi.geojson", out=d / "out", buffer=50.0, rasters="none",
                  device="cpu", keep=True)
    tab, pat, meta = F.extract(res, d / "out")
    n = len(res.layers["crossings"])
    assert n > 0 and len(tab) == n and pat.shape == (n, len(F.PATCH_BANDS), 64, 64) and pat.dtype == np.float16
    assert np.allclose(tab.x, res.layers["crossings"].x) and np.allclose(tab.y, res.layers["crossings"].y)
    for col in ("dem_h_b", "tier", "rem_at", "log10_upa_max5", "relief_med11_mean10", "outlet_dist_m"):
        assert col in tab, col
    assert tab.rem_at.notna().all() and tab.tier.between(1, 5).all()
    t2 = pd.read_parquet(d / "out" / "ml_features.parquet")
    assert list(t2.columns) == list(tab.columns) and np.load(d / "out" / "ml_patches.npy").shape == pat.shape
    assert json.loads((d / "out" / "ml_patches.json").read_text())["bands"] == F.PATCH_BANDS
    zr = pat[:, F.PATCH_BANDS.index("z_rel")].astype(float)
    assert np.allclose(zr[:, 31:33, 31:33].mean(axis=(1, 2)), 0, atol=0.5)    # relative to the crest
    # the channel axis runs along the patch x axis: centreline cells lie near the middle rows
    cl = pat[:, F.PATCH_BANDS.index("centreline")].astype(float)
    has = np.nansum(cl, axis=(1, 2)) > 5
    rows = np.arange(64)[None, :]
    mean_row = (np.nansum(cl, axis=2) * rows).sum(1) / np.maximum(np.nansum(cl, axis=(1, 2)), 1)
    assert np.median(np.abs(mean_row[has] - 31.5)) < 8


def test_axis_angle_of_a_straight_channel():
    from affine import Affine
    sk = np.zeros((100, 100), bool)
    for i in range(10, 90):
        sk[i, i] = True                                          # a diagonal: from NW to SE
    T = Affine(1, 0, 0, 0, -1, 100)
    a = F.axis_angles(sk, T, np.array([50.5]), np.array([49.5]), 1.0)
    assert np.isclose(abs(np.tan(a[0])), 1.0, atol=0.1)
