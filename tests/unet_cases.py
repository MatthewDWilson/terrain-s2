"""ML-3b checks, run in their own process by tests/test_ml_unet.py (not collected by pytest).

PyTorch is imported here only: loaded inside the pytest process it brought its own OpenMP runtime next to numpy's,
and on Windows a later numpy.linalg call (rasterio's rowcol in test_acquire) aborted the whole run (17 Oct 2026).
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


from terrain_s2.ml import chips as C
from terrain_s2.ml import unet as U

S = 32


def _dataset(root, tiles=("BW24_10000_0101", "BW24_10000_0102", "BX24_10000_0101", "BX24_10000_0102"), n=12):
    rng = np.random.default_rng(0)
    (root / "chips").mkdir(parents=True)
    rows = []
    for t in tiles:
        x = rng.normal(0, 1, (n, len(C.BANDS), S, S)).astype(np.float16)
        lab = np.zeros((n, S, S), np.uint8)
        for i in range(n):
            r = int(rng.integers(6, S - 6))
            lab[i, r - 1:r + 2, :] = 1                                   # a channel across the chip
            x[i, C.BANDS.index("tophat_black11"), r - 1:r + 2, :] += 4   # visible in the inputs
            lab[i, :3, :3] = 255
            rows.append(dict(tile_id=t, i=i, cx=0.0, cy=0.0, kind="positive", pos_frac=0.1, ignored_frac=0.01,
                             council="a" if t.startswith("BW") else "b", holdout=t.startswith("BX24_10000_0102")))
        w = np.where(lab == 1, 1.0, np.where(lab == 255, 0.0, 0.05)).astype(np.float16)
        phys = (lab == 1).astype(np.uint8)
        if t.endswith("0102"):
            phys[:, :, S // 2:] = 0                                     # the physics map misses half these channels
        np.savez_compressed(root / "chips" / f"{t}.npz", x=x, label=lab, weight=w, physics=phys)
    pd.DataFrame(rows).to_parquet(root / "chips.parquet")
    (root / "card.json").write_text(json.dumps(dict(name="t", bands=list(C.BANDS), mean=[0.0] * len(C.BANDS),
                                                    std=[1.0] * len(C.BANDS))))
    return root


def test_soft_skeleton_of_a_bar_is_thin():
    img = torch.zeros(1, 16, 16)
    img[0, 6:11, 2:14] = 1
    sk = U.soft_skeleton(img, 5)[0]
    assert sk[8, 5:11].mean() > 0.5 and sk[6, 5:11].mean() < 0.5 and sk.sum() < img.sum() / 2


def test_loss_ignores_ignored_cells():
    y = torch.zeros(1, 8, 8)
    y[0, 4] = 1
    w = torch.where(y == 1, 1.0, 0.05)
    v = torch.ones(1, 8, 8)
    good = torch.where(y == 1, 5.0, -5.0)
    bad = -good
    assert U.loss_fn(good, y, w, v)[0] < U.loss_fn(bad, y, w, v)[0]
    v2 = v.clone()
    v2[0, 0] = 0                                                         # an ignored row with a wrong prediction
    w2 = w * v2
    wrong = good.clone()
    wrong[0, 0] = 5.0
    assert abs(float(U.loss_fn(wrong, y, w2, v2, cldice=0)[0]) - float(U.loss_fn(good, y, w2, v2, cldice=0)[0])) < 1e-6


def test_train_and_evaluate(tmp_path):
    ds = _dataset(tmp_path / "ds")
    cfg = dict(U.DEFAULTS, base=8, depth=2, epochs=3, batch=8, workers=0, folds=["holdout"], val_share=0.25,
               lr=3e-3, patience=5)
    res = U.run(ds, tmp_path / "models" / "m", cfg, folds=["holdout"], device="cpu", log=lambda m: None)
    assert len(res) == 1 and res[0]["test_chips"] == 12
    m = res[0]["metrics"]
    assert set(m) == {"all", "b"} and {"model", "physics"} <= set(m["all"])
    assert m["all"]["physics"]["recall_3m"] < 1.0                        # it misses half of one tile's channels
    assert m["all"]["model"]["off_physics_cells"] > 0 and m["all"]["model"]["recall_off_physics"] is not None
    assert set(m["all"]["sweep"]) == {str(t) for t in U.SWEEP}
    assert m["all"]["physics"]["precision_3m"] is None                   # 1 by construction: not reported
    assert 0 <= m["all"]["model"]["ap"] <= 1 and len(res[0]["history"]) >= 1
    assert (tmp_path / "models" / "m" / "holdout" / "model.pt").exists()
    assert "holdout" in (tmp_path / "models" / "m" / "report.md").read_text()
    again = U.run(ds, tmp_path / "models" / "m", cfg, folds=["holdout"], device="cpu", log=lambda m: None,
                  evaluate_only=True)
    assert again[0]["history"] == res[0]["history"] and "recall off physics" in (tmp_path / "models" / "m" / "report.md").read_text()
    model, meta = U.load_model(tmp_path / "models" / "m" / "holdout", torch.device("cpu"))
    assert meta["threshold"] == res[0]["threshold"]


def test_predict_tile_and_review(tmp_path, model_dir):
    """Whole-tile prediction (blended windows) and the review reaches, on a synthetic run."""
    import geopandas as gpd
    from affine import Affine
    from shapely.geometry import LineString

    from terrain_s2 import dataio
    from terrain_s2.ml import channel_predict as P
    N, X0, Y0 = 96, 1_570_000.0, 5_200_100.0
    T = Affine(1.0, 0, X0, 0, -1.0, Y0)
    run = tmp_path / "runs" / "A"
    run.mkdir(parents=True)
    rng = np.random.default_rng(1)
    f = {b: rng.normal(0, 1, (N, N)).astype(np.float32) for b in C.FEATURE_BANDS}
    f["tophat_black11"][40:43, :] += 4
    dataio.write_stack(run / "features.tif", f, T, "EPSG:2193")
    h = {b: np.zeros((N, N), np.float32) for b in C.HYDRO_BANDS}
    mask = np.zeros((N, N), np.float32)
    mask[40:43, :48] = 1
    h["channel_mask"] = mask
    dataio.write_stack(run / "hydro.tif", h, T, "EPSG:2193")
    dataio.write_stack(run / "ml_ways.tif", {C.WAY_BAND: np.full((N, N), 500, np.float32)}, T, "EPSG:2193")
    lab = np.zeros((N, N), np.float32)
    lab[40:43, :] = 1
    dataio.write_stack(run / "ml_channel_labels.tif", {"label": lab, "weight": lab}, T, "EPSG:2193")
    line = LineString([(X0 + 2, Y0 - 41.5), (X0 + 94, Y0 - 41.5)])
    gpd.GeoDataFrame({"snapped": [True], "source_id": ["d1"], "council": ["a"]}, geometry=[line], crs=2193).to_file(
        run / "ml_channels.gpkg", layer="council_channels", driver="GPKG")
    model, meta = U.load_model(model_dir, torch.device("cpu"))
    M = C.Mosaic({"A": run})
    prob, T2, crs = P.predict_tile(run, M, model, meta, torch.device("cpu"), window=32, step=16, batch=8)
    assert prob.shape == (N, N) and np.isfinite(prob).all() and 0 <= prob.min() and prob.max() <= 1
    P.write_prob(run, prob, T2, crs, model_dir, 0.5)
    r = P.review_reaches(run, prob, T2, crs, threshold=2.0)             # nothing predicted: the council line is missed
    assert set(r.kind) == {"council_missed"} and set(r.exists) == {""}


def test_train_v2_settings(tmp_path):
    """Off-physics weight and sampling, and the threshold chosen for recall off the physics map."""
    ds = _dataset(tmp_path / "ds")
    cfg = dict(U.DEFAULTS, base=8, depth=2, epochs=2, batch=8, workers=0, folds=["holdout"], val_share=0.25,
               lr=3e-3, patience=5, off_physics_weight=3.0, off_physics_sampling=1.0,
               threshold_objective="off_physics", precision_floor=0.0)
    res = U.run(ds, tmp_path / "models" / "v2", cfg, folds=["holdout"], device="cpu", log=lambda m: None)
    st = U.chip_stats(U.ChipStore(ds))
    assert (st.off_physics_cells > 0).any() and (st.off_physics_cells <= st.positive_cells).all()
    assert res[0]["threshold"] in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def test_channel_mask_in_run():
    """ML-3c: the U-Net over in-memory bands (a random model: shapes, threshold, fragment filter)."""
    from terrain_s2.ml import channel_predict as P
    model = U.build_model(len(C.BANDS), 8, 2).eval()
    meta = dict(bands=list(C.BANDS), mean=[0.0] * len(C.BANDS), std=[1.0] * len(C.BANDS), threshold=0.5)
    rng = np.random.default_rng(0)
    bands = [rng.normal(0, 1, (150, 170)).astype(np.float32) for _ in C.BANDS]
    bands[0][:5, :5] = np.nan
    prob, m, info = P.channel_mask_in_run("x", bands, "cpu", threshold=0.0, min_cells=1, model=model, meta=meta)
    assert prob.shape == (150, 170) and np.isnan(prob[:5, :5]).all() and m[10:, 10:].all()
    _, m2, info2 = P.channel_mask_in_run("x", bands, "cpu", threshold=2.0, model=model, meta=meta)
    assert not m2.any() and info2["cells"] == 0 and info["threshold"] == 0.0


if __name__ == "__main__":
    test_soft_skeleton_of_a_bar_is_thin()
    test_loss_ignores_ignored_cells()
    test_channel_mask_in_run()
    with tempfile.TemporaryDirectory() as d:
        test_train_and_evaluate(Path(d))
        test_predict_tile_and_review(Path(d), Path(d) / "models" / "m" / "holdout")
    with tempfile.TemporaryDirectory() as d:
        test_train_v2_settings(Path(d))
    print("unet checks ok")
