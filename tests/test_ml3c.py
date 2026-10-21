"""ML-3c: the channel U-Net's channels in the run (the model is stood in for), and the end-to-end comparison."""
import json

import numpy as np
import pandas as pd
import rasterio

import terrain_s2.ml.channel_predict as CP
from terrain_s2.run import network

from test_run import _write


def test_model_channels_join_the_network(tmp_path, monkeypatch):
    d = _write(tmp_path)
    seen = {}

    def fake(model_dir, bands, device="auto", threshold=None, min_cells=30, model=None, meta=None):
        seen["bands"] = len(bands)
        h, w = bands[0].shape
        prob = np.zeros((h, w), np.float32)
        prob[h // 2 - 1:h // 2 + 2, w // 4:3 * w // 4] = 0.9          # a drain only the "model" sees
        return prob, prob >= 0.5, dict(model=str(model_dir), threshold=0.5, device="cpu", cells=int((prob >= 0.5).sum()),
                                       seconds=0.0, dataset="test")
    monkeypatch.setattr(CP, "channel_mask_in_run", fake)
    run = network(d / "dem.tif", d / "dsm.tif", d / "aoi.geojson", out=d / "ml", buffer=50.0, rasters="hydro",
                  device="cpu", channel_model=str(tmp_path / "model"))
    assert seen["bands"] == 23
    ch = run.layers["channels"]
    assert "channel_source" in ch and (ch.channel_source == "ml").any()
    info = json.loads((d / "ml" / "network.json").read_text())["channel_model"]
    assert info["ml_reaches"] >= 1 and info["ml_km"] > 0 and info["ml_only_cells"] > 0
    with rasterio.open(d / "ml" / "hydro.tif") as s:
        assert "ml_channel_prob" in s.descriptions
    base = network(d / "dem.tif", d / "dsm.tif", d / "aoi.geojson", out=d / "base", buffer=50.0, rasters="none",
                   device="cpu")
    assert "channel_source" not in base.layers["channels"] and run.info["channels_km"] > base.info["channels_km"]
    # the physics network's candidates are kept: every baseline candidate has one within 8 m in the run with the model
    from scipy.spatial import cKDTree
    a, b = base.layers["crossings"], run.layers["crossings"]
    if len(a):
        dist, _ = cKDTree(np.column_stack([b.geometry.x, b.geometry.y])).query(np.column_stack([a.geometry.x, a.geometry.y]))
        assert (dist <= 8.0).all() and len(b) >= len(a)
    assert info["candidates_physics_network"] >= 0


def test_recall_at_k_and_summary():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("cmp", Path(__file__).resolve().parents[1] / "scripts" / "compare_ml3c.py")
    cmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp)
    cands = np.array([[0, 0], [100, 0], [200, 0]])
    assert cmp.recall_at_k(cands, [0.9, 0.1, 0.8], np.array([[5, 0], [100, 5]]), k=2, radius=25) == (1, 2)
    assert cmp.recall_at_k(cands, [0.9, 0.1, 0.8], np.zeros((0, 2)), k=2) == (0, 0)
    d = pd.DataFrame([dict(tile_id="t", council="c", run=r, kind="register", labels=10, found=f, no_channel=nc,
                           recall_hits=h, candidates=50, missing=False)
                      for r, f, nc, h in (("A", 5, 4, 3), ("B", 8, 1, 4))])
    s = cmp.summarise(d, 100).iloc[0]
    assert s.found_A == 0.5 and s.found_B == 0.8 and s.no_channel_B == 0.1 and s["recall@100_B"] == 0.4


def test_component_quality_separates_crisp_from_smudged():
    from affine import Affine
    T = Affine(1.0, 0, 0, 0, -1.0, 100)
    yy, xx = np.mgrid[:100, :200]
    crisp = np.where(np.abs(yy - 25) <= 1, 0.95, 0.02).astype(np.float32)          # a drain: a tight line
    smudge = np.exp(-((yy - 75) / 12.0) ** 2).astype(np.float32) * 0.9              # a flow path: smudged
    prob = np.maximum(crisp, smudge)
    mask = prob >= 0.7
    cover = np.zeros(prob.shape, np.float32)
    cover[60:90, :] = 0.4                                                              # buildings around the smudge
    lab, t = CP.component_quality(prob, mask, T, cover)
    assert len(t) == 2
    line, blob = t[t.component == lab[25, 100]].iloc[0], t[t.component == lab[75, 100]].iloc[0]
    assert line.contrast > 0.8 and blob.contrast < 0.5 and line.width_m < blob.width_m and blob.urban > line.urban
    keep, dropped = CP.keep_components(lab, t, min_contrast=0.5)
    assert keep[25, 100] and not keep[75, 100] and dropped == 1
    keep2, _ = CP.keep_components(lab, t, max_urban=0.1)
    assert keep2[25, 100] and not keep2[75, 100]


def test_filtered_model_channels_in_the_run(tmp_path, monkeypatch):
    d = _write(tmp_path)

    def fake(model_dir, bands, device="auto", threshold=None, min_cells=30, model=None, meta=None):
        h, w = bands[0].shape
        prob = np.zeros((h, w), np.float32)
        prob[h // 2 - 1:h // 2 + 2, w // 4:3 * w // 4] = 0.9
        return prob, prob >= 0.5, dict(model="m", threshold=0.5, device="cpu", cells=int((prob >= 0.5).sum()),
                                       seconds=0.0, dataset="t")
    monkeypatch.setattr(CP, "channel_mask_in_run", fake)
    run = network(d / "dem.tif", d / "dsm.tif", d / "aoi.geojson", out=d / "f", buffer=50.0, rasters="none",
                  device="cpu", channel_model="m", channel_min_contrast=0.95)          # 0.9 inside, 0 outside: < 0.95
    info = json.loads((d / "f" / "network.json").read_text())["channel_model"]
    assert info["components_dropped"] == info["components"] >= 1 and info["ml_only_cells"] == 0


def test_review_auc():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("rf", Path(__file__).resolve().parents[1] / "scripts" / "review_features.py")
    rf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rf)
    assert rf.auc([0.9, 0.8], [0.1, 0.2]) == 1.0 and rf.auc([1, 2], [3, 4], higher=False) == 1.0
    assert rf.auc([0.5], [0.5]) == 0.5


def test_sharpness_map_and_cell_filter():
    from affine import Affine
    T = Affine(1.0, 0, 0, 0, -1.0, 100)
    yy, xx = np.mgrid[:100, :200]
    crisp = np.where(np.abs(yy - 25) <= 1, 0.95, 0.02).astype(np.float32)
    smudge = (np.exp(-((yy - 75) / 12.0) ** 2) * 0.9).astype(np.float32)
    prob = np.maximum(crisp, smudge)
    S = CP.sharpness_map(prob, T)
    assert S[25, 100] > 0.6 and S[75, 100] < 0.2
    keep = CP.keep_sharp(prob >= 0.7, S, min_sharpness=0.4)
    assert keep[25, 100] and not keep[75, 100]
    cover = np.zeros(prob.shape, np.float32)
    cover[20:30, :100] = 0.5
    keep2 = CP.keep_sharp(prob >= 0.7, S, cover, min_sharpness=0.4, max_urban=0.1)
    assert not keep2[25, 50] and keep2[25, 150]


def test_channel_features_for_the_scorer(tmp_path):
    import geopandas as gpd
    from affine import Affine
    from shapely.geometry import Point

    from terrain_s2 import dataio
    from terrain_s2.ml.dataset import channel_features
    T = Affine(1.0, 0, 1_570_000, 0, -1.0, 5_200_100)
    run = tmp_path / "runs_b" / "BX00"
    run.mkdir(parents=True)
    gpd.GeoDataFrame({"ml_channel": [True, False]}, geometry=[Point(1_570_020.5, 5_200_049.5), Point(1_570_080.5, 5_200_020.5)],
                     crs=2193).to_file(run / "network.gpkg", layer="crossings", driver="GPKG")
    probs = tmp_path / "runs" / "BX00"
    probs.mkdir(parents=True)
    prob = np.zeros((100, 100), np.float32)
    prob[48:53, :] = 0.9                                           # a channel along row 50
    dataio.write_stack(probs / "ml_channel_prob.tif", {"channel_prob": prob}, T, "EPSG:2193")
    D = pd.DataFrame({"x": [1_570_020.5, 1_570_080.5], "y": [5_200_049.5, 5_200_020.5]})
    out = channel_features(run, D, tmp_path / "runs")
    assert out.ml_channel.tolist() == [1.0, 0.0]
    assert out.ml_prob_max.iloc[0] == np.float32(0.9) and out.ml_prob_max.iloc[1] == 0.0
    assert out.ml_sharp_max.iloc[0] > 0.5 > out.ml_sharp_max.iloc[1]
