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
