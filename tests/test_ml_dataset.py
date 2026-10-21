"""Step 5: a dataset from two synthetic tiles run by the batch runner (features), labelled from a fake bundle."""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from terrain_s2.ml import dataset as DS
from terrain_s2.ml import labels as L
from terrain_s2.ml.batch import run_batch
from test_ml_batch import _bundles


def test_dataset_from_batch_tiles(tmp_path):
    cfg = _bundles(tmp_path, ["T1", "T2"])
    assert all(r["ok"] for r in run_batch(["T1", "T2"], cfg, log=lambda *a: None))
    # a council culvert at a candidate of T1, and one far from any candidate (a miss)
    c1 = gpd.read_file(tmp_path / "runs" / "T1" / "network.gpkg", layer="crossings")
    x, y = c1.x.iloc[0], c1.y.iloc[0]
    gpd.GeoDataFrame(dict(source=["wdc_culverts", "wdc_culverts"], source_id=["a", "b"]),
                     geometry=[Point(x + 2, y), Point(x + 5000, y)], crs=c1.crs).to_file(
        tmp_path / "bundles" / "T1" / "labels.gpkg", layer="crossings")
    pd.DataFrame([dict(tile_id="T1", council="wdc", wave="w", sheet="BW24", split="test", holdout=True,
                       region="r", latest_survey="s"),
                  dict(tile_id="T2", council="wdc", wave="w", sheet="BW23", split="train", holdout=False,
                       region="r", latest_survey="s"),
                  dict(tile_id="T3", council="wdc", wave="w", sheet="BW23", split="train", holdout=False,
                       region="r", latest_survey="s")]).to_csv(tmp_path / "m.csv", index=False)
    lcfg = L.load_config()
    lcfg["council_sources"] = ["wdc_culverts"]
    import pytest
    with pytest.raises(ValueError, match="not ready"):           # 1 of 3 tiles missing: more than 2 %
        DS.build([tmp_path / "m.csv"], tmp_path / "ds", tmp_path / "runs", tmp_path / "bundles", lcfg, log=lambda *a: None)
    card = DS.build([tmp_path / "m.csv"], tmp_path / "ds", tmp_path / "runs", tmp_path / "bundles", lcfg, log=lambda *a: None,
                    max_skipped=None)
    d = pd.read_parquet(tmp_path / "ds" / "candidates.parquet")
    assert card["tiles"]["included"] == 2 and card["tiles"]["skipped"] == ["T3"]
    assert card["positives"] == 1 and card["misses"]["total"] == 0                 # the far culvert is outside the core
    row = d[d.struct == 1].iloc[0]
    assert row.tile_id == "T1" and row.holdout and row.label_kind == "council"
    p = np.load(row.patch_file, mmap_mode="r")
    assert p.shape[0] == (d.tile_id == "T1").sum() and p.shape[1] == len(card["patches"]["bands"])
    assert "x" not in card["features"]["columns"] and "struct" not in card["features"]["columns"]
    assert "rem_at" in card["features"]["columns"] and json.loads((tmp_path / "ds" / "card.json").read_text())["name"] == "ds"
    assert (tmp_path / "ds" / "card.md").read_text().startswith("# Dataset ds")
