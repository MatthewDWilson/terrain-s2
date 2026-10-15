"""ML-0 steps 1 and 3 on synthetic inputs: tile selection and folds; labels per candidate."""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from terrain_s2.ml import labels as L
from terrain_s2.ml import select as S


def _index():
    def rec(name, provides, tiles, enabled=True):
        return (box(0, 0, 1, 1), dict(name=name, provides=provides, enabled=enabled, tiles=tiles))
    return {"wdc_culverts": rec("wdc_culverts", ["crossings"], {"BW24_10000_0402": 5, "BW23_10000_0101": 2}),
            "wdc_channels": rec("wdc_channels", ["channels"], {"BW24_10000_0402": 9, "BW22_10000_0101": 4}),
            "hb_culverts": rec("hb_culverts", ["crossings"], {"BH41_10000_0303": 3}),
            "kiwirail_culverts": rec("kiwirail_culverts", ["crossings"], {"AW29_10000_0202": 47, "BW24_10000_0402": 1}),
            "off": rec("off", ["crossings"], {"BW24_10000_0402": 99}, enabled=False)}


def test_select_by_council_crossings_with_folds():
    tiles = pd.DataFrame([dict(tile_id=t, linz_tile=t, sheet=t[:4], split="test" if t.startswith("BW24") else "train",
                               region="r", latest_survey="s", has_dsm=True, n_neighbours_same_survey=8, survey_edge=False)
                          for t in ["BW24_10000_0402", "BW23_10000_0101", "BW22_10000_0101", "BH41_10000_0303", "AW29_10000_0202"]])
    cfg = dict(name="w", select_by="crossings", min_features=1, max_tiles_per_council=None, seed=0, holdout_sheets=["BW24"],
               councils={"wdc": {"crossings": ["wdc_culverts"], "channels": ["wdc_channels"]}, "hb": {"crossings": ["hb_culverts"]}})
    m = S.select(tiles, _index(), cfg)
    assert sorted(m.tile_id) == ["BH41_10000_0303", "BW23_10000_0101", "BW24_10000_0402"]   # KiwiRail does not select
    r = m.set_index("tile_id")
    assert r.loc["BW24_10000_0402", "council_crossings"] == 5 and r.loc["BW24_10000_0402", "council_channels"] == 9
    assert r.loc["BW24_10000_0402", "holdout"] and not r.loc["BW23_10000_0101", "holdout"]
    assert "kiwirail_culverts=1" in r.loc["BW24_10000_0402", "label_sources"] and "off=" not in r.loc["BW24_10000_0402", "label_sources"]
    cfg["min_features"] = 3
    assert sorted(S.select(tiles, _index(), cfg).tile_id) == ["BH41_10000_0303", "BW24_10000_0402"]
    cfg["councils"]["x"] = {"crossings": ["nope"]}
    with pytest.raises(ValueError, match="not in the coverage index"):
        S.select(tiles, _index(), cfg)


def _run_and_bundle(tmp_path):
    """Candidates at x = 10, 50, 52, 100, 150, 200 (y = 0); floodplain everywhere; AOI x < 190."""
    run, bundle = tmp_path / "run", tmp_path / "bundle"
    run.mkdir()
    bundle.mkdir()
    xs = [10, 50, 52, 100, 150, 200]
    gpd.GeoDataFrame(dict(x=xs, y=[0.0] * 6, tier=["high"] * 6), geometry=[Point(x, 0) for x in xs], crs=2193).to_file(
        run / "network.gpkg", layer="crossings")
    gpd.GeoDataFrame(geometry=[box(-50, -50, 300, 50)], crs=2193).to_file(run / "network.gpkg", layer="floodplain")
    gpd.GeoDataFrame(geometry=[box(-50, -50, 190, 50)], crs=2193).to_file(bundle / "aoi.geojson")
    gpd.GeoDataFrame(dict(source=["wdc_culverts", "kiwirail_culverts", "wdc_culverts"], source_id=["c1", "k1", "c2"]),
                     geometry=[LineString([(40, 0), (64, 0)]), Point(150, 3), Point(120, 30)], crs=2193).to_file(
        bundle / "labels.gpkg", layer="crossings")
    gpd.GeoDataFrame(dict(source=["linz_bridges"], source_id=["b1"]), geometry=[LineString([(95, -60), (95, 60)])],
                     crs=2193).to_file(bundle / "labels.gpkg", layer="bridges")
    gpd.GeoDataFrame(dict(barrier=["road", "road"], channel_id=["ch1", "ch2"], channel_source=["wdc_channels"] * 2),
                     geometry=[Point(12, 4), Point(175, 0)], crs=2193).to_file(bundle / "labels.gpkg", layer="crossings_implied")
    manual = gpd.GeoDataFrame(dict(value=[0.0, np.nan], raw=["n", "unsure"], file=["r.geojson"] * 2, round=["r1"] * 2,
                                   order=[0, 0]), geometry=[Point(10, 0), Point(200, 0)], crs=2193)
    return run, bundle, manual


def test_labels_priorities_duplicates_misses_conflicts(tmp_path):
    run, bundle, manual = _run_and_bundle(tmp_path)
    cfg = L.load_config()
    cfg["council_sources"] = ["wdc_culverts"]
    c, misses, conflicts, rep = L.label_run(run, bundle, manual, cfg)
    r = c.set_index("x")
    assert r.loc[52, "label_kind"] == "council" and r.loc[52, "struct"] == 1 and r.loc[52, "weight"] == 1.0  # crest x = 52
    assert r.loc[50, "near_positive"] and np.isnan(r.loc[50, "struct"])          # the same culvert, second candidate
    assert r.loc[100, "label_kind"] == "bridge" and r.loc[100, "struct_class"] == "bridge"   # 5 m from the bridge line
    assert r.loc[150, "label_kind"] == "asset"
    assert r.loc[10, "label_kind"] == "manual" and r.loc[10, "struct"] == 0 and r.loc[10, "weight"] == 0.6  # beats implied
    assert len(conflicts) == 1 and "implied:road=1" in conflicts.labels.iloc[0] and "manual:r1=0" in conflicts.labels.iloc[0]
    assert r.loc[200, "label_kind"] == "manual" and np.isnan(r.loc[200, "struct"]) and r.loc[200, "weight"] == 0
    assert not r.loc[200, "in_core"] and rep["candidates_in_core"] == 5
    # c2 (council, 30 m off) and the implied crossing at x = 175 (25 m from x = 150 and 200) are misses
    assert sorted(misses.source_id) == ["c2", "ch2"]
    assert rep["positives"] == 3 and rep["negatives"] == 1 and rep["misses"] == {"council": 1, "implied": 1}
    assert json.loads((run / "ml_labels.json").read_text())["conflicts"] == 1


def test_manual_files_newest_wins(tmp_path):
    a = gpd.GeoDataFrame(dict(lab=["y", "n", "skip"]), geometry=[Point(0, 0), Point(100, 0), Point(200, 0)], crs=2193)
    b = gpd.GeoDataFrame(dict(prev=["n"]), geometry=[Point(2, 0)], crs=2193)
    a.to_file(tmp_path / "a.geojson")
    b.to_file(tmp_path / "b.geojson")
    m = L.load_manual(dict(manual_root=str(tmp_path), manual=[dict(file="a.geojson", label="lab", round="v1"),
                                                             dict(file="b.geojson", label="prev", round="v2")]))
    assert len(m) == 2 and set(m["round"]) == {"v1", "v2"}
    assert m.loc[m.geometry.x < 50, "raw"].tolist() == ["n"]                      # the later file wins at 2 m


def test_select_excludes_earlier_waves_and_holds_out_councils():
    tiles = pd.DataFrame([dict(tile_id=t, linz_tile=t, sheet=t[:4], split="train", region="r", latest_survey="s",
                               has_dsm=True, n_neighbours_same_survey=8, survey_edge=False)
                          for t in ["BW24_10000_0402", "BW23_10000_0101", "BW22_10000_0101", "BH41_10000_0303"]])
    cfg = dict(name="w3", select_by="channels", min_features=1, max_tiles_per_council=None, seed=0, holdout_sheets=[],
               holdout_councils=["hb"], exclude_tiles=["BW22_10000_0101"],
               councils={"wdc": {"channels": ["wdc_channels"]}, "hb": {"channels": ["hb_culverts"]}})
    m = S.select(tiles, _index(), cfg).set_index("tile_id")
    assert "BW22_10000_0101" not in m.index and "BW24_10000_0402" in m.index
    assert bool(m.loc["BH41_10000_0303", "holdout"]) and not bool(m.loc["BW24_10000_0402", "holdout"])
