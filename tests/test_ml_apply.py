"""ML-1 step 9: run.network(crossing_model=...) scores the crossings and adds the recorded structures."""
import pickle

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from terrain_s2.ml import apply as A
from terrain_s2.ml import train as TR
from terrain_s2.run import network
from test_run import _write


def test_scored_crossings_and_records(tmp_path):
    d = _write(tmp_path)
    base = network(d / "dem.tif", d / "dsm.tif", d / "aoi.geojson", out=d / "base", buffer=50.0, rasters="none",
                   device="cpu", keep=True)
    tab = A.model_features(base, d)                                     # the features as the scorer sees them
    cols = [c for c in TR.feature_columns(tab) if c not in ("x", "y")][:20]
    rng = np.random.default_rng(0)
    tab = tab.assign(struct=np.where(rng.random(len(tab)) < 0.3, 1.0, np.nan), n_labels=0, near_positive=False,
                     weight=1.0, label_kind="council", struct_class="culvert")
    tab["n_labels"] = tab.struct.notna().astype(int)
    tab.loc[tab.struct == 1, "struct_class"] = np.where(rng.random((tab.struct == 1).sum()) < 0.5, "bridge", "culvert")
    cfg = TR.load_config()
    cfg["model"]["max_iter"] = 20
    model = dict(structure=TR.fit(tab, cols, cfg), bridge=TR.fit(tab, cols, cfg, target="bridge"), features=cols,
                 meta=dict(name="t", code="c"))
    (tmp_path / "m").mkdir()
    with open(tmp_path / "m" / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    c0 = base.layers["crossings"]
    x, y = c0.x.iloc[0], c0.y.iloc[0]
    aoi = gpd.read_file(d / "aoi.geojson").union_all()
    inside_far = aoi.centroid if aoi.centroid.distance(Point(x, y)) > 50 else Point(aoi.bounds[0] + 5, aoi.bounds[1] + 5)
    gpd.GeoDataFrame(dict(source=["wdc_culverts", "wdc_culverts", "other"], source_id=["a", "b", "c"]),
                     geometry=[Point(x + 3, y), inside_far, Point(x, y)], crs=c0.crs).to_file(d / "labels.gpkg", layer="crossings")
    res = network(d / "dem.tif", d / "dsm.tif", d / "aoi.geojson", out=d / "scored", buffer=50.0, rasters="none",
                  device="cpu", crossing_model=tmp_path / "m", record_sources=["wdc_culverts"])
    s = res.layers["crossings"]
    assert {"p_structure", "p_bridge", "model_version", "recorded", "record"} <= set(s.columns)
    assert len(s) == len(c0) + 1 and s.recorded.sum() == 2                     # one matched, one added
    added = s[s.tier == "recorded"].iloc[0]
    assert added.p_structure == 1.0 and added.record == "wdc_culverts:b" and added.source == "record:wdc_culverts"
    assert s.iloc[0].recorded and s.iloc[0].record == "wdc_culverts:a"           # "other" is not a register source
    assert s.p_structure.iloc[:len(c0)].between(0, 1).all() and res.info["scored"]["added_records"] == 1
    assert gpd.read_file(d / "scored" / "network.gpkg", layer="crossings").shape == s.shape
