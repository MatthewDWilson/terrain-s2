"""Miss diagnosis: reasons at labelled points, from a synthetic run."""
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from terrain_s2.ml import labels as L
from terrain_s2.ml import probes as P
from terrain_s2.run import network
from test_run import _write


def test_reasons(tmp_path):
    d = _write(tmp_path)
    res = network(d / "dem.tif", d / "dsm.tif", d / "aoi.geojson", out=d / "o", buffer=50.0, rasters="none",
                  device="cpu", keep=True)
    c = res.layers["crossings"]
    x, y = c.x.iloc[0], c.y.iloc[0]
    aoi = gpd.read_file(d / "aoi.geojson").union_all()
    cx, cy = aoi.bounds[0] + 3, aoi.bounds[1] + 3                      # a corner: far from everything
    gpd.GeoDataFrame(dict(source=["wdc", "wdc", "wdc"], source_id=["at", "near", "far"]),
                     geometry=[Point(x + 2, y), Point(x + 22, y), Point(cx, cy)], crs=c.crs).to_file(d / "labels.gpkg", layer="crossings")
    cfg = L.load_config()
    cfg["council_sources"] = ["wdc"]
    t = P.write(res, d, d / "o", cfg).set_index("source_id")
    assert t.loc["at", "reason"] == "found"
    assert t.loc["near", "reason"] in ("found", "position")             # 22 m from this candidate (another may be nearer)
    assert t.loc["far", "reason"] in ("no_channel", "not_in_network", "outside_floodplain", "no_crossing_test")
    assert np.isfinite(t.loc["far", "dist_channel_m"]) and "rem_at" in t
    s = P.summarise([t.reset_index()])
    assert s.to_numpy().sum() == 3 and list(s.columns) == [r for r in P.REASONS if r in s.columns]
