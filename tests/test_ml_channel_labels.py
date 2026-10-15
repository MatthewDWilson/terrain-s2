"""ML-3a: council channel lines re-harmonised from the raw layers, widths with fallbacks, weights."""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString, box

from terrain_s2.acquire.harmonise import harmonise
from terrain_s2.acquire.site import load_sources
from terrain_s2.ml import channel_labels as CL

ROOT = Path(__file__).resolve().parents[1]
X0, Y0 = 1_570_000.0, 5_200_100.0
T = Affine(1.0, 0, X0, 0, -1.0, Y0)            # 1 m grid, 100 x 100


def _bundle(tmp_path, sources):
    """A bundle with Auckland's raw layer (widths in mm, forms) and, for HBRC, only the harmonised layer."""
    ak = gpd.GeoDataFrame({"SW_WIDTH_TOP_MM": [3500, None, None, None], "SW_WIDTH_BASE_MM": [None, 2, None, 0],
                           "SW_ASSET_TYPE": ["UNLI", "SWAL", "LINE", "XXXX"], "SW_GIS_ID": ["a1", "a2", "a3", "a4"]},
                          geometry=[LineString([(X0 + 10, Y0 - 10 - 10 * i), (X0 + 60, Y0 - 10 - 10 * i)]) for i in range(4)],
                          crs=2193)
    hb_raw = gpd.GeoDataFrame({"AssetType": ["Drain", "River Channel"], "OBJECTID": [1, 2]},
                              geometry=[LineString([(X0, Y0 - 80), (X0 + 50, Y0 - 80)]),
                                        LineString([(X0, Y0 - 90), (X0 + 50, Y0 - 90)])], crs=2193)
    hb = harmonise(hb_raw, sources["hbrc_channels"], "hbrc_channels", "2026-10-17")
    d = tmp_path / "BX00_10000_0101"
    d.mkdir()
    ak.to_file(d / "labels.gpkg", layer="auckland_channels_raw", driver="GPKG")
    hb.to_file(d / "labels.gpkg", layer="channels", driver="GPKG")       # an old bundle: no hbrc raw layer
    return d


def test_council_lines_reharmonised_with_forms_and_weights(tmp_path):
    sources = load_sources(ROOT / "sites" / "sources.yml")
    cfg = CL.load_config(ROOT / "ml" / "config" / "channels.yml")
    g = CL.council_lines(_bundle(tmp_path, sources), sources, cfg)
    ak = g[g.council == "auckland"].set_index("source_id")
    assert ak.loc["a1", "width_m"] == pytest.approx(3.5) and ak.loc["a1", "form"] == "unlined"
    assert ak.loc["a2", "base_width_m"] == pytest.approx(0.002) and np.isnan(ak.loc["a4", "base_width_m"])   # 0 = unknown
    assert ak.loc["a4", "form"] == "other" and set(ak["class"]) == {"drain"}
    assert ak.loc["a1", "weight"] == pytest.approx(0.7) and ak.loc["a2", "weight"] == pytest.approx(0.35)   # swale
    hb = g[g.council == "hbrc"]
    assert sorted(hb["class"]) == ["drain", "stream"] and set(hb.weight) == {1.0}
    assert set(g.council) == {"auckland", "hbrc"}


def test_widths_council_river_physics_default():
    cfg = CL.load_config(ROOT / "ml" / "config" / "channels.yml")
    mask = np.zeros((100, 100), bool)
    mask[18:24, 5:95] = True                      # a 6 m wide drain along y = Y0 - 21
    edt = CL.edge_distance(mask, T)
    lines = gpd.GeoDataFrame({
        "class": ["drain", "drain", "drain", "stream", "drain"],
        "width_m": [3.0, np.nan, np.nan, np.nan, np.nan],
        "base_width_m": [np.nan, 0.002, np.nan, np.nan, np.nan]},
        geometry=[LineString([(X0 + 10, Y0 - 21), (X0 + 90, Y0 - 21)]),       # council width
                  LineString([(X0 + 10, Y0 - 21), (X0 + 90, Y0 - 21)]),       # implausible 2 mm: physics
                  LineString([(X0 + 10, Y0 - 60), (X0 + 90, Y0 - 60)]),       # off the channel map: default
                  LineString([(X0 + 10, Y0 - 61), (X0 + 90, Y0 - 61)]),       # stream default
                  LineString([(X0 + 10, Y0 - 85), (X0 + 40, Y0 - 85)])],      # inside a river polygon
        crs=2193)
    rivers = gpd.GeoSeries([box(X0, Y0 - 95, X0 + 50, Y0 - 75)], crs=2193)
    r = CL.resolve_widths(lines, cfg, edt=edt, transform=T, rivers=rivers)
    assert r.width_source.tolist() == ["council", "physics", "default", "default", "river"]
    assert r.half_width_m.iloc[0] == pytest.approx(1.5)
    assert 2.5 <= r.half_width_m.iloc[1] <= 3.5              # half of 6 m, within a cell
    assert r.half_width_m.iloc[2] == pytest.approx(1.0) and r.half_width_m.iloc[3] == pytest.approx(2.0)
    assert np.isnan(r.half_width_m.iloc[4])
    r2 = CL.resolve_widths(lines, cfg)                       # no rasters, no polygons
    assert r2.width_source.tolist() == ["council", "default", "default", "default", "default"]


def test_missing_bundle_gives_empty(tmp_path):
    g = CL.council_lines(tmp_path / "nothing", {}, CL.load_config())
    assert len(g) == 0 and "half_width_m" not in g
