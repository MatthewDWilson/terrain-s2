"""ML-3a step 3: council lines snapped to the bed; the independent bed check; the tile step end to end."""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import LineString

from terrain_s2 import dataio
from terrain_s2.acquire.site import load_sources
from terrain_s2.ml import channel_labels as CL
from terrain_s2.ml import snap as S

ROOT = Path(__file__).resolve().parents[1]
X0, Y0, N = 1_570_000.0, 5_200_400.0, 400
T = Affine(1.0, 0, X0, 0, -1.0, Y0)


def _scene():
    """A 4 m wide drain, 0.5 m deep in the middle, along row 100 (it wiggles by +-3 m), on gently undulating ground."""
    rng = np.random.default_rng(0)
    z = 10.0 + 0.03 * rng.standard_normal((N, N))
    cols = np.arange(N)
    centre = 100 + np.round(3 * np.sin(cols / 40)).astype(int)
    tb = np.zeros((N, N), np.float32)
    prof = np.array([0.25, 0.5, 0.5, 0.25])                          # deepest in the middle, as a real drain
    for c, r in zip(cols, centre):
        z[r - 2:r + 2, c] -= prof
        tb[r - 2:r + 2, c] = prof - 0.05
    return z.astype(np.float32), tb, centre


def test_snap_moves_line_onto_bed_and_flags_flat_ground():
    z, tb, centre = _scene()
    depth, bed = S.bed_depth({"tophat_black11": tb, "tophat_black21": tb})
    off = LineString([(X0 + 5, Y0 - 108), (X0 + 395, Y0 - 108)])           # 8 m off the drain's mean line
    flat = LineString([(X0 + 5, Y0 - 300), (X0 + 395, Y0 - 300)])          # no bed within 15 m
    g = gpd.GeoDataFrame({"council": ["w", "w"]}, geometry=[off, flat], crs=2193)
    out = S.snap_lines(g, depth, T, bed=bed)
    assert out.snapped.tolist() == [True, False]
    assert out.geometry.iloc[1].equals(flat)                               # unsnapped lines keep their geometry
    snapped = out.geometry.iloc[0]
    rows = np.array([(Y0 - y) for x, y in snapped.coords])
    assert np.median(np.abs(rows - 100.5)) <= 3.5                          # on the drain, not 8 m away
    assert out.bed_frac.iloc[0] >= 0.9 and out.offset_med_m.iloc[0] > 4
    assert S.bed_check(snapped, z, T) >= 0.9 > 0.3 >= S.bed_check(off, z, T)


def test_long_line_snapped_in_pieces_is_continuous():
    z, tb, _ = _scene()
    depth, bed = S.bed_depth({"tophat_black11": tb})
    line = LineString([(X0 + 5, Y0 - 104), (X0 + 395, Y0 - 104)])
    g, info = S.snap_line(line, depth, T, dict(piece_m=60.0), bed=bed)
    assert g is not None and info["bed_frac"] >= 0.9
    xy = np.array(g.coords)
    assert g.is_simple and abs(g.length - line.length) < 30                # one path, no doubling back or jumps
    assert xy[:, 0].min() < X0 + 20 and xy[:, 0].max() > X0 + 380


def test_tile_channels_writes_layer_and_summary(tmp_path):
    z, tb, centre = _scene()
    run, bundle = tmp_path / "runs" / "BX00", tmp_path / "tiles" / "BX00"
    run.mkdir(parents=True), bundle.mkdir(parents=True)
    dataio.write_stack(run / "features.tif", {"tophat_black11": tb, "tophat_black21": tb}, T, "EPSG:2193")
    mask = np.zeros((N, N), np.float32)
    mask[96:105, :200] = 1                                                 # the physics map has the west half
    dataio.write_stack(run / "hydro.tif", {"channel_mask": mask}, T, "EPSG:2193")
    dataio.write_stack(bundle / "dem.tif", {"z": z}, T, "EPSG:2193")
    raw = gpd.GeoDataFrame({"CLASSIFICATION3": ["Network Drain", "Network Drain"], "ASSET_ID": ["d1", "f1"]},
                           geometry=[LineString([(X0 - 50, Y0 - 107), (X0 + 450, Y0 - 107)]),   # crosses the tile edge
                                     LineString([(X0 + 5, Y0 - 300), (X0 + 395, Y0 - 300)])], crs=2193)
    raw.to_file(bundle / "labels.gpkg", layer="waimakariri_channels_raw", driver="GPKG")
    sources = load_sources(ROOT / "sites" / "sources.yml")
    cfg = CL.load_config(ROOT / "ml" / "config" / "channels.yml")
    out, summ = CL.tile_channels(run, bundle, sources, cfg)
    d = out.set_index("source_id")
    assert bool(d.loc["d1", "snapped"]) and not bool(d.loc["f1", "snapped"])
    assert d.loc["d1", "geometry"].length < 420                           # clipped to the tile
    assert d.loc["d1", "width_source"] in ("physics", "default") and d.loc["f1", "width_source"] == "default"
    assert d.loc["f1", "weight"] == 0.5 and d.loc["d1", "weight"] == 1.0  # unsnapped lines down-weighted
    w = summ["councils"]["waimakariri"]
    assert w["lines"] == 2 and 0.4 < w["snapped_share"] < 0.6 and w["check_snapped"] > w["check_council_snapped"]
    back = gpd.read_file(run / "ml_channels.gpkg", layer="council_channels")
    assert len(back) == 2 and back.geometry_council.str.startswith("LINESTRING").all()
    assert json.loads((run / "ml_channels.json").read_text())["snap_version"] == CL.SNAP_VERSION
    foot = gpd.read_file(run / "ml_channels.gpkg", layer="council_footprints")
    assert len(foot) == 2 and back.crs.to_epsg() == 2193
    assert 0.5 * d.loc["d1", "geometry"].length < foot.area.max() < 10 * d.loc["d1", "geometry"].length


def test_per_council_snap_overrides_and_river_footprints():
    from shapely.geometry import box
    cfg = CL.load_config(ROOT / "ml" / "config" / "channels.yml")
    assert cfg["councils"]["environment_southland"]["snap"]["corridor_m"] > cfg["snap"]["corridor_m"]
    lines = gpd.GeoDataFrame({"council": ["w"], "source_id": ["r1"], "width_source": ["river"], "snapped": [True],
                              "weight": [1.0], "half_width_m": [np.nan]},
                             geometry=[LineString([(0, 0), (30, 0)])], crs=2193)
    river = gpd.GeoSeries([box(-500, -20, 500, 20)], crs=2193)           # a long river; the line covers 30 m of it
    f = CL.footprints(lines, river, cfg)
    assert f.geometry.iloc[0].bounds[0] >= -20.5 and f.geometry.iloc[0].bounds[2] <= 50.5   # within 20 m of the line
