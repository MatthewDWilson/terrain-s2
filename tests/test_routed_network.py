"""Stage A version 2 (routed_network): one outlet at the coast, the network routed from it on a conditioned coarse DEM,
heads by probability, drains joined by probability, classes in order, crossings."""
import json

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString as L
from shapely.geometry import box

from terrain_s2 import dataio
from terrain_s2 import routed_network as RN

B = 60


def _reach(g, cls="stream", src="physics"):
    return dict(geometry=g, width_m=4.0, incision_m=0.8, **{"class": cls}, river=False, channel_source=src,
                osm_waterway=None, p_stream=0.8)


def _tile(root, name, x0, reaches, crossings=None, roads=None, coast=None, prob_boxes=()):
    import rasterio
    run, b = root / "runs" / name, root / "tiles" / name
    run.mkdir(parents=True)
    b.mkdir(parents=True)
    gpd.GeoDataFrame(geometry=[box(x0, 0, x0 + 1000, 1000)], crs=2193).to_file(b / "aoi.geojson")
    n = 1000 + 2 * B
    T = Affine(1, 0, x0 - B, 0, -1, 1000 + B)
    xs = x0 - B + np.arange(n) + 0.5
    ys = 1000 + B - np.arange(n) - 0.5
    X, Y = np.meshgrid(xs, ys)
    z = 20.0 - X / 100.0 + 0.02 * np.abs(Y - 500.0)               # a valley along y = 500, falling east
    for xr in (700.0, 1500.0):                                    # two road embankments across it, 3 m high
        z = np.where(np.abs(X - xr) <= 5.0, z + 3.0, z)
    dataio.write_stack(b / "dem.tif", {"z": z.astype(np.float32)}, T, "EPSG:2193")
    p = np.zeros((n, n), np.uint8)
    for (bx0, by0, bx1, by1) in prob_boxes:
        p[(Y >= by0) & (Y <= by1) & (X >= bx0) & (X <= bx1)] = 230
    with rasterio.open(run / "channel_prob.tif", "w", driver="GTiff", height=n, width=n, count=1, dtype="uint8",
                       crs="EPSG:2193", transform=T, nodata=255) as f:
        f.write(p, 1)
    g = gpd.GeoDataFrame(reaches, geometry="geometry", crs=2193)
    g.to_file(run / "network_b.gpkg", layer="channels", driver="GPKG")
    g.to_file(run / "network.gpkg", layer="channels", driver="GPKG")
    if crossings is not None:
        crossings.to_file(run / "network.gpkg", layer="crossings", driver="GPKG")
    if roads is not None:
        roads.to_file(b / "roads.gpkg", layer="roads", driver="GPKG")
    if coast is not None:
        gpd.GeoDataFrame(geometry=[coast], crs=2193).to_file(b / "context.gpkg", layer="osm_coastline", driver="GPKG")


def _catchment(root):
    cd = root / "catchments" / "t"
    cd.mkdir(parents=True)
    gpd.GeoDataFrame(dict(catchment_id=["c1"]), geometry=[box(0, 0, 1990, 1000)], crs=2193).to_file(
        cd / "catchment.gpkg", layer="catchment", driver="GPKG")
    gpd.GeoDataFrame(dict(tile_id=["T1", "T2"]), geometry=[box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000)],
                     crs=2193).to_file(cd / "catchment.gpkg", layer="tiles", driver="GPKG")
    return cd


def test_stage_a_v2_routes_one_network_to_the_coast(tmp_path):
    roads = gpd.GeoDataFrame(geometry=[L([(700, 0), (700, 1000)]), L([(1500, 0), (1500, 1000)])], crs=2193)
    culvert = gpd.GeoDataFrame(dict(tier=["high"], p_structure=[0.9]), geometry=[L([(1500, 490), (1500, 510)])],
                               crs=2193)
    t1 = [_reach(L([(300, 500), (694, 500)])), _reach(L([(706, 500), (1000, 500)])),
          _reach(L([(800, 600), (800, 545)]), cls="drain"),           # a small drain 45 m short of the stream
          _reach(L([(100, 950), (300, 950)]), cls="drain")]           # a drain far up the slope
    t2 = [_reach(L([(1000, 500), (1494, 500)])), _reach(L([(1506, 500), (1985, 500)]))]
    _tile(tmp_path, "T1", 0, t1, roads=roads, prob_boxes=[(797, 500, 803, 545)])
    _tile(tmp_path, "T2", 1000, t2, crossings=culvert, roads=roads, coast=L([(1990, 0), (1990, 1000)]))
    cd = _catchment(tmp_path)
    s = RN.run_catchment(cd, tmp_path / "runs", tmp_path / "tiles", log=lambda m: None,
                         trace_points=[(1500.0, 500.0), (700.0, 500.0)])
    assert s["outlet"]["chosen"]["coast"] and s["outlet"]["chosen"]["x"] > 1950          # one outlet, at the mouth
    N = gpd.read_file(cd / "network_a.gpkg", layer="channels")
    routed = N[N.channel_source == "routed"]
    from shapely import union_all
    u = union_all(routed.geometry.to_numpy())
    assert u.distance(gpd.points_from_xy([1900], [500])[0]) < 6 and u.distance(gpd.points_from_xy([400], [500])[0]) < 6
    assert all(g.coords[0][0] <= g.coords[-1][0] + 1 for g in routed.geometry if g.length > 50)   # drawn to the east
    assert u.distance(gpd.points_from_xy([30], [500])[0]) > 50                                  # the head is trimmed
    valley = routed[routed.geometry.apply(lambda g: abs(g.centroid.y - 500) < 10)]
    assert set(valley.class_v2) <= {"stream", "river"}                         # the valley's channel
    X = gpd.read_file(cd / "network_a.gpkg", layer="crossings")
    at_culvert = X[(X.geometry.x - 1500).abs() < 10]
    at_road = X[(X.geometry.x - 700).abs() < 10]
    assert len(at_culvert) >= 1 and at_culvert.p_structure.max() == pytest.approx(0.9)
    assert len(at_road) >= 1 and at_road.needs_structure.all()                   # no candidate explains it
    assert s["joined"]["accepted"] >= 1 and (N.channel_source == "joined drain").any()
    J = gpd.read_file(cd / "network_a.gpkg", layer="joins")
    assert (J.reason == "no_network_near").any() and (J.p_join.max() >= 0.5)
    tr = s["trace"]
    assert tr[0]["crossings"]["distance_m"] < 10 and tr[0]["candidates"]["p_structure"] == pytest.approx(0.9)
    assert (cd / "dem_routed.tif").exists() and json.loads((cd / "stage_a.json").read_text())["tiles"] == 2


def test_classes_are_consistent_downstream():
    geoms = [L([(0, 0), (100, 0)]), L([(0, 100), (100, 0)]), L([(100, 0), (200, 0)])]
    S = gpd.GeoDataFrame(dict(idx=[1, 2, 3], idx_ds=[3, 3, 99], upa_m2=[1e4, 2e4, 5e4]), geometry=geoms, crs=2193)
    R = gpd.GeoDataFrame({"class": ["stream", "drain", "drain"]}, geometry=geoms, crs=2193)
    cls, src = RN.classes(S, R, None, None, RN.DEFAULTS)
    assert cls == ["stream", "drain", "stream"] and src[2] == "below a stream"


def test_point_probability_and_trim():
    P = dict(RN.DEFAULTS)
    assert RN.point_p(np.array([0.0]), np.array([1e5]), P)[0] == pytest.approx(0.5)
    assert RN.point_p(np.array([0.9]), np.array([10.0]), P)[0] > 0.9
    g = L([(0, 0), (100, 0)])
    d = np.linspace(0, 100, 51)
    e = np.where(d >= 40, 1.0, 0.0)
    t = RN.trim_head(g, (d, e, np.full(51, 10.0)), P)
    assert t is not None and 25 < t.coords[0][0] < 45
    assert RN.trim_head(g, (d, np.zeros(51), np.full(51, 10.0)), P) is None


def test_join_probability_counts_a_short_gap():
    P = dict(RN.DEFAULTS)
    assert np.exp(-2.5 / P["gap_scale_m"]) > 0.7 and np.exp(-40 / P["gap_scale_m"]) < 0.02
