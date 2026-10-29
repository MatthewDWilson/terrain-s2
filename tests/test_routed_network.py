"""Stage A (routed_network, version 3): one outlet where the main OSM waterway leaves the catchment, the network routed
from it on a conditioned coarse DEM; mapped lines always represented; detected reaches added by evidence; routed paths
without evidence only as flow paths; heads where the evidence ends; classes in order; crossings; widths and depths."""
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


def _tile(root, name, x0, reaches, crossings=None, roads=None, coast=None, prob_boxes=(), waterways=None):
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
    if waterways is not None:
        waterways.to_file(b / "context.gpkg", layer="osm_waterways", driver="GPKG")


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
    river = gpd.GeoDataFrame(dict(waterway=["river"], name=["Test River"]), geometry=[L([(100, 503), (1995, 503)])],
                             crs=2193)
    _tile(tmp_path, "T1", 0, t1, roads=roads, prob_boxes=[(797, 500, 803, 545)], waterways=river)
    _tile(tmp_path, "T2", 1000, t2, crossings=culvert, roads=roads, coast=L([(1990, 0), (1990, 1000)]), waterways=river)
    cd = _catchment(tmp_path)
    s = RN.run_catchment(cd, tmp_path / "runs", tmp_path / "tiles", log=lambda m: None,
                         trace_points=[(1500.0, 500.0), (700.0, 500.0)])
    o = s["outlet"]["chosen"]
    assert o["method"] == "osm" and o["waterway"] == "Test River" and o["x"] > 1950 and abs(o["y"] - 500) < 10
    off = s["offsets"]
    assert off["stage_b"]["median_m"] < 1.0 and s["mapped"]["covered"] > 0.97 and s["version"] == 3
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
    assert s["detected"]["joined"] >= 1 and (N.channel_source == "detected").any() and (N.channel_source == "join").any()
    assert N.width_m.notna().all() and set(N.width_source) <= {"stage B", "cross-sections"}
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


def test_osm_outlet_prefers_the_main_waterway_inside_and_its_lowest_exit():
    cat = box(0, 0, 1000, 1000)
    T5 = Affine(5, 0, -10, 0, -5, 1010)
    xs = -10 + 5 * np.arange(204) + 2.5
    z5 = np.tile(20.0 - xs / 100.0, (204, 1)).astype(np.float32)        # falling east
    W = gpd.GeoDataFrame(dict(waterway=["stream", "river"], name=["Inner Creek", "Edge River"]),
                         geometry=[L([(100, 500), (1050, 500)]),          # a stream across the catchment
                                   L([(-50, 990), (1050, 1010)])],        # a river hugging the boundary, mostly outside
                         crs=2193)
    xy, rec = RN.osm_outlet(cat, W, [], z5, T5, RN.DEFAULTS)
    assert rec["waterway"] == "Inner Creek" and xy[0] == pytest.approx(1000.0) and xy[1] == pytest.approx(500.0)
    xy2, rec2 = RN.osm_outlet(cat, W.iloc[0:0], [], z5, T5, RN.DEFAULTS)
    assert xy2 is None and "no OSM" in rec2["reason"]


def test_catchment_by_id(tmp_path):
    from terrain_s2 import catchments
    f = tmp_path / "c.gpkg"
    gpd.GeoDataFrame(dict(catch_id=[1373, 1589]), geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)], crs=2193).to_file(f)
    assert catchments.by_id(1589, f).geometry.bounds[0] == 10
    assert catchments.by_id(1, f, field="fid").catchment_id == "1373"          # GeoPackage feature ids start at 1
    assert catchments.by_id(9999, f) is None


def test_flow_paths_heads_and_mapped_lines(tmp_path):
    """No OSM river: detected channel from x 300 to 600 and from 1200 to the coast; between them the routed path has
    no evidence (a flow path); above 300 the head is trimmed; an OSM drain on the slope is always represented."""
    t1 = [_reach(L([(300, 500), (600, 500)]))]
    t2 = [_reach(L([(1200, 500), (1985, 500)]))]
    drain = gpd.GeoDataFrame(dict(waterway=["drain"], name=[None]), geometry=[L([(1300, 900), (1300, 760)])], crs=2193)
    _tile(tmp_path, "T1", 0, t1)
    _tile(tmp_path, "T2", 1000, t2, coast=L([(1990, 0), (1990, 1000)]), waterways=drain)
    cd = _catchment(tmp_path)
    s = RN.run_catchment(cd, tmp_path / "runs", tmp_path / "tiles", log=lambda m: None)
    N = gpd.read_file(cd / "network_a.gpkg", layer="channels")
    fp = N[N.class_v2 == "flow path"]
    assert len(fp) and all(abs(g.centroid.y - 500) < 10 and 550 < g.centroid.x < 1250 for g in fp.geometry)
    ch = N[(N.class_v2 != "flow path") & N.geometry.apply(lambda g: abs(g.centroid.y - 500) < 10)]
    assert ch.geometry.bounds.minx.min() > 250                                   # the head ends near the evidence
    assert s["mapped"]["covered"] > 0.97 and s["class_km"]["flow path"] > 0.4
    from shapely import union_all
    assert union_all(N.geometry.to_numpy()).distance(L([(1300, 880), (1300, 780)])) < 6     # the OSM drain is there


def test_catchment_id_field_ignores_case(tmp_path):
    from terrain_s2 import catchments
    f = tmp_path / "c.gpkg"
    gpd.GeoDataFrame(dict(Catch_ID=[1589, 1373]), geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)], crs=2193).to_file(f)
    r = catchments.by_id(1589, f)
    assert r.geometry.bounds[0] == 0 and r.id_field == "Catch_ID" and catchments.by_id(2, f, field="fid").catchment_id == "1373"
