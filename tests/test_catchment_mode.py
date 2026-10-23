"""Stage A of channel network v2 (catchment_mode): a synthetic two-tile catchment draining east to the coast."""
import json

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString as L
from shapely.geometry import box

from terrain_s2 import catchment_mode as A
from terrain_s2 import dataio

B = 60                                                       # the tiles' buffer (m)


def _reach(g, w=4.0, d=0.8, cls="stream", ps=0.8):
    return dict(geometry=g, width_m=w, incision_m=d, **{"class": cls}, river=False, channel_source="physics",
                osm_waterway=None, p_stream=ps)


def _tile(root, name, x0, reaches, prob_rows=(), prob_cols=(), coast=None):
    run = root / "runs" / name
    b = root / "tiles" / name
    run.mkdir(parents=True)
    b.mkdir(parents=True)
    core = box(x0, 0, x0 + 1000, 1000)
    gpd.GeoDataFrame(geometry=[core], crs=2193).to_file(b / "aoi.geojson")
    n_x, n_y = 1000 + 2 * B, 1000 + 2 * B
    T = Affine(1, 0, x0 - B, 0, -1, 1000 + B)
    xs = x0 - B + np.arange(n_x) + 0.5
    z = np.tile(20.0 - xs / 100.0, (n_y, 1)).astype(np.float32)        # falling east, 1 in 100
    dataio.write_stack(b / "dem.tif", {"z": z}, T, "EPSG:2193")
    import rasterio
    p = np.zeros((n_y, n_x), np.uint8)
    for (r0, r1, c0, c1) in prob_rows:
        p[r0:r1, c0:c1] = 230
    with rasterio.open(run / "channel_prob.tif", "w", driver="GTiff", height=n_y, width=n_x, count=1, dtype="uint8",
                       crs="EPSG:2193", transform=T, nodata=255) as f:
        f.write(p, 1)
    gpd.GeoDataFrame(reaches, geometry="geometry", crs=2193).to_file(run / "network_b.gpkg", layer="channels",
                                                                    driver="GPKG")
    gpd.GeoDataFrame(reaches[:1], geometry="geometry", crs=2193).to_file(run / "network.gpkg", layer="channels",
                                                                        driver="GPKG")
    if coast is not None:
        gpd.GeoDataFrame(geometry=[coast], crs=2193).to_file(b / "context.gpkg", layer="osm_coastline", driver="GPKG")
    return T


def _catchment(root):
    cd = root / "catchments" / "t"
    cd.mkdir(parents=True)
    gpd.GeoDataFrame(dict(catchment_id=["c1"]), geometry=[box(0, 0, 2000, 1000)], crs=2193).to_file(
        cd / "catchment.gpkg", layer="catchment", driver="GPKG")
    gpd.GeoDataFrame(dict(tile_id=["T1", "T2"]), geometry=[box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000)],
                     crs=2193).to_file(cd / "catchment.gpkg", layer="tiles", driver="GPKG")
    return cd


def test_stage_a_on_a_small_catchment(tmp_path):
    t1 = [_reach(L([(10, 500), (1040, 500)])),                    # the main stream (runs into T2's buffer)
          _reach(L([(300, 800), (300, 500)])),                    # a tributary ending on the main stream's side
          _reach(L([(200, 100), (800, 100)]), ps=0.2),            # 600 m, far from everything: kept, isolated
          _reach(L([(600, 500), (600, 520)]), w=1.0, d=0.1, cls="drain", ps=0.1)]   # a minor drain leaf
    t2 = [_reach(L([(960, 501.5), (1995, 501.5)])),               # the main stream again, 1.5 m off at the edge
          _reach(L([(1700, 560), (1700, 530)]), cls="drain", ps=0.2),  # 30 m short of the stream, a channel between
          _reach(L([(1500, 700), (1500, 650)]), cls="drain", ps=0.2)]  # 150 m short: dropped
    _tile(tmp_path, "T1", 0, t1)
    # T2's grid: column = x - 940, row = 1060 - y; the U-Net sees a channel from y 530 down to the stream at x 1700
    _tile(tmp_path, "T2", 1000, t2, prob_rows=[(1060 - 531, 1060 - 500, 1700 - 940 - 1, 1700 - 940 + 2)],
          coast=L([(1998, 0), (1998, 1000)]))
    cd = _catchment(tmp_path)
    s = A.run_catchment(cd, tmp_path / "runs", tmp_path / "tiles", log=lambda m: None)
    R = gpd.read_file(cd / "network_a.gpkg", layer="channels")
    assert s["outlets"].get("coast", 0) >= 1
    main = R[(R.geometry.bounds.miny >= 499) & (R.geometry.bounds.maxy <= 502) & (R.length > 100)]
    assert len(main) >= 3                                          # split at the tributary, cut at the tile edge
    assert all(g.coords[0][0] < g.coords[-1][0] for g in main.geometry)       # drawn downstream, to the east
    below = main[main.geometry.bounds.minx >= 1100]
    assert (below.strahler == 2).all() and (main[main.geometry.bounds.maxx <= 300].strahler == 1).all()
    assert (below.upstream_km > 1.5).all() and not R.uphill.any()
    trib = R[(R.geometry.bounds.minx == 300) & (R.geometry.bounds.maxx == 300)]
    assert len(trib) == 1 and trib.geometry.iloc[0].coords[0][1] > trib.geometry.iloc[0].coords[-1][1]   # flows south
    assert s["connections"]["accepted"] >= 1 and (R.channel_source == "a_link").any()
    joined = R[(R.geometry.bounds.minx >= 1699) & (R.geometry.bounds.maxx <= 1701) & (R.geometry.bounds.maxy >= 559)]
    assert len(joined) == 1 and not bool(joined.isolated.iloc[0])
    D = gpd.read_file(cd / "network_a.gpkg", layer="dropped")
    assert ((D.reason == "unconnected") & (D.geometry.bounds.minx == 1500)).any()
    assert ((D.reason == "minor_drain") & (D.geometry.bounds.minx == 600)).any()
    iso = R[R.isolated]
    assert len(iso) == 1 and iso.length.iloc[0] == pytest.approx(600.0) and s["isolated_kept"] == 1
    assert set(R.class_v2) <= {"river", "stream", "drain"} and "stitch" in set(R.channel_source)
    assert (tmp_path / "runs" / "T1" / "network_a.gpkg").exists() and (tmp_path / "runs" / "T2" / "network_a.gpkg").exists()
    assert json.loads((cd / "stage_a.json").read_text())["tiles"] == 2


def test_orient_prefers_downhill():
    ends = np.array([[0, 1], [1, 2]])
    z = np.array([0.0, 5.0, 10.0])
    dist, pred, src = A.orient(ends, np.array([100.0, 100.0]), z, {0: "coast"}, A.DEFAULTS)
    assert np.isfinite(dist).all() and pred[2] == 1 and pred[1] == 0
    down, upn, order, up = A.order_and_upstream(ends, np.array([100.0, 100.0]), dist, pred)
    assert list(down) == [0, 1] and list(order) == [1, 1] and up[0] == pytest.approx(200.0)


def test_split_at_touches():
    g, src = A.split_at_touches([L([(0, 0), (100, 0)]), L([(50, 30), (50, 0.2)])], 0.5)
    assert len(g) == 3 and sorted(round(x.length) for x in g) == [30, 50, 50] and list(src) == [0, 0, 1]
