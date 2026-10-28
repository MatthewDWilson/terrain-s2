"""Stage B of channel network v2 (cleanup): supported joins, barriers only with crossing evidence, minor drains from
the ends inwards, fragments."""
import json

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import LineString as L
from shapely.geometry import box

from terrain_s2 import cleanup as B
from terrain_s2 import connectivity as C

AOI = box(0, 0, 1000, 1000)
T = Affine(1, 0, 0, 0, -1, 1000)


def _reach(g, w=3.0, d=0.6, cls="drain", src="physics"):
    return dict(geometry=g, width_m=w, incision_m=d, **{"class": cls}, river=False, channel_source=src,
                osm_waterway=None)


def _run(tmp_path, reaches, crossing=None, lake=None):
    run = tmp_path / "run"
    run.mkdir()
    f = run / "network.gpkg"
    ch = gpd.GeoDataFrame(reaches, geometry="geometry", crs=2193)
    ch.to_file(f, layer="channels", driver="GPKG")
    z = np.full((1000, 1000), 10.0, np.float32)
    z[:, 498:503] = 11.0                                          # a road embankment along x = 500
    roads = gpd.GeoDataFrame(geometry=[L([(500, 0), (500, 1000)])], crs=2193)
    cr = gpd.GeoDataFrame({"tier": ["high"]}, geometry=[crossing], crs=2193) if crossing is not None else None
    s, comps = C.measure(run, AOI)
    links = C.candidate_links(f, comps, z, T, crossings=cr, roads=roads)
    links.to_file(f, layer="links", driver="GPKG")
    ctx = None
    if lake is not None:
        ctx = tmp_path / "context.gpkg"
        gpd.GeoDataFrame({"water_kind": ["lake"]}, geometry=[lake], crs=2193).to_file(ctx, layer="osm_water", driver="GPKG")
    return run, ctx


def test_stage_b_joins_peels_and_drops(tmp_path):
    reaches = [
        _reach(L([(0, 500), (300, 500)])),                         # 0: main drain from the tile edge
        _reach(L([(304, 500), (480, 500)])),                       # 1: 4 m flat gap: joined
        _reach(L([(520, 500), (700, 500)])),                       # 2: across the road (40 m): too long, stays isolated
        _reach(L([(100, 500), (100, 530)]), w=1.0, d=0.1),         # 3: a minor drain leaf on the main drain: peeled
        _reach(L([(200, 500), (200, 560)]), w=1.0, d=0.1),         # 4: minor, but carries 5 to the network: kept
        _reach(L([(200, 560), (200, 800)])),                       # 5: a larger drain beyond the minor one
        _reach(L([(800, 100), (850, 100)])),                       # 6: a 50 m isolated piece: a fragment
        _reach(L([(600, 800), (800, 800)])),                       # 7: 200 m isolated: kept for stage A
        _reach(L([(600, 300), (650, 300)])),                       # 8: 50 m, but into a lake: an outlet, kept
    ]
    lake = box(650, 280, 700, 320)
    run, ctx = _run(tmp_path, reaches, lake=lake)
    s = B.clean_tile(run, AOI, context=ctx, window=(0, 0, 1000, 1000))
    out = gpd.read_file(run / "network_b.gpkg", layer="channels")
    rej = gpd.read_file(run / "network_b.gpkg", layer="rejected")
    links = gpd.read_file(run / "network_b.gpkg", layer="links")
    gap = links[links.length_m.round(1) == 4.0]
    assert len(gap) == 1 and bool(gap.accepted.iloc[0]) and not bool(gap.is_crossing.iloc[0])
    assert (out.channel_source == "link").sum() >= 1
    minor = rej[rej.reason == "minor_drain"]
    assert len(minor) == 1 and minor.geometry.iloc[0].bounds[0] == 100       # only the leaf went
    assert ((out.geometry.bounds.minx == 200) & (out.geometry.bounds.maxy == 560)).any()   # the carrier stayed
    frag = rej[rej.reason == "fragment"]
    assert len(frag) == 1 and frag.geometry.iloc[0].bounds == (800.0, 100.0, 850.0, 100.0)
    kept7 = out[(out.geometry.bounds.minx == 600) & (out.geometry.bounds.miny == 800)]
    assert len(kept7) == 1 and bool(kept7.isolated.iloc[0])
    kept8 = out[(out.geometry.bounds.minx == 600) & (out.geometry.bounds.miny == 300)]
    assert len(kept8) == 1 and kept8.anchor.iloc[0] == "outlet"
    assert s["after"]["isolated_share"] < s["before"]["isolated_share"] and s["outlets"] == 1
    assert json.loads((run / "stage_b.json").read_text())["removed"]["fragment_reaches"] == 1


def test_barrier_needs_a_crossing(tmp_path):
    reaches = [_reach(L([(0, 500), (494, 500)])), _reach(L([(506, 500), (700, 500)]))]   # 12 m across the road
    P = B._params({})
    (tmp_path / "a").mkdir()
    run, _ = _run(tmp_path / "a", reaches)
    j = B.judge_links(C._layer(run / "network.gpkg", "links"), P)
    e = j[j.length_m.round(1) == 12.0]
    assert len(e) == 1 and not e.accepted.iloc[0] and e.reason.iloc[0] == "improbable" and e.p_link.iloc[0] < 0.05
    (tmp_path / "b").mkdir()
    run2, _ = _run(tmp_path / "b", reaches, crossing=L([(494, 500), (506, 500)]).centroid)
    j2 = B.choose_links(B.judge_links(C._layer(run2 / "network.gpkg", "links"), P))
    e2 = j2[j2.length_m.round(1) == 12.0]
    assert e2.accepted.iloc[0] and e2.is_crossing.iloc[0] and e2.p_link.iloc[0] == 0.8     # the candidate's tier


def test_link_probability_without_cutoffs():
    import geopandas as gpd
    P = B._params({})
    L0 = gpd.GeoDataFrame(dict(kind=["end_end"] * 4, length_m=[3.0, 25.0, 25.0, 6.0], prob_mean=[0.0, 0.0, 0.9, 0.0],
                               mask_share=[0.0] * 4, rise_m=[0.0, 0.0, 0.1, 1.5], crosses_road=[False] * 4,
                               crossing_tier=[None] * 4), geometry=[L([(0, 0), (1, 0)])] * 4, crs=2193)
    j = B.judge_links(L0, P)
    assert list(j.accepted) == [True, False, True, False]     # short gap; long without evidence; long seen; a climb
    assert j.p_link.iloc[3] < j.p_link.iloc[0]


def test_minor_sweep(tmp_path):
    reaches = [_reach(L([(0, 500), (300, 500)])), _reach(L([(100, 500), (100, 530)]), w=1.2, d=0.25)]
    run, _ = _run(tmp_path, reaches)
    sw = {(r["width_m"], r["depth_m"]): r["removed_km"] for r in B.minor_sweep(run, AOI)}
    assert sw[(1.0, 0.2)] == 0.0 and sw[(1.5, 0.2)] == 0.03 and sw[(1.0, 0.3)] == 0.03
