"""Integration steps 2 and 3: the production tile route (run.tile, tiles.ensure_bundle) and the connectivity
measures of a run's channel network (connectivity.measure)."""
import json

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon, box

from terrain_s2 import connectivity as C
from terrain_s2 import tiles
from terrain_s2.run import tile
from test_run import _write


def _site_json(d):
    (d / "site.json").write_text(json.dumps({"errors": []}))


def test_ensure_bundle_reuses_builds_and_refuses(tmp_path, monkeypatch):
    b = tmp_path / "tiles" / "T1"
    b.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="build=False"):
        tiles.ensure_bundle("T1", tmp_path / "tiles", build=False)
    built = {}

    def fake_build(site, sources, bundles, cache, index=None, mirror=True, log=print):
        built["site"] = site
        (tmp_path / "tiles" / "T1" / "dem.tif").write_bytes(b"x")
        _site_json(tmp_path / "tiles" / "T1")
        return {"errors": [{"step": "osm", "error": "timeout"}]}
    import terrain_s2.acquire.build as B
    import terrain_s2.acquire.registry as R
    import terrain_s2.acquire.site as S
    monkeypatch.setattr(B, "build_site", fake_build)
    monkeypatch.setattr(S, "tile_site", lambda t, p, buf: dict(id=t, partition=str(p), buffer=buf))
    monkeypatch.setattr(S, "load_sources", lambda p: {})
    monkeypatch.setattr(R, "load_index", lambda p: None)
    bundle, errs = tiles.ensure_bundle("T1", tmp_path / "tiles", partition=tmp_path / "p.gpkg")
    assert bundle == b and errs[0]["step"] == "osm" and built["site"]["buffer"] == 200.0
    built.clear()
    assert tiles.ensure_bundle("T1", tmp_path / "tiles")[1] == [] and not built        # usable: not rebuilt

    def fails(*a, **k):
        return {"errors": [{"step": "elevation:dem", "error": "no survey"}]}
    (b / "dem.tif").unlink()
    monkeypatch.setattr(B, "build_site", fails)
    with pytest.raises(RuntimeError, match="elevation failed"):
        tiles.ensure_bundle("T1", tmp_path / "tiles", partition=tmp_path / "p.gpkg")


def test_tile_runs_the_network_on_its_bundle(tmp_path):
    b = tmp_path / "tiles" / "T2"
    b.mkdir(parents=True)
    _write(b)
    _site_json(b)
    res = tile("T2", bundles=tmp_path / "tiles", out=tmp_path / "out" / "T2", build=False, buffer=50.0, device="cpu")
    info = json.loads((tmp_path / "out" / "T2" / "network.json").read_text())
    assert info["tile"]["id"] == "T2" and info["tile"]["bundle"] == str(b) and info["channels_km"] > 0
    assert res.info["dsm"] == [str(b / "dsm.tif")] and not (tmp_path / "out" / "T2" / "hydro.tif").exists()


def _net(tmp_path, channels, rivers=None, floodplain=None, ponds=None):
    run = tmp_path / "run"
    run.mkdir()
    f = run / "network.gpkg"
    gpd.GeoDataFrame({"channel_source": [s for _, s, _ in channels], "class": [c for _, _, c in channels]},
                     geometry=[g for g, _, _ in channels], crs=2193).to_file(f, layer="channels", driver="GPKG")
    for name, geoms in (("rivers", rivers), ("floodplain", floodplain), ("residual_ponds", ponds)):
        if geoms:
            gpd.GeoDataFrame(geometry=geoms, crs=2193).to_file(f, layer=name, driver="GPKG")
    return run


def test_components_anchors_and_gaps(tmp_path):
    aoi = box(0, 0, 1000, 1000)
    L = LineString
    ch = [(L([(0, 500), (300, 500)]), "physics", "drain"),            # 0: from the tile edge
          (L([(300, 500), (300, 700)]), "physics", "drain"),          #    joined at a junction: same component
          (L([(303, 700), (303, 800)]), "ml", "drain"),               # 1: 3 m end-to-end gap from component 0
          (L([(320, 600), (320, 650)]), "ml", "drain"),               # 2: 20 m beside component 0's side
          (L([(500, 300), (600, 300), (650, 340)]), "physics", "stream"),   # 3: into a river
          (L([(700, 100), (750, 100)]), "physics", "drain"),          # 4: ends at a residual pond
          (L([(600, 600), (650, 640), (700, 600)]), "ml", "drain"),   # 5: isolated, an arch ...
          (L([(600, 700), (650, 646), (700, 700)]), "physics", "drain")]  # 6: ... meeting a V 6 m off, mid-line
    river = [Polygon([(648, 330), (700, 330), (700, 400), (648, 400)])]
    pond = [box(752, 95, 760, 105)]
    s, comps = C.measure(_net(tmp_path, ch, river, None, pond), aoi)
    assert s["components"] == 7 and s["reaches"] == 8
    a = dict(zip(comps.component, comps.anchor))
    by_first = {round(g.bounds[0]): k for k, g in zip(comps.component, comps.geometry)}
    assert a[by_first[0]] == "edge" and a[by_first[500]] == "river" and a[by_first[700]] == "pond"
    k1 = comps[comps.geometry.bounds.minx.round() == 303].iloc[0]
    assert k1.isolated and k1.gap_kind == "end_end" and k1.gap_m == pytest.approx(3.0)
    k2 = comps[comps.geometry.bounds.minx.round() == 320].iloc[0]
    assert k2.isolated and k2.gap_kind == "end_line" and k2.gap_m == pytest.approx(20.0)
    k5 = comps[(comps.geometry.bounds.minx.round() == 600) & (comps.ml_share == 1.0)].iloc[0]
    assert k5.isolated and k5.gap_m == pytest.approx(6.0) and k5.gap_kind == "line_line"
    assert s["isolated"] == 4 and s["gap_end_end"] >= 1 and s["gap_end_line"] >= 1 and s["gap_line_line"] == 2
    # joining: 5 m joins component 1 to the anchored network; 20 m also component 2; 5 and 6 stay isolated
    assert s["isolated_share_join_2m"] > s["isolated_share_join_5m"] > s["isolated_share_join_20m"] > 0
    con = C.connectors(comps)
    assert len(con) == len(comps) and set(con.gap_kind) <= {"end_end", "end_line", "line_line"}


def test_floodplain_limit_anchor(tmp_path):
    aoi = box(0, 0, 1000, 1000)
    fp = [box(0, 0, 400, 1000)]
    ch = [(LineString([(350, 500), (445, 500)]), "physics", "stream")]   # ends 45 m outside the floodplain
    s, comps = C.measure(_net(tmp_path, ch, None, fp, None), aoi)
    assert comps.anchor.iloc[0] == "floodplain_limit" and s["isolated"] == 0
