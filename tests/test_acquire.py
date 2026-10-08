"""Acquisition (R2/M1) with a fake HTTP layer: no network needed. Real endpoints are exercised by
``scripts/build_site.py --dry-run`` on a machine that can reach them."""
import json

import numpy as np
import pytest
import rasterio
from affine import Affine

from terrain_s2.acquire import arcgis, linz_elevation, linz_wfs, osm
from terrain_s2.acquire.harmonise import harmonise
from terrain_s2.acquire.snapshot import SnapshotStore

WINDOW = (1_570_000.0, 5_200_000.0, 1_570_200.0, 5_200_100.0)      # EPSG:2193, near Kaiapoi


class FakeHttp:
    """Routes: list of (predicate(url, params, data) -> bool, response bytes or callable)."""

    def __init__(self, routes, store=None):
        self.routes, self.store, self.calls = routes, store, []

    def get(self, url, params=None, source=None, ext="json", method="GET", data=None):
        self.calls.append((url, params, data))
        for pred, resp in self.routes:
            if pred(url, params or {}, data or {}):
                body = resp(url, params or {}, data or {}) if callable(resp) else resp
                body = body if isinstance(body, bytes) else json.dumps(body).encode()
                if source and self.store is not None:
                    self.store.put(source, body, url, params or data, ext)
                return body
        raise IOError(f"no fake route for {url} {params}")

    def json(self, url, params=None, source=None, **kw):
        return json.loads(self.get(url, params, source, **kw))


def _lonlat(x, y):
    from pyproj import Transformer
    return Transformer.from_crs(2193, 4326, always_xy=True).transform(x, y)


def _feature(x, y, **props):
    lon, lat = _lonlat(x, y)
    lon2, lat2 = _lonlat(x + 10, y)
    return {"type": "Feature", "properties": props,
            "geometry": {"type": "LineString", "coordinates": [[lon, lat], [lon2, lat2]]}}


def test_snapshot_store(tmp_path):
    s = SnapshotStore(tmp_path)
    r = s.put("src one", b'{"a": 1}', "https://example/x", {"q": 1})
    assert s.path(r).read_bytes() == b'{"a": 1}'
    line = json.loads((tmp_path / "manifest.jsonl").read_text().splitlines()[0])
    assert line["sha256"] == r["sha256"] and line["url"] == "https://example/x" and line["source"] == "src one"


def test_arcgis_paging_harmonise_and_hub_item():
    layer = "https://council/arcgis/rest/services/SW/MapServer/19"
    meta = {"maxRecordCount": 2, "advancedQueryCapabilities": {"supportsPagination": True}}
    pages = {0: {"type": "FeatureCollection", "exceededTransferLimit": True,
                 "features": [_feature(1_570_010, 5_200_010, ASSNBRI="SW1", DIAMETER_mm=300, CLASSIFICATION2="Culvert"),
                              _feature(1_570_050, 5_200_050, ASSNBRI="SW2", DIAMETER_mm=0, CLASSIFICATION2="Culvert")]},
             2: {"type": "FeatureCollection",
                 "features": [_feature(1_570_100, 5_200_080, ASSNBRI="SW3", DIAMETER_mm=600, CLASSIFICATION2="Culvert")]}}
    http = FakeHttp([
        (lambda u, p, d: u.endswith("/content/items/abcdefabcdefabcdefabcdefabcdefab"), {"url": "https://council/arcgis/rest/services/SW/MapServer"}),
        (lambda u, p, d: u == layer and p.get("f") == "json", meta),
        (lambda u, p, d: u == layer + "/query", lambda u, p, d: pages[p["resultOffset"]]),
    ])
    assert arcgis.resolve_layer(http, "abcdefabcdefabcdefabcdefabcdefab_19") == layer
    g, m = arcgis.query(http, layer, WINDOW, "CLASSIFICATION2 = 'Culvert'")
    assert len(g) == 3 and m["paging"] and g.crs.to_epsg() == 2193
    assert abs(g.geometry.iloc[0].coords[0][0] - 1_570_010) < 0.01        # reprojected back exactly
    h = harmonise(g, {"layer": "crossings", "const": {"type": "culvert"}, "scale": {"diameter_m": ["DIAMETER_mm", 0.001]},
                      "id_field": "ASSNBRI"}, "waimakariri_culverts", "2026-10-07")
    assert list(h["type"]) == ["culvert"] * 3 and h.source_id.tolist() == ["SW1", "SW2", "SW3"]
    assert h.diameter_m.iloc[0] == pytest.approx(0.3) and np.isnan(h.diameter_m.iloc[1])   # 0 = unknown


def test_harmonise_class_map():
    import geopandas as gpd
    from shapely.geometry import LineString
    g = gpd.GeoDataFrame({"CLASSIFICATION3": ["Network Drain", "Receiving Waterway", "Something"]},
                         geometry=[LineString([(0, 0), (1, 1)])] * 3, crs=2193)
    h = harmonise(g, {"layer": "channels", "const": {"exists": "y"},
                      "class_map": {"field": "CLASSIFICATION3", "values": {"Network Drain": "drain", "Receiving Waterway": "stream"}}},
                  "waimakariri_channels", "2026-10-07")
    assert h["class"].tolist() == ["drain", "stream", "other"] and set(h.exists) == {"y"}


def test_linz_wfs_redacts_key_and_guards_axis_order(tmp_path):
    store = SnapshotStore(tmp_path)
    near = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"t50_fid": 1}, "geometry": {"type": "LineString",
         "coordinates": [[1_570_000, 5_200_050], [1_570_200, 5_200_050]]}}]}
    http = FakeHttp([(lambda u, p, d: "wfs" in u, near)])
    g, m = linz_wfs.get_layer(http, 50329, WINDOW, key="SECRET", store=store, source="linz_roads")
    assert len(g) == 1 and g.crs.to_epsg() == 2193
    assert http.calls[0][1]["version"] == "1.0.0" and http.calls[0][1]["bbox"].startswith("1570000.0,5200000.0")
    manifest = (tmp_path / "manifest.jsonl").read_text()
    assert "SECRET" not in manifest and "<redacted>" in manifest
    far = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "LineString",
         "coordinates": [[5_200_050, 1_570_000], [5_200_060, 1_570_000]]}}]}        # swapped axes
    with pytest.raises(ValueError, match="axis order"):
        linz_wfs.get_layer(FakeHttp([(lambda u, p, d: True, far)]), 50329, WINDOW, key="K")


def test_osm_parse_layers():
    pts = lambda *xy: [{"lon": _lonlat(x, y)[0], "lat": _lonlat(x, y)[1]} for x, y in xy]   # noqa: E731
    resp = {"elements": [
        {"type": "way", "id": 1, "tags": {"highway": "residential", "name": "Mill Rd"}, "geometry": pts((1_570_000, 5_200_050), (1_570_200, 5_200_050))},
        {"type": "way", "id": 2, "tags": {"railway": "rail"}, "geometry": pts((1_570_100, 5_200_000), (1_570_100, 5_200_100))},
        {"type": "way", "id": 3, "tags": {"waterway": "drain"}, "geometry": pts((1_570_000, 5_200_020), (1_570_200, 5_200_020))},
        {"type": "way", "id": 4, "tags": {"waterway": "drain", "tunnel": "culvert"}, "geometry": pts((1_570_050, 5_200_045), (1_570_050, 5_200_055))},
        {"type": "node", "id": 5}]}
    http = FakeHttp([(lambda u, p, d: "interpreter" in u, resp)])
    parts, meta = osm.query(http, WINDOW)
    assert {k: len(v) for k, v in parts.items()} == {"road": 1, "rail": 1, "waterway": 1, "culvert": 1}
    assert parts["road"].name.iloc[0] == "Mill Rd" and parts["road"].crs.to_epsg() == 2193
    assert "way[\"tunnel\"=\"culvert\"]" in meta["query"]


def _fake_linz(tmp_path):
    """A fake nz-elevation bucket: catalogue, two Canterbury DEM surveys, two tiles of a known plane."""
    root = "https://fake-bucket/"
    tiles = {}
    for name, x0 in (("BW24_10000_0101", 1_569_900), ("BW24_10000_0102", 1_570_100)):
        n = 200
        yy, xx = np.mgrid[0:n, 0:n]
        z = (0.01 * (x0 + xx + 0.5 - 1_569_900) + 5.0).astype(np.float32)
        p = tmp_path / f"{name}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32", crs="EPSG:2193",
                           transform=Affine(1, 0, x0, 0, -1, 5_200_150), nodata=-9999) as d:
            d.write(z, 1)
        tiles[name] = (p, (x0, 5_200_150 - n, x0 + n, 5_200_150))
    coll_url = root + "canterbury/canterbury_2020-2023/dem_1m/2193/collection.json"
    old_url = root + "canterbury/canterbury_2010/dem_1m/2193/collection.json"
    bb = list(linz_elevation.to_lonlat((1_560_000, 5_190_000, 1_580_000, 5_210_000)))
    cat = {"links": [{"rel": "child", "href": "./canterbury/canterbury_2020-2023/dem_1m/2193/collection.json"},
                     {"rel": "child", "href": "./canterbury/canterbury_2010/dem_1m/2193/collection.json"},
                     {"rel": "child", "href": "./canterbury/canterbury_2020-2023/dsm_1m/2193/collection.json"},
                     {"rel": "child", "href": "./wellington/x/dem_1m/2193/collection.json"}]}
    coll = {"linz:slug": "canterbury_2020-2023", "linz:geospatial_category": "dem", "title": "Canterbury LiDAR 1m DEM",
            "license": "CC-BY-4.0", "updated": "2025-06-01", "extent": {"spatial": {"bbox": [bb]},
            "temporal": {"interval": [["2020-04-30T12:00:00Z", "2025-03-19T11:00:00Z"]]}},
            "links": [{"rel": "item", "href": f"./{k}.json"} for k in tiles]}
    old = dict(coll, **{"linz:slug": "canterbury_2010", "extent": {"spatial": {"bbox": [bb]},
               "temporal": {"interval": [["2010-01-01T00:00:00Z", "2010-12-31T00:00:00Z"]]}}, "links": []})
    items = {k: {"id": k, "bbox": list(linz_elevation.to_lonlat(b)),
                 "assets": {"visual": {"href": f"file://{p}", "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                                       "file:checksum": "1220" + k}}} for k, (p, b) in tiles.items()}
    routes = [(lambda u, p, d: u == root + "catalog.json", cat),
              (lambda u, p, d: u == coll_url, coll), (lambda u, p, d: u == old_url, old)]
    for k, it in items.items():
        routes.append((lambda u, p, d, k=k: u == root + f"canterbury/canterbury_2020-2023/dem_1m/2193/{k}.json", it))
    return root, routes


def test_linz_elevation_latest_survey_tile_index_and_window(tmp_path):
    root, routes = _fake_linz(tmp_path)
    http = FakeHttp(routes)
    win = (1_570_050.0, 5_200_000.0, 1_570_150.0, 5_200_100.0)                 # straddles both tiles
    rec = linz_elevation.fetch_product(http, "canterbury", "dem", win, tmp_path / "out" / "dem.tif",
                                       tmp_path / "cache", root=root)
    assert rec["survey"] == "canterbury_2020-2023" and rec["n_tiles"] == 2      # latest capture chosen
    assert {c["slug"] for c in rec["considered"]} == {"canterbury_2020-2023", "canterbury_2010"}
    with rasterio.open(tmp_path / "out" / "dem.tif") as s:
        z = s.read(1)
        assert s.width == 100 and s.height == 100 and s.crs.to_epsg() == 2193
        assert np.allclose(z[50], 0.01 * (np.arange(100) + 150.5) + 5.0, atol=1e-4)   # seamless across the join
    n_calls = len(http.calls)
    linz_elevation.fetch_product(http, "canterbury", "dem", win, tmp_path / "out" / "dem2.tif", tmp_path / "cache", root=root)
    assert not any(c[0].endswith("_0101.json") for c in http.calls[n_calls:])   # tile index came from the cache


def test_linz_named_survey_missing_is_an_error(tmp_path):
    root, routes = _fake_linz(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        linz_elevation.fetch_product(FakeHttp(routes), "canterbury", "dem", WINDOW, tmp_path / "d.tif",
                                     tmp_path / "c", survey="nope", root=root)


def test_build_site_end_to_end(tmp_path, monkeypatch):
    import geopandas as gpd
    from terrain_s2.acquire import build
    from terrain_s2.acquire.site import Site
    root, routes = _fake_linz(tmp_path)
    monkeypatch.setattr(linz_elevation, "ROOT", root)
    orig = linz_elevation.fetch_product
    monkeypatch.setattr(build.linz_elevation, "fetch_product", lambda *a, **k: orig(*a, **dict(k, root=root)))
    layer = "https://council/arcgis/rest/services/SW/MapServer/19"
    routes += [(lambda u, p, d: u == layer and p.get("f") == "json", {"maxRecordCount": 10, "advancedQueryCapabilities": {"supportsPagination": True}}),
               (lambda u, p, d: u == layer + "/query", {"type": "FeatureCollection", "features": [
                   _feature(1_570_080, 5_200_040, ASSNBRI="SW9", DIAMETER_mm=450)]})]
    sources = {"waimakariri_culverts": {"kind": "arcgis", "url": layer, "layer": "crossings", "const": {"type": "culvert"},
                                        "scale": {"diameter_m": ["DIAMETER_mm", 0.001]}, "id_field": "ASSNBRI"},
               "kiwirail_culverts": {"kind": "arcgis", "url": "TODO", "layer": "crossings", "enabled": False, "note": "URL to resolve"}}
    site = Site(id="t1", bounds=(1_570_070.0, 5_200_020.0, 1_570_130.0, 5_200_080.0), buffer_m=20,
                elevation={"region": "canterbury", "products": ["dem"]}, sources=["waimakariri_culverts", "kiwirail_culverts"])
    prov = build.build_site(site, sources, tmp_path / "sites", tmp_path / "cache", http=FakeHttp(routes), log=lambda *a: None)
    out = tmp_path / "sites" / "t1"
    assert prov["errors"] == [] and (out / "dem.tif").exists() and (out / "site.json").exists()
    lab = gpd.read_file(out / "labels.gpkg", layer="crossings")
    assert lab.source_id.tolist() == ["SW9"] and lab.diameter_m.iloc[0] == pytest.approx(0.45)
    assert gpd.read_file(out / "labels.gpkg", layer="waimakariri_culverts_raw").ASSNBRI.tolist() == ["SW9"]
    meta = json.loads((out / "site.json").read_text())
    assert meta["elevation"][0]["survey"] == "canterbury_2020-2023"
    assert {s["name"] for s in meta["sources"]} == {"waimakariri_culverts", "kiwirail_culverts"}
    assert any("skipped" in s for s in meta["sources"])
    manifest = [json.loads(l) for l in (out / "snapshots" / "manifest.jsonl").read_text().splitlines()]
    assert any(r["source"] == "waimakariri_culverts" for r in manifest)


def test_arcgis_error_body_raises_and_falls_back_to_esri_json():
    layer = "https://council/arcgis/rest/services/SW/MapServer/18"
    meta = {"maxRecordCount": 5, "advancedQueryCapabilities": {"supportsPagination": True}}
    err = {"error": {"code": 400, "message": "Invalid or missing input parameters.", "details": []}}
    esri = {"features": [{"attributes": {"ASSNBRI": "CH1", "CLASSIFICATION3": "Network Drain"},
                          "geometry": {"paths": [[[1_570_010, 5_200_010], [1_570_090, 5_200_010]]]}}]}
    http = FakeHttp([(lambda u, p, d: u == layer and p.get("f") == "json" and "where" not in p, meta),
                     (lambda u, p, d: u == layer + "/query" and p.get("f") == "geojson", err),
                     (lambda u, p, d: u == layer + "/query" and p.get("f") == "json", esri)])
    g, m = arcgis.query(http, layer, WINDOW)
    assert m["format"] == "json" and len(g) == 1 and g.crs.to_epsg() == 2193
    assert g.geometry.iloc[0].length == pytest.approx(80) and g.ASSNBRI.iloc[0] == "CH1"
    with pytest.raises(arcgis.ArcGISError, match="Invalid or missing"):
        arcgis._read(json.dumps(err).encode(), layer)


def test_arcgis_count_for_dry_run():
    layer = "https://council/arcgis/rest/services/SW/MapServer/19"
    http = FakeHttp([(lambda u, p, d: u == layer + "/query" and p.get("returnCountOnly") == "true", {"count": 42})])
    n, url = arcgis.count(http, layer, WINDOW, "CLASSIFICATION2 = 'Culvert'")
    assert n == 42 and "returnCountOnly=true" in url and "CLASSIFICATION2" in url


def test_osm_buildings_are_polygons():
    ring = [(1_570_010, 5_200_010), (1_570_030, 5_200_010), (1_570_030, 5_200_025), (1_570_010, 5_200_025), (1_570_010, 5_200_010)]
    geom = [{"lon": _lonlat(x, y)[0], "lat": _lonlat(x, y)[1]} for x, y in ring]
    http = FakeHttp([(lambda u, p, d: "interpreter" in u,
                      {"elements": [{"type": "way", "id": 9, "tags": {"building": "house"}, "geometry": geom}]})])
    parts, _ = osm.query(http, WINDOW)
    b = parts["building"]
    assert len(b) == 1 and b.geometry.iloc[0].geom_type == "Polygon" and b.geometry.iloc[0].area == pytest.approx(300, rel=1e-3)


def test_arcgis_fallback_url_when_primary_is_gone():
    gone = "https://council/arcgis/rest/services/Old/MapServer/19"
    live = "https://council/arcgis/rest/services/BUD/MapServer/10"
    http = FakeHttp([(lambda u, p, d: u == gone, {"error": {"code": 404, "message": "Service not found"}}),
                     (lambda u, p, d: u == live and p.get("f") == "json", {"name": "Main_SW"})])
    assert arcgis.first_working(http, {"url": live, "fallback_urls": [gone]}) == (live, 0)
    assert arcgis.first_working(http, {"url": gone, "fallback_urls": [live]}) == (live, 1)
    with pytest.raises(arcgis.ArcGISError, match="no working URL.*Service not found"):
        arcgis.first_working(http, {"url": gone})


def test_arcgis_field_values_pages_and_picks_text_fields():
    layer = "https://council/arcgis/rest/services/BUD/MapServer/10"
    meta = {"maxRecordCount": 2, "fields": [{"name": "OBJECTID", "type": "esriFieldTypeOID"},
                                            {"name": "CLASSIFICATION2", "type": "esriFieldTypeString"},
                                            {"name": "ASSNBRI", "type": "esriFieldTypeString"}]}
    rows = [{"OBJECTID": i, "CLASSIFICATION2": c, "ASSNBRI": f"SW{i}"} for i, c in enumerate(["Pipe", "Pipe", "Culvert"])]

    def page(u, p, d):
        off = int(p["resultOffset"])
        return {"features": [{"attributes": r} for r in rows[off:off + 2]], "exceededTransferLimit": off + 2 < len(rows)}
    http = FakeHttp([(lambda u, p, d: u == layer and "where" not in p, meta),
                     (lambda u, p, d: u == layer + "/query" and p.get("returnGeometry") == "false", page)])
    n, vals = arcgis.field_values(http, layer, WINDOW, max_distinct=2)
    assert n == 3 and vals == {"CLASSIFICATION2": [("Pipe", 2), ("Culvert", 1)]}     # ASSNBRI: 3 distinct > 2


def test_hub_item_whose_url_is_already_a_layer():
    item = "9ab2c4f660dd4dc4a6b3a1ab3efea019"
    layer = "https://council/arcgis/rest/services/3Waters/Assets_Stormwater/MapServer/14"
    svc = "https://council/arcgis/rest/services/3Waters/Assets_Stormwater/MapServer"
    http = FakeHttp([(lambda u, p, d: u.endswith(item), {"url": layer}),
                     (lambda u, p, d: u.endswith("abcdefabcdefabcdefabcdefabcdefab"), {"url": svc + "/"})])
    assert arcgis.resolve_layer(http, f"{item}_14") == layer                      # not .../14/14
    assert arcgis.resolve_layer(http, "abcdefabcdefabcdefabcdefabcdefab_3") == svc + "/3"


def test_missing_fields_are_reported():
    spec = {"where": "CLASSIFICATION3 = 'Culvert' AND OWNERSHIP IN ('Council')", "id_field": "ASSNBRI",
            "fields": {"length_m": "LENGTH_m"}, "scale": {"diameter_m": ["DIAMETER_mm", 0.001]}}
    assert arcgis.mapped_fields(spec) == ["LENGTH_m", "DIAMETER_mm", "ASSNBRI", "CLASSIFICATION3", "OWNERSHIP"]
    meta = {"fields": [{"name": n} for n in ("OBJECTID", "ASSET_ID", "classification3", "OWNERSHIP", "LENGTH_m")]}
    assert arcgis.missing_fields(meta, spec) == ["DIAMETER_mm", "ASSNBRI"]
    import geopandas as gpd
    from shapely.geometry import Point
    g = gpd.GeoDataFrame({"ASSET_ID": ["1"], "LENGTH_m": [12.0]}, geometry=[Point(0, 0)], crs=2193)
    h = harmonise(g, dict(spec, layer="crossings"), "src", "now")
    assert h.attrs["missing_fields"] == ["DIAMETER_mm", "ASSNBRI"] and h.source_id.tolist() == ["0"]


def test_load_site_rejects_the_source_registry(tmp_path):
    from terrain_s2.acquire.site import load_site
    f = tmp_path / "sources.yml"
    f.write_text("sources:\n  a: {kind: osm}\n")
    with pytest.raises(ValueError, match="not a site definition"):
        load_site(f, tmp_path)


def test_id_field_alternatives():
    import geopandas as gpd
    from shapely.geometry import Point
    spec = {"layer": "crossings", "id_field": ["ASSET_ID", "ASSNBRI"]}
    old = gpd.GeoDataFrame({"ASSNBRI": ["SW1"]}, geometry=[Point(0, 0)], crs=2193)
    assert harmonise(old, spec, "s", "t").source_id.tolist() == ["SW1"]
    assert harmonise(old, spec, "s", "t").attrs["missing_fields"] == []
    none = gpd.GeoDataFrame({"X": [1]}, geometry=[Point(0, 0)], crs=2193)
    assert harmonise(none, spec, "s", "t").attrs["missing_fields"] == ["ASSET_ID | ASSNBRI"]
    assert arcgis.missing_fields({"fields": [{"name": "ASSET_ID"}]}, spec) == []
    assert arcgis.missing_fields({"fields": [{"name": "OBJECTID"}]}, spec) == ["ASSET_ID | ASSNBRI"]


def test_linz_wfs_long_line_whose_bbox_only_overlaps_is_not_an_axis_error():
    # an L-shaped railway: its bounding box covers the window, the line passes 5 km away (canterbury2, 8 Oct)
    rail = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"t50_fid": 7}, "geometry": {"type": "LineString",
         "coordinates": [[1_560_000, 5_190_000], [1_560_000, 5_210_000], [1_580_000, 5_210_000]]}}]}
    g, m = linz_wfs.get_layer(FakeHttp([(lambda u, p, d: True, rail)]), 50319, WINDOW, key="K")
    assert len(g) == 0 and m["count"] == 0 and m["bbox_only"] == 1


def test_context_clipped_to_window_labels_kept_whole(tmp_path):
    import geopandas as gpd
    from shapely.geometry import LineString
    from terrain_s2.acquire import build
    long_road = gpd.GeoDataFrame({"source_id": ["r"]}, geometry=[LineString([(1_569_000, 5_200_050), (1_571_000, 5_200_050)])], crs=2193)
    channel = gpd.GeoDataFrame({"source_id": ["c"]}, geometry=[LineString([(1_570_100, 5_199_000), (1_570_100, 5_200_050)])], crs=2193)
    build._write_layers(tmp_path, {"roads": [long_road], "channels": [channel]}, WINDOW)
    assert gpd.read_file(tmp_path / "roads.gpkg").length.sum() == pytest.approx(200)            # window is 200 m wide
    assert gpd.read_file(tmp_path / "context.gpkg", layer="roads").length.sum() == pytest.approx(200)
    assert gpd.read_file(tmp_path / "labels.gpkg", layer="channels").length.sum() == pytest.approx(1050)


def test_crossing_length_from_geometry_when_blank():
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    g = gpd.GeoDataFrame({"LENGTH_m": [None, 12.0, None]},
                         geometry=[LineString([(0, 0), (9, 0)]), LineString([(0, 0), (5, 0)]), Point(0, 0)], crs=2193)
    h = harmonise(g, {"layer": "crossings", "fields": {"length_m": "LENGTH_m"}}, "s", "t")
    assert h.length_m.tolist()[:2] == [9.0, 12.0] and np.isnan(h.length_m.iloc[2])
    assert h.length_from_geometry.tolist() == [True, False, False]


def test_implied_crossings_where_channels_cross_roads_and_rail():
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    from terrain_s2.acquire.build import implied_crossings
    L = lambda *xy: LineString(xy)  # noqa: E731
    ch = gpd.GeoDataFrame({"source": ["es"], "source_id": ["d1"], "class": [None]},
                          geometry=[L((0, -100), (0, 100))], crs=2193)
    roads = gpd.GeoDataFrame({"source": ["linz_roads"] * 2, "source_id": ["r1", "r2"]},
                             geometry=[L((-50, 0), (50, 0)), L((-50, 2), (50, 2))], crs=2193)  # dual carriageway
    rail = gpd.GeoDataFrame({"source": ["linz_rail"], "source_id": ["k1"]}, geometry=[L((-50, 60), (50, 60))], crs=2193)
    rec = gpd.GeoDataFrame({"source": ["council"], "source_id": ["c1"]}, geometry=[Point(1, 59)], crs=2193)
    g = implied_crossings({"channels": [ch], "roads": [roads], "rail": [rail], "crossings": [rec]})
    assert sorted(g.barrier.tolist()) == ["rail", "road"]                   # the two carriageways merge (5 m)
    assert g.set_index("barrier").recorded.to_dict() == {"road": False, "rail": True}
    assert implied_crossings({"channels": [ch]}).empty
