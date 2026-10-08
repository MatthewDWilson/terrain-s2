"""Source registry: coverage by map sheet, auto-resolution for a window, and the tile pool."""
import importlib.util
from pathlib import Path

import pytest

from terrain_s2 import grid
from terrain_s2.acquire import registry
from test_acquire import FakeHttp

ROOT = Path(__file__).resolve().parents[1]
LAYER = "https://council/arcgis/rest/services/SW/MapServer/3"
BW24 = grid.tile_bounds("BW24")
CANTERBURY1 = (1570444.7, 5198416.6, 1573131.4, 5200182.5)


def _arcgis_http(feature_sheets):
    """A layer whose extent spans BW23-BW25 and BV24-BX24, with features only in ``feature_sheets``."""
    x0, y0, x1, y1 = BW24
    ext = {"xmin": x0 - 24000, "ymin": y0 - 36000, "xmax": x1 + 24000, "ymax": y1 + 36000, "spatialReference": {"wkid": 2193}}

    def count(u, p, d):
        g = [float(v) for v in p["geometry"].split(",")]
        sheet = grid.NZSheetGrid(50000).tile_id((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)
        return {"count": 7 if sheet in feature_sheets else 0}
    return FakeHttp([(lambda u, p, d: u == LAYER and "where" not in p, {"name": "x", "extent": ext}),
                     (lambda u, p, d: u == LAYER + "/query" and p.get("returnCountOnly"), count)])


def test_coverage_is_the_sheets_with_features():
    f = registry.coverage_record(_arcgis_http({"BW24"}), "council", {"kind": "arcgis", "url": LAYER, "layer": "channels"},
                                 log=lambda *a: None)
    p = f["properties"]
    assert p["coverage"] == "tiles" and p["sheets"] == {"BW24": 7} and p["n_features"] == 7
    assert len(p["tiles"]) == 25 and all(t.startswith("BW24_10000_") for t in p["tiles"])
    assert p["provides"] == ["channels"] and len(p["extent"]) == 4
    only = registry.coverage_record(_arcgis_http({"BW24"}), "c", {"kind": "arcgis", "url": LAYER, "layer": "channels"},
                                    sheets={"BW23"}, log=lambda *a: None)
    assert only["properties"]["sheets"] == {} and only["properties"]["coverage"].startswith("extent")


def test_national_global_and_explicit_coverage():
    http = FakeHttp([])
    assert registry.coverage_record(http, "r", {"kind": "linz_wfs", "layer": "roads"})["properties"]["coverage"] == "nz"
    o = registry.coverage_record(http, "o", {"kind": "osm"})["properties"]
    assert o["coverage"] == "global" and "culverts" in o["provides"]
    e = registry.coverage_record(http, "e", {"kind": "arcgis", "url": "x", "layer": "channels", "coverage": [170, -44, 171, -43]})
    assert e["properties"]["coverage"] == "explicit"


def test_resolve_auto_by_window_layers_and_exclude(tmp_path):
    feats = [registry.coverage_record(_arcgis_http({"BW24"}), "waimak", {"kind": "arcgis", "url": LAYER, "layer": "crossings"},
                                      log=lambda *a: None),
             registry.coverage_record(_arcgis_http({"BW24"}), "south",
                                      {"kind": "arcgis", "url": LAYER, "layer": "channels", "coverage": [167.5, -46.7, 169.5, -45.5]}),
             registry.coverage_record(FakeHttp([]), "linz_roads", {"kind": "linz_wfs", "layer": "roads"}),
             registry.coverage_record(FakeHttp([]), "osm", {"kind": "osm"}),
             registry.coverage_record(FakeHttp([]), "off", {"kind": "osm", "enabled": False})]
    registry.write_index(feats, tmp_path / "cov.geojson")
    idx = registry.load_index(tmp_path / "cov.geojson")
    sources = {"waimak": {"kind": "arcgis", "layer": "crossings"}, "south": {"kind": "arcgis", "layer": "channels"},
               "linz_roads": {"kind": "linz_wfs", "layer": "roads"}, "osm": {"kind": "osm"}, "off": {"kind": "osm", "enabled": False}}
    assert registry.resolve(sources, idx, CANTERBURY1) == ["waimak", "linz_roads", "osm"]
    assert registry.resolve(sources, idx, CANTERBURY1, layers=["crossings", "roads"], exclude=["osm"]) == ["waimak", "linz_roads"]
    with pytest.raises(ValueError, match="not in the coverage index.*new"):
        registry.resolve(dict(sources, new={"kind": "arcgis", "layer": "channels"}), idx, CANTERBURY1)


def test_tile_pool_joins_partition_and_coverage(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box
    spec = importlib.util.spec_from_file_location("tile_pool", ROOT / "scripts" / "tile_pool.py")
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)
    ids = ["BW24_10000_0402", "BW25_10000_0101", "AW27_10000_0302"]
    tiles = gpd.GeoDataFrame({"tile_id": ids, "linz_tile": ids, "sheet": [i[:4] for i in ids], "split": ["test", "train", "train"]},
                             geometry=[box(*grid.tile_bounds(i)) for i in ids], crs=2193)
    feats = [registry.coverage_record(_arcgis_http({"BW24"}), "waimak", {"kind": "arcgis", "url": LAYER, "layer": "crossings"},
                                      log=lambda *a: None),
             registry.coverage_record(FakeHttp([]), "osm", {"kind": "osm"})]
    registry.write_index(feats, tmp_path / "cov.geojson")
    p = tp.pool(tiles, registry.load_index(tmp_path / "cov.geojson"))
    assert p.tile_id.tolist() == ["BW24_10000_0402"] and p.crossings.iloc[0] == "waimak"
    assert p.label_features.iloc[0] == 7 and p.n_label_types.iloc[0] == 1


def test_build_site_auto_resolves_sources(tmp_path):
    from terrain_s2.acquire import build
    from terrain_s2.acquire.site import Site
    far = {"kind": "arcgis", "url": LAYER, "layer": "channels", "coverage": [167.5, -46.7, 169.5, -45.5]}
    registry.write_index([registry.coverage_record(FakeHttp([]), "elsewhere", far)], tmp_path / "cov.geojson")
    site = Site(id="t", bounds=CANTERBURY1, buffer_m=0, sources="auto")
    prov = build.build_site(site, {"elsewhere": {"kind": "arcgis", "url": LAYER, "layer": "channels"}}, tmp_path / "s",
                            tmp_path / "c", http=FakeHttp([]), dry_run=True, log=lambda *a: None,
                            index=registry.load_index(tmp_path / "cov.geojson"))
    assert prov["sources_auto"]["resolved"] == [] and prov["errors"] == []
    with pytest.raises(ValueError, match="needs the coverage index"):
        build.build_site(site, {}, tmp_path / "s", tmp_path / "c", http=FakeHttp([]), dry_run=True, log=lambda *a: None)


def test_tile_counts_only_where_features_and_unresolved_urls_are_not_queried():
    b = grid.tile_bounds("BW24_10000_0402")
    feature = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)                    # 3 features at the centre of one tile

    def count(u, p, d):
        g = [float(v) for v in p["geometry"].split(",")]
        return {"count": 3 if g[0] <= feature[0] <= g[2] and g[1] <= feature[1] <= g[3] else 0}
    x0, y0, x1, y1 = BW24
    http = FakeHttp([(lambda u, p, d: u == LAYER and "where" not in p,
                      {"extent": {"xmin": x0 + 10, "ymin": y0 + 10, "xmax": x1 - 10, "ymax": y1 - 10, "spatialReference": {"wkid": 2193}}}),
                     (lambda u, p, d: u.endswith("/query"), count)])
    p = registry.coverage_record(http, "k", {"kind": "arcgis", "url": LAYER, "layer": "crossings"}, log=lambda *a: None)["properties"]
    assert p["tiles"] == {"BW24_10000_0402": 3}                       # a sheet with features, one tile with them
    u = registry.coverage_record(FakeHttp([]), "t", {"kind": "arcgis", "url": "TODO", "layer": "rail"})["properties"]
    assert u["coverage"] == "unresolved"
