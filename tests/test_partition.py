"""Partition from a fake nz-elevation catalogue (no network)."""
import importlib.util
import json
from pathlib import Path

from test_acquire import FakeHttp

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("partition", ROOT / "scripts" / "partition.py")
partition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(partition)


def _coll(slug, end, tiles):
    return {"linz:slug": slug, "updated": "2026-01-01", "title": slug, "license": "CC-BY-4.0",
            "extent": {"temporal": {"interval": [["2020-01-01T00:00:00Z", end]]}},
            "links": [{"rel": "item", "href": f"./{t}.json"} for t in tiles]}


def test_partition_latest_survey_dsm_and_split(tmp_path):
    S = partition.STAC
    cat = {"links": [{"rel": "child", "href": "./canterbury/canterbury_2018-2019/dem_1m/2193/collection.json"},
                     {"rel": "child", "href": "./canterbury/canterbury_2020-2023/dem_1m/2193/collection.json"},
                     {"rel": "child", "href": "./canterbury/canterbury_2020-2023/dsm_1m/2193/collection.json"},
                     {"rel": "child", "href": "./canterbury/canterbury_2020-2023/dem_50cm/2193/collection.json"},
                     {"rel": "child", "href": "./northland/northland_2018-2020/dem_1m/2193/collection.json"},
                     {"rel": "child", "href": "./new-zealand/new-zealand/dem_1m/2193/collection.json"}]}
    colls = {"canterbury/canterbury_2018-2019/dem_1m": _coll("canterbury_2018-2019", "2019-04-30T12:00:00Z", ["BW24_10000_0402"]),
             "canterbury/canterbury_2020-2023/dem_1m": _coll("canterbury_2020-2023", "2025-03-19T11:00:00Z",
                                                             ["BW24_10000_0402", "BW24_10000_0401", "odd_name"]),
             "canterbury/canterbury_2020-2023/dsm_1m": _coll("canterbury_2020-2023", "2025-03-19T11:00:00Z", ["BW24_10000_0402"]),
             "northland/northland_2018-2020/dem_1m": _coll("northland_2018-2020", "2020-01-31T11:00:00Z", ["AW27_10000_0302"]),
             "new-zealand/new-zealand/dem_1m": _coll("new-zealand", "2026-06-01T00:00:00Z", ["BW24", "BW25"])}
    routes = [(lambda u, p, d: u == S + "catalog.json", cat)]
    for k, v in colls.items():
        routes.append((lambda u, p, d, k=k: u == S + k + "/2193/collection.json", v))
    http = FakeHttp(routes)
    cs = partition.collections(http, tmp_path / "c", log=lambda *a: None)
    assert len(cs) == 5                                                  # dem_50cm is not a 1 m collection
    lt = partition.linz_tiles(cs, log=lambda *a: None)
    assert set(lt) == {"BW24_10000_0402", "BW24_10000_0401", "AW27_10000_0302"}
    t = lt["BW24_10000_0402"]
    assert t["latest_survey"] == "canterbury_2020-2023" and t["n_surveys"] == 2 and t["has_dsm"] is True
    assert t["in_national"] is True and lt["AW27_10000_0302"]["in_national"] is False   # the mosaic never picks
    assert lt["BW24_10000_0401"]["has_dsm"] is False
    pt = partition.processing_tiles(lt, "quadrant", (0.8, 0.1, 0.1))
    assert len(pt) == 12 and pt["BW24_10000_0402_NW"]["linz_tile"] == "BW24_10000_0402"
    assert set(partition.processing_tiles(lt, "tile", (0.8, 0.1, 0.1))) == set(lt)
    assert len({pt[i]["split"] for i in pt if pt[i]["sheet"] == "BW24"}) == 1     # one split per sheet
    n_calls = len(http.calls)
    partition.collections(http, tmp_path / "c", log=lambda *a: None)
    assert len(http.calls) == n_calls + 1                                 # cached: only the catalogue again
    g = partition.to_gdf(pt)
    assert g.area.round().unique().tolist() == [2400 * 3600]           # quadrants


def test_check_against_cached_footprints(tmp_path):
    from rasterio.warp import transform_bounds
    from terrain_s2 import grid
    good = transform_bounds("EPSG:2193", "EPSG:4326", *grid.tile_bounds("BW24_10000_0402"), densify_pts=21)
    off = transform_bounds("EPSG:2193", "EPSG:4326", *grid.tile_bounds("BW24_10000_0403"), densify_pts=21)
    (tmp_path / "tiles_x.json").write_text(json.dumps([{"id": "BW24_10000_0402", "bbox": good},
                                                       {"id": "BW24_10000_0401", "bbox": off}]))
    n, bad = partition.check(tmp_path, log=lambda *a: None)
    assert n == 2 and [b[0] for b in bad] == ["BW24_10000_0401"]


def test_neighbours_for_the_buffer():
    lt = {t: dict(latest_survey="a", latest_end="2025", region="r") for t in ("BW24_10000_0402", "BW24_10000_0403", "BW24_10000_0302")}
    lt["BW24_10000_0502"] = dict(latest_survey="b", latest_end="2019", region="r")
    pt = partition.add_neighbours(partition.processing_tiles(lt, "tile", (0.8, 0.1, 0.1)), lt, "tile")
    t = pt["BW24_10000_0402"]
    assert t["n_neighbours_lidar"] == 3 and t["n_neighbours_same_survey"] == 2 and t["survey_edge"] is True
    q = partition.add_neighbours(partition.processing_tiles(lt, "quadrant", (0.8, 0.1, 0.1)), lt, "quadrant")
    nw = q["BW24_10000_0402_NW"]       # neighbours: NE, SW, SE of its own tile, 2 of 0302 above (survey a), 0 left
    assert nw["n_neighbours_lidar"] == 5 and nw["n_neighbours_same_survey"] == 5


def test_held_out_sites_fix_their_sheets_split():
    forced = partition.site_sheets(sorted((ROOT / "sites").glob("*.yml")))
    assert forced == {"BW24": "test"}                       # canterbury1/2 are test; sh12 and aoi2 are train
    lt = {"BW24_10000_0402": dict(latest_survey="a", latest_end="2025", region="r")}
    assert partition.processing_tiles(lt, "tile", (1.0, 0.0, 0.0), forced)["BW24_10000_0402"]["split"] == "test"
