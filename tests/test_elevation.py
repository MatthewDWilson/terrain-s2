"""The shared elevation reader (S2-9): survey order, fill, lidar_source, DEM/DSM pairing, the COG mirror."""
import hashlib
import json

import numpy as np
import pytest
import rasterio
from affine import Affine

from terrain_s2.acquire import elevation as E
from terrain_s2.acquire import linz_elevation as le
from test_acquire import FakeHttp

ROOT = "https://fake-bucket/"
X0, Y1 = 1_570_000, 5_200_200


def _cog(path, value, x0, y1, w, h):
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32", crs="EPSG:2193",
                       transform=Affine(1, 0, x0, 0, -1, y1), nodata=-9999, tiled=True, blockxsize=64, blockysize=64) as d:
        d.write(np.full((h, w), value, np.float32), 1)
    return path


def _catalogue(tmp_path, surveys, http_items=False):
    """surveys: [(region, slug, end, products, tiles=[(id, x0, y1, w, h, value)], national)]"""
    routes, links, binary = [], [], {}
    for region, slug, end, products, tiles, national in surveys:
        for prod in products:
            rel = f"{region}/{slug}/{prod}_1m/2193/collection.json"
            links.append({"rel": "child", "href": "./" + rel})
            items, bbs = [], []
            for tid, x0, y1, w, h, v in tiles:
                p = _cog(tmp_path / f"{slug}_{prod}_{tid}.tif", v + (100 if prod == "dsm" else 0), x0, y1, w, h)
                if http_items:
                    href = ROOT + f"{region}/{slug}/{prod}_1m/2193/{tid}.tiff"
                    binary[href] = p.read_bytes()
                    ck = "1220" + hashlib.sha256(binary[href]).hexdigest()
                else:
                    href, ck = f"file://{p}", "1220" + "0" * 64
                b = (x0, y1 - h, x0 + w, y1)
                bbs.append(b)
                it = {"id": tid, "bbox": list(le.to_lonlat(b)),
                      "assets": {"visual": {"href": href, "type": "image/tiff; application=geotiff", "file:checksum": ck}}}
                routes.append((lambda u, p_, d, k=ROOT + f"{region}/{slug}/{prod}_1m/2193/{tid}.json": u == k, it))
                items.append({"rel": "item", "href": f"./{tid}.json"})
            bb = (min(b[0] for b in bbs), min(b[1] for b in bbs), max(b[2] for b in bbs), max(b[3] for b in bbs))
            coll = {"linz:slug": slug, "title": slug, "license": "CC-BY-4.0", "updated": "2025-01-01",
                    "extent": {"spatial": {"bbox": [list(le.to_lonlat(bb))]},
                               "temporal": {"interval": [["2010-01-01T00:00:00Z", end]]}}, "links": items}
            routes.append((lambda u, p_, d, k=ROOT + rel: u == k, coll))
    routes.append((lambda u, p_, d: u == ROOT + "catalog.json", {"links": links}))
    http = FakeHttp(routes)
    http.binary = binary
    get = http.get

    def get_binary(url, params=None, source=None, ext="json", **kw):
        if url in binary:
            http.calls.append((url, params))
            return binary[url]
        return get(url, params, source, ext, **kw)
    http.get = get_binary
    return http


def _profile(national=None):
    p = json.loads(json.dumps(E.DEFAULT_PROFILE))
    p["elevation"]["stac_root"] = ROOT
    p["elevation"]["national"] = {"dem": national} if national else {}
    return p


def test_straddling_surveys_newest_first_fill_and_dsm_pairing(tmp_path):
    http = _catalogue(tmp_path, [
        ("canterbury", "new_2022", "2022-12-31T00:00:00Z", ["dem", "dsm"], [("A", X0, Y1, 100, 200, 2.0)], False),
        ("canterbury", "old_2015", "2015-12-31T00:00:00Z", ["dem", "dsm"], [("B", X0, Y1, 200, 200, 1.0)], False)])
    src = E.ElevationSource(_profile(), http=http, cache_dir=tmp_path / "stac")
    bounds = (X0 + 50, Y1 - 150, X0 + 150, Y1 - 50)                         # half in each
    dem_plan, dsm_plan = src.pair(bounds)
    assert [s.slug for s in dem_plan.surveys] == ["new_2022", "old_2015"]
    assert [s.slug for s in dsm_plan.surveys] == ["new_2022", "old_2015"]
    dem, dsm = src.read(dem_plan), src.read(dsm_plan)
    assert np.isfinite(dem.z).all() and [s.slug for s in dem.used] == ["new_2022", "old_2015"]
    assert set(np.unique(dem.lidar_source)) == {1, 2}
    assert (dem.z[:, :50] == 2.0).all() and (dem.z[:, 50:] == 1.0).all()     # newest where it exists
    assert np.array_equal(dsm.lidar_source, dem.lidar_source) and (dsm.z - dem.z == 100).all()
    # a named survey first (a site's or tile's survey; profile priority works the same way)
    d2, _ = src.pair(bounds, priority=["old_2015"])
    assert [s.slug for s in d2.surveys] == ["old_2015", "new_2022"] and (src.read(d2).z == 1.0).all()
    v = dem_plan.source_version()
    assert [s["collection"].split("/")[-4] for s in v["surveys"]] == ["new_2022", "old_2015"]


def test_national_mosaic_is_fill_only_and_exclude(tmp_path):
    nat = "new-zealand/new-zealand/dem_1m/2193/collection.json"
    http = _catalogue(tmp_path, [
        ("new-zealand", "new-zealand", "2026-01-01T00:00:00Z", ["dem"], [("N", X0, Y1, 200, 200, 9.0)], True),
        ("canterbury", "c_2020", "2020-12-31T00:00:00Z", ["dem"], [("C", X0, Y1, 100, 200, 3.0)], False)])
    src = E.ElevationSource(_profile(nat), http=http, cache_dir=tmp_path / "stac")
    r = src.read(src.plan((X0, Y1 - 200, X0 + 200, Y1)))
    assert [s.slug for s in r.plan.surveys] == ["c_2020", "new-zealand"]     # newer national date, still last
    assert (r.z[:, :100] == 3.0).all() and (r.z[:, 100:] == 9.0).all() and r.plan.surveys[1].national
    p = _profile(nat)
    p["elevation"]["exclude"] = ["c_2020"]
    assert [s.slug for s in E.ElevationSource(p, http=http, cache_dir=tmp_path / "s2").plan(
        (X0, Y1 - 200, X0 + 200, Y1)).surveys] == ["new-zealand"]


def test_mirror_downloads_once_and_checks_the_checksum(tmp_path):
    http = _catalogue(tmp_path, [("canterbury", "c_2020", "2020-12-31T00:00:00Z", ["dem"],
                                  [("C", X0, Y1, 200, 200, 3.0)], False)], http_items=True)
    src = E.ElevationSource(_profile(), http=http, cache_dir=tmp_path / "data" / "stac")
    plan = src.plan((X0 + 10, Y1 - 100, X0 + 60, Y1 - 50))
    r = src.read(plan)
    assert (r.z == 3.0).all()
    files = list((tmp_path / "data" / "cog").rglob("*.tiff"))
    assert len(files) == 1 and json.loads(files[0].with_suffix(".tiff.json").read_text())["verified"] is True
    n = sum(1 for u, *_ in http.calls if u.endswith(".tiff"))
    src.read(plan)
    assert sum(1 for u, *_ in http.calls if u.endswith(".tiff")) == n == 1        # read from the mirror
    plan.surveys[0].items[0].checksum = "1220" + "f" * 64                          # a changed tile
    with pytest.raises(IOError, match="does not match"):
        src.read(plan)


def test_whole_metre_bounds_and_stage1_uses_the_shared_reader(tmp_path):
    assert E.whole_metre_bounds((1.2, 2.7, 3.1, 4.0)) == (1.0, 2.0, 4.0, 4.0)
    from terrain_s2.stage1.source import get_source
    assert isinstance(get_source(_profile(), http=FakeHttp([]), cache_dir=tmp_path), E.ElevationSource)


def test_stage1_catches_the_shared_readers_no_data():
    from terrain_s2.stage1 import source
    assert source.NoElevationData is E.NoElevationData
