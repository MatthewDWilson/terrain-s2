"""LINZ imagery reader (acquire/imagery.py) and the cheap imagery check on the reviews (scripts/review_imagery.py)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from affine import Affine

from terrain_s2 import grid
from terrain_s2.acquire import imagery as IM
from terrain_s2.acquire import linz_elevation as le
from test_acquire import FakeHttp

ROOT = "https://fake-imagery/"
TILE = "BW24_10000_0402"
SHEET = "BW24_1000_3111"                    # top-left 1:1k sheet of the tile: 480 x 720 m
SX0, SY0, SX1, SY1 = grid.tile_bounds(SHEET)

spec = importlib.util.spec_from_file_location("review_imagery", Path(__file__).resolve().parents[1] / "scripts" / "review_imagery.py")
RI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RI)


def _rgba(path, rgb, alpha=None, res=0.5, bounds=(SX0, SY0, SX1, SY1)):
    x0, y0, x1, y1 = bounds
    w, h = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    a = np.zeros((4, h, w), np.uint8)
    a[:3] = rgb if np.ndim(rgb) == 3 else np.asarray(rgb, np.uint8)[:, None, None]
    a[3] = 255 if alpha is None else alpha
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=4, dtype="uint8", crs="EPSG:2193",
                       transform=Affine(res, 0, x0, 0, -res, y1), photometric="RGB", alpha="YES",
                       tiled=True, blockxsize=256, blockysize=256) as d:
        d.write(a)
    return path


def _catalogue(tmp_path, surveys):
    """surveys: [(region, slug, (start, end), category, product, {sheet id: file path or None})]: every sheet
    of the tile is listed as an item; only those with a file get an item document."""
    routes, links = [], []
    for region, slug, (start, end), cat, product, files in surveys:
        rel = f"{region}/{slug}/{product}/2193/collection.json"
        links.append({"rel": "child", "href": "./" + rel})
        items = []
        for sid in grid.children(TILE, 1000):
            items.append({"rel": "item", "href": f"./{sid}.json"})
            if files.get(sid):
                it = {"id": sid, "bbox": list(le.to_lonlat(grid.tile_bounds(sid))),
                      "assets": {"visual": {"href": f"file://{files[sid]}", "type": "image/tiff; application=geotiff; profile=cloud-optimized"}}}
                routes.append((lambda u, p, d, k=ROOT + f"{region}/{slug}/{product}/2193/{sid}.json": u == k, it))
        coll = {"linz:slug": slug, "title": slug, "license": "CC-BY-4.0", "linz:geospatial_category": cat,
                "providers": [{"name": "Toitū Te Whenua LINZ"}],
                "extent": {"spatial": {"bbox": [list(le.to_lonlat(grid.tile_bounds(TILE)))]},
                           "temporal": {"interval": [[start, end]]}}, "links": items}
        routes.append((lambda u, p, d, k=ROOT + rel: u == k, coll))
    routes.append((lambda u, p, d: u == ROOT + "catalog.json", {"links": links}))
    return FakeHttp(routes)


def _src(tmp_path, http):
    return IM.ImagerySource(http=http, cache_dir=tmp_path / "stac", root=ROOT)


def test_surveys_order_exclusions_and_only_the_items_needed(tmp_path):
    f20 = _rgba(tmp_path / "a.tif", (100, 120, 80))
    f15 = _rgba(tmp_path / "b.tif", (90, 110, 70))
    http = _catalogue(tmp_path, [
        ("canterbury", "canterbury_2020-2021_0.3m", ("2020-11-01T00:00:00Z", "2021-02-01T00:00:00Z"), "rural-aerial-photos", "rgb", {SHEET: f20}),
        ("canterbury", "canterbury_2015-2016_0.3m", ("2015-12-01T00:00:00Z", "2016-01-01T00:00:00Z"), "rural-aerial-photos", "rgb", {SHEET: f15}),
        ("canterbury", "canterbury_2022_10m", ("2022-01-01T00:00:00Z", "2022-03-01T00:00:00Z"), "satellite-imagery", "rgb", {SHEET: f20}),
        ("canterbury", "christchurch_2023_1m", ("2023-01-01T00:00:00Z", "2023-03-01T00:00:00Z"), "rural-aerial-photos", "rgb", {SHEET: f20}),
        ("otago", "otago_2021_0.3m", ("2021-01-01T00:00:00Z", "2021-02-01T00:00:00Z"), "rural-aerial-photos", "rgb", {SHEET: f20}),
    ])
    src = _src(tmp_path, http)
    box = (SX0 + 100, SY0 + 100, SX0 + 200, SY0 + 200)
    s = src.surveys(box, regions=["canterbury"])
    assert [x.slug for x in s] == ["canterbury_2020-2021_0.3m", "canterbury_2015-2016_0.3m"]   # newest; 10 m, 1 m out
    assert [i["id"] for i in s[0].items] == [SHEET] and s[0].gsd == 0.3 and s[0].providers
    item_calls = [u for u, *_ in http.calls if u.endswith("_1000_3111.json") or "_1000_" in u]
    assert len(item_calls) == 2                       # one sheet per survey fetched, not all 100 listed
    near = src.surveys(box, regions=["canterbury"], prefer="date", target="2016-03-01")
    assert near[0].slug == "canterbury_2015-2016_0.3m"
    assert IM.gsd_of("auckland_2010-2011_0.125m") == 0.125 and IM.gsd_of("x_rgbnir") is None


def test_read_fills_survey_by_survey_and_respects_alpha(tmp_path):
    alpha = np.full((1440, 960), 255, np.uint8)
    alpha[:, :480] = 0                                 # the newer survey has no data on the sheet's west half
    f20 = _rgba(tmp_path / "a.tif", (200, 0, 0), alpha)
    f15 = _rgba(tmp_path / "b.tif", (0, 200, 0))
    http = _catalogue(tmp_path, [
        ("canterbury", "canterbury_2020_0.3m", ("2020-01-01", "2020-02-01"), "rural-aerial-photos", "rgb", {SHEET: f20}),
        ("canterbury", "canterbury_2015_0.3m", ("2015-01-01", "2015-02-01"), "rural-aerial-photos", "rgb", {SHEET: f15}),
    ])
    src = _src(tmp_path, http)
    box = (SX0 + 200, SY0 + 300, SX0 + 280, SY0 + 340)       # straddles the west / east halves (x = SX0 + 240)
    s = src.surveys(box)
    r = src.read(s, box, res=1.0)
    assert r.data.shape == (3, 40, 80) and r.bands == ("R", "G", "B")
    assert np.all(r.data[0, :, 45:] == 200) and np.all(r.data[1, :, :35] == 200)
    assert set(np.unique(r.survey)) == {1, 2} and [x.slug for x in r.surveys] == ["canterbury_2020_0.3m", "canterbury_2015_0.3m"]
    off = src.read(s, (SX1 + 10, SY0 + 10, SX1 + 20, SY0 + 20), res=1.0)      # beyond the only sheet with files
    assert not off.valid.any() and np.all(off.survey == -1)


def _scene(path):
    """Pasture with three dark drains (y), two road edges and a fence line nothing shows (n)."""
    res = 0.5
    w, h = int((SX1 - SX0) / res), int((SY1 - SY0) / res)
    rng = np.random.default_rng(1)
    img = np.empty((3, h, w), np.float32)
    img[:] = np.array([90, 120, 70], np.float32)[:, None, None]
    img += rng.normal(0, 6, img.shape)
    xs = SX0 + (np.arange(w) + 0.5) * res
    for xd in (SX0 + 50, SX0 + 100, SX0 + 150):                  # drains 3 m wide, darker and greener banks
        img[:, :, np.abs(xs - xd) <= 1.5] = np.array([40, 70, 40])[:, None, None]
    for x0r in (SX0 + 250, SX0 + 350):                           # gravel roads 12 m wide
        img[:, :, (xs >= x0r) & (xs <= x0r + 12)] = np.array([170, 160, 140])[:, None, None]
    return _rgba(path, img.clip(0, 255).astype(np.uint8), res=res)


def test_review_measures_separate_drains_from_road_edges(tmp_path):
    import geopandas as gpd
    from shapely.geometry import LineString
    f = _scene(tmp_path / "scene.tif")
    http = _catalogue(tmp_path, [("canterbury", "canterbury_2020_0.3m", ("2020-01-01", "2020-02-01"),
                                  "rural-aerial-photos", "rgb", {SHEET: f})])
    src = _src(tmp_path, http)
    bundles = tmp_path / "tiles"
    (bundles / TILE).mkdir(parents=True)
    (bundles / TILE / "site.json").write_text(json.dumps({"elevation": [
        {"product": "dem", "capture": ["2019-06-01T00:00:00Z", "2019-07-01T00:00:00Z"],
         "surveys": [{"region": "canterbury", "national": False}]}]}))
    y0, y1 = SY0 + 200, SY0 + 500
    lines = [(SX0 + 50.6, "y"), (SX0 + 100, "y"), (SX0 + 149.4, "y"),     # drains, two a little off (registration)
             (SX0 + 250, "n"), (SX0 + 362, "n"), (SX0 + 420, "n")]         # road edges, a fence nothing shows
    rows = gpd.GeoDataFrame({"review_id": range(1, 7), "tile": TILE, "kind": "model_only", "review": "r",
                             "answer": [a for _, a in lines]},
                            geometry=[LineString([(x, y0), (x, y1)]) for x, _ in lines], crs=2193)
    a = SimpleNamespace(prefer="lidar", res=0.5, step=2.0, core=1.0, ring=(5.0, 8.0), shift=2.0)
    recs, summ = RI.tile_imagery(TILE, rows, bundles, src, a)
    d = {r["review_id"]: r for r in recs}
    assert summ["used"] == ["canterbury_2020_0.3m"] and summ["prefer"] == "date"
    for i in (1, 2, 3):                                              # a drain: darker than both banks, all along
        assert d[i]["bright_line"] < -1.0 and d[i]["bright_twoside"] > 0.9 and d[i]["bright_edge"] < 1.0
        assert d[i]["img_valid"] == 1.0 and d[i]["img_gap_years"] == pytest.approx(0.6, abs=0.1)
    assert abs(d[1]["img_shift_m"]) >= 0.5 and d[2]["img_shift_m"] == 0.0
    for i in (4, 5):                                                 # a road edge: a step, not a line
        assert d[i]["bright_edge"] > 3.0 and abs(d[i]["bright_line"]) < 0.5
    assert abs(d[6]["bright_line"]) < 0.5 and d[6]["bright_edge"] < 1.0   # a fence: nothing
    yy = [d[i]["bright_twoside"] for i in (1, 2, 3)]
    nn = [d[i]["bright_twoside"] for i in (4, 5, 6)]
    assert RI.RF.auc(yy, nn) == 1.0
    sc, v, dirn = RI.best_threshold(yy, nn)
    assert sc == pytest.approx(1.0) and dirn == "high"


def test_main_end_to_end(tmp_path, monkeypatch, capsys):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import LineString
    f = _scene(tmp_path / "scene.tif")
    http = _catalogue(tmp_path, [("canterbury", "canterbury_2020_0.3m", ("2020-01-01", "2020-02-01"),
                                  "rural-aerial-photos", "rgb", {SHEET: f})])
    data = tmp_path / "data"
    manual = data / "labels" / "manual"
    manual.mkdir(parents=True)
    (data / "tiles" / TILE).mkdir(parents=True)
    (data / "tiles" / TILE / "site.json").write_text(json.dumps({"elevation": [
        {"product": "dem", "capture": ["2019-06-01", "2019-07-01"], "surveys": [{"region": "canterbury"}]}]}))
    xs = [(SX0 + 50, "y"), (SX0 + 100, "y"), (SX0 + 150, "y"), (SX0 + 250, "n"), (SX0 + 362, "n"), (SX0 + 420, "unsure")]
    g = gpd.GeoDataFrame({"review_id": range(1, 7), "tile": TILE, "kind": "model_only", "exists": [a for _, a in xs]},
                         geometry=[LineString([(x, SY0 + 200), (x, SY0 + 500)]) for x, _ in xs], crs=2193)
    g.to_file(manual / "review_test.gpkg", layer="reaches", driver="GPKG")
    pd.DataFrame({"review": "review_test", "review_id": range(1, 7), "sharpness": 0.5, "urban_reach": 0.0,
                  "component": 1}).to_csv(manual / "review_features.csv", index=False)
    monkeypatch.setenv("TERRAIN_DATA_DIR", str(data))
    real = IM.ImagerySource
    monkeypatch.setattr(IM, "ImagerySource", lambda **kw: real(http=http, cache_dir=tmp_path / "stac", root=ROOT, **kw))
    monkeypatch.setattr("sys.argv", ["review_imagery.py", "--boot", "50"])
    RI.main()
    out = capsys.readouterr().out
    d = pd.read_csv(manual / "review_imagery.csv")
    assert len(d) == 5 and d.img_survey.eq("canterbury_2020_0.3m").all() and "sharpness" in d and "wkt" in d
    assert "model-only reaches: 3 y, 2 n" in out and "bright_twoside" in out and "leave-one-tile-out" in out
