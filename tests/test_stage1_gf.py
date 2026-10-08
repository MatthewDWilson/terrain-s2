"""Stage 1 GeoFabrics backend (W4): templates, otCatalog ranking, orchestration, and the real GeoFabrics 1.1.30
runner on its coarse-DEM path (no point clouds or network needed; PDAL stubbed when absent)."""
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from terrain_s2.io import contract as C
from terrain_s2.stage1 import catalogue, gf

from stage1_fakes import FakeHttp, write_cog

X0, Y1 = 1_641_800.0, 6_076_100.0
STUBS = Path(__file__).parent / "stubs"


def test_fill_template():
    t = {"_comment": "x", "a": "{n}", "b": {"c": "{gone}", "d": "{s}/x"}, "e": {"f": "{gone}"}, "g": [1, "{n}"]}
    out = gf.fill(t, {"n": 3, "gone": None, "s": "p"})
    assert out == {"a": 3, "b": {"d": "p/x"}, "g": [1, 3]}           # comment and emptied dict removed
    with pytest.raises(KeyError):
        gf.fill({"a": "{missing}"}, {})


def test_templates_fill_for_both_products(tmp_path):
    from terrain_s2.settings import Settings
    from terrain_s2.stage1.profile import load
    ds = [catalogue.Dataset("NZ20_Northland", None, "OT.1", None, "2020-12", None, None, "x", True)]
    for product in ("dem", "geofabric"):
        s = Settings(backend="geofabrics", product=product, resolution=8, data_dir=tmp_path)
        ins = gf.instructions(settings=s, profile=load("nz"), out_dir=tmp_path / "p", name="g", datasets=ds,
                              mapping={"NZ20_Northland": 1}, land_file=None, cache_dir=tmp_path / "gfc")
        d = ins["default"]
        assert d["output"] == {"crs": {"horizontal": 2193, "vertical": 7839}, "grid_params": {"resolution": 8}}
        assert d["datasets"]["lidar"]["open_topography"] == {"NZ20_Northland": {}}
        assert d["dataset_mapping"]["lidar"] == {"NZ20_Northland": 1} and "land" not in d["data_paths"]
        assert ins["dem"]["general"]["zero_positive_foreshore"] is False
        assert ins["dem"]["general"]["lidar_classifications_to_keep"] == [2]
        assert ("roughness" in ins) == (product == "geofabric")
        assert gf.result_path(ins, product) == (tmp_path / "p").resolve() / "g.nc"
        assert not any(k.startswith("_") for k in ins)


OT_RESPONSE = {"Datasets": [
    {"Dataset": {"name": "Northland 2018-2020", "alternateName": "NZ18_Northland", "identifier": {"value": "OT.042021.2193.1"},
                 "url": "https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.042021.2193.1",
                 "temporalCoverage": "2018-11-01/2020-03-31", "datePublished": "2021-04-01"}},
    {"Dataset": {"name": "Whirinaki 2022", "alternateName": "NZ22_Whirinaki", "identifier": {"value": "OT.052023.2193.1"},
                 "temporalCoverage": "2022-01-01/2022-02-01", "datePublished": "2023-05-01"}},
    {"Dataset": {"name": "Undated", "alternateName": "NZ_Undated", "identifier": {"value": "OT.010101.2193.1"}}},
    {"Dataset": {"name": "USGS thing", "alternateName": "USGS_X", "identifier": {"value": "USGS.1"}}},
]}


def test_catalogue_parse_and_rank():
    ds = catalogue.rank(catalogue.parse(OT_RESPONSE))
    assert [d.name for d in ds] == ["NZ22_Whirinaki", "NZ18_Northland", "NZ_Undated"]   # newest first, undated last
    assert catalogue.mapping(ds) == {"NZ22_Whirinaki": 1, "NZ18_Northland": 2, "NZ_Undated": 3}
    assert ds[1].survey_end == "2020-03-31" and ds[1].date_source == "temporalCoverage"
    assert ds[2].date_source is None
    pr = catalogue.rank(catalogue.parse(OT_RESPONSE), priority=["NZ18_Northland"], exclude=["NZ_Undated"])
    assert [d.name for d in pr] == ["NZ18_Northland", "NZ22_Whirinaki"]


def _tile_index_zip(tmp, name, bounds):
    import geopandas as gpd
    from shapely.geometry import box
    d = tmp / f"ti_{name}"
    d.mkdir()
    gpd.GeoDataFrame({"name": ["t1"]}, geometry=[box(*bounds)], crs=2193).to_file(d / f"{name}_TileIndex.shp")
    z = tmp / f"{name}_TileIndex.zip"
    with zipfile.ZipFile(z, "w") as f:
        for p in d.iterdir():
            f.write(p, p.name)
    return z.read_bytes()


def _fake_gf_runner(ins):
    """Writes what GeoFabrics would: a netCDF at result_dem with its own description, no CF attributes."""
    import xarray as xr
    d = ins["default"]
    res = d["output"]["grid_params"]["resolution"]
    ext = Path(d["data_paths"]["local_cache"]) / d["data_paths"]["subfolder"] / d["data_paths"]["extents"]
    import geopandas as gpd
    x0, y0, x1, y1 = gpd.read_file(ext).total_bounds
    nx, ny = int((x1 - x0) / res), int((y1 - y0) / res)
    z = np.full((ny, nx), 7.0, np.float32)
    ds = xr.Dataset({"z": (("y", "x"), z), "data_source": (("y", "x"), np.ones_like(z)),
                     "lidar_source": (("y", "x"), np.ones_like(z), {"mapping": str(d["dataset_mapping"]["lidar"])})},
                    coords={"x": x0 + res / 2 + res * np.arange(nx), "y": y1 - res / 2 - res * np.arange(ny)},
                    attrs={"description": f"geofabrics:HydrologicDemGenerator resolution {res}",
                           "geofabrics_instructions": json.dumps(ins)})
    ds.to_netcdf(gf.result_path(ins, "dem"))


def test_make_geofabrics_orchestration(tmp_path):
    from terrain_s2.acquire.snapshot import SnapshotStore
    from terrain_s2.settings import Settings
    from terrain_s2.stage1.dem import make
    tis = {catalogue.tile_index_url(n): _tile_index_zip(tmp_path, n, (X0 - 50, Y1 - 300, X0 + 300, Y1 + 50))
           for n in ("NZ22_Whirinaki", "NZ18_Northland", "NZ_Undated")}
    http = FakeHttp({catalogue.OT_CATALOG: OT_RESPONSE}, binary=tis)
    s = Settings(backend="geofabrics", resolution=8, buffer_m=0, data_dir=tmp_path / "data")
    aoi = f"POLYGON(({X0} {Y1 - 160}, {X0 + 160} {Y1 - 160}, {X0 + 160} {Y1}, {X0} {Y1}, {X0} {Y1 - 160}))"
    kw = dict(aoi_crs="EPSG:2193", http=http, store=SnapshotStore(tmp_path / "snap"), gf_runner=_fake_gf_runner,
              log=lambda *a: None)
    r = make(aoi, s, **kw)
    assert r.backend == "geofabrics" and r.resolution == 8
    assert r.info["source_version"]["mapping"] == {"NZ22_Whirinaki": 1, "NZ18_Northland": 2, "NZ_Undated": 3}
    ins = json.loads(Path(r.paths["instructions"]).read_text())
    assert ins["default"]["data_paths"]["land"] == r.paths["land"]
    import geopandas as gpd
    land = gpd.read_file(r.paths["land"]).geometry.iloc[0]          # tile-index coverage clipped to AOI + 100 m
    assert land.bounds == pytest.approx((X0 - 50, Y1 - 260, X0 + 260, Y1 + 50))
    p = C.read_dem(r.paths["netcdf"])
    assert p.resolution == 8 and p.crs == "EPSG:2193+7839" and p.provenance["generator_key"] == r.key
    assert "cog" not in r.paths
    assert make(aoi, s, **kw).reused
    # changing the land source changes the key
    r2 = make(aoi, s.replace(land_source="file", land_file=r.paths["land"]), **kw)
    assert r2.key != r.key


def test_no_point_clouds_raises_without_fallback(tmp_path):
    from terrain_s2.settings import Settings
    from terrain_s2.stage1.dem import make
    http = FakeHttp({catalogue.OT_CATALOG: {"Datasets": []}})
    s = Settings(backend="geofabrics", resolution=8, data_dir=tmp_path)
    aoi = f"POLYGON(({X0} {Y1 - 160}, {X0 + 160} {Y1 - 160}, {X0 + 160} {Y1}, {X0} {Y1 - 160}))"
    with pytest.raises(catalogue.NoPointClouds):
        make(aoi, s, aoi_crs="EPSG:2193", http=http, log=lambda *a: None)


def _geofabrics_importable():
    try:
        import geofabrics  # noqa: F401
        import distributed  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _geofabrics_importable(), reason="geofabrics not installed (terrain worker / [geofabrics] extra)")
@pytest.mark.parametrize("res", [1, 8])
def test_geofabrics_runner_coarse_dem(tmp_path, monkeypatch, res):
    """The pinned GeoFabrics runs our template end to end (raw then hydrologic DEM) and its output is finished
    to the contract: grid on the snapped bounds, compound CRS, integer description, COG at 1 m."""
    try:
        import pdal  # noqa: F401
    except ImportError:
        monkeypatch.syspath_prepend(str(STUBS))
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(STUBS), os.environ.get("PYTHONPATH", "")]))
        sys.modules.pop("pdal", None)
    import geofabrics
    if getattr(geofabrics, "__version__", gf.PIN) != gf.PIN:
        pytest.skip("not the pinned geofabrics")
    from shapely.geometry import box

    from terrain_s2.settings import Settings
    from terrain_s2.stage1.profile import load
    yy, xx = np.mgrid[Y1 - 0.5:Y1 - 400:-1, X0 + 0.5:X0 + 400:1]
    write_cog(tmp_path / "coarse.tif", (5 + 0.01 * (xx - X0)).astype(np.float32), X0, Y1)
    s = Settings(backend="geofabrics", resolution=res, buffer_m=0, data_dir=tmp_path / "data", gf_cores=1,
                 gf_memory_limit="1GiB")
    bounds = (X0 + 40, Y1 - 200, X0 + 200, Y1 - 40)
    r = gf.build(settings=s, profile=load("nz"), aoi_geom=box(*bounds), bounds=bounds, out_dir=tmp_path / "prod",
                 name=f"dem_{res}m", key="k", info={"code": {}}, datasets=None, mapping=None, land_file=None,
                 land_rec={}, cache_dir=tmp_path / "data" / "geofabrics", log=lambda *a: None,
                 overrides={"default": {"data_paths": {"coarse_dems": [str(tmp_path / "coarse.tif")]}}})
    p = C.read_dem(r["paths"]["netcdf"])
    assert p.transform == (res, 0, X0 + 40, 0, -res, Y1 - 40) and p.crs == "EPSG:2193+7839"
    assert p.resolution == res and np.all(p.data_source == 5) and r["gap_fraction"] == 0
    x = X0 + 40 + res / 2 + res * np.arange(p.z.shape[1])
    np.testing.assert_allclose(p.z[0], 5 + 0.01 * (x - X0), atol=1e-4)
    assert ("cog" in r["paths"]) == (res == 1)
    import rasterio
    with rasterio.open(C.gdal_name(r["paths"]["netcdf"])) as src:
        assert tuple(src.transform)[:6] == (res, 0, X0 + 40, 0, -res, Y1 - 40)
