"""Stage 1 (W2, W3): contract, key, aggregation, AOI snapping and the raster backend on a fake STAC."""
import json

import numpy as np
import pytest

# Stage 1 needs the light base install (xarray, netCDF4, rioxarray, SQLAlchemy) beside the Stage 2 stack, and
# GDAL's netCDF driver (conda-forge: libgdal-netcdf). Skipped where they are missing.
for _m in ("xarray", "netCDF4", "rioxarray", "sqlalchemy"):
    pytest.importorskip(_m)

from terrain_s2 import key as K
from terrain_s2.io import contract as C
from terrain_s2.stage1 import aggregate, aoi as A

from stage1_fakes import FakeHttp, nz_profile, stac

X0, Y1 = 1_641_800.0, 6_076_100.0          # near SH12 (Whirinaki), EPSG:2193


# --- contract (W2) ---------------------------------------------------------------------------------------
def _product(res=1, shape=(24, 32)):
    z = np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape) / 10
    z[0, 0] = np.nan
    return C.DemProduct(z=z, transform=(res, 0, X0, 0, -res, Y1), crs="EPSG:2193+7839", resolution=res,
                        generator="terrain_s2:raster", lidar_source=np.ones(shape), lidar_mapping={"s": 1},
                        provenance={"generator_key": "abc"})


def test_contract_round_trip_and_gdal_subdataset(tmp_path):
    import rasterio
    p = _product()
    out = C.write_dem(tmp_path / "dem_1m.nc", p)
    assert set(out) == {"netcdf", "cog", "provenance"}
    with rasterio.open(C.gdal_name(out["netcdf"])) as s:          # GDAL reads netcdf:"<file>":z
        assert s.count == 1 and tuple(s.transform)[:6] == (1, 0, X0, 0, -1, Y1)
        assert "NZVD2016" in s.crs.to_wkt()                         # compound CRS kept
        np.testing.assert_array_equal(s.read(1), p.z)
    with rasterio.open(out["cog"]) as s:
        assert s.profile["blockxsize"] == 512 and tuple(s.transform)[:6] == (1, 0, X0, 0, -1, Y1)
        np.testing.assert_array_equal(s.read(1), p.z)
    r = C.read_dem(out["netcdf"])
    assert r.transform == p.transform and r.crs == "EPSG:2193+7839" and r.resolution == 1
    np.testing.assert_array_equal(r.z, p.z)
    assert r.lidar_mapping == {"s": 1, "no LiDAR": -1}
    assert np.all(r.data_source[np.isfinite(p.z)] == 5) and r.data_source[0, 0] == -1
    assert r.provenance == {"generator_key": "abc"}
    assert json.loads((tmp_path / "dem_1m.provenance.json").read_text()) == {"generator_key": "abc"}


def test_contract_description_and_resolution_guard(tmp_path):
    import xarray as xr
    out = C.write_dem(tmp_path / "d8.nc", _product(res=8))
    assert "cog" not in out                                          # COG only beside 1 m products
    with xr.open_dataset(out["netcdf"]) as ds:
        assert int(ds.attrs["description"].split()[-1]) == 8        # what FReDT does
        assert ds["x"].attrs["standard_name"] == "projection_x_coordinate" and ds["z"].attrs["grid_mapping"] == "spatial_ref"
    for bad in (2.5, "8.0x", 0, -1, True):
        with pytest.raises(ValueError):
            C.check_resolution(bad)
    assert C.check_resolution(8.0) == 8
    with pytest.raises(ValueError):
        C.write_dem(tmp_path / "bad.nc", C.DemProduct(z=np.zeros((2, 2), np.float32), transform=(2, 0, 0, 0, -2, 0),
                                                       crs="EPSG:2193", resolution=1))


def test_finish_netcdf_adds_conventions(tmp_path):
    """A netCDF without CF attributes (as GeoFabrics may write it) has no geotransform in GDAL until finished."""
    import rasterio
    import xarray as xr
    z = np.ones((4, 6), np.float32)
    ds = xr.Dataset({"z": (("y", "x"), z)}, coords={"x": X0 + 0.5 + np.arange(6), "y": Y1 - 0.5 - np.arange(4)},
                    attrs={"description": "geofabrics:HydrologicDemGenerator resolution 1"})
    f = tmp_path / "gf.nc"
    ds.to_netcdf(f)
    C.finish_netcdf(f, "EPSG:2193+7839", {"k": 1})
    with rasterio.open(C.gdal_name(f)) as s:
        assert tuple(s.transform)[:6] == (1, 0, X0, 0, -1, Y1) and "NZVD2016" in s.crs.to_wkt()
    with xr.open_dataset(f) as d2:
        assert json.loads(d2.attrs["terrain_provenance"]) == {"k": 1}
        assert d2.attrs["description"].endswith("resolution 1")


def test_valid_footprint_and_decimation():
    z = np.full((64, 64), np.nan, np.float32)
    z[8:40, 16:48] = 1
    tr = (1, 0, 0, 0, -1, 64)
    g = C.valid_footprint(z, tr)
    assert g.area == 32 * 32
    g8 = C.valid_footprint(z, tr, decimate=8)
    assert g8.contains(g) and g8.area <= (32 + 8) ** 2


# --- key (W2) ---------------------------------------------------------------------------------------------
def _components(**kw):
    c = dict(product="dem", profile="nz", backend="raster", resolution=8, land_source="coverage", buffer_m=10,
             source_version={"items": [["AW27", "1220ab"]]}, configuration={"lidar_classes": [2]},
             code={"terrain_s2": "0.1.0", "commit": "abc"})
    c.update(kw)
    return K.stage1_components(**c)


def test_key_deterministic_and_sensitive():
    k1, info = K.generator_key(_components())
    k2, _ = K.generator_key(_components(buffer_m=10.0))             # 10 == 10.0 after normalisation
    assert k1 == k2 and len(k1) == 16 and info["config_key"]
    for change in (dict(resolution=4), dict(land_source="topo50"), dict(backend="geofabrics"),
                   dict(source_version={"items": [["AW27", "1220cd"]]}), dict(configuration={"lidar_classes": [2, 9]}),
                   dict(code={"terrain_s2": "0.1.1", "commit": "abc"})):
        assert K.generator_key(_components(**change))[0] != k1, change
    # the config key ignores the source version only
    assert K.generator_key(_components(source_version={"x": 1}))[1]["config_key"] == info["config_key"]
    assert K.generator_key(_components(resolution=4))[1]["config_key"] != info["config_key"]
    assert K.canonical({"b": 1.0, "a": (1, 2.50000000000001)}) == '{"a":[1,2.5],"b":1}'


# --- aggregation and snapping (W2) -----------------------------------------------------------------------
def test_block_mean_min_valid_and_mode():
    z = np.arange(16, dtype=np.float32).reshape(4, 4)
    z[0, 0:2] = np.nan                                  # 2 of 4 valid in the first block: kept (50 %)
    z[2:4, 0] = np.nan
    z[3, 1] = np.nan                                    # 1 of 4 valid: no data
    m = aggregate.block_mean(z, 2)
    assert m[0, 0] == pytest.approx((4 + 5) / 2) and np.isnan(m[1, 0]) and m[1, 1] == pytest.approx(12.5)
    src = np.array([[1, 1, 2, 2], [2, 1, 2, 2], [1, 1, 1, 2], [2, 2, 1, 2]])
    assert aggregate.block_mode(src, 2).tolist() == [[1, 2], [1, 1]]   # tie (2-2) goes to the higher priority


def test_snap_to_resolution():
    assert A.snap((1641770.3, 6075482.2, 1642737.9, 6076173.1), 8) == (1641768, 6075480, 1642744, 6076176)
    assert A.snap((10.5, 10.5, 11.2, 11.2), 1) == (10, 10, 12, 12)
    g = A.to_working("POLYGON((172.7 -43.4, 172.71 -43.4, 172.71 -43.41, 172.7 -43.41, 172.7 -43.4))", "EPSG:2193")
    b = A.grid_bounds(g, 10, 4)
    assert all(v % 4 == 0 for v in b)


# --- raster backend (W3) ------------------------------------------------------------------------------------
def _surveys(edge=False):
    """Survey 'new' (2020) covers the west part; 'old' (2015) covers all (or only the east, for ``edge``);
    the national mosaic covers all with a constant."""
    w = X0 + 120
    old_x0 = w if edge else X0
    return [
        dict(region="northland", slug="northland_2020", end="2020-12-31T00:00:00Z",
             tiles=[("N1", X0, Y1 - 160, w, Y1, lambda x, y: 10 + 0 * x)]),
        dict(region="northland", slug="northland_2015", end="2015-12-31T00:00:00Z",
             tiles=[("O1", old_x0, Y1 - 160, X0 + 160, Y1 - 80, lambda x, y: 20 + 0 * x),
                    ("O2", old_x0, Y1 - 80, X0 + 160, Y1, lambda x, y: 20 + 0 * x)]),
        dict(region="new-zealand", slug="new-zealand", end="2024-12-31T00:00:00Z",
             tiles=[("AW27", X0 - 400, Y1 - 400, X0 + 400, Y1 + 400, lambda x, y: 99 + 0 * x)]),
    ]


def _run(tmp_path, res, aoi=None, edge=False, **settings):
    from terrain_s2.acquire.snapshot import SnapshotStore
    from terrain_s2.settings import Settings
    from terrain_s2.stage1.dem import make
    from terrain_s2.stage1.source import InterimLinzSource
    docs = stac(tmp_path, _surveys(edge))
    profile = nz_profile(national="new-zealand/new-zealand/dem_1m/2193/collection.json")
    http = FakeHttp(docs)
    src = InterimLinzSource(profile, http=http, cache_dir=tmp_path / "stac")
    s = Settings(resolution=res, buffer_m=0, data_dir=tmp_path / "data", **settings)
    aoi = aoi or f"POLYGON(({X0 + 8} {Y1 - 152}, {X0 + 152} {Y1 - 152}, {X0 + 152} {Y1 - 8}, {X0 + 8} {Y1 - 8}, {X0 + 8} {Y1 - 152}))"
    return make(aoi, s, aoi_crs="EPSG:2193", http=http, source=src, store=SnapshotStore(tmp_path / "snap"),
                log=lambda *a: None), http


def test_raster_1m_newest_first_with_fill(tmp_path):
    r, _ = _run(tmp_path, 1)
    p = C.read_dem(r.paths["netcdf"])
    assert p.z.shape == (144, 144) and p.transform == (1, 0, X0 + 8, 0, -1, Y1 - 8)
    assert not np.isnan(p.z).any()                                   # no gaps
    west, east = p.z[:, :112], p.z[:, 112:]
    assert np.all(west == 10) and np.all(east == 20)                 # newest survey wins where both cover
    m = p.lidar_mapping
    assert m["northland_2020"] == 1 and m["northland_2015"] == 2 and m["new-zealand"] == 3   # national last
    assert set(np.unique(p.lidar_source)) == {1, 2}
    prov = json.loads((tmp_path / "data" / "products" / r.key / r.product_dir.split("/")[-1] /
                       "dem_1m.provenance.json").read_text())
    used = {s["survey"]: s for s in prov["inputs"]}
    assert set(used) == {"northland_2020", "northland_2015"}         # the national mosaic was not needed
    assert {i["checksum"] for i in used["northland_2015"]["items"]} == {"1220O1northland_2015", "1220O2northland_2015"}
    assert r.paths["cog"].endswith(".tif") and r.gap_fraction == 0
    assert r.info["source_version"]["surveys"][0]["items"] == [["N1", "1220N1northland_2020"]]


def test_raster_8m_area_average_and_reuse(tmp_path):
    r, http = _run(tmp_path, 8)
    p = C.read_dem(r.paths["netcdf"])
    assert p.z.shape == (19, 18) and p.resolution == 8   # y snapped outward: 6075944-6076096 and "cog" not in r.paths
    assert np.allclose(p.z[:, :14], 10) and np.allclose(p.z[:, 14:], 20)   # 112 m = 14 cells of 8 m
    n = len(http.calls)
    r2, _ = _run(tmp_path, 8)
    assert r2.reused and r2.key == r.key and r2.paths == r.paths
    r3, _ = _run(tmp_path, 4)
    assert r3.key != r.key and not r3.reused
    del n


def test_raster_across_survey_edge_no_gaps(tmp_path):
    r, _ = _run(tmp_path, 1, edge=True)
    p = C.read_dem(r.paths["netcdf"])
    assert not np.isnan(p.z).any()
    assert set(np.unique(p.lidar_source)) == {1, 2}
    assert np.all(p.z[:, :112] == 10) and np.all(p.z[:, 112:] == 20)


def test_raster_national_fill_and_gap_report(tmp_path):
    aoi = f"POLYGON(({X0 + 100} {Y1 + 50}, {X0 + 200} {Y1 + 50}, {X0 + 200} {Y1 + 100}, {X0 + 100} {Y1 + 100}, {X0 + 100} {Y1 + 50}))"
    r, _ = _run(tmp_path, 1, aoi=aoi)                                 # north of both surveys: national only
    p = C.read_dem(r.paths["netcdf"])
    assert np.all(p.z == 99) and set(np.unique(p.lidar_source)) == {1}
    assert p.lidar_mapping == {"new-zealand": 1, "no LiDAR": -1}       # only surveys with tiles here are listed


def test_raster_no_survey(tmp_path):
    from terrain_s2.stage1.source import NoElevationData
    aoi = f"POLYGON(({X0 + 5000} {Y1}, {X0 + 5100} {Y1}, {X0 + 5100} {Y1 + 100}, {X0 + 5000} {Y1}))"
    with pytest.raises(NoElevationData):
        _run(tmp_path, 1, aoi=aoi)
