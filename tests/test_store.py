"""W5: product registry, reuse, clipping, and the NewZeaLiDAR-compatible shim (SQLite; PostGIS on the workstation)."""
import numpy as np
import pytest

from terrain_s2.io import contract as C
from terrain_s2.store import Store, sources_compatible

from stage1_fakes import FakeHttp, nz_profile, stac
from test_stage1 import X0, Y1, _surveys


def _box(x0, y0, x1, y1):
    return f"POLYGON(({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"


BIG = _box(X0 + 8, Y1 - 152, X0 + 152, Y1 - 8)
SMALL = _box(X0 + 40, Y1 - 120, X0 + 100, Y1 - 40)


@pytest.fixture
def env(tmp_path):
    from terrain_s2.acquire.snapshot import SnapshotStore
    from terrain_s2.stage1.source import InterimLinzSource
    http = FakeHttp(stac(tmp_path, _surveys()))
    src = InterimLinzSource(nz_profile(national="new-zealand/new-zealand/dem_1m/2193/collection.json"), http=http,
                            cache_dir=tmp_path / "stac")
    reg = Store(f"sqlite:///{tmp_path / 'reg.sqlite'}")
    kw = dict(aoi_crs="EPSG:2193", http=http, source=src, store=SnapshotStore(tmp_path / "snap"), registry=reg,
              log=lambda *a: None)
    return tmp_path, reg, kw


def _settings(tmp, **kw):
    from terrain_s2.settings import Settings
    return Settings(**{"resolution": 8, "buffer_m": 0, "data_dir": tmp / "data", **kw})


def test_register_and_exact_hit(env):
    from terrain_s2.stage1.dem import ensure
    tmp, reg, kw = env
    p1 = ensure(BIG, _settings(tmp), **kw)
    assert p1.id == 1 and p1.parent_id is None and p1.resolution == 8 and p1.product == "dem"
    assert p1.aoi.bounds == (X0 + 8, Y1 - 152, X0 + 152, Y1 - 8) and p1.extent.area > 0
    p2 = ensure(BIG, _settings(tmp), **kw)
    assert p2.id == p1.id                                            # identical request: cache hit
    p3 = ensure(BIG, _settings(tmp, resolution=4), **kw)
    assert p3.id != p1.id and p3.generator_key != p1.generator_key     # changed resolution: new product
    p4 = ensure(BIG, _settings(tmp, land_source="file", land_file=str(_land(tmp))), **kw)
    assert p4.generator_key not in (p1.generator_key, p3.generator_key)


def _land(tmp):
    import geopandas as gpd
    from shapely.geometry import box
    f = tmp / "land.geojson"
    gpd.GeoDataFrame(geometry=[box(X0, Y1 - 200, X0 + 100, Y1)], crs=2193).to_file(f)
    return f


def test_spatial_reuse_clips_a_containing_product(env):
    from terrain_s2.stage1.dem import ensure
    tmp, reg, kw = env
    big = ensure(BIG, _settings(tmp), **kw)
    no_reuse = ensure(SMALL, _settings(tmp), **kw)
    assert no_reuse.parent_id is None and no_reuse.id != big.id
    child = ensure(SMALL, _settings(tmp, spatial_reuse=True), **kw)
    # the small request's own product (exact key) exists now, so it is found first
    assert child.id == no_reuse.id
    small2 = _box(X0 + 24, Y1 - 136, X0 + 64, Y1 - 24)
    c2 = ensure(small2, _settings(tmp, spatial_reuse=True), **kw)
    assert c2.parent_id == big.id and c2.generator_key == big.generator_key
    pb, pc = C.read_dem(big.netcdf), C.read_dem(c2.netcdf)
    a, _, c, _, e, f = pb.transform
    r0, c0 = int((f - pc.transform[5]) / -e), int((pc.transform[2] - c) / a)
    np.testing.assert_array_equal(pc.z, pb.z[r0:r0 + pc.z.shape[0], c0:c0 + pc.z.shape[1]])
    assert pc.provenance["clipped_from"]["id"] == big.id
    assert ensure(small2, _settings(tmp, spatial_reuse=True), **kw).id == c2.id     # the clip is reused


def test_sources_compatible():
    a = {"source_version": {"surveys": [{"collection": "c1", "items": [["A", "1"]]},
                                        {"collection": "c2", "items": [["B", "2"]]}]}}
    b = {"source_version": {"surveys": [{"collection": "c1", "items": [["A", "1"], ["A2", "3"]]},
                                        {"collection": "c2", "items": [["B", "2"]]}]}}
    assert sources_compatible(a, b)
    changed = {"source_version": {"surveys": [{"collection": "c1", "items": [["A", "9"]]}]}}
    assert not sources_compatible(changed, b)
    swapped = {"source_version": {"surveys": [{"collection": "c2", "items": [["B", "2"]]},
                                              {"collection": "c1", "items": [["A", "1"]]}]}}
    assert not sources_compatible(swapped, b)
    gfa = {"source_version": {"mapping": {"X": 1, "Y": 2}}}
    assert sources_compatible(gfa, {"source_version": {"mapping": {"X": 1, "Y": 2, "Z": 3}}})
    assert not sources_compatible(gfa, {"source_version": {"mapping": {"Y": 1, "X": 2}}})


def test_compat_shim(env, monkeypatch):
    from shapely import from_wkt

    from terrain_s2.client import compat
    from terrain_s2.stage1.dem import ensure
    tmp, reg, kw = env
    for k, v in dict(TERRAIN_RESOLUTION="8", TERRAIN_BUFFER_M="0", TERRAIN_DATA_DIR=str(tmp / "data"),
                     TERRAIN_DB_URL=f"sqlite:///{tmp / 'reg.sqlite'}").items():
        monkeypatch.setenv(k, v)
    with pytest.raises(compat.DemNotFound):
        compat.get_dem_by_geometry(None, from_wkt(BIG))
    p = ensure(BIG, _settings(tmp), **kw)
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(geometry=[from_wkt(BIG)], crs=2193)      # as FReDT passes it
    hydro, raw, extent, res = compat.get_dem_by_geometry(None, gdf)
    assert hydro == p.netcdf and raw == p.netcdf and extent.endswith("_extents.geojson") and res == 8
    dem, r = compat.get_dem_band_and_resolution_by_geometry(None, gdf)
    assert r == 8 and "z" in dem and dem.rio.bounds() == p.grid.bounds
    import pyproj
    assert pyproj.CRS(dem.spatial_ref.crs_wkt).sub_crs_list[0].to_epsg() == 2193   # FReDT reads crs_wkt
    # a smaller catchment gets a clip, registered as a child
    hydro2, _, _, _ = compat.get_dem_by_geometry(None, gpd.GeoDataFrame(geometry=[from_wkt(SMALL)], crs=2193))
    assert hydro2 != p.netcdf and C.read_dem(hydro2).z.shape == (11, 8)
    # other settings do not match
    monkeypatch.setenv("TERRAIN_RESOLUTION", "4")
    with pytest.raises(compat.DemNotFound):
        compat.get_dem_by_geometry(None, gdf)


def test_dem_signature():
    pytest.importorskip("celery")
    from terrain_s2.client.compat import dem_signature
    s = dem_signature("POLYGON((0 0,1 0,1 1,0 0))", resolution=4)
    assert s.task == "eddie_terrain.tasks.ensure_dem" and s.options["queue"] == "terrain" and s.immutable
    assert s.args == ("POLYGON((0 0,1 0,1 1,0 0))", {"resolution": 4})


def test_geometry_column_round_trip(tmp_path):
    from shapely.geometry import box
    reg = Store(f"sqlite:///{tmp_path / 'g.sqlite'}").ensure()
    rec = dict(key="k" * 16, info={"config_key": "c", "request": {}}, product="dem", resolution=8,
               aoi_wkt=box(0, 0, 10, 10).wkt, extent_wkt=None, grid_bounds=[0, 0, 16, 16],
               paths={"netcdf": str(tmp_path / "missing.nc")})
    p = reg.register(rec)
    assert p.aoi.equals(box(0, 0, 10, 10)) and p.grid.bounds == (0, 0, 16, 16)
    assert reg.get(p.id).generator_info == {"config_key": "c", "request": {}}
