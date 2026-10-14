"""Stage 1 river and waterway bed estimation (GeoFabrics rivers / waterways stages): network mapping, river
selection (checked against GeoFabrics' own upstream trace), instructions, and the orchestration with a fake
runner. The live run (LiDAR, REC1, Overpass) is on the workstation (docs/stage1_rivers.md)."""
import json
from pathlib import Path

import pytest

for _m in ("xarray", "netCDF4", "rioxarray", "sqlalchemy", "geopandas"):
    pytest.importorskip(_m)

import geopandas as gpd
import yaml
from shapely.geometry import LineString

from terrain_s2.stage1 import catalogue
from terrain_s2.stage1 import rivers as R

from stage1_fakes import FakeHttp

X0, Y1 = 1_641_800.0, 6_076_100.0
GRID = (X0, Y1 - 160, X0 + 160, Y1)


def _network(raw_names=True, reverse=()):
    """River A crosses the grid (A3 -> A2 -> A1, outlet A1, tributary T1 below the threshold); river B lies
    in the margin but not on the grid. Reaches digitised upstream to downstream, except those in ``reverse``.
    ``raw_names``: as NIWA's flood-statistics REC1 layer (NZREACH, Areakm2, H_C18_MAF; no node fields);
    otherwise GeoFabrics' names with nodes."""
    x, xb = X0 + 80, X0 + 900
    rows = [  # id, from, to, area m2, flow, geometry
        (9000003, 1, 2, 30e6, 12.0, LineString([(x, Y1 + 800), (x, Y1 + 400)])),
        (9000002, 2, 3, 32e6, 13.0, LineString([(x, Y1 + 400), (x, Y1 + 100)])),
        (9000009, 9, 3, 2e6, 1.0, LineString([(x - 300, Y1 + 300), (x, Y1 + 100)])),
        (9000001, 3, 4, 35e6, 14.5, LineString([(x, Y1 + 100), (x, Y1 - 700)])),
        (9000021, 20, 21, 50e6, 20.0, LineString([(xb, Y1 + 500), (xb, Y1 - 500)])),
    ]
    geoms = [LineString(list(r[5].coords)[::-1]) if r[0] in reverse else r[5] for r in rows]
    if raw_names:
        return gpd.GeoDataFrame([{"NZREACH": float(r[0]), "Areakm2": r[3] / 1e6, "H_C18_MAF": r[4], "Strm_Order": 1}
                                 for r in rows], geometry=geoms, crs=2193)
    return gpd.GeoDataFrame([dict(zip(("id", "from_node", "to_node", "area", "flow"), r[:5])) for r in rows],
                            geometry=geoms, crs=2193)


def _nz():
    import copy

    from terrain_s2.stage1.profile import load
    return copy.deepcopy(load("nz"))                                 # load() is cached: never mutate its result


def _profile(tmp_path, src_path, **rivers):
    p = _nz()
    p["rivers"]["network"]["source"] = {"kind": "file", "path": str(src_path)}
    p["rivers"].update(rivers)
    f = tmp_path / "nz_test.yml"
    f.write_text(yaml.safe_dump(p))
    return f


def test_map_columns():
    p = _nz()
    g = R.map_columns(_network(), p)                                # the profile's REC1 mapping
    assert list(g.columns) == ["id", "to_node", "from_node", "flow", "mannings_n", "area", "geometry"]
    assert (g["mannings_n"] == 0.035).all() and g["flow"].tolist()[0] == 12.0   # constant n while no field
    assert g["area"].tolist()[0] == pytest.approx(30e6)               # km2 -> m2
    assert g.attrs["topology"]["topology"] == "geometry"
    p["rivers"]["network"]["columns"]["flow"] = "h_c18_maf"         # case differs from the layer: still found
    R.map_columns(_network(), p)
    p["rivers"]["network"]["columns"]["flow"] = "MALF"
    with pytest.raises(R.RiverNetworkError, match="'MALF'.*not found.*H_C18_MAF"):
        R.map_columns(_network(), p)                                 # names the layer's fields
    p["rivers"]["network"]["columns"]["flow"] = None
    with pytest.raises(R.RiverNetworkError, match="columns.flow is not set"):
        R.map_columns(_network(), p)


@pytest.mark.parametrize("reverse", [(), (9000002, 9000009), (9000001,), (9000003, 9000002, 9000001, 9000009, 9000021)])
def test_topology_from_geometry(reverse):
    """Nodes and direction from the geometry, whatever the digitising direction: the trace is the same as with
    the source's own nodes, and every reach ends up running downstream."""
    A1, A2, A3, T1, B = 9000001, 9000002, 9000003, 9000009, 9000021
    g = R.map_columns(_network(reverse=reverse), _nz())
    assert R.upstream(g, A1, 1e6) == [A1, A2, A3]
    assert R.upstream(g, T1, 1e6) == [T1]
    t = g.attrs["topology"]
    assert t["connections"] == 3 and t["connections_downstream_larger"] == 3
    ys = {int(r.id): (r.geometry.coords[0][1], r.geometry.coords[-1][1]) for r in g.itertuples()}
    assert all(ys[i][0] > ys[i][1] for i in (A1, A2, A3, B))       # all run south (downstream)
    assert t["undecided"] == 1                                       # B, isolated: the majority direction
    assert t["reversed"] - (B in reverse) == len([i for i in reverse if i != B])


def test_select_rivers_and_upstream_trace():
    net = _network(raw_names=False)
    A1, A2, A3, T1 = 9000001, 9000002, 9000003, 9000009
    assert R.upstream(net, A1, 10e6) == [A1, A2, A3]                # main stem, downstream first
    assert R.upstream(net, A1, 1e6) == [A1, A2, A3]                 # the larger branch only: T1 not followed
    assert R.upstream(net, T1, 1e6) == [T1]
    rivers = R.select_rivers(net, GRID, 10e6)
    assert [r.network_id for r in rivers] == [A1]                   # B is in the margin, not on the grid
    assert (rivers[0].n_reaches, rivers[0].flow_min, rivers[0].flow_max) == (3, 12.0, 14.5)
    assert R.select_rivers(net, GRID, 40e6) == []                    # nothing that large on the grid


def test_upstream_matches_geofabrics(tmp_path):
    """Our trace selects the same reaches as GeoFabrics' own (Channel._get_up_stream_reaches)."""
    bathy = pytest.importorskip("geofabrics.bathymetry_estimation")
    net = _network(raw_names=False)
    for start, thr in ((9000001, 10e6), (9000001, 1e6), (9000002, 10e6), (9000009, 1e6)):
        theirs = bathy.Channel._get_up_stream_reaches(network=net, starting_id=start, area_threshold=thr)
        assert list(theirs["id"].astype(int)) == R.upstream(net, start, thr)


def test_plan_and_instructions(tmp_path):
    src = tmp_path / "rec1.gpkg"
    _network().to_file(src)
    from terrain_s2.stage1.profile import load
    p = load(str(_profile(tmp_path, src)))
    plan = R.plan(None, GRID, p, log=lambda *a: None)
    assert [r.network_id for r in plan.rivers] == [9000001] and plan.source["kind"] == "file"
    assert plan.source["digest"] and plan.source["sha256"]
    assert plan.network.total_bounds[3] <= Y1 + 1000                # clipped to the grid + margin
    default = {"output": {"grid_params": {"resolution": 8}}, "processing": {"number_of_cores": 2},
               "data_paths": {"local_cache": str(tmp_path / "prod"), "subfolder": ".", "downloads": "/d",
                              "extents": "dem_8m.geojson", "land": "/l.geojson", "raw_dem": "r.nc",
                              "result_dem": "dem_8m.nc"},
               "datasets": {"lidar": {"open_topography": {"NZ18_North": {}}}}, "dataset_mapping": {"lidar": {"NZ18_North": 1}}}
    ins = R.river_instructions(default, plan.rivers[0], tmp_path / "net.geojson", p)
    assert set(ins) == {"rivers"}                                   # one stage only: no DEM auto-added
    st = ins["rivers"]
    assert st["data_paths"]["subfolder"] == "rivers/9000001"
    assert not {"extents", "land", "raw_dem", "result_dem"} & set(st["data_paths"])
    assert st["output"]["grid_params"]["resolution"] == 1           # channel characterised at 1 m
    assert st["rivers"]["network_id"] == 9000001 and st["rivers"]["area_threshold"] == 10e6
    assert "osm" not in st["rivers"]                                # no OSM id given: aligned to the DEM
    assert st["datasets"] == default["datasets"]
    p["rivers"]["osm"] = {9000001: {"type": "way", "id": 132793862}}
    plan2 = R.plan(None, GRID, p, log=lambda *a: None)
    assert R.river_instructions(default, plan2.rivers[0], tmp_path / "n", p)["rivers"]["rivers"]["osm"]["id"] == 132793862
    ww = R.waterways_instructions(default, p)["waterways"]
    assert ww["data_paths"]["subfolder"] == "waterways" and Path(ww["data_paths"]["extents"]).is_absolute()
    assert R.dem_paths(plan.rivers, True)["rivers"][0]["elevations"] == "rivers/9000001/river_bathymetry.geojson"


def _fake_runner(calls, fail=()):
    """Writes what each stage would: river polygon and bathymetry, waterway files, or the DEM."""
    from test_stage1_gf import _fake_gf_runner

    def run(ins):
        calls.append(ins)
        if "rivers" in ins:
            st = ins["rivers"]
            if st["rivers"]["network_id"] in fail:
                raise RuntimeError("no channel found")
            d = Path(st["data_paths"]["local_cache"]) / st["data_paths"]["subfolder"]
            d.mkdir(parents=True, exist_ok=True)
            for n in ("river_polygon", "river_bathymetry"):
                (d / f"{n}.geojson").write_text('{"type": "FeatureCollection", "features": []}')
        elif "waterways" in ins:
            st = ins["waterways"]
            d = Path(st["data_paths"]["local_cache"]) / st["data_paths"]["subfolder"]
            d.mkdir(parents=True, exist_ok=True)
        else:
            _fake_gf_runner(ins)
    return run


def test_make_with_rivers(tmp_path):
    from test_stage1_gf import OT_RESPONSE, _tile_index_zip

    from terrain_s2.acquire.snapshot import SnapshotStore
    from terrain_s2.settings import Settings
    from terrain_s2.stage1.dem import make
    src = tmp_path / "rec1.gpkg"
    _network().to_file(src)
    prof = _profile(tmp_path, src)
    tis = {catalogue.tile_index_url(n): _tile_index_zip(tmp_path, n, (X0 - 50, Y1 - 300, X0 + 300, Y1 + 50))
           for n in ("NZ22_Whirinaki", "NZ18_Northland", "NZ_Undated")}
    http = FakeHttp({catalogue.OT_CATALOG: OT_RESPONSE}, binary=tis)
    aoi = f"POLYGON(({X0} {Y1 - 160}, {X0 + 160} {Y1 - 160}, {X0 + 160} {Y1}, {X0} {Y1}, {X0} {Y1 - 160}))"
    s = Settings(backend="geofabrics", product="dem", resolution=8, buffer_m=0, data_dir=tmp_path / "data",
                 profile=str(prof))
    kw = dict(aoi_crs="EPSG:2193", http=http, store=SnapshotStore(tmp_path / "snap"), log=lambda *a: None)
    base = make(aoi, s, gf_runner=_fake_runner([]), **kw)
    calls = []
    r = make(aoi, s.replace(rivers=True, waterways=True), gf_runner=_fake_runner(calls), **kw)
    assert r.key != base.key                                        # rivers and waterways are in the key
    assert r.info["source_version"]["rivers"]["rivers"][0]["network_id"] == 9000001
    assert [next(iter(c)) for c in calls] == ["rivers", "waterways", "default"]   # bed stages before the DEM
    ins = json.loads(Path(r.paths["instructions"]).read_text())
    dp = ins["dem"]["data_paths"]
    assert dp["rivers"] == [{"extents": "rivers/9000001/river_polygon.geojson",
                             "elevations": "rivers/9000001/river_bathymetry.geojson"}]
    assert len(dp["waterways"]) == 2
    assert ins["dem"]["general"]["z_labels"]["rivers"] == "bed_elevation_Rupp_and_Smart"
    assert Path(r.paths["rivers"], "network.geojson").exists() and Path(r.paths["rivers"], "9000001", "instructions.json").exists()
    prov = json.loads(Path(r.paths["provenance"]).read_text())
    assert prov["rivers"]["runs"][0]["ok"] and prov["rivers"]["network"]["kind"] == "file"
    # every river failing fails the build (nothing cached); with rivers only, no waterway call
    with pytest.raises(RuntimeError, match="every selected river"):
        make(aoi, s.replace(rivers=True, resolution=4), gf_runner=_fake_runner([], fail=(9000001,)), **kw)


def test_rivers_need_geofabrics():
    from terrain_s2.settings import from_env
    with pytest.raises(ValueError, match="need backend 'geofabrics'"):
        from_env({"TERRAIN_GF_RIVERS": "true"})
    assert from_env({"TERRAIN_BACKEND": "geofabrics", "TERRAIN_GF_RIVERS": "1"}).rivers
