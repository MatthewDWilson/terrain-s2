"""W8 (integration plan 9.3, sandbox part): the module against the EDDIE core's plugin mechanism.

Run once with core v4.0.0 installed and once with v5-integration. The live check (Postgres, Redis and the terrain
worker; ensure_dem by name; a cache hit on repeat) is on the workstation (docs/stage1_workstation.md).
"""
import json
import subprocess
import sys

import pytest

HEAVY = ("geofabrics", "terrain_s2.stage1.gf", "terrain_s2.stage1.raster", "numba", "scipy", "pyflwdir", "dask",
         "pdal", "terrain_s2.pipeline")


def _fresh(code: str) -> dict:
    """Run ``code`` in a fresh interpreter (the core's imports have side effects) and return its JSON output."""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_core_tasks_discovers_module_without_heavy_imports():
    out = _fresh(
        "import json, sys\n"
        "import eddie.tasks as t\n"
        "names = sorted(n for n in t.app.tasks if n.startswith('eddie_terrain'))\n"
        f"heavy = sorted(m for m in {HEAVY!r} if m in sys.modules)\n"
        "print(json.dumps({'names': names, 'routes': t.app.conf.task_routes, 'heavy': heavy,"
        " 'plugins': sorted(t.eddie_plugins)}))")
    assert "eddie_terrain" in out["plugins"]
    assert out["names"] == ["eddie_terrain.tasks.ensure_dem", "eddie_terrain.tasks.ensure_network",
                            "eddie_terrain.tasks.refresh_catalogue"]
    assert out["routes"]["eddie_terrain.tasks.*"] == {"queue": "terrain"}
    assert out["heavy"] == []


def test_core_app_registers_blueprint_without_heavy_imports():
    out = _fresh(
        "import json, sys\n"
        "import eddie.app as a\n"
        "rules = sorted(str(r) for r in a.app.url_map.iter_rules() if 'terrain' in str(r))\n"
        f"heavy = sorted(m for m in {HEAVY!r} if m in sys.modules)\n"
        "print(json.dumps({'rules': rules, 'heavy': heavy}))")
    assert out["rules"] == ["/terrain/dem", "/terrain/products/<int:product_id>", "/terrain/tasks/<task_id>"]
    assert out["heavy"] == []
    assert not any(r.endswith("/wps") for r in out["rules"])


def test_v5_attributes():
    import eddie_terrain
    assert eddie_terrain.blueprint.blueprint.name == "eddie_terrain"
    assert [p.identifier for p in eddie_terrain.processes] == ["terrain.dem"]
    assert eddie_terrain.Config.__name__ == "TerrainConfig"
    assert hasattr(eddie_terrain.tasks, "ensure_dem")


@pytest.fixture
def client(monkeypatch, tmp_path):
    import eddie.app as a

    import eddie_terrain.tables as tables
    from terrain_s2.store import Store
    reg = Store(f"sqlite:///{tmp_path / 'r.sqlite'}").ensure()
    monkeypatch.setattr(tables, "get_store", lambda srid=2193: reg)
    sent = []

    class _T:
        id = "task-1"

    import eddie.tasks as t
    monkeypatch.setattr(t.app, "send_task", lambda name, args=None, queue=None, **kw: sent.append((name, args, queue)) or _T())
    a.app.config["TESTING"] = True
    return a.app.test_client(), sent, reg


def test_post_dem_validates_and_sends_to_terrain_queue(client):
    c, sent, _ = client
    wkt = "POLYGON((174.88 -35.77, 174.89 -35.77, 174.89 -35.76, 174.88 -35.76, 174.88 -35.77))"
    assert c.post("/terrain/dem", json={}).status_code == 400
    assert c.post("/terrain/dem", json={"aoi": wkt, "params": {"resolution": 2.5}}).status_code == 400
    assert c.post("/terrain/dem", json={"aoi": wkt, "params": {"data_dir": "/etc"}}).status_code == 400
    r = c.post("/terrain/dem", json={"aoi": wkt, "params": {"resolution": 4}})
    assert r.status_code == 202 and r.get_json() == {"taskId": "task-1"}
    assert sent == [("eddie_terrain.tasks.ensure_dem", [wkt, {"resolution": 4}], "terrain")]


def test_get_product(client, tmp_path):
    c, _, reg = client
    assert c.get("/terrain/products/1").status_code == 404
    from shapely.geometry import box
    nc = tmp_path / "x.nc"
    nc.write_bytes(b"")
    reg.register(dict(key="k" * 16, info={"config_key": "c", "request": {}}, product="dem", resolution=8,
                      aoi_wkt=box(0, 0, 8, 8).wkt, extent_wkt=None, grid_bounds=[0, 0, 8, 8], paths={"netcdf": str(nc)}))
    r = c.get("/terrain/products/1")
    assert r.status_code == 200 and r.get_json()["paths"]["netcdf"] == str(nc)


def test_ensure_dem_task_body(monkeypatch, tmp_path):
    """The task body end to end on the fake STAC (the raster backend), registry on SQLite: product id returned,
    a repeat is a cache hit, and FReDT's catchment rectangle finds the product through the shim."""
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2] / "tests"))
    from stage1_fakes import FakeHttp, nz_profile, stac
    from test_stage1 import X0, Y1, _surveys

    import eddie_terrain.tables as tables
    import eddie_terrain.tasks as tasks
    from terrain_s2.stage1 import source
    from terrain_s2.store import Store
    http = FakeHttp(stac(tmp_path, _surveys()))
    prof = nz_profile(national="new-zealand/new-zealand/dem_1m/2193/collection.json")
    monkeypatch.setattr(source, "get_source", lambda profile, **kw: source.InterimLinzSource(
        prof, http=http, cache_dir=tmp_path / "stac"))
    reg = Store(f"sqlite:///{tmp_path / 'r.sqlite'}").ensure()
    monkeypatch.setattr(tables, "get_store", lambda srid=2193: reg)
    monkeypatch.setenv("TERRAIN_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TERRAIN_BUFFER_M", "0")
    monkeypatch.setenv("TERRAIN_DB_URL", f"sqlite:///{tmp_path / 'r.sqlite'}")
    import geopandas as gpd
    from shapely.geometry import box
    rect = box(X0 + 8, Y1 - 152, X0 + 152, Y1 - 8)
    wkt = gpd.GeoSeries([rect], crs=2193).to_crs(4326).iloc[0].wkt
    pid = tasks.ensure_dem.run(wkt, {"resolution": 8})
    assert pid == tasks.ensure_dem.run(wkt, {"resolution": 8})
    row = reg.get(pid)
    assert row.resolution == 8 and row.aoi.equals(row.aoi.envelope)       # bbox mode: a rectangle
    # FReDT: wkt_to_gdf gives the projected bounding rectangle; the shim finds this product for it
    import eddie.tasks as core
    from terrain_s2.client import compat
    monkeypatch.setenv("TERRAIN_RESOLUTION", "8")
    hydro, _, _, res = compat.get_dem_by_geometry(None, core.wkt_to_gdf(wkt))
    assert hydro == row.netcdf and res == 8
