"""Terrain tasks, routed to the ``terrain`` queue (the terrain worker).

- ``ensure_dem(aoi_wkt, params=None) -> product_id``: the ``terrain.dem`` process (Stage 1).
- ``ensure_network(aoi_wkt=None, params=None, tile_id=None) -> {structures, dem_honest, dem_burned, ...}``:
  the ``terrain.network`` process (Stage 2, Input B) once the Stage 2 entry point exists (S2-1, S2-6).
- ``refresh_catalogue() -> dict``: drop the cached STAC catalogue and collection documents, so the next request
  sees new surveys (the item indexes are keyed by each collection's ``updated`` and refresh themselves).

Callers send them by name (``eddie_terrain.tasks.ensure_dem``), e.g. through
``terrain_s2.client.compat.dem_signature``. Only the core Celery app and light modules are imported here; the
library is imported inside the task bodies, because v4 core imports every plugin's ``tasks`` in every container.

AOIs arrive as WKT in EPSG:4326, as elsewhere in EDDIE. With ``TERRAIN_AOI_MODE=bbox`` (the default) the AOI is
the bounding rectangle of the polygon in the working CRS, which is what ``eddie.tasks.wkt_to_gdf`` gives FReDT,
so the compatibility shim finds the product for FReDT's catchment area. ``polygon`` keeps the polygon.
"""
from __future__ import annotations

import logging

from eddie.tasks import OnFailureStateTask, app  # pylint: disable=cyclic-import

from . import QUEUE

log = logging.getLogger(__name__)

# Route every terrain task to the terrain worker, wherever it is sent from.
_routes = app.conf.task_routes or {}
if isinstance(_routes, dict):
    app.conf.task_routes = {**_routes, "eddie_terrain.tasks.*": {"queue": QUEUE}}
else:  # a list or tuple of routers: put ours first
    app.conf.task_routes = ({"eddie_terrain.tasks.*": {"queue": QUEUE}},) + tuple(_routes)

# Settings fields a request may set (paths, credentials and the database stay with the deployment).
REQUEST_FIELDS = ("backend", "resolution", "product", "land_source", "lidar_classes", "buffer_m", "spatial_reuse",
                  "profile", "allow_fallback")


def request_settings(params: dict | None):
    """Validated library settings for a request: the deployment's ``TERRAIN_*`` plus allowed ``params``."""
    from .config import TerrainConfig
    params = dict(params or {})
    aoi_mode = params.pop("aoi_mode", None)
    bad = set(params) - set(REQUEST_FIELDS)
    if bad:
        raise ValueError(f"unknown or disallowed terrain parameters: {sorted(bad)}; allowed: {REQUEST_FIELDS}")
    if "lidar_classes" in params:
        params["lidar_classes"] = tuple(int(c) for c in params["lidar_classes"])
    if "resolution" in params:
        from terrain_s2.io.contract import check_resolution
        params["resolution"] = check_resolution(params["resolution"])
    return TerrainConfig.settings(**params), (aoi_mode or TerrainConfig.TERRAIN_AOI_MODE)


def aoi_geometry(aoi_wkt: str, settings, aoi_mode: str = "bbox"):
    """The AOI in the working CRS: the polygon, or (``bbox``) its bounding rectangle there."""
    from shapely.geometry import box

    from terrain_s2.stage1.aoi import to_working
    from terrain_s2.stage1.profile import load
    p = load(settings.profile)
    g = to_working(aoi_wkt, f"EPSG:{p['crs']['horizontal']}", "EPSG:4326")
    if aoi_mode == "bbox":
        return box(*g.bounds)
    if aoi_mode != "polygon":
        raise ValueError(f"aoi_mode must be 'bbox' or 'polygon', got {aoi_mode!r}")
    return g


def _registry(settings):
    from terrain_s2.stage1.profile import load

    from .tables import get_store
    return get_store(srid=int(load(settings.profile)["crs"]["horizontal"]))


@app.task(base=OnFailureStateTask, name="eddie_terrain.tasks.ensure_dem")
def ensure_dem(aoi_wkt: str, params: dict | None = None) -> int:
    """Make or find the Stage 1 DEM for an AOI (WKT, EPSG:4326); returns the ``terrain_product`` id."""
    from terrain_s2.stage1.dem import ensure
    s, mode = request_settings(params)
    geom = aoi_geometry(aoi_wkt, s, mode)
    row = ensure(geom, s, aoi_crs=f"EPSG:{_epsg(s)}", registry=_registry(s), log=log.info)
    log.info(f"terrain.dem: product {row.id} ({row.generator_key}) {row.paths.get('netcdf')}")
    return int(row.id)


@app.task(base=OnFailureStateTask, name="eddie_terrain.tasks.ensure_network")
def ensure_network(aoi_wkt: str | None = None, params: dict | None = None, tile_id: str | None = None) -> dict:
    """Stage 2 on a 1 m DEM with a 200 m buffer (Input B through the shared reader). Returns the paths of the
    structures layer and the honest and burned DEMs. Needs the Stage 2 library entry point
    ``terrain_s2.run.network`` (S2-1) and conditioning (S2-6)."""
    try:
        from terrain_s2 import run as s2run
    except ImportError:
        s2run = None
    if s2run is None or not hasattr(s2run, "network"):
        raise NotImplementedError("terrain.network needs terrain_s2.run.network (Stage 2 item S2-1), which is not "
                                  "in this terrain-s2 version")
    from terrain_s2.stage1.dem import ensure
    p = dict(params or {})
    stage2 = p.pop("stage2", {}) or {}
    if tile_id:
        from terrain_s2 import grid
        from shapely.geometry import box
        geom = box(*grid.tile_bounds(tile_id))
        s, _ = request_settings({**p, "resolution": 1, "buffer_m": 200})
    else:
        if not aoi_wkt:
            raise ValueError("give aoi_wkt or tile_id")
        s, mode = request_settings({**p, "resolution": 1, "buffer_m": 200})
        geom = aoi_geometry(aoi_wkt, s, mode)
    row = ensure(geom, s, aoi_crs=f"EPSG:{_epsg(s)}", registry=_registry(s), log=log.info)
    from pathlib import Path
    out = Path(row.paths["netcdf"]).parent / "network"
    dem = row.paths.get("cog") or row.paths["netcdf"]
    res = s2run.network(dem, None, geom, params=stage2.get("params"), out=out)
    paths = getattr(res, "paths", None) or (res if isinstance(res, dict) else {})
    return {"dem_product_id": int(row.id), "out": str(out), **{k: str(v) for k, v in paths.items()}}


@app.task(base=OnFailureStateTask, name="eddie_terrain.tasks.refresh_catalogue")
def refresh_catalogue() -> dict:
    """Drop the cached STAC catalogue and collection documents (``<TERRAIN_DATA_DIR>/stac/json``)."""
    import shutil

    from .config import TerrainConfig
    s = TerrainConfig.settings()
    d = s.data_dir / "stac" / "json"
    n = len(list(d.glob("*.json"))) if d.exists() else 0
    if d.exists():
        shutil.rmtree(d)
    return {"removed": n, "folder": str(d)}


def _epsg(settings) -> int:
    from terrain_s2.stage1.profile import load
    return int(load(settings.profile)["crs"]["horizontal"])
