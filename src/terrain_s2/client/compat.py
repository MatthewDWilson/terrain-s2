"""NewZeaLiDAR-compatible shim for FReDT and Smart Ideas (plan 6.6, section 10).

The call sites keep their signatures; parameters come from the ``TERRAIN_*`` settings:

- ``get_dem_by_geometry(conn, geometry, index=None)`` -> ``(hydro_dem_path, raw_dem_path, extent_path,
  resolution)``. In the raster route, and for our products generally, ``raw_dem_path`` is the product itself.
- ``get_dem_band_and_resolution_by_geometry(conn, geometry, band=1)`` -> ``(Dataset, resolution)``: the netCDF
  opened with ``rioxarray.open_rasterio(...).sel(band=band)``, as NewZeaLiDAR does, after checking the
  ``description`` resolution against the grid.
- ``dem_signature(wkt, **overrides)`` -> a Celery signature for ``eddie_terrain.tasks.ensure_dem`` on queue
  ``terrain``, for FReDT's ``create_model_for_area`` chain (``process_dem.si(wkt)`` is replaced by it).

The DEM is the newest stored product made with the current settings whose AOI contains the geometry (it was made
by ``ensure_dem`` earlier in the chain). When the product is larger than the geometry by more than 10 m^2, a
clip is made and registered, as NewZeaLiDAR's ``clip_dem`` did.
"""
from __future__ import annotations

from pathlib import Path

TASK = "eddie_terrain.tasks.ensure_dem"
QUEUE = "terrain"


class DemNotFound(LookupError):
    pass


def _geometry(geometry, srid: int):
    """A shapely geometry in the working CRS from a shapely geometry (assumed working CRS), GeoDataFrame,
    GeoSeries or Series (one row)."""
    import geopandas as gpd
    import pandas as pd
    if isinstance(geometry, (gpd.GeoSeries, gpd.GeoDataFrame)):
        if len(geometry) != 1:
            raise ValueError(f"only one geometry is allowed, got {len(geometry)}")
        g = geometry.to_crs(srid) if geometry.crs is not None else geometry
        return (g.geometry if isinstance(g, gpd.GeoDataFrame) else g).iloc[0]
    if isinstance(geometry, (pd.Series, pd.DataFrame)):
        return geometry["geometry"].values[0] if "geometry" in geometry else geometry.values[0]
    return geometry


def _client(conn, **overrides):
    from ..settings import from_env
    from . import TerrainClient
    return TerrainClient(conn, settings=from_env(**overrides))


def find_product(conn, geometry, **overrides):
    """The stored product for ``geometry`` under the current settings (clipped when much larger)."""
    c = _client(conn, **overrides)
    from ..stage1.profile import load
    srid = int(load(c.settings.profile)["crs"]["horizontal"])
    geom = _geometry(geometry, srid)
    p = c.find(geom)
    if p is None:
        raise DemNotFound("no terrain product covers this geometry with the current TERRAIN_* settings "
                          f"({c.request()}); run eddie_terrain.tasks.ensure_dem first (dem_signature)")
    if p.aoi.area - geom.area > 10:
        p = c.clip(p, geom)
    return p


def get_dem_by_geometry(conn, geometry, index=None):
    """NewZeaLiDAR signature: ``(hydro_dem_path, raw_dem_path, extent_path, resolution)``. ``index`` is
    accepted for compatibility and unused."""
    p = find_product(conn, geometry)
    nc = p.paths["netcdf"]
    raw = p.paths.get("raw_dem")
    if raw is None:
        # GeoFabrics' raw DEM sits beside the product as <name>_raw_dem.nc; the raster route has none
        cand = Path(nc).with_name(Path(nc).stem + "_raw_dem.nc")
        raw = str(cand) if cand.exists() else nc
    return nc, raw, p.paths.get("extents", ""), int(p.resolution)


def get_dem_band_and_resolution_by_geometry(conn, geometry, band: int = 1):
    """NewZeaLiDAR signature: ``(Dataset, resolution)``; refuses a product whose ``description`` resolution
    disagrees with its grid."""
    import rioxarray as rxr
    nc, _, _, _ = get_dem_by_geometry(conn, geometry)
    with rxr.open_rasterio(Path(nc)) as f:
        dem = f.sel(band=band)
        res = {abs(r) for r in dem.rio.resolution()}
        res_no = res.pop() if len(res) == 1 else None
        res_desc = int(str(dem.attrs.get("description", "")).split()[-1])
        if res_no != res_desc:
            raise ValueError("Inconsistent resolution between metadata and actual resolution of the Hydro DEM.")
        dem.load()
    return dem, int(res_no)


def dem_signature(wkt: str, **overrides):
    """Celery signature for ``ensure_dem(wkt, params)`` on queue ``terrain`` (immutable, like ``.si``).
    ``overrides`` are settings fields (e.g. ``resolution=4``) sent as the task's parameters."""
    from celery import signature
    return signature(TASK, args=(wkt, {k: v for k, v in overrides.items()}), queue=QUEUE, immutable=True)
