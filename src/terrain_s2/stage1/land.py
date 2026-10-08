"""Land polygon for Stage 1 (plan 6.4; Document 2 S1.4). Every source writes exactly one polygon file,
clipped to the AOI buffered by 100 m, because GeoFabrics accepts only one land file.

============================  =======================================================================
``land_source``               Polygon
============================  =======================================================================
``coverage``                  raster: valid-data footprint of the 1 m read; GeoFabrics: union of the
                              selected datasets' tile-index footprints (supplied by the caller)
``topo50``                    LINZ 51153 (Topo50 coastline polygons, MHW). Regression only
``topo50_mangrove``           union of LINZ 51153 and 50296 (Topo50 mangrove polygons). Interim NZ
``file``                      ``LAND_FILE``
============================  =======================================================================

LINZ layers are fetched through :mod:`terrain_s2.acquire.linz_wfs` and the snapshot store (S2-10), so every
response is kept with its URL, parameters, time and sha256. Coastline polygons come back island-sized and are
clipped at once.
"""
from __future__ import annotations

from pathlib import Path


def build(source: str, aoi_geom, crs, out_path, *, profile: dict, coverage=None, http=None, store=None,
          linz_key: str | None = None, land_file: str | None = None):
    """Write the land polygon for ``source`` and return ``(path or None, record)``.

    ``aoi_geom`` is in the working CRS ``crs``. ``coverage`` (a shapely geometry in ``crs``) is required for
    ``coverage``. Returns ``None`` as the path when the polygon is empty (the caller decides whether that is
    an error); the record lists what was used, with snapshot hashes.
    """
    import geopandas as gpd
    from shapely.ops import unary_union

    buf = float(profile.get("land", {}).get("clip_buffer_m", 100))
    clip = aoi_geom.buffer(buf)
    rec = {"land_source": source, "clip_buffer_m": buf}
    if source == "coverage":
        if coverage is None:
            raise ValueError("land_source 'coverage' needs the coverage footprint")
        geom = coverage.intersection(clip)
    elif source in ("topo50", "topo50_mangrove"):
        layers = [profile["land"]["topo50_coastline"]]
        if source == "topo50_mangrove":
            layers.append(profile["land"]["topo50_mangrove"])
        parts, rec["layers"] = [], []
        for layer in layers:
            gdf, info = _linz_layer(layer, clip, crs, http=http, store=store, key=linz_key)
            rec["layers"].append(info)
            parts.extend(p.intersection(clip) for p in gdf.geometry if p is not None and not p.is_empty)
        geom = unary_union(parts) if parts else None
    elif source == "file":
        if not land_file:
            raise ValueError("land_source 'file' needs LAND_FILE")
        g = gpd.read_file(land_file)
        if g.crs is None:
            raise ValueError(f"{land_file}: no CRS")
        geom = unary_union(list(g.to_crs(crs).geometry)).intersection(clip)
        from ..key import file_sha256
        rec.update(file=str(land_file), sha256=file_sha256(land_file))
    else:
        raise ValueError(f"unknown land_source {source!r}")
    if geom is None or geom.is_empty or geom.area <= 0:
        rec["empty"] = True
        return None, rec
    geom = geom.buffer(0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from ..io.contract import _horizontal
    gpd.GeoDataFrame({"source": [source]}, geometry=[geom], crs=_horizontal(crs)).to_file(out_path, driver="GeoJSON")
    rec.update(path=str(out_path), area_m2=float(geom.area))
    return out_path, rec


def _linz_layer(layer: int, clip, crs, http=None, store=None, key=None):
    """One LINZ Data Service layer over the clip window, in ``crs`` (the WFS client returns EPSG:2193)."""
    import geopandas as gpd
    from pyproj import CRS

    from ..acquire import linz_wfs
    if http is None:
        from ..acquire.http import Http
        http = Http(store=store)
    c = CRS.from_user_input(crs)
    h = c.sub_crs_list[0] if c.is_compound else c
    window = gpd.GeoSeries([clip], crs=h).to_crs(2193).total_bounds
    try:
        g, info = linz_wfs.get_layer(http, layer, tuple(window), key=key, store=store, source=f"linz-{layer}")
    except ValueError as e:
        if "none near the window" in str(e):       # offshore window: no land
            return gpd.GeoDataFrame(geometry=[], crs=h), dict(layer=layer, count=0)
        raise
    if len(g):
        g = g.to_crs(h)
    return g, info


def tile_index_coverage(index_paths, crs):
    """Union of OpenTopography tile-index footprints (``<name>_TileIndex.zip`` shapefiles), in ``crs``."""
    import geopandas as gpd
    from shapely.ops import unary_union

    from ..io.contract import _horizontal
    parts = []
    for p in index_paths:
        p = Path(p)
        g = gpd.read_file(f"zip://{p}") if p.suffix == ".zip" else gpd.read_file(p)
        if g.crs is None:
            raise ValueError(f"{p}: tile index has no CRS")
        parts.extend(g.to_crs(_horizontal(crs)).geometry)
    return unary_union(parts) if parts else None
