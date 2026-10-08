"""AOI handling for Stage 1: any CRS in, the profile's working CRS out; buffering and snapping."""
from __future__ import annotations

import math


def to_working(aoi, working_crs, aoi_crs=None):
    """A shapely polygon in ``working_crs`` from a shapely geometry, WKT, GeoJSON dict, GeoDataFrame/GeoSeries
    or a vector file path. ``aoi_crs`` is the CRS of a bare geometry or WKT (default EPSG:4326, as EDDIE sends
    WKT in WGS84)."""
    import geopandas as gpd
    import shapely
    from shapely.geometry import shape
    from shapely.ops import unary_union

    if isinstance(aoi, (gpd.GeoDataFrame, gpd.GeoSeries)):
        g = aoi
        if g.crs is None:
            if aoi_crs is None:
                raise ValueError("the AOI has no CRS; pass aoi_crs")
            g = g.set_crs(aoi_crs)
        geom = unary_union(list(g.to_crs(working_crs).geometry))
    else:
        if isinstance(aoi, str) and not aoi.lstrip().upper().startswith(("POLYGON", "MULTIPOLYGON", "{")):
            return to_working(gpd.read_file(aoi), working_crs, aoi_crs)
        if isinstance(aoi, str) and aoi.lstrip().startswith("{"):
            import json
            aoi = json.loads(aoi)
        if isinstance(aoi, dict):
            if aoi.get("type") == "FeatureCollection":
                return to_working(gpd.GeoDataFrame.from_features(aoi["features"], crs=aoi_crs or 4326),
                                  working_crs)
            geom = shape(aoi.get("geometry", aoi))
        elif isinstance(aoi, str):
            geom = shapely.from_wkt(aoi)
        else:
            geom = aoi
        geom = gpd.GeoSeries([geom], crs=aoi_crs or 4326).to_crs(working_crs).iloc[0]
    geom = shapely.force_2d(geom)                  # a Z AOI (KML, some GeoJSON) would not fit 2D columns
    if geom.is_empty or not geom.is_valid or geom.area <= 0:
        geom = geom.buffer(0)
    if geom.is_empty or geom.area <= 0:
        raise ValueError("the AOI is empty or has no area")
    b = geom.bounds
    if not all(math.isfinite(v) for v in b):
        raise ValueError("the AOI bounds are not finite in the working CRS (check its CRS)")
    return geom


def snap(bounds, resolution: int):
    """Bounds expanded outward to multiples of ``resolution``. At 1 m the cell edges fall on whole metres,
    as in LINZ tiles, so the Stage 1 DEM shares the LINZ DEM and DSM grid."""
    r = float(resolution)
    x0, y0, x1, y1 = bounds
    return (math.floor(x0 / r) * r, math.floor(y0 / r) * r, math.ceil(x1 / r) * r, math.ceil(y1 / r) * r)


def grid_bounds(aoi_geom, buffer_m: float, resolution: int):
    """The product grid: the AOI's bounding box, buffered, snapped outward to the resolution."""
    x0, y0, x1, y1 = aoi_geom.bounds
    return snap((x0 - buffer_m, y0 - buffer_m, x1 + buffer_m, y1 + buffer_m), resolution)


def shape_of(bounds, resolution: int):
    x0, y0, x1, y1 = bounds
    return int(round((y1 - y0) / resolution)), int(round((x1 - x0) / resolution))


def transform_of(bounds, resolution: int):
    x0, _, _, y1 = bounds
    r = float(resolution)
    return (r, 0.0, x0, 0.0, -r, y1)


def lonlat_bounds(bounds, crs):
    from pyproj import Transformer
    t = Transformer.from_crs(crs, 4326, always_xy=True)
    return t.transform_bounds(*bounds, densify_pts=21)
