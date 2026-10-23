"""OSM guidance for the channel network (channel network v2, stage C): main rivers always present, placed on the LiDAR.

The current network finds rivers only where the channel mask has wide cells, so a river on a wide, flat bed can
vanish (CF15_10000_0501). Here OSM helps find rivers and streams, and the LiDAR decides where they are:

- **River extents** (``river_extent``): OSM river areas (``osm_water`` with ``water_kind = river_area``) refined on
  the DEM. Each area's water level is estimated locally (``level_window_m``: the mean of its lower cells), carried
  out to a band around the area (``band_m``), and the extent is every cell in the area or the band no higher than
  the level plus ``tol_m``, connected to the area. A misplaced or out-of-date polygon is thus moved to the low,
  flat bed beside it; one with no low bed nearby adds little.
- **Waterway lines** (``snap_waterways``): OSM waterway lines of the given kinds (default river, stream, canal,
  drain), snapped onto the bed by the council-line snapping (``ml.snap``: a least-cost path through a corridor,
  on the 11 and 21 m top-hats), with a wider corridor for rivers. Only lines that land on a bed (``min_bed_frac``)
  are used; parts inside a river extent are left to the river's centreline.

``load`` reads the OSM layers for a window from the bundle's ``context.gpkg``, or (for bundles built before the
water layers, Oct 2026) from the OSM extract named in ``sites/sources.yml``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
LINE_KINDS = ("river", "stream", "canal", "drain")
CORRIDOR_M = dict(river=30.0, canal=20.0, stream=20.0, drain=15.0)
SNAP = dict(min_bed_frac=0.5)


def _extract_path():
    import yaml
    from .acquire.elevation import default_data_dir
    f = ROOT / "sites" / "sources.yml"
    if not f.exists():
        return None
    ext = ((yaml.safe_load(f.read_text()) or {}).get("sources", {}).get("osm") or {}).get("extract")
    if not ext:
        return None
    p = Path(ext)
    return p if p.is_absolute() else default_data_dir() / p


def load(context=None, window=None, extract=None):
    """OSM guidance for a window (NZTM bounds): dict(lines=waterway lines, areas=water polygons (``water_kind``),
    coastline=lines, source=where they came from). Empty frames when there is nothing."""
    import geopandas as gpd
    import pyogrio
    out = dict(lines=None, areas=None, coastline=None, source=None)
    have = set()
    if context is not None and Path(context).exists():
        have = {n for n, _ in pyogrio.list_layers(context)}
        for key, layer in (("lines", "osm_waterways"), ("areas", "osm_water"), ("coastline", "osm_coastline")):
            if layer in have:
                out[key] = gpd.read_file(context, layer=layer, bbox=tuple(window) if window is not None else None)
        out["source"] = str(context)
    if "osm_water" not in have and window is not None:
        ext = Path(extract) if extract else _extract_path()
        if ext is not None and ext.exists():
            from .acquire import osm
            parts, meta = osm.from_extract(ext, window)
            if meta.get("water_layer"):
                out["areas"] = parts.get("water")
                out["coastline"] = parts.get("coastline")
                out["lines"] = out["lines"] if out["lines"] is not None else parts.get("waterway")
                out["source"] = f"{out['source'] or ''} + extract {ext}".strip(" +")
    for k in ("lines", "areas", "coastline"):
        if out[k] is None:
            out[k] = gpd.GeoDataFrame(geometry=[], crs=2193)
        elif out[k].crs is not None and out[k].crs.to_epsg() != 2193:
            out[k] = out[k].to_crs(2193)
    return out


def river_extent(areas, z, T, band_m: float = 15.0, tol_m: float = 0.5, level_window_m: float = 50.0,
                 min_area_m2: float = 500.0):
    """Bool mask of the river extents: OSM river areas refined on the DEM ``z`` (see the module docstring)."""
    from rasterio.features import rasterize
    shape = z.shape
    if areas is None or not len(areas):
        return np.zeros(shape, bool)
    riv = areas[areas.get("water_kind", "river_area") == "river_area"] if "water_kind" in areas else areas
    if not len(riv):
        return np.zeros(shape, bool)
    A = rasterize([(g, 1) for g in riv.geometry if g is not None and not g.is_empty], out_shape=shape, transform=T,
                  fill=0, dtype="uint8").astype(bool)
    valid = np.isfinite(z)
    A &= valid
    if not A.any():
        return np.zeros(shape, bool)
    cs = abs(T.a)
    k = max(3, int(round(level_window_m / cs)) | 1)
    zA = np.where(A, z, 0.0)
    n = ndi.uniform_filter(A.astype(np.float64), k)
    m1 = np.where(n > 0, ndi.uniform_filter(zA, k) / np.maximum(n, 1e-12), np.nan)
    low = A & (z <= m1)                                      # the lower half of the area's cells nearby
    nl = ndi.uniform_filter(low.astype(np.float64), k)
    level = np.where(nl > 0, ndi.uniform_filter(np.where(low, z, 0.0), k) / np.maximum(nl, 1e-12), m1)
    dist, (ri, ci) = ndi.distance_transform_edt(~A, return_indices=True)
    lvl = level[ri, ci]                                      # each cell takes the level of the nearest area cell
    cand = valid & (dist * cs <= band_m) & np.isfinite(lvl) & (z <= lvl + tol_m)
    lab, nlab = ndi.label(cand, np.ones((3, 3), bool))
    if not nlab:
        return np.zeros(shape, bool)
    touch = np.zeros(nlab + 1, bool)
    touch[np.unique(lab[A & cand])] = True
    area = np.bincount(lab.ravel(), minlength=nlab + 1) * cs * cs
    keep = touch & (area >= min_area_m2)
    keep[0] = False
    return keep[lab]


def snap_waterways(lines, bands, T, kinds=LINE_KINDS, river_mask=None, params=None):
    """OSM waterway lines of ``kinds`` snapped onto the bed (``ml.snap.snap_lines``, corridor by kind); returns
    the lines with ``snapped``, ``bed_frac``, ``offset_med_m`` and ``waterway``; parts inside ``river_mask`` are cut."""
    import geopandas as gpd
    import pandas as pd

    from .ml import snap
    if lines is None or not len(lines) or "waterway" not in lines:
        return gpd.GeoDataFrame(geometry=[], crs=2193)
    L = lines[lines.waterway.isin(kinds)].copy()
    L = L[L.geometry.notna() & ~L.geometry.is_empty]
    if not len(L):
        return gpd.GeoDataFrame(geometry=[], crs=2193)
    L = L.explode(index_parts=False).reset_index(drop=True)
    depth, bed = snap.bed_depth(bands)
    parts = []
    for kind, g in L.groupby("waterway"):
        P = dict(SNAP, corridor_m=CORRIDOR_M.get(kind, 15.0), **(params or {}))
        parts.append(snap.snap_lines(g, depth, T, P, bed))
    S = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True).drop(columns="geometry_council", errors="ignore"),
                         geometry="geometry", crs=2193)
    if river_mask is not None and river_mask.any():
        rp = _polys(river_mask, T)
        if len(rp):
            from shapely import union_all
            u = union_all(rp)
            S["geometry"] = S.geometry.difference(u)
            S = S[~S.geometry.is_empty].explode(index_parts=False).reset_index(drop=True)
    return S


def _polys(mask, T):
    from rasterio.features import shapes
    from shapely.geometry import shape
    return [shape(g) for g, v in shapes(mask.astype(np.uint8), mask=mask, transform=T) if v == 1]


def polygons(mask, T, simplify_m: float = 1.0):
    """Polygons of a bool mask (rows as dicts with geometry), as ``network.river_polygons`` returns them."""
    return [dict(geometry=g.simplify(simplify_m)) for g in _polys(mask, T)]
