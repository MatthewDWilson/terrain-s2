"""``raster`` backend (plan 6.2): the LINZ 1 m DEM, no GeoFabrics. Writes the section 4.3 contract itself.

1. Snap the buffered AOI outward to multiples of the resolution (done by the caller).
2. Read the 1 m DEM window through the elevation source: newest survey first, national-mosaic fill,
   per-cell survey index; record collections, item ids, ``file:checksum`` and collection ``updated``.
3. Aggregate above 1 m by area averaging (cells under 50 % valid are no data).
4. Land: ``coverage`` removes nothing beyond no data; other land sources clip to the polygon.
5. Write ``data_source = 5`` where valid, ``lidar_source`` = survey index with the mapping in attributes,
   the ``description`` ending in the resolution, the COG copy (1 m), the provenance JSON and the extents.

Gaps are left as no data and reported as a fraction; there is no interpolation.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..io import contract
from . import aggregate, land
from .source import ElevationRead, SourcePlan

GENERATOR = "terrain_s2:raster"


def configuration(settings) -> dict:
    return {"lidar_classes": list(settings.lidar_classes), "min_valid": 0.5, "aggregation": "area_mean"}


def build(plan: SourcePlan, src, *, settings, profile: dict, aoi_geom, bounds, out_dir, name: str, key: str,
          info: dict, http=None, store=None, log=print) -> dict:
    """Read, aggregate, clip and write one product. Returns the paths and a summary."""
    t0 = time.perf_counter()
    rd: ElevationRead = src.read(plan)
    t_read = time.perf_counter() - t0
    res = int(settings.resolution)
    z, ls = rd.z, rd.lidar_source
    if res > 1:
        z, ls = aggregate.aggregate(z, ls, res)
    transform = (float(res), 0.0, float(bounds[0]), 0.0, -float(res), float(bounds[3]))
    out_dir = Path(out_dir)
    land_rec = {"land_source": settings.land_source}
    if settings.land_source != "coverage":
        lp, land_rec = land.build(settings.land_source, aoi_geom, rd.crs, out_dir / f"{name}_land.geojson",
                                  profile=profile, http=http, store=store, linz_key=settings.linz_api_key,
                                  land_file=settings.land_file)
        if lp is None:
            z = np.full_like(z, np.nan)
        else:
            from rasterio.features import geometry_mask
            from affine import Affine
            import geopandas as gpd
            g = gpd.read_file(lp).geometry
            outside = geometry_mask(list(g), out_shape=z.shape, transform=Affine(*transform), invert=False)
            z = np.where(outside, np.nan, z).astype(np.float32)
    valid = np.isfinite(z)
    ls = np.where(valid, ls, -1)
    ds = np.where(valid, contract.DATA_SOURCE["coarse DEM"], contract.NO_DATA)
    gap = float(1.0 - valid.mean()) if valid.size else 1.0
    if not valid.any():
        raise RuntimeError(f"no valid DEM cells in {tuple(round(b) for b in bounds)} after reading and land clip")
    from ..key import provenance
    inputs = [s.record() for s in rd.used]
    prov = provenance(key, info, inputs=inputs,
                      parameters={"settings": settings.public(), "grid_bounds": list(bounds)},
                      timings={"read_s": round(t_read, 2), "total_s": round(time.perf_counter() - t0, 2)},
                      extra={"backend": "raster", "gap_fraction": round(gap, 6), "land": land_rec,
                             "surveys_considered": [s.record(with_items=False) for s in plan.surveys]})
    p = contract.DemProduct(z=z, transform=transform, crs=rd.crs, resolution=res, product=settings.product,
                            generator=GENERATOR, data_source=ds, lidar_source=ls,
                            lidar_mapping={s.slug: s.index for s in plan.surveys}, provenance=prov)
    paths = contract.write_dem(out_dir / f"{name}.nc", p)
    geom = contract.valid_footprint(z, transform)
    paths["extents"] = str(contract.write_extents(out_dir / f"{name}_extents.geojson", geom, rd.crs))
    if settings.land_source != "coverage" and land_rec.get("path"):
        paths["land"] = land_rec["path"]
    log(f"raster: {z.shape[1]} x {z.shape[0]} cells at {res} m, {len(rd.used)} survey(s), gaps {gap:.2%}")
    return {"paths": paths, "extent": geom, "gap_fraction": gap, "crs": rd.crs, "provenance": prov}
