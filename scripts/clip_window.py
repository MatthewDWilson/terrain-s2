#!/usr/bin/env python
"""Clip AOI + buffer from large DEM/DSM tiles into small GeoTIFFs (for sharing or fast iteration).

    python scripts/clip_window.py --aoi data/aoi.geojson --buffer 200 \
        --src data/AW27.tif --src data/AW27_dsm.tif --out data/clips

The test-site window (AOI + 200 m) is ~1.5 M cells, i.e. a few MB per raster.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from terrain_s2 import dataio  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--aoi", required=True)
ap.add_argument("--buffer", type=float, default=200.0)
ap.add_argument("--src", action="append", required=True, help="repeat per raster (each clipped separately)")
ap.add_argument("--out", required=True)
ap.add_argument("--assume-crs", help="relabel inputs with this CRS (no reprojection), e.g. EPSG:2193+7839 "
                                     "for LINZ tiles mislabelled EPSG:9528")
ap.add_argument("--aoi-crs", help="relabel the AOI with this CRS (no reprojection), e.g. EPSG:2193")
ap.add_argument("--reproject-to", help="warp inputs to this projected CRS (resamples; prefer native data), "
                                       "e.g. EPSG:2193 for LDS exports downloaded in EPSG:9528")
ap.add_argument("--resolution", type=float, default=1.0, help="output cell size (m) when reprojecting")
ap.add_argument("--resampling", default="bilinear", help="resampling when reprojecting (bilinear, cubic, nearest)")
ap.add_argument("--prefix", default="", help="prefix for output file names")
a = ap.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
stems = [Path(s).stem for s in a.src]
for src in a.src:
    stem = Path(src).stem
    if stems.count(stem) > 1:                     # e.g. DEM and DSM tiles with the same name
        stem = f"{Path(src).parent.name}_{stem}"
    with rasterio.open(src) as s:
        nd = s.nodata
    hcrs = dataio.processing_crs(src, a.assume_crs, a.reproject_to)   # AOI goes into this CRS
    aoi = dataio.read_aoi(a.aoi, hcrs, a.aoi_crs)
    w = dataio.read_window([src], tuple(aoi.total_bounds), a.buffer, assume_crs=a.assume_crs,
                           reproject_to=a.reproject_to, resolution=a.resolution, resampling=a.resampling)
    nodata = nd if nd is not None else -9999.0
    z = np.where(np.isfinite(w.z), w.z, nodata).astype(np.float32)
    dst = out / f"{a.prefix}{stem}_clip.tif"
    with rasterio.open(dst, "w", driver="GTiff", height=z.shape[0], width=z.shape[1], count=1,
                       dtype="float32", crs=w.crs, transform=w.transform, nodata=nodata,
                       compress="deflate", predictor=3, tiled=True) as d:
        d.write(z, 1)
        tags = dict(source_file=Path(src).name)
        if a.reproject_to:
            tags.update(reprojected_to=a.reproject_to, resolution=str(a.resolution), resampling=a.resampling)
        with rasterio.open(src) as s:
            d.update_tags(**s.tags(), **tags)
    print(f"{dst}: {z.shape[1]}x{z.shape[0]} cells, {dst.stat().st_size / 1e6:.1f} MB, CRS {dataio.horizontal_crs(w.crs).name}")
