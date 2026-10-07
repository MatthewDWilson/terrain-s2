#!/usr/bin/env python
"""Write the synthetic scene as GeoTIFFs (DEM + DSM-like) plus its ground truth, for end-to-end tests."""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from terrain_s2.synthetic import make_scene  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
out.mkdir(parents=True, exist_ok=True)
s = make_scene()
prof = dict(driver="GTiff", height=s.z.shape[0], width=s.z.shape[1], count=1, dtype="float32",
            crs=s.crs, transform=s.transform, nodata=-9999.0, compress="deflate", tiled=True)
z = np.where(np.isfinite(s.z), s.z, -9999.0).astype(np.float32)
with rasterio.open(out / "synthetic_dem.tif", "w", **prof) as d:
    d.write(z, 1)
with rasterio.open(out / "synthetic_dsm.tif", "w", **prof) as d:
    d.write(np.where(np.isfinite(s.dsm), s.dsm, -9999.0).astype(np.float32), 1)
b0 = rasterio.transform.array_bounds(s.z.shape[0], s.z.shape[1], s.transform)
aoi = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::2193"}},
       "features": [{"type": "Feature", "properties": {"id": 1}, "geometry": {"type": "Polygon", "coordinates": [[
           [b0[0] + 60, b0[1] + 60], [b0[2] - 60, b0[1] + 60], [b0[2] - 60, b0[3] - 60],
           [b0[0] + 60, b0[3] - 60], [b0[0] + 60, b0[1] + 60]]]}}]}  # (west, south, east, north)
(out / "aoi.geojson").write_text(json.dumps(aoi))
(out / "truth.json").write_text(json.dumps({k: list(map(float, v)) for k, v in s.truth.items()}, indent=2))
b = rasterio.transform.array_bounds(s.z.shape[0], s.z.shape[1], s.transform)
print("wrote", out, "bounds", b)
