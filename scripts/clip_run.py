#!/usr/bin/env python
"""A small window of one tile's run and bundle, for sharing (e.g. with the Stage 2 agent's sandbox).

    python scripts/clip_run.py --tile BA31_10000_0305 --out /mnt/b/terrain-s2            # 1 km, at the council lines
    python scripts/clip_run.py --tile CD08_10000_0502 --size 1500 --centre 1234567 4876543

Clips the saved rasters (features.tif, hydro.tif, conditioned.tif) and the bundle's dem.tif and dsm.tif to the
window, copies network.gpkg, labels.gpkg, ml_channels.gpkg and rasters.json whole, and zips the folder. Without
--centre the window is centred where the council channel lines are densest (ml_channels.gpkg), else on the tile.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", required=True)
    ap.add_argument("--runs")
    ap.add_argument("--bundles")
    ap.add_argument("--size", type=float, default=1000.0, help="window side, m")
    ap.add_argument("--centre", type=float, nargs=2, metavar=("X", "Y"))
    ap.add_argument("--out", default=".", help="folder for clip_<tile>/ and its zip")
    a = ap.parse_args()
    import numpy as np
    import rasterio
    import rasterio.windows as W
    from terrain_s2.acquire.elevation import default_data_dir
    data = default_data_dir()
    run = Path(a.runs or data / "runs") / a.tile
    bundle = Path(a.bundles or data / "tiles") / a.tile
    with rasterio.open(run / "features.tif") as s:
        b = s.bounds
    cx, cy = a.centre if a.centre else ((b.left + b.right) / 2, (b.bottom + b.top) / 2)
    if not a.centre and (run / "ml_channels.gpkg").exists():
        import geopandas as gpd
        mid = gpd.read_file(run / "ml_channels.gpkg", layer="council_channels").geometry.interpolate(0.5, normalized=True)
        if len(mid):                                   # the densest 500 m cell of line midpoints
            k = np.floor(np.column_stack([mid.x, mid.y]) / 500.0).astype(int)
            cells, n = np.unique(k, axis=0, return_counts=True)
            cx, cy = (cells[n.argmax()] + 0.5) * 500.0
    h = a.size / 2
    box = (max(cx - h, b.left), max(cy - h, b.bottom), min(cx + h, b.right), min(cy + h, b.top))
    out = Path(a.out) / f"clip_{a.tile}"
    out.mkdir(parents=True, exist_ok=True)
    for f in [run / n for n in ("features.tif", "hydro.tif", "conditioned.tif")] + [bundle / "dem.tif", bundle / "dsm.tif"]:
        if not f.exists():
            continue
        with rasterio.open(f) as s:
            win = W.from_bounds(*box, transform=s.transform).round_offsets().round_lengths()
            p = s.profile | dict(width=int(win.width), height=int(win.height), transform=s.window_transform(win))
            p.pop("blockxsize", None), p.pop("blockysize", None), p.pop("tiled", None)
            with rasterio.open(out / f.name, "w", **p) as o:
                o.write(s.read(window=win, boundless=True, fill_value=s.nodata if s.nodata is not None else 0))
                o.descriptions = s.descriptions
    for f in (run / "network.gpkg", bundle / "labels.gpkg", run / "ml_channels.gpkg", run / "rasters.json"):
        if f.exists():
            shutil.copy(f, out)
    z = shutil.make_archive(str(out), "zip", out)
    print(f"window {[round(v) for v in box]}: {z} ({Path(z).stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
