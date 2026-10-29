#!/usr/bin/env python
"""Aerial imagery on each run's grid (``ml_rgb.tif``), for the imagery test (stage 2: the U-Net with RGB bands).

    python scripts/imagery_rasters.py --manifest ml/manifests/wave1.csv ml/manifests/wave2.csv ml/manifests/wave3.csv

For each tile with saved rasters in ``--runs`` (default ``<data>/runs``): LINZ aerial imagery (``acquire.imagery``), the
survey closest to the tile's LiDAR capture (``--prefer newest`` for the newest), read at 1 m on the run's grid and
written as 8-bit RGB (0 = no data) with ``ml_rgb.json`` (surveys used, valid share). Tiles already done are skipped
(``--rerun`` to redo). Prints a line per tile and a summary; failures are listed, not fatal.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _one(job):
    tile, runs, bundles, prefer = job
    from terrain_s2.ml.chips import imagery_raster
    try:
        return tile, imagery_raster(Path(runs) / tile, Path(bundles) / tile, prefer=prefer), None
    except Exception as e:  # noqa: BLE001 - recorded per tile
        return tile, None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--runs")
    ap.add_argument("--bundles")
    ap.add_argument("--prefer", choices=["lidar", "newest"], default="lidar")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    import pandas as pd

    from terrain_s2.acquire.elevation import default_data_dir
    data = default_data_dir()
    runs = Path(a.runs or data / "runs")
    bundles = Path(a.bundles or data / "tiles")
    m = pd.concat([pd.read_csv(f) for f in a.manifest], ignore_index=True).drop_duplicates("tile_id")
    tiles = [t for t in m.tile_id if (runs / t / "features.tif").exists()]
    todo = [t for t in tiles if a.rerun or not (runs / t / "ml_rgb.tif").exists()]
    print(f"{len(m)} tiles: {len(tiles)} with saved rasters, {len(todo)} to do")
    bad, low = [], []
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = [ex.submit(_one, (t, str(runs), str(bundles), a.prefer)) for t in todo]
        for i, f in enumerate(as_completed(futs), 1):
            t, rec, err = f.result()
            if err:
                bad.append((t, err))
                print(f"[{i}/{len(todo)}] {t}: FAILED {err}", flush=True)
                continue
            if rec["valid"] < 0.9:
                low.append((t, rec["valid"]))
            print(f"[{i}/{len(todo)}] {t}: {', '.join(rec['surveys']) or 'no imagery'}; valid {rec['valid']:.0%}", flush=True)
    print(f"\ndone; {len(bad)} failed, {len(low)} under 90 % covered" + (f" (e.g. {low[:3]})" if low else ""))
    sys.exit(1 if len(bad) > 0.05 * max(len(todo), 1) else 0)


if __name__ == "__main__":
    main()
