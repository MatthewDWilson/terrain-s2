#!/usr/bin/env python
"""Fetch one source again for built bundles, one tile at a time, and mark those tiles for a rerun.

    python scripts/refresh_source.py --source osm --tiles-file ~/terrain/osm_redo.txt
    python scripts/refresh_source.py --source osm --manifest ml/manifests/wave3.csv     # tiles whose site.json has its error

For a source that failed in a batch (Overpass refusing many parallel workers, wave 3). Each tile in turn: the source
is fetched again (``acquire.build.refresh_source``: its rows replaced in the bundle's GeoPackages, its error removed
from site.json), then the run's done.json is removed, so the next batch reruns the network and features with the
bundle as it now is (the bundle counts as built, so it is not rebuilt). A tile that fails again is reported and left
as it was; --pause waits between tiles, and --tries retries a tile after a longer wait.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--tiles-file")
    ap.add_argument("--manifest", nargs="+", default=[])
    ap.add_argument("--bundles")
    ap.add_argument("--runs")
    ap.add_argument("--sources", default=str(ROOT / "sites" / "sources.yml"))
    ap.add_argument("--pause", type=float, default=10.0, help="seconds between tiles")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--keep-done", action="store_true", help="do not remove done.json")
    a = ap.parse_args()
    import pandas as pd
    from terrain_s2.acquire.build import refresh_source
    from terrain_s2.acquire.elevation import default_data_dir
    from terrain_s2.acquire.site import load_sources
    data = default_data_dir()
    bundles, runs = Path(a.bundles or data / "tiles"), Path(a.runs or data / "runs")
    sources = load_sources(a.sources)
    tiles = []
    if a.tiles_file:
        tiles += [t.strip() for t in Path(a.tiles_file).expanduser().read_text().split() if t.strip()]
    step = f"source:{a.source}"
    for m in a.manifest:
        for t in pd.read_csv(m).tile_id:
            try:
                errs = json.loads((bundles / t / "site.json").read_text()).get("errors", [])
            except (OSError, ValueError):
                continue
            if any(e.get("step") == step for e in errs) and t not in tiles:
                tiles.append(t)
    print(f"{len(tiles)} tiles to refresh {a.source} for")
    failed = []
    for i, t in enumerate(tiles, 1):
        ok = False
        for k in range(a.tries):
            try:
                refresh_source(bundles / t, a.source, sources, data, log=lambda m: print(f"[{i}/{len(tiles)}] {m}"))
                ok = True
                break
            except Exception as e:  # noqa: BLE001 - reported; next try, then next tile
                print(f"[{i}/{len(tiles)}] {t}: try {k + 1} failed: {str(e)[:300]}")
                time.sleep(a.pause * 6 * (k + 1))
        if ok and not a.keep_done:
            (runs / t / "done.json").unlink(missing_ok=True)
        if not ok:
            failed.append(t)
        time.sleep(a.pause)
    print(f"refreshed {len(tiles) - len(failed)}, failed {len(failed)}" + (f": {' '.join(failed)}" if failed else ""))
    if len(tiles) > len(failed) and not a.keep_done:
        print("now rerun those tiles: terrain_hpc.sh start <manifest> with the batch's parameters")


if __name__ == "__main__":
    main()
