#!/usr/bin/env python
"""ML-0 step 2: build bundles and run the network chain for many tiles (terrain_s2.ml.batch).

    python scripts/run_tiles.py --manifest ml/manifests/wave1.csv --partition D:/eddie_data/partition/nz_tiles.gpkg \\
        --workers 6 --gpu-workers 2
    python scripts/run_tiles.py --tiles BW24_10000_0402 BW24_10000_0401 --partition ... --workers 2

Bundles go to --bundles (default TERRAIN_DATA_DIR/tiles), runs to --runs (default TERRAIN_DATA_DIR/runs);
finished tiles are skipped (done.json with the same parameters). --param key=value sets run.network options
(e.g. --param rasters=hydro --param buffer=200). A summary CSV of the tiles run is written to <runs>.
"""
import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from terrain_s2.ml.batch import BatchConfig, run_batch  # noqa: E402


def _value(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="manifest CSV from select_tiles.py (column tile_id)")
    ap.add_argument("--tiles", nargs="+", default=[])
    ap.add_argument("--council", nargs="+", help="only these councils of the manifest")
    ap.add_argument("--limit", type=int, help="at most this many tiles (in manifest order)")
    ap.add_argument("--partition", required=True)
    ap.add_argument("--bundles")
    ap.add_argument("--runs")
    ap.add_argument("--cache")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gpu-workers", type=int, default=0)
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--rerun", action="store_true", help="rerun tiles already done")
    ap.add_argument("--rerun-older", action="store_true", help="rerun tiles done by another code version")
    ap.add_argument("--no-build", action="store_true", help="never build bundles (they must exist)")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--no-features", action="store_true", help="skip step 4 (features and patches)")
    a = ap.parse_args()
    from terrain_s2.acquire.elevation import default_data_dir
    data = default_data_dir()
    tiles = list(a.tiles)
    if a.manifest:
        import pandas as pd
        m = pd.read_csv(a.manifest)
        if a.council:
            m = m[m.council.isin(a.council)]
        tiles += [t for t in m.tile_id if t not in tiles]
    if a.limit:
        tiles = tiles[:a.limit]
    if not tiles:
        ap.error("no tiles (give --manifest and/or --tiles)")
    params = dict(kv.split("=", 1) for kv in a.param)
    cfg = BatchConfig(partition=a.partition, bundles=a.bundles or str(data / "tiles"), runs=a.runs or str(data / "runs"),
                      cache=a.cache, params={k: _value(v) for k, v in params.items()}, build=not a.no_build,
                      rerun=a.rerun, rerun_older=a.rerun_older, mirror=not a.no_mirror,
                      features=not a.no_features)
    recs = run_batch(tiles, cfg, workers=a.workers, gpu_workers=a.gpu_workers)
    if recs:
        import pandas as pd
        f = Path(cfg.runs) / f"batch_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items() if k != "traceback"}
                      for r in recs]).to_csv(f, index=False)
        bad = [r for r in recs if not r.get("ok")]
        print(f"{len(recs) - len(bad)} ok, {len(bad)} failed -> {f}")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
