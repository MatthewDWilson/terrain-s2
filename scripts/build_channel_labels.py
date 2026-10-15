#!/usr/bin/env python
"""ML-3a step 4: the channel label raster of each tile (terrain_s2.ml.channel_rasters; ml/config/channels.yml).

    python scripts/build_channel_labels.py --manifest ml/manifests/wave1.csv ml/manifests/wave2.csv --workers 12

Needs each run's saved rasters and ml_channels.gpkg (snap_channels.py); other tiles are skipped. Tiles done with this
LABEL_VERSION are skipped unless --rerun. Writes ml_channel_labels.tif (label 1 / 0 / 255 ignored, weight) and
ml_channel_labels.json beside each run, and a summary (--summary, default <runs>/channel_labels_summary.csv).
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _one(job):
    run, council, config_path = job
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from terrain_s2.ml import channel_labels as CL
    from terrain_s2.ml import channel_rasters as CR
    try:
        return CR.tile_labels(run, CL.load_config(config_path), council=council)
    except Exception as e:  # noqa: BLE001 - reported, the others carry on
        return dict(tile=Path(run).name, error=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", default=[])
    ap.add_argument("--tiles", nargs="+", default=[])
    ap.add_argument("--runs")
    ap.add_argument("--config", default=str(ROOT / "ml" / "config" / "channels.yml"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--summary")
    a = ap.parse_args()
    import pandas as pd
    from terrain_s2.acquire.elevation import default_data_dir
    from terrain_s2.ml.channel_rasters import LABEL_VERSION
    runs = Path(a.runs or default_data_dir() / "runs")
    tiles = {t: None for t in a.tiles}
    for m in a.manifest:
        for r in pd.read_csv(m).itertuples():
            tiles.setdefault(r.tile_id, getattr(r, "council", None))
    if not tiles:
        ap.error("no tiles (--manifest or --tiles)")
    jobs, skipped = [], dict(not_snapped=0, done=0)
    for t, council in tiles.items():
        r = runs / t
        if not ((r / "ml_channels.json").exists() and (r / "features.tif").exists() and (r / "hydro.tif").exists()):
            skipped["not_snapped"] += 1
            continue
        j = r / "ml_channel_labels.json"
        if not a.rerun and j.exists() and (r / "ml_channel_labels.tif").exists():
            try:
                if json.loads(j.read_text()).get("label_version") == LABEL_VERSION:
                    skipped["done"] += 1
                    continue
            except ValueError:
                pass
        jobs.append((str(r), council, a.config))
    print(f"{len(tiles)} tiles: {len(jobs)} to label; skipped {skipped['not_snapped']} not snapped, {skipped['done']} done")
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for i, f in enumerate(as_completed([ex.submit(_one, j) for j in jobs]), 1):
            s = f.result()
            rows.append(s)
            if "error" in s:
                print(f"[{i}/{len(jobs)}] {s['tile']}: FAILED {s['error']}")
                continue
            n = s["positive"] + s["negative"] + s["ignored"]
            print(f"[{i}/{len(jobs)}] {s['tile']} ({s['council']}): positive {s['positive'] / n:.2%}, ignored "
                  f"{s['ignored'] / n:.1%}, negative {s['negative'] / n:.1%}; weight positive / negative "
                  f"{s['positive_weight_sum'] / max(s['negative_weight_sum'], 1e-9):.2f}")
    if rows:
        df = pd.DataFrame(rows)
        out = Path(a.summary) if a.summary else runs / "channel_labels_summary.csv"
        df.to_csv(out, index=False)
        ok = df[df.get("error").isna()] if "error" in df else df
        if len(ok):
            print("\nper council:")
            for c, g in ok.groupby("council"):
                n = (g.positive + g.negative + g.ignored).sum()
                print(f"  {c}: {len(g)} tiles; positive {g.positive.sum() / n:.2%}, ignored {g.ignored.sum() / n:.1%}, "
                      f"negative {g.negative.sum() / n:.1%}; weight positive / negative "
                      f"{g.positive_weight_sum.sum() / max(g.negative_weight_sum.sum(), 1e-9):.2f}")
        print(f"summary: {out}")


if __name__ == "__main__":
    main()
