#!/usr/bin/env python
"""ML-3a step 3: snap council channel lines to the DEM bed, with widths and weights, for each tile of a manifest
(terrain_s2.ml.channel_labels.tile_channels; configuration ml/config/channels.yml).

    python scripts/snap_channels.py --manifest ml/manifests/wave1.csv --workers 8
    python scripts/snap_channels.py --tiles BW24_10000_0402 --rerun

Needs the runs' saved rasters (run_tiles.py --param rasters=all; done.json says so): other tiles are skipped, so it
can run beside the batch and again later for the rest; tiles done with this SNAP_VERSION are skipped unless --rerun. Writes ml_channels.gpkg and ml_channels.json beside each run
and a summary (--summary, default <runs>/channels_summary.csv): per tile and council, km of council line, the length
share snapped, the bed check before and after (share of check points whose DEM low lies within 2 m of the line).
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _one(job):
    run, bundle, sources_path, config_path = job
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from terrain_s2.acquire.site import load_sources
    from terrain_s2.ml import channel_labels as CL
    try:
        _, summary = CL.tile_channels(run, bundle, load_sources(sources_path), CL.load_config(config_path))
        return summary
    except Exception as e:  # noqa: BLE001 - reported, the others carry on
        return dict(tile=Path(run).name, error=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", default=[])
    ap.add_argument("--tiles", nargs="+", default=[])
    ap.add_argument("--runs")
    ap.add_argument("--bundles")
    ap.add_argument("--config", default=str(ROOT / "ml" / "config" / "channels.yml"))
    ap.add_argument("--sources", default=str(ROOT / "sites" / "sources.yml"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--summary")
    a = ap.parse_args()
    import pandas as pd
    from terrain_s2.acquire.elevation import default_data_dir
    from terrain_s2.ml.channel_labels import SNAP_VERSION
    data = default_data_dir()
    runs, bundles = Path(a.runs or data / "runs"), Path(a.bundles or data / "tiles")
    tiles = list(a.tiles)
    for m in a.manifest:
        tiles += [t for t in pd.read_csv(m).tile_id if t not in tiles]
    if not tiles:
        ap.error("no tiles (--manifest or --tiles)")
    jobs, skipped = [], dict(no_rasters=0, done=0)
    for t in tiles:
        r = runs / t
        try:                                  # finished with saved rasters (not a tile the batch is still writing)
            ok = json.loads((r / "done.json").read_text()).get("params", {}).get("rasters") == "all"
        except (OSError, ValueError):
            ok = False
        if not (ok and (r / "features.tif").exists() and (r / "hydro.tif").exists()):
            skipped["no_rasters"] += 1
            continue
        j = r / "ml_channels.json"
        if not a.rerun and j.exists():
            try:
                if json.loads(j.read_text()).get("snap_version") == SNAP_VERSION:
                    skipped["done"] += 1
                    continue
            except ValueError:
                pass
        jobs.append((str(r), str(bundles / t), a.sources, a.config))
    print(f"{len(tiles)} tiles: {len(jobs)} to snap; skipped {skipped['no_rasters']} without finished rasters, {skipped['done']} done")
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            s = f.result()
            if "error" in s:
                print(f"[{i}/{len(jobs)}] {s['tile']}: FAILED {s['error']}")
                rows.append(dict(tile=s["tile"], error=s["error"]))
                continue
            parts = []
            for c, v in s["councils"].items():
                rows.append(dict(tile=s["tile"], council=c, **{k: v[k] for k in v if k != "width_sources"},
                                 **{f"width_{k}": x for k, x in v["width_sources"].items()}))
                parts.append(f"{c} {v['km']} km, snapped {v['snapped_share']:.0%}, check {v['check_council_snapped']} -> "
                             f"{v['check_snapped']}")
            print(f"[{i}/{len(jobs)}] {s['tile']}: " + ("; ".join(parts) or "no council lines"))
    if rows:
        df = pd.DataFrame(rows)
        out = Path(a.summary) if a.summary else runs / "channels_summary.csv"
        df.to_csv(out, index=False)
        ok = df[df.get("km", pd.Series(dtype=float)).notna()] if "km" in df else df.iloc[0:0]
        ok = ok.fillna({c: 0.0 for c in ok.columns if c.startswith("width_")})   # a source absent from a tile is 0 %
        if len(ok):
            print("\nper council (length-weighted):")
            for c, g in ok.groupby("council"):
                w = g.km.clip(lower=1e-6)

                def avg(col):
                    v = g[col].astype(float)
                    k = v.notna()
                    return float((v[k] * w[k]).sum() / w[k].sum()) if k.any() else float("nan")
                print(f"  {c}: {len(g)} tiles, {g.km.sum():.0f} km, snapped {avg('snapped_share'):.0%}, check council "
                      f"{avg('check_council_snapped'):.2f} -> snapped {avg('check_snapped'):.2f}; widths: "
                      + ", ".join(f"{k[6:]} {avg(k):.0%}" for k in g.columns if k.startswith("width_")))
        print(f"summary: {out}")


if __name__ == "__main__":
    main()
