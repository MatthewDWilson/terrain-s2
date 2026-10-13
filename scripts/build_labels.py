#!/usr/bin/env python
"""ML-0 step 3: labels for every crossing candidate of each run (terrain_s2.ml.labels).

    python scripts/build_labels.py --manifest ml/manifests/wave1.csv          # tiles: TERRAIN_DATA_DIR/runs, /tiles
    python scripts/build_labels.py --run D:/eddie_data/runs/canterbury1 --bundle D:/eddie_data/sites/canterbury1 \\
        --aoi examples/aoi_canterbury1.geojson

Writes ml_labels.gpkg and ml_labels.json beside each run, and a summary CSV (--summary, default
<runs>/labels_summary.csv). Manual review files are read from manual_root in the config (default
TERRAIN_DATA_DIR/labels/manual).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from terrain_s2.ml import labels as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "ml" / "config" / "labels.yml"))
    ap.add_argument("--manifest")
    ap.add_argument("--runs")
    ap.add_argument("--bundles")
    ap.add_argument("--run", action="append", default=[], help="a run folder (with --bundle, repeatable in pairs)")
    ap.add_argument("--bundle", action="append", default=[])
    ap.add_argument("--aoi", action="append", default=[], help="core AOI per --run (default: the bundle's aoi.geojson)")
    ap.add_argument("--summary")
    a = ap.parse_args()
    import pandas as pd
    from terrain_s2.acquire.elevation import default_data_dir
    data = default_data_dir()
    cfg = L.load_config(a.config)
    cfg["manual_root"] = cfg.get("manual_root") or str(data / "labels" / "manual")
    manual = L.load_manual(cfg) if cfg.get("manual") else None
    jobs = []
    if a.manifest:
        runs, bundles = Path(a.runs or data / "runs"), Path(a.bundles or data / "tiles")
        for t in pd.read_csv(a.manifest).tile_id:
            if (runs / t / "network.gpkg").exists():
                jobs.append((runs / t, bundles / t, None))
            else:
                print(f"{t}: no run yet, skipped")
    if len(a.run) != len(a.bundle):
        ap.error("give --run and --bundle in pairs")
    for i, (r, b) in enumerate(zip(a.run, a.bundle)):
        jobs.append((Path(r), Path(b), a.aoi[i] if i < len(a.aoi) else None))
    if not jobs:
        ap.error("nothing to label (--manifest, or --run with --bundle)")
    reps = []
    for run, bundle, aoi in jobs:
        _, _, _, rep = L.label_run(run, bundle, manual, cfg, aoi=aoi)
        reps.append(rep)
        print(f"{run.name}: {rep['candidates_in_core']} candidates in the core; +{rep['positives']} -{rep['negatives']} "
              f"?{rep['unsure']}, near {rep['near_positive']}, unlabelled {rep['unlabelled']}; labels in core "
              f"{rep['labels_in_core']}; misses {rep['misses']}; conflicts {rep['conflicts']}")
    out = Path(a.summary) if a.summary else (Path(a.runs or data / "runs") / "labels_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: (str(v) if isinstance(v, dict) else v) for k, v in r.items()} for r in reps]).to_csv(out, index=False)
    tot = pd.DataFrame(reps)[["candidates_in_core", "positives", "negatives", "unsure", "unlabelled", "conflicts"]].sum()
    print("total: " + ", ".join(f"{k} {int(v)}" for k, v in tot.items()) + f" -> {out}")


if __name__ == "__main__":
    main()
