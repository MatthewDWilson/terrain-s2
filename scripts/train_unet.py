#!/usr/bin/env python
"""ML-3b: train and evaluate the channel U-Net on a chip dataset (terrain_s2.ml.unet; ml/config/unet.yml).

    python scripts/train_unet.py --dataset ml3_waves123 --name unet_v1                       # first block fold + holdout
    python scripts/train_unet.py --dataset ml3_waves123 --name unet_lc --folds blocks:1/5 --subsets 0.25 0.5 1
    python scripts/train_unet.py --dataset ml3_waves123 --name unet_v1 --folds all          # every fold, then final
    python scripts/train_unet.py --dataset ml3_waves123 --name unet_v1 --list-folds
    python scripts/train_unet.py --dataset ml3_waves123 --name unet_v1 --evaluate       # saved models, measures again

Folds as train.folds names them (blocks:k/5, council:<name>, holdout) and "final" (everything outside the holdout,
no test). Writes <models>/<name>/<fold>/ (model.pt, model.json, history.json, metrics.json) and report.md. The
first run unpacks each tile's chips into <dataset>/cache (memory-mapped .npy, ~230 MB a tile).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default=str(ROOT / "ml" / "config" / "unet.yml"))
    ap.add_argument("--folds", nargs="+")
    ap.add_argument("--subsets", nargs="+", type=float)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--device")
    ap.add_argument("--datasets")
    ap.add_argument("--models")
    ap.add_argument("--list-folds", action="store_true")
    ap.add_argument("--evaluate", action="store_true", help="evaluate the saved models of these runs again (no training)")
    a = ap.parse_args()
    from terrain_s2.acquire.elevation import default_data_dir
    from terrain_s2.ml import unet as U
    data = default_data_dir()
    ds = Path(a.datasets or data / "datasets") / a.dataset
    cfg = U.load_config(a.config)
    if a.epochs:
        cfg["epochs"] = a.epochs
    if a.workers is not None:
        cfg["workers"] = a.workers
    if a.list_folds or (a.folds and a.folds == ["all"]):
        import pandas as pd
        names = [n for n, _, _ in U.fold_masks(pd.read_parquet(ds / "chips.parquet"), cfg)]
        if a.list_folds:
            print("\n".join(names))
            return
        a.folds = names
    out = Path(a.models or data / "models") / a.name
    print(f"{ds} -> {out}")
    if a.evaluate and not a.folds:                       # every run saved in the folder
        import json
        runs = [json.loads(f.read_text()) for f in sorted(out.glob("*/metrics.json"))]
        a.folds = sorted({r["fold"] for r in runs})
        a.subsets = a.subsets or sorted({r["subset"] for r in runs})
    U.run(ds, out, cfg, folds=a.folds, subsets=a.subsets, device=a.device, evaluate_only=a.evaluate)
    print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
