#!/usr/bin/env python
"""ML-1: train and evaluate the candidate scorer on a dataset (terrain_s2.ml.train).

    python scripts/train_scorer.py --dataset D:/eddie_data/datasets/wave1 --name scorer_wave1

Cross-validates (leave one council out, sheets within councils, the holdout sheets), compares with the rule-based
tiers, trains the final models outside the holdout, and writes TERRAIN_DATA_DIR/models/<name>/ (or --out):
model.pkl, report.md, report.json, fold_scores.parquet.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from terrain_s2.ml import train as TR  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out")
    ap.add_argument("--config", default=str(ROOT / "ml" / "config" / "scorer.yml"))
    a = ap.parse_args()
    from terrain_s2.acquire.elevation import default_data_dir
    out = Path(a.out) if a.out else default_data_dir() / "models" / a.name
    TR.run(a.dataset, out, TR.load_config(a.config if Path(a.config).exists() else None), name=a.name)
    print((out / "report.md").read_text())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
