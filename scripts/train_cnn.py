#!/usr/bin/env python
"""ML-1b: train and evaluate the patch CNN on a dataset (terrain_s2.ml.cnn); needs PyTorch.

    python scripts/train_cnn.py --dataset D:/eddie_data/datasets/wave1 --name cnn_wave1 \\
        --scorer D:/eddie_data/models/scorer_wave1_physical

--scorer adds the gradient-boosting model's fold scores, for the comparison and the ensemble. Writes
TERRAIN_DATA_DIR/models/<name>/ (or --out): cnn.pt, report.md, report.json, fold_scores.parquet.
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
    ap.add_argument("--out")
    ap.add_argument("--scorer", help="a scorer run folder (fold_scores.parquet) for the comparison and ensemble")
    ap.add_argument("--config", default=str(ROOT / "ml" / "config" / "cnn.yml"))
    ap.add_argument("--epochs", type=int)
    a = ap.parse_args()
    try:
        import torch
    except ImportError:
        sys.exit("PyTorch is not installed: python -m pip install torch --index-url https://download.pytorch.org/whl/cu128")
    from terrain_s2.acquire.elevation import default_data_dir
    from terrain_s2.ml import cnn
    cfg = cnn.load_config(a.config if Path(a.config).exists() else None)
    if a.scorer:
        cfg["scorer"] = a.scorer
    if a.epochs:
        cfg["epochs"] = a.epochs
    print(f"PyTorch {torch.__version__}; device {cnn._device(cfg)}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    out = Path(a.out) if a.out else default_data_dir() / "models" / a.name
    cnn.run(a.dataset, out, cfg, name=a.name)
    print((out / "report.md").read_text())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
