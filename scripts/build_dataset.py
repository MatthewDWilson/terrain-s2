#!/usr/bin/env python
"""ML-0 step 5: assemble a dataset from the tiles of one or more manifests (terrain_s2.ml.dataset).

    python scripts/build_dataset.py --manifest ml/manifests/wave1.csv --name wave1

Writes TERRAIN_DATA_DIR/datasets/<name>/ (or --out): candidates.parquet, misses.parquet, tiles.csv, card.json,
card.md. Labels are rebuilt for tiles whose run is newer than their labels (--relabel always|never to change).
Tiles without a finished run with features are listed as skipped; run it again when the batch is done.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from terrain_s2.ml import dataset as DS  # noqa: E402
from terrain_s2.ml import labels as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out")
    ap.add_argument("--runs")
    ap.add_argument("--bundles")
    ap.add_argument("--labels-config", default=str(ROOT / "ml" / "config" / "labels.yml"))
    ap.add_argument("--relabel", choices=["stale", "always", "never"], default="stale")
    ap.add_argument("--allow-missing", action="store_true", help="build even if more than 2 %% of the tiles are not ready")
    ap.add_argument("--channel-probs", help="runs folder holding each tile's ml_channel_prob.tif (the out-of-fold "
                                            "U-Net prediction), for the channel-model features (ML-3c)")
    a = ap.parse_args()
    from terrain_s2.acquire.elevation import default_data_dir
    data = default_data_dir()
    cfg = L.load_config(a.labels_config)
    cfg["manual_root"] = cfg.get("manual_root") or str(data / "labels" / "manual")
    manual = L.load_manual(cfg) if cfg.get("manual") else None
    out = Path(a.out) if a.out else data / "datasets" / a.name
    DS.build(a.manifest, out, a.runs or data / "runs", a.bundles or data / "tiles", cfg, manual,
                    relabel=a.relabel, name=a.name, channel_probs=a.channel_probs,
                    max_skipped=None if a.allow_missing else 0.02)
    print((out / "card.md").read_text())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
