#!/usr/bin/env python
"""ML-0 step 1: select training tiles where council data exist, with their folds (terrain_s2.ml.select).

    python scripts/select_tiles.py --config ml/config/wave1.yml --partition D:/eddie_data/partition/nz_tiles.gpkg

Writes ``ml/manifests/<wave>.csv`` (or --out) and prints a summary per council and split.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from terrain_s2.acquire import registry  # noqa: E402
from terrain_s2.ml import select as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--partition", required=True, help="partition GeoPackage (layer tiles)")
    ap.add_argument("--index", default=str(ROOT / "sites" / "sources_coverage.geojson"))
    ap.add_argument("--out", help="manifest CSV (default ml/manifests/<wave>.csv)")
    a = ap.parse_args()
    import geopandas as gpd
    cfg = S.load_config(a.config)
    import pandas as pd
    for m in cfg.get("exclude_manifests") or []:              # tiles of earlier waves keep their labels as they are
        cfg["exclude_tiles"] = list(cfg["exclude_tiles"]) + list(pd.read_csv(ROOT / m).tile_id)
    if cfg["exclude_tiles"]:
        print(f"excluding {len(set(cfg['exclude_tiles']))} tiles of {', '.join(cfg.get('exclude_manifests') or [])}")
    tiles = gpd.read_file(a.partition, layer="tiles", ignore_geometry=True)
    m = S.select(tiles, registry.load_index(a.index), cfg)
    out = Path(a.out) if a.out else ROOT / "ml" / "manifests" / f"{cfg['name']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out, index=False)
    print(S.summary(m))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
