#!/usr/bin/env python
"""Train the channel presence model (``ml.channel_presence``) on stage B reaches, and set its threshold on the reviews.

    python scripts/train_channel_presence.py --manifest ml/manifests/wave1.csv ml/manifests/wave2.csv \\
        ml/manifests/wave3.csv --runs ~/terrain_data/runs_c3 --out ~/terrain_data/models/channel_presence

Each run needs stage B's ``network_b.gpkg`` (``stage_b.py``) and stage C's ``channel_prob.tif``; labels come from the
bundle's council lines and OSM waterways. Cross-validated by council (``held_out``, per council in ``folds``), then a
final model on all tiles. The threshold: the lowest probability at which the reviewed reaches
(``<data>/labels/manual/review_*.gpkg``, matched to the reaches of their tiles) reach ``--precision`` (default 0.7).
Writes ``model.pkl``, ``report.json`` and ``rows.parquet`` (the table, with the out-of-fold probability) to ``--out``.
"""
import argparse
import glob
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _one(job):
    tile, runs, bundles = job
    from terrain_s2.ml import channel_presence as CP
    try:
        return tile, CP.table(Path(runs) / tile, Path(bundles) / tile), None
    except Exception as e:  # noqa: BLE001 - recorded per tile
        return tile, None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--bundles")
    ap.add_argument("--reviews")
    ap.add_argument("--precision", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args()
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    from terrain_s2.acquire.elevation import default_data_dir
    from terrain_s2.ml import channel_presence as CP
    data = default_data_dir()
    bundles = a.bundles or str(data / "tiles")
    out = Path(a.out or data / "models" / "channel_presence")
    out.mkdir(parents=True, exist_ok=True)
    m = pd.concat([pd.read_csv(f) for f in a.manifest], ignore_index=True).drop_duplicates("tile_id")
    council = dict(zip(m.tile_id, m.council if "council" in m else [""] * len(m)))
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        res = list(ex.map(_one, [(t, a.runs, bundles) for t in m.tile_id]))
    bad = [(t, e) for t, _, e in res if e]
    parts = [T.assign(council=council.get(t, "")) for t, T, _ in res if T is not None and len(T)]
    if not parts:
        sys.exit(f"no reaches (failed: {bad[:3]})")
    T = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=2193)
    print(f"{len(parts)} tiles, {len(T):,} reaches, {T.recorded.mean():.1%} recorded (council or OSM); {len(bad)} failed")
    M, rep, oof = CP.fit(T, dict(precision_target=a.precision), groups=T.council.to_numpy())
    T["p_oof"] = oof
    # the reviews: reaches of the reviewed tiles, matched to the answers
    files = sorted(glob.glob(a.reviews or str(data / "labels" / "manual" / "review_*.gpkg")))
    rev = None
    if files:
        import importlib.util
        spec = importlib.util.spec_from_file_location("review_features", ROOT / "scripts" / "review_features.py")
        RF = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(RF)
        rv = RF.load_reviews(files).to_crs(2193)
        mm = []
        for t, g in rv.groupby("tile"):
            sub = T[T.tile_id == t]
            hit = CP.match_reviews(sub.reset_index(drop=True), g.reset_index(drop=True))
            if hit is not None:
                mm.append(hit)
        rev = pd.concat(mm, ignore_index=True) if mm else None
    cal = CP.calibrate(M, rev, dict(precision_target=a.precision))
    rep.update(calibration=cal, failed=[t for t, _ in bad], features=list(CP.FEATURES))
    with open(out / "model.pkl", "wb") as f:
        pickle.dump(M, f)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    T.drop(columns="geometry").to_parquet(out / "rows.parquet")
    print(json.dumps({k: rep[k] for k in ("rows", "recorded_share", "held_out") if k in rep}, default=str))
    for k, v in sorted(rep.get("folds", {}).items()):
        print(f"  {k}: {v}")
    print(f"calibration on the reviews: {cal}")
    print(f"-> {out / 'model.pkl'}")


if __name__ == "__main__":
    main()
