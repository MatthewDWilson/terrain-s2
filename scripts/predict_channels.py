#!/usr/bin/env python
"""ML-3b: the channel U-Net over whole tiles (ml_channel_prob.tif per run) and a review set for Matt.

    python scripts/predict_channels.py --dataset ml3_waves123 --name unet_v1 --fold blocks:1/5 \
        --pick waimakariri environment_southland hauraki --review
    python scripts/predict_channels.py --dataset ml3_waves123 --name unet_v1 --fold holdout --tiles BW24_10000_0402

The model is <models>/<name>/<fold> (use a fold whose test tiles you predict: the model has not seen them).
--pick chooses, per council, the fold's test tile with most positive chips. --review writes
<models>/<name>/<fold>/review.gpkg (layer reaches: model_only and council_missed, each with exists = y / n / unsure
to fill, and notes); the model's threshold unless --threshold.
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
    ap.add_argument("--fold", required=True)
    ap.add_argument("--tiles", nargs="+", default=[])
    ap.add_argument("--pick", nargs="+", default=[], metavar="COUNCIL")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--max-per-kind", type=int, default=40)
    ap.add_argument("--config", default=str(ROOT / "ml" / "config" / "unet.yml"))
    ap.add_argument("--device")
    ap.add_argument("--datasets")
    ap.add_argument("--models")
    ap.add_argument("--runs")
    a = ap.parse_args()
    import geopandas as gpd
    import pandas as pd
    import torch
    from terrain_s2.acquire.elevation import default_data_dir
    from terrain_s2.ml import channel_predict as P
    from terrain_s2.ml import unet as U
    from terrain_s2.ml.chips import Mosaic
    data = default_data_dir()
    ds = Path(a.datasets or data / "datasets") / a.dataset
    runs = Path(a.runs or data / "runs")
    model_dir = Path(a.models or data / "models") / a.name / U.fold_tag(a.fold)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = U.load_config(a.config)
    idx = pd.read_parquet(ds / "chips.parquet")
    tiles = list(a.tiles)
    if a.pick:
        masks = {n: te for n, _, te in U.fold_masks(idx, cfg)}
        if a.fold not in masks:
            ap.error(f"no fold {a.fold}")
        test = idx[masks[a.fold]]
        for c in a.pick:
            g = test[(test.council == c) & (test.kind == "positive")]
            if not len(g):
                print(f"{c}: no test tiles in {a.fold}")
                continue
            t = g.tile_id.value_counts().index[0]
            print(f"{c}: {t} ({int((g.tile_id == t).sum())} positive chips)")
            tiles.append(t)
    if not tiles:
        ap.error("no tiles (--tiles or --pick)")
    model, meta = U.load_model(model_dir, device)
    thr = a.threshold if a.threshold is not None else meta["threshold"]
    mosaic = Mosaic({t: runs / t for t in sorted(set(idx.tile_id) | set(tiles))})
    parts = []
    for t in tiles:
        prob, T, crs = P.predict_tile(runs / t, mosaic, model, meta, device)
        P.write_prob(runs / t, prob, T, crs, model_dir, thr)
        print(f"{t}: ml_channel_prob.tif written; {float((prob >= thr).mean()):.2%} of cells >= {thr}")
        if a.review:
            r = P.review_reaches(runs / t, prob, T, crs, thr, max_per_kind=a.max_per_kind)
            print(f"  review: {r.kind.value_counts().to_dict() if len(r) else {}}")
            parts.append(r)
    if a.review and parts:
        rv = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=parts[0].crs)
        rv.insert(0, "review_id", range(1, len(rv) + 1))
        out = model_dir / "review.gpkg"
        rv.to_file(out, layer="reaches", driver="GPKG")
        print(f"review set: {out} ({len(rv)} reaches; fill 'exists' with y / n / unsure)")


if __name__ == "__main__":
    main()
