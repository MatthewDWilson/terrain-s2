#!/usr/bin/env python
"""ML-3b: the channel U-Net over whole tiles (ml_channel_prob.tif per run) and a review set for Matt.

    python scripts/predict_channels.py --dataset ml3_waves123 --name unet_v1 --fold blocks:1/5 \
        --pick waimakariri environment_southland hauraki --review
    python scripts/predict_channels.py --dataset ml3_waves123 --name unet_v1 --fold holdout --tiles BW24_10000_0402
    python scripts/predict_channels.py --dataset ml3_waves123 --name unet_v1 --fold oof --workers 4

The model is <models>/<name>/<fold> (use a fold whose test tiles you predict: the model has not seen them).
--fold oof predicts every tile of the dataset with the model of the fold that held it out (blocks folds; the holdout
model for the holdout tiles): out-of-fold probabilities, which labels v3 uses to clean council lines; tiles already
done by the same model are skipped unless --rerun; --workers runs that many processes on the GPU. It also writes
<models>/<name>/oof_map.json (tile -> fold), with which a batch run given channel_model=<models>/<name> uses each
tile's held-out model (ML-3c).
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
    ap.add_argument("--workers", type=int, default=1, help="--fold oof: processes on the GPU")
    ap.add_argument("--rerun", action="store_true")
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
    if a.fold == "oof":
        return oof(a, ds, runs, Path(a.models or data / "models") / a.name, cfg, idx)
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
        held = None
        if t in set(idx.tile_id):
            masks = {n: te for n, _, te in U.fold_masks(idx, cfg)}
            held = bool(t in set(idx[masks[a.fold]].tile_id)) if a.fold in masks else None
        P.write_prob(runs / t, prob, T, crs, model_dir, thr, fold=a.fold, held_out=held)
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


def _oof_worker(job):
    """One process: its share of the tiles, each with the model of the fold that held it out."""
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import torch
    from terrain_s2.ml import channel_predict as P
    from terrain_s2.ml import unet as U
    from terrain_s2.ml.chips import Mosaic
    tiles, all_tiles, runs, models_dir, device = job
    device = torch.device(device)
    mosaic = Mosaic({t: Path(runs) / t for t in all_tiles})
    loaded, done = {}, []
    for t, fold in tiles:
        md = Path(models_dir) / U.fold_tag(fold)
        if fold not in loaded:
            loaded.clear()
            loaded[fold] = U.load_model(md, device)
        model, meta = loaded[fold]
        try:
            prob, T, crs = P.predict_tile(Path(runs) / t, mosaic, model, meta, device)
            P.write_prob(Path(runs) / t, prob, T, crs, md, meta["threshold"], fold=fold, held_out=True)
            done.append((t, fold, None))
        except Exception as e:  # noqa: BLE001 - reported
            done.append((t, fold, f"{type(e).__name__}: {e}"))
        print(f"  {t} ({fold}): " + ("done" if done[-1][2] is None else f"FAILED {done[-1][2]}"), flush=True)
    return done


def oof(a, ds, runs, models_dir, cfg, idx):
    """Every tile of the dataset predicted by the model of the fold that held it out."""
    import json
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    import torch
    from terrain_s2.ml import unet as U
    owner = {}
    for name, _, te in U.fold_masks(idx, cfg):
        if name.startswith("blocks") or name == "holdout":
            for t in idx[te].tile_id.unique():
                owner.setdefault(t, name)
    missing = sorted({f for f in owner.values() if not (models_dir / U.fold_tag(f) / "model.pt").exists()})
    if missing:
        print(f"no saved model for {', '.join(missing)}: their tiles are skipped")
    omap = {t: U.fold_tag(f) for t, f in owner.items() if f not in missing}   # for runs with this model set (ML-3c)
    (models_dir / "oof_map.json").write_text(json.dumps(omap, indent=1, sort_keys=True))
    print(f"oof_map.json: {len(omap)} tiles -> their held-out fold ({models_dir / 'oof_map.json'})")
    jobs = []
    for t, f in sorted(owner.items(), key=lambda kv: (kv[1], kv[0])):
        if f in missing:
            continue
        j = runs / t / "ml_channel_prob.json"
        if not a.rerun and j.exists():
            try:
                rec = json.loads(j.read_text())
                if rec.get("held_out") and rec.get("model") == str(models_dir / U.fold_tag(f)):
                    continue
            except ValueError:
                pass
        jobs.append((t, f))
    print(f"{len(owner)} tiles with a held-out fold; {len(jobs)} to predict with {a.workers} process(es)")
    if not jobs:
        return
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = max(1, a.workers)
    shares = [jobs[k::n] for k in range(n)]
    all_tiles = sorted(idx.tile_id.unique())
    with ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn")) as ex:
        results = list(ex.map(_oof_worker, [(sh, all_tiles, str(runs), str(models_dir), device) for sh in shares if sh]))
    failed = [r for part in results for r in part if r[2]]
    print(f"predicted {len(jobs) - len(failed)}, failed {len(failed)}")


if __name__ == "__main__":
    main()
