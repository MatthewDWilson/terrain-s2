#!/usr/bin/env python
"""Compare channel U-Nets on whole test tiles against the same labels (the council labels by default).

    python scripts/compare_channel_models.py --dataset ml3_waves123_v5 --fold blocks:1/5 \\
        --models ~/terrain_data/models/unet_v2/blocks_1of5 ~/terrain_data/models/unet_v4/blocks_1of5 --drop-osm

Why: each model's report is measured on its own chips' labels, and unet_v4's (labels v5) add OSM river banks and
OSM waterways, which are easy and which its OSM input band points at, so its report cannot be set beside unet_v2's.
Here every model predicts the same whole tiles (the fold's test tiles, ``--per-council`` of them per council, from
the chip dataset's index and ``--config``'s folds), and is scored on one label raster (``--labels``, default
``ml_channel_labels.tif``: council lines, cleaning and reviews, no OSM) as the U-Net's evaluation scores chips:
centreline recall and precision within 3 m, recall off the physics map, at each model's own threshold and at
0.5-0.8. ``--drop-osm`` scores a model with the OSM band once more with that band set to "no OSM within 500 m"
everywhere: how it does where OSM maps nothing. Writes ``--out`` (CSV, one row per model, variant, council and
threshold) and prints the pooled table.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
THRESHOLDS = (0.5, 0.6, 0.7, 0.8)


class DropOSM:
    """A Mosaic whose OSM band reads as "no OSM within the cap"."""

    def __init__(self, mosaic):
        self.m = mosaic
        self.bands = mosaic.bands

    def read_inputs(self, *a, **k):
        import numpy as np

        from terrain_s2.ml.chips import OSM_BAND, OSM_CAP_M
        x = self.m.read_inputs(*a, **k)
        i = list(self.bands).index(OSM_BAND)
        x[i] = np.where(np.isfinite(x[i]), OSM_CAP_M, np.nan)
        return x


def score_tile(run, prob, label_file, thresholds, tol_m=3.0):
    """{threshold: Scores} for one tile's probability against its label raster (and physics map)."""
    import numpy as np
    import rasterio

    from terrain_s2.ml import unet as U
    with rasterio.open(Path(run) / label_file) as s:
        lab = s.read(1)
    with rasterio.open(Path(run) / "hydro.tif") as s:
        names = list(s.descriptions)
        phys = s.read(names.index("channel_mask") + 1) > 0
        la = s.read(names.index("log10_upstream_area_m2") + 1) if "log10_upstream_area_m2" in names else None
    lab = np.where(np.isfinite(lab), lab, 255).astype(np.uint8)
    ctx = U.chip_context(lab, phys.astype(np.uint8), tol_m)
    p = np.nan_to_num(prob, nan=0.0)
    out = {}
    for t in thresholds:
        sc = U.Scores(hist=False)
        sc.add(p, ctx, la, t, tol_m)
        out[t] = sc
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="a chip dataset (its index gives the folds' tiles)")
    ap.add_argument("--fold", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="model folders (model.pt, model.json)")
    ap.add_argument("--config", default=str(ROOT / "ml" / "config" / "unet_v4.yml"), help="for the folds")
    ap.add_argument("--labels", default="ml_channel_labels.tif")
    ap.add_argument("--per-council", type=int, default=4)
    ap.add_argument("--drop-osm", action="store_true")
    ap.add_argument("--runs")
    ap.add_argument("--datasets")
    ap.add_argument("--device")
    ap.add_argument("--out")
    a = ap.parse_args()
    import numpy as np
    import pandas as pd
    import torch

    from terrain_s2.acquire.elevation import default_data_dir
    from terrain_s2.ml import channel_predict as P
    from terrain_s2.ml import unet as U
    from terrain_s2.ml.chips import OSM_BAND, Mosaic
    data = default_data_dir()
    runs = Path(a.runs or data / "runs")
    ds = Path(a.datasets or data / "datasets") / a.dataset
    idx = pd.read_parquet(ds / "chips.parquet")
    cfg = U.load_config(a.config)
    masks = {n: te for n, _, te in U.fold_masks(idx, cfg)}
    if a.fold not in masks:
        ap.error(f"no fold {a.fold}; known: {sorted(masks)}")
    test = idx[masks[a.fold]]
    rng = np.random.default_rng(0)
    tiles = []
    for c, g in test.groupby(test.council.fillna("unknown")):
        tt = sorted(g.tile_id.unique())
        tt = [t for t in tt if (runs / t / a.labels).exists()]
        tiles += [(t, c) for t in rng.permutation(tt)[:a.per_council]]
    print(f"{a.fold}: {len(tiles)} test tiles ({a.per_council} per council at most), labels {a.labels}")
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    all_tiles = sorted(idx.tile_id.unique())
    acc = {}
    for md in a.models:
        model, meta = U.load_model(md, device)
        name = Path(md).parent.name + "/" + Path(md).name
        mosaic = Mosaic({t: runs / t for t in all_tiles}, bands=meta["bands"])
        variants = [("as trained", mosaic)]
        if a.drop_osm and OSM_BAND in meta["bands"]:
            variants.append(("OSM band dropped", DropOSM(mosaic)))
        ths = tuple(sorted(set(THRESHOLDS) | {float(meta.get("threshold", 0.5))}))
        for vname, mo in variants:
            for t, council in tiles:
                prob, _, _ = P.predict_tile(runs / t, mo, model, meta, device)
                for th, sc in score_tile(runs / t, prob, a.labels, ths, cfg.get("tolerance_m", 3.0)).items():
                    for grp in ("all", council):
                        key = (name, vname, grp, th)
                        if key not in acc:
                            acc[key] = U.Scores(hist=False)
                        for k, v in sc.c.items():
                            acc[key].c[k] += v
                        for n_, (h, tot) in sc.area.items():
                            acc[key].area[n_][0] += h
                            acc[key].area[n_][1] += tot
                print(f"  {name} ({vname}): {t} done", flush=True)
        mosaic.close()
    rows = []
    for (name, vname, grp, th), sc in acc.items():
        r = sc.result()
        rows.append(dict(model=name, variant=vname, council=grp, threshold=th, recall=r["recall_3m"],
                         precision=r["precision_3m"], f1=r["f1_3m"], recall_off_physics=r["recall_off_physics"],
                         off_physics_cells=r["off_physics_cells"], new_cells=r["new_cells"],
                         new_on_council=r["new_on_council"]))
    d = pd.DataFrame(rows).sort_values(["council", "model", "variant", "threshold"])
    out = Path(a.out) if a.out else Path(a.models[-1]).parent / f"compare_{a.fold.replace(':', '_').replace('/', 'of')}.csv"
    d.to_csv(out, index=False)
    print("\nall test tiles:\n" + d[d.council == "all"].drop(columns="council").to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
