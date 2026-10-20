#!/usr/bin/env python
"""Calibrate the ML-3c filter on the model's channels against Matt's review answers.

    python scripts/review_features.py                      # every <data>/labels/manual/review_*.gpkg

For each reviewed reach (exists y / n; unsure left out), on its tile's current probability raster
(``ml_channel_prob.tif``, the out-of-fold prediction) at the raster's threshold: the model's channel component it
lies on and that component's quality (``channel_predict.component_quality``: contrast between the probability on
the line and 2-5 m outside it, width, mean probability, urban building cover). Then, for the model-only reaches
(the model's new detections), how well each measure separates y from n (AUC), the best single threshold, and the
best combined rules: the share of real drains kept and of false detections dropped. The rules go to
``run.network`` as channel_min_contrast, channel_max_width_m, channel_max_urban. Writes --out (CSV, one row per
reviewed reach).
"""
import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MEASURES = (("contrast", "high"), ("mean_prob", "high"), ("width_m", "low"), ("urban", "low"))


def auc(pos, neg, higher=True):
    """Probability that a real drain scores better than a false detection (ties count half)."""
    import numpy as np
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    if not higher:
        pos, neg = -pos, -neg
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def tile_features(tile, rows, runs, bundles, min_cells):
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.features import rasterize
    from scipy import ndimage as ndi

    from terrain_s2.ml import channel_predict as CP
    from terrain_s2.ml.channel_labels import _horizontal
    pf = Path(runs) / tile / "ml_channel_prob.tif"
    if not pf.exists():
        return [dict(r, component=None) for r in rows.to_dict("records")]
    with rasterio.open(pf) as s:
        prob, T, crs = s.read(1), s.transform, _horizontal(s.crs)
    thr = json.loads((pf.with_suffix(".json")).read_text()).get("threshold", 0.5)
    m = np.nan_to_num(prob) >= thr
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n:
        sizes = np.bincount(lab.ravel())
        keep = sizes >= min_cells
        keep[0] = False
        m = keep[lab]
    cover = CP.building_cover(prob.shape, T, crs, Path(bundles) / tile / "context.gpkg")
    lab, qt = CP.component_quality(prob, m, T, cover)
    qt = qt.set_index("component")
    out = []
    g = gpd.GeoDataFrame(rows, geometry="geometry", crs=rows.crs).to_crs(crs)
    for r, geom in zip(g.drop(columns="geometry").to_dict("records"), g.geometry):
        cells = rasterize([(geom.buffer(2.0), 1)], out_shape=prob.shape, transform=T, fill=0, dtype="uint8") > 0
        ids = lab[cells]
        ids = ids[ids > 0]
        rec = dict(r, threshold=thr, prob_on_reach=float(np.nanmean(prob[cells])) if cells.any() else np.nan)
        if len(ids):
            c = int(np.bincount(ids).argmax())
            rec.update(component=c, **qt.loc[c].to_dict())
        else:
            rec.update(component=None)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviews", default=None, help="glob of review GeoPackages (default <data>/labels/manual/review_*.gpkg)")
    ap.add_argument("--runs")
    ap.add_argument("--bundles")
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--out")
    a = ap.parse_args()
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from terrain_s2.acquire.elevation import default_data_dir
    data = default_data_dir()
    runs, bundles = Path(a.runs or data / "runs"), Path(a.bundles or data / "tiles")
    files = sorted(glob.glob(a.reviews or str(data / "labels" / "manual" / "review_*.gpkg")))
    if not files:
        ap.error("no review files")
    parts = []
    for f in files:
        g = gpd.read_file(f, layer="reaches")
        g["review"] = Path(f).stem
        parts.append(g)
    rv = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=parts[0].crs)
    rv["answer"] = rv.exists.fillna("").astype(str).str.strip().str.lower()
    rv = rv[rv.answer.isin(["y", "n"])]
    print(f"{len(files)} review files, {len(rv)} confident answers on {rv.tile.nunique()} tiles")
    rows = []
    for t, g in rv.groupby("tile"):
        rows += tile_features(t, g, runs, bundles, a.min_cells)
    d = pd.DataFrame(rows)
    out = Path(a.out) if a.out else data / "labels" / "manual" / "review_features.csv"
    d.drop(columns=[c for c in ("geometry",) if c in d]).to_csv(out, index=False)
    mo = d[(d.kind == "model_only") & d.component.notna()]
    y, n = mo[mo.answer == "y"], mo[mo.answer == "n"]
    print(f"\nmodel-only reaches on a model component: {len(y)} y, {len(n)} n "
          f"({int(((d.kind == 'model_only') & d.component.isna()).sum())} not on one at the current threshold)")
    print(mo.groupby(["review", "answer"])[[m for m, _ in MEASURES]].median().round(3).to_string())
    print("\nmeasure      AUC   best threshold  keeps y  drops n")
    best = {}
    for m, sense in MEASURES:
        vals = np.unique(np.round(mo[m].dropna().to_numpy(), 3))
        top = (0.0, None, 0, 0)
        for v in vals:
            ky = (y[m] >= v).mean() if sense == "high" else (y[m] <= v).mean()
            dn = (n[m] < v).mean() if sense == "high" else (n[m] > v).mean()
            if ky + dn - 1 > top[0]:
                top = (ky + dn - 1, v, ky, dn)
        best[m] = top[1]
        print(f"{m:11s} {auc(y[m], n[m], sense == 'high'):.3f}  {sense} {'>=' if sense == 'high' else '<='} "
              f"{top[1] if top[1] is not None else '-':>8}   {top[2]:.0%}     {top[3]:.0%}")
    print("\nrules (contrast >=, width <=, urban <=): keeps y, drops n, best first")
    cq = np.quantile(mo.contrast.dropna(), [0.1, 0.2, 0.3, 0.4, 0.5]) if len(mo) else []
    wq = list(np.quantile(mo.width_m.dropna(), [0.6, 0.8, 0.9])) + [None] if len(mo) else []
    uq = [0.05, 0.1, 0.2, None]
    res = []
    for c, w, u in itertools.product(list(cq) + [None], wq, uq):
        def keep(t):
            k = np.ones(len(t), bool)
            if c is not None:
                k &= t.contrast.to_numpy() >= c
            if w is not None:
                k &= t.width_m.to_numpy() <= w
            if u is not None:
                k &= t.urban.to_numpy() <= u
            return k
        ky, dn = keep(y).mean() if len(y) else np.nan, 1 - keep(n).mean() if len(n) else np.nan
        res.append(dict(min_contrast=None if c is None else round(float(c), 3),
                        max_width_m=None if w is None else round(float(w), 1), max_urban=u,
                        keeps_y=round(float(ky), 3), drops_n=round(float(dn), 3), score=round(float(ky + dn - 1), 3)))
    r = pd.DataFrame(res).sort_values("score", ascending=False).drop_duplicates(["keeps_y", "drops_n"]).head(10)
    print(r.to_string(index=False))
    cm = d[d.kind == "council_missed"]
    if len(cm):
        print("\ncouncil-missed pieces (for interest): median probability on the reach, by answer")
        print(cm.groupby("answer").prob_on_reach.median().round(3).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
