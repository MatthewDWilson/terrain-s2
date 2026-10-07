#!/usr/bin/env python
"""Train the stream/drain segment classifier on reference channel data (e.g. Waimakariri
CLASSIFICATION3: Network Drain / Receiving Waterway) and evaluate leave-one-site-out.

    python scripts/train_stream_model.py --site NAME network.gpkg reference.geojson [--site ...] \
        --class-field CLASSIFICATION3 --stream-value "Receiving Waterway" --out stream_model.joblib

Works on *reaches* (channels layer of run_network.py). A reach is labelled if >= 60% of its length
lies within 5 m of reference channels of one class. Reports, leave-one-site-out and weighted by
length: the transparent prior (raw and after network smoothing, from the channels layer) and a
logistic-regression model on the reach features.
Reference channels are used only to train and evaluate, never as pipeline inputs.
"""
import argparse

import numpy as np



def label_segments(ch, ref, field, stream_value, dist=5.0, frac=0.6):
    ref = ref.to_crs(ch.crs)
    is_stream = ref[field].astype(str) == stream_value
    S, D = ref[is_stream].union_all(), ref[~is_stream].union_all()
    y = np.full(len(ch), np.nan)
    for i, g in enumerate(ch.geometry):
        if g.length == 0:
            continue
        pts = [g.interpolate(s) for s in np.arange(0, g.length, 2.0)]
        ns = np.mean([p.distance(S) <= dist for p in pts]) if not S.is_empty else 0
        nd = np.mean([p.distance(D) <= dist for p in pts]) if not D.is_empty else 0
        if ns >= frac and ns > nd:
            y[i] = 1
        elif nd >= frac:
            y[i] = 0
    return y


def main():
    import geopandas as gpd
    import joblib
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", nargs=3, action="append", metavar=("NAME", "NETWORK_GPKG", "REFERENCE"), required=True)
    ap.add_argument("--class-field", default="CLASSIFICATION3")
    ap.add_argument("--stream-value", default="Receiving Waterway")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from terrain_s2 import network
    data = {}
    for name, gpkg, ref in a.site:
        ch = gpd.read_file(gpkg, layer="channels")
        y = label_segments(ch, gpd.read_file(ref), a.class_field, a.stream_value)
        data[name] = (network.design_matrix(ch), y, ch.length.values, ch)
        lab = np.isfinite(y)
        print(f"{name}: {len(ch)} reaches, labelled {lab.sum()} ({ch.length.values[lab].sum() / 1000:.1f} km): "
              f"streams {int((y == 1).sum())}, drains {int((y == 0).sum())}")

    def model():
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000))

    def acc(p, y, w):
        return round(float(np.average((p >= 0.5) == y, weights=w)), 2)

    rows = []
    for name, (X, y, w, ch) in data.items():
        lab = np.isfinite(y)
        yl, wl = y[lab], w[lab]
        row = dict(site=name, labelled=int(lab.sum()))
        for col, tag in (("p_stream_raw", "prior"), ("p_stream", "prior+smoothing")):
            if col in ch:
                row[f"{tag}_acc"] = acc(ch[col].values[lab], yl, wl)
                if len(np.unique(yl)) == 2:
                    row[f"{tag}_auc"] = round(roc_auc_score(yl, ch[col].values[lab], sample_weight=wl), 2)
        others = [(Xo[np.isfinite(yo)], yo[np.isfinite(yo)], wo[np.isfinite(yo)]) for n, (Xo, yo, wo, _) in data.items() if n != name]
        if others:
            Xt = pd.concat([o[0] for o in others]); yt = np.concatenate([o[1] for o in others]); wt = np.concatenate([o[2] for o in others])
            if len(np.unique(yt)) == 2:
                m = model().fit(Xt, yt, logisticregression__sample_weight=wt)
                pm = m.predict_proba(X[lab])[:, 1]
                row["model_acc"] = acc(pm, yl, wl)
                if len(np.unique(yl)) == 2:
                    row["model_auc"] = round(roc_auc_score(yl, pm, sample_weight=wl), 2)
                joblib.dump(m, f"{a.out}.without_{name}.joblib")
        rows.append(row)
    print("\nleave-one-site-out (length-weighted):"); print(pd.DataFrame(rows).to_string(index=False))
    X = pd.concat([d[0][np.isfinite(d[1])] for d in data.values()])
    y = np.concatenate([d[1][np.isfinite(d[1])] for d in data.values()])
    w = np.concatenate([d[2][np.isfinite(d[1])] for d in data.values()])
    m = model().fit(X, y, logisticregression__sample_weight=w)
    joblib.dump(m, a.out)
    coef = dict(zip(network.REACH_FEATS, np.round(m[-1].coef_[0], 2)))
    print(f"\nfinal model (all sites) written to {a.out}; standardised coefficients: {coef}")


if __name__ == "__main__":
    main()
