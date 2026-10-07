#!/usr/bin/env python
"""Rank crossing candidates with a learned model; evaluate leave-one-site-out (Phase 2b, first form).

    python scripts/rank_candidates.py --config sites.json --out out/ranking

``sites.json`` lists sites, each with a run folder (from ``run_aoi.py --no-gates``), label files
(review GeoJSONs with ``culvert_y_n_unsure``) and/or inventories (e.g. council culvert lines):

    {"sites": [
      {"name": "SH12", "run": "out/sh12_ng", "labels": ["sh12_v1.geojson", "sh12_v2.geojson"]},
      {"name": "CANT1", "run": "out/cant1_ng",
       "inventory": ["cant1_stormwater.geojson"], "inventory_filter": "CLASSIFI_2=Culvert"}]}

Candidates are matched to labels by crest within 10 m (y = 1, n = 0, unsure and unlabelled are not
used for training) and to inventory lines by cutline within 10 m (y = 1). For each site with labels,
a model trained on the *other* sites scores it: ROC AUC, average precision, and how many candidates
must be reviewed, in score order, to reach 80% / 90% / 100% of the labelled culverts (against the
unranked list). A final model trained on all labelled sites writes ``<site>_ranked.geojson`` with a
``score`` per candidate for every site.

The hard gates of earlier iterations become features here: kind (one-hot), vegetation fraction,
approach length and angle, L_b. Missing values are allowed (HistGradientBoosting handles NaN).
"""
import argparse
import json
from pathlib import Path

import numpy as np

NUMERIC = ["h_b", "L_b", "width_half_height", "crest_width", "raised_h", "raised_run", "elongation",
           "cross_angle", "approach_length", "channel_angle_up", "channel_angle_dn", "veg_frac",
           "dsm_minus_dem_max", "dsm_minus_dem_mean", "upstream_area", "dep_area", "dep_volume",
           "dep_depth", "max_cut", "inc_up", "inc_dn"]
KINDS = ["crossing", "bank", "overflow", "channel_obstruction"]
MATCH_M = 10.0


def features(c):
    import pandas as pd
    X = pd.DataFrame(index=c.index)
    for f in NUMERIC:
        X[f] = pd.to_numeric(c[f], errors="coerce") if f in c else np.nan
    for f in ("upstream_area", "dep_area", "dep_volume"):
        X[f] = np.log10(1.0 + X[f].clip(lower=0))
    X["test_A"] = c["test"].isin(["A", "A+B"]).astype(float)
    X["test_B"] = c["test"].isin(["B", "A+B"]).astype(float)
    for k in KINDS:
        X[f"kind_{k}"] = (c.get("kind", "") == k).astype(float)
    return X


def load_site(s):
    import geopandas as gpd
    import pandas as pd
    from scipy.spatial import cKDTree
    c = gpd.read_file(Path(s["run"]) / "candidates.gpkg", layer="candidates")
    y = np.full(len(c), np.nan)
    if s.get("labels"):
        L = pd.concat([gpd.read_file(f).to_crs(c.crs) for f in s["labels"]], ignore_index=True)
        L = L[L["culvert_y_n_unsure"].astype(str).str.strip().str.len() > 0].copy()
        lab = L["culvert_y_n_unsure"].str.strip().str.lower()
        rank = lab.map({"y": 0, "unsure": 1, "n": 2}).fillna(3).values
        xy = np.column_stack([L.crest_x.fillna(L.geometry.x), L.crest_y.fillna(L.geometry.y)])
        if len(L):
            t = cKDTree(xy)
            for i, (x0, y0) in enumerate(zip(c.crest_x, c.crest_y)):
                hits = t.query_ball_point([x0, y0], MATCH_M)
                if hits:
                    best = min(hits, key=lambda j: rank[j])          # y > unsure > n
                    y[i] = {0: 1.0, 1: np.nan, 2: 0.0}.get(rank[best], np.nan)
    if s.get("inventory"):
        inv = pd.concat([gpd.read_file(f).to_crs(c.crs) for f in s["inventory"]], ignore_index=True)
        if s.get("inventory_filter"):
            col, val = s["inventory_filter"].split("=", 1)
            inv = inv[inv[col].astype(str).str.lower() == val.lower()]
        for g in inv.geometry:
            d = c.geometry.distance(g)
            y[(d <= MATCH_M).values] = 1.0
    return c, y


def review_to_reach(scores, y, frac):
    """Candidates to review in score order to find frac of the labelled positives."""
    order = np.argsort(-scores)
    pos = np.cumsum(y[order] == 1)
    need = int(np.ceil(frac * (y == 1).sum()))
    return int(np.argmax(pos >= need)) + 1 if need else 0


def main():
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sites = {}
    for s in cfg["sites"]:
        c, y = load_site(s)
        sites[s["name"]] = (c, features(c), y)
        print(f"{s['name']}: {len(c)} candidates, labelled {int(np.isfinite(y).sum())} "
              f"(culverts {int((y == 1).sum())}, not {int((y == 0).sum())})")

    def model():
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                              l2_regularization=1.0, min_samples_leaf=10,
                                              class_weight="balanced", random_state=a.seed)

    rows, oos = [], {}
    for name, (c, X, y) in sites.items():
        others = [(Xo[np.isfinite(yo)], yo[np.isfinite(yo)]) for n, (_, Xo, yo) in sites.items() if n != name]
        Xt = pd.concat([t[0] for t in others]); yt = np.concatenate([t[1] for t in others])
        if len(np.unique(yt)) < 2:
            continue
        m = model().fit(Xt, yt)
        sc_all = m.predict_proba(X)[:, 1]                  # out-of-site scores for every candidate
        oos[name] = sc_all
        row = dict(site=name, candidates=len(c), labelled=int(np.isfinite(y).sum()), culverts=int((y == 1).sum()))
        lab = np.isfinite(y)
        yl, sc = y[lab], sc_all[lab]
        if len(np.unique(yl)) == 2:                        # labelled negatives exist
            rnd = np.random.default_rng(1).random(len(sc))
            row.update(auc=round(roc_auc_score(yl, sc), 2), avg_precision=round(average_precision_score(yl, sc), 2),
                       base_rate=round(float((yl == 1).mean()), 2))
            for f in (0.8, 0.9, 1.0):
                row[f"review@{int(f * 100)}%"] = f"{review_to_reach(sc, yl, f)} (random {review_to_reach(rnd, yl, f)})"
        if (y == 1).any():                                 # ranks of known culverts among ALL candidates
            ranks = np.argsort(np.argsort(-sc_all)) + 1
            row["culvert_ranks_of_all"] = sorted(ranks[y == 1].tolist())[:12]
        rows.append(row)
    print("\nleave-one-site-out (model trained on the other sites only):")
    print(pd.DataFrame(rows).to_string(index=False))

    Xall = pd.concat([X[np.isfinite(y)] for _, X, y in sites.values()])
    yall = np.concatenate([y[np.isfinite(y)] for _, _, y in sites.values()])
    m = model().fit(Xall, yall)                            # for feature importance and future sites
    for name, (c, X, y) in sites.items():
        r = c.copy()
        r["score"] = oos.get(name, m.predict_proba(X)[:, 1])   # out-of-site score where available
        r["label_used"] = y
        r = r.sort_values("score", ascending=False)
        r["rank"] = np.arange(1, len(r) + 1)
        r.to_file(out / f"{name}_ranked.geojson", driver="GeoJSON")
    try:
        from sklearn.inspection import permutation_importance
        imp = permutation_importance(m, Xall, yall, n_repeats=10, random_state=a.seed, scoring="average_precision")
        top = sorted(zip(Xall.columns, imp.importances_mean), key=lambda t: -t[1])[:10]
        print("\nmost useful features (permutation importance, average precision):")
        for f, v in top:
            print(f"  {f:22s} {v:+.3f}")
    except Exception as e:  # noqa: BLE001
        print("importance not computed:", e)


if __name__ == "__main__":
    main()
