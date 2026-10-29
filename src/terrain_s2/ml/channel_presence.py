"""The probability that a detected reach is a real channel (channel network v2, stage A version 3; Matt, 29 Oct).

Stage A version 3 adds detected channels (stage B: physics and U-Net) to the OSM and council lines where the evidence
supports them; this model says how much it does. One row per stage B reach; features from the reach itself
(``FEATURES``: width and depth on the LiDAR, length, bendiness, straightness, corners, slope from the bed heights,
roadside share, the U-Net's mean probability along it, whether the physics chain or only the U-Net found it, the
stream classifier's probability, the U-Net's class when there is one). OSM tags are left out: OSM is a label source.

Labels: a reach is *recorded* when council channel lines or OSM waterways run along it (within ``label_dist_m`` over
at least ``label_share`` of its length). Unrecorded is not the same as false: many real drains are on neither, so the
model learns ``P(recorded | features)`` (positive-unlabelled), which ranks reaches by how channel-like they are but
understates real channels. The threshold is therefore set on the channel reviews (answered real or false by Matt):
``calibrate`` picks the lowest probability whose reviewed precision reaches ``precision_target``, and stage A adds a
detected-only reach when its probability reaches that threshold.

``table`` builds the rows of one run; ``fit`` trains (scikit-learn's HistGradientBoostingClassifier, as the scorer) with
cross-validation by council, on the features that vary (``usable``; the model keeps their list); ``predict`` applies
a model to a reach table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FEATURES = ("width_m", "incision_m", "length_m", "bend_deg_per_100m", "straight_frac", "corners", "slope",
            "roadside_frac", "unet_prob", "physics", "unet_only", "p_stream", "ml_class_river", "ml_class_stream",
            "ml_class_drain", "log10_length")
DEFAULTS = dict(label_dist_m=5.0, label_share=0.5, precision_target=0.7, max_iter=300, learning_rate=0.05,
                max_leaf_nodes=31, l2_regularization=1.0, seed=0)


def _layer(f, name):
    import geopandas as gpd
    import pyogrio
    f = Path(f)
    if not f.exists() or name not in {n for n, _ in pyogrio.list_layers(f)}:
        return None
    g = gpd.read_file(f, layer=name)
    return g.to_crs(2193) if g.crs is not None and g.crs.to_epsg() != 2193 else g


def sample_mean(lines, prob_file, step=2.0):
    """Mean of a 0-250 uint8 probability raster along each line (0-1; NaN when it does not cover the line)."""
    import rasterio
    out = np.full(len(lines), np.nan)
    if prob_file is None or not Path(prob_file).exists() or not len(lines):
        return out
    with rasterio.open(prob_file) as s:
        for i, g in enumerate(lines):
            n = max(int(g.length // step), 1)
            pts = [g.interpolate(t).coords[0][:2] for t in np.linspace(0, g.length, n + 1)]
            v = np.array([x[0] for x in s.sample(pts)], float)
            v = v[v <= 250]
            if len(v):
                out[i] = float(v.mean() / 250.0)
    return out


def features(R, prob_file=None):
    """The model's features for a reach table (a stage B channels layer)."""
    import pandas as pd
    n = len(R)
    col = lambda c, d=np.nan: R[c].to_numpy(float) if c in R else np.full(n, d)
    src = R["channel_source"].astype(str).to_numpy() if "channel_source" in R else np.full(n, "physics")
    L = R.geometry.length.to_numpy(float)
    bs, be = col("bed_start"), col("bed_end")
    slope = np.abs(bs - be) / np.maximum(L, 1.0)
    mlc = R["ml_class"].astype(object).to_numpy() if "ml_class" in R else np.full(n, None)
    X = pd.DataFrame(dict(width_m=col("width_m"), incision_m=col("incision_m"), length_m=L,
                          bend_deg_per_100m=col("bend_deg_per_100m"), straight_frac=col("straight_frac"),
                          corners=col("corners"), slope=slope, roadside_frac=col("roadside_frac"),
                          unet_prob=sample_mean(list(R.geometry), prob_file), physics=(src == "physics").astype(float),
                          unet_only=(src == "ml").astype(float), p_stream=col("p_stream"),
                          ml_class_river=(mlc == "river").astype(float), ml_class_stream=(mlc == "stream").astype(float),
                          ml_class_drain=(mlc == "drain").astype(float), log10_length=np.log10(np.maximum(L, 1.0))))
    return X[list(FEATURES)]


def recorded(R, refs, P=None):
    """Bool per reach: council lines or OSM waterways (``refs``) run along it."""
    from shapely import STRtree
    P = dict(DEFAULTS, **(P or {}))
    if refs is None or not len(refs) or not len(R):
        return np.zeros(len(R), bool)
    tree = STRtree(refs.geometry.to_numpy())
    out = np.zeros(len(R), bool)
    for i, g in enumerate(R.geometry):
        idx = tree.query(g, predicate="dwithin", distance=P["label_dist_m"])
        if not len(idx):
            continue
        buf = __import__("shapely").union_all([refs.geometry.iloc[j] for j in idx]).buffer(P["label_dist_m"])
        out[i] = g.intersection(buf).length >= P["label_share"] * g.length
    return out


def table(run_dir, bundle_dir, prob_file=None):
    """The rows of one run: its stage B reaches (``network_b.gpkg``, else the run's channels) with features, the label
    and the geometry (for the reviews)."""
    import geopandas as gpd
    import pandas as pd
    run, b = Path(run_dir), Path(bundle_dir)
    R = _layer(run / "network_b.gpkg", "channels")
    if R is None:
        R = _layer(run / "network.gpkg", "channels")
    if R is None or not len(R):
        return None
    if "channel_source" in R:
        R = R[R.channel_source.astype(str) != "link"]
    R = R[R.geometry.notna() & ~R.geometry.is_empty].reset_index(drop=True)
    pf = prob_file or (run / "channel_prob.tif")
    X = features(R, pf)
    refs = []
    for f, lyr in ((b / "labels.gpkg", "channels"), (b / "context.gpkg", "osm_waterways")):
        g = _layer(f, lyr)
        if g is not None and len(g):
            refs.append(g[["geometry"]])
    ref = gpd.GeoDataFrame(pd.concat(refs, ignore_index=True), geometry="geometry", crs=2193) if refs else None
    X["recorded"] = recorded(R, ref)
    X["tile_id"] = run.name
    return gpd.GeoDataFrame(X, geometry=list(R.geometry), crs=2193)


def fit(T, P=None, groups=None):
    """Cross-validated by group (council) and a final model on everything: (model dict, report dict)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    P = dict(DEFAULTS, **(P or {}))
    feats = usable(T)
    X = T[feats].to_numpy(float)
    y = T["recorded"].to_numpy(bool)
    mk = lambda: HistGradientBoostingClassifier(max_iter=P["max_iter"], learning_rate=P["learning_rate"],
                                                max_leaf_nodes=P["max_leaf_nodes"],
                                                l2_regularization=P["l2_regularization"], random_state=P["seed"])
    rep = dict(rows=int(len(T)), recorded_share=round(float(y.mean()), 4), folds={},
               features_used=feats, features_constant=[f for f in FEATURES if f not in feats])
    oof = np.full(len(T), np.nan)
    if groups is not None:
        g = np.asarray(groups)
        for k in np.unique(g):
            te = g == k
            if te.all() or not y[~te].any() or y[~te].all():
                continue
            m = mk().fit(X[~te], y[~te])
            p = m.predict_proba(X[te])[:, 1]
            oof[te] = p
            if y[te].any() and not y[te].all():
                rep["folds"][str(k)] = dict(rows=int(te.sum()), auc=round(float(roc_auc_score(y[te], p)), 4),
                                            ap=round(float(average_precision_score(y[te], p)), 4),
                                            base=round(float(y[te].mean()), 4))
        ok = np.isfinite(oof)
        if ok.any() and y[ok].any() and not y[ok].all():
            rep["held_out"] = dict(auc=round(float(roc_auc_score(y[ok], oof[ok])), 4),
                                   ap=round(float(average_precision_score(y[ok], oof[ok])), 4))
    m = mk().fit(X, y)
    return dict(model=m, features=feats, threshold=0.5, params=P), rep, oof


def usable(T):
    """The features with at least two distinct values in ``T`` (constant or empty ones carry nothing, and some
    scikit-learn versions fail to bin them: "window shape cannot be larger than input array shape", Windows, 30 Oct)."""
    out = []
    for f in FEATURES:
        v = T[f].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(np.unique(v)) >= 2:
            out.append(f)
    if not out:
        raise ValueError("no feature varies across the reaches")
    return out


def predict(M, X):
    """Probabilities for a feature table (``features``)."""
    return M["model"].predict_proba(X[M["features"]].to_numpy(float))[:, 1]


def calibrate(M, reviewed, P=None):
    """The threshold from reviewed reaches (``reviewed``: a table with the model's features and ``real``, bool): the
    lowest probability whose reviewed precision (real among those at or above it) reaches ``precision_target``; with
    the reviewed precision and recall at it. Updates ``M["threshold"]``; returns the report."""
    P = dict(DEFAULTS, **(P or {}))
    if reviewed is None or not len(reviewed):
        return dict(reviewed=0)
    p = predict(M, reviewed)
    y = reviewed["real"].to_numpy(bool)
    order = np.argsort(-p)
    ps, ys = p[order], y[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    rec = np.cumsum(ys) / max(ys.sum(), 1)
    ok = np.flatnonzero(prec >= P["precision_target"])
    if len(ok):
        k = ok[-1]
        thr = float(ps[k])
    else:
        k = int(np.argmax(prec))
        thr = float(ps[k])
    M["threshold"] = round(thr, 4)
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, p)) if y.any() and not y.all() else None
    return dict(reviewed=int(len(y)), real=int(y.sum()), auc=None if auc is None else round(auc, 4),
                threshold=M["threshold"], precision=round(float(prec[k]), 4), recall=round(float(rec[k]), 4),
                precision_target=P["precision_target"])


def match_reviews(T, reviews, dist_m=3.0, share=0.5):
    """The rows of ``T`` (one tile's reaches) that reviewed reaches cover, with ``real`` from their answers."""
    from shapely import STRtree
    if reviews is None or not len(reviews) or T is None or not len(T):
        return None
    tree = STRtree(reviews.geometry.to_numpy())
    rows = []
    for i, g in enumerate(T.geometry):
        idx = tree.query(g, predicate="dwithin", distance=dist_m)
        best, bl = None, 0.0
        for j in idx:
            l_ = g.intersection(reviews.geometry.iloc[j].buffer(dist_m)).length
            if l_ > bl:
                best, bl = j, l_
        if best is not None and bl >= share * g.length:
            rows.append((i, reviews.answer.iloc[best] == "y"))
    if not rows:
        return None
    out = T.iloc[[i for i, _ in rows]].copy()
    out["real"] = [r for _, r in rows]
    return out
