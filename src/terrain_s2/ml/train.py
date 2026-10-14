"""ML-1: the candidate scorer (gradient boosting, weighted positive-unlabelled) and its evaluation.

Targets, from the dataset (``ml.dataset``):
- ``p_structure``: a structure (culvert or bridge) at the candidate. Positives: council and asset registers,
  bridges, implied crossings, Matt's "y"; negatives: Matt's "n"; unlabelled candidates are weak negatives only
  inside the inventories' scope (``unlabelled_scope_m`` of a road or railway: council registers hold road culverts,
  KiwiRail rail ones), with total weight ``unlabelled_weight_ratio`` x the positives'; elsewhere (farm and private
  crossings, which no inventory records) they get no weight, so the model is not taught that an unrecorded
  structure is no structure (scope-aware PU; ``unlabelled_scope_m: null`` treats every unlabelled candidate as a
  weak negative, the first run). ``manual_weight`` overrides the weight of Matt's labels. ``near_positive`` (a second candidate of one structure) and
  "unsure" are left out of training;
- ``p_bridge``: bridge rather than culvert, among positives of known class.

Model: scikit-learn's HistGradientBoostingClassifier (missing values handled natively). Features: the dataset's
feature columns (never location, identifiers or labels).

Evaluation, for the model and for the rule-based tiers (the baseline: tier, then the DEM-only tier, then h_b):
- folds: leave one council out (train on the others, outside the holdout); blocked by sheet within the training
  councils (``GroupKFold``); the holdout sheets (BW24), trained on everything else;
- recall at fixed review budgets per tile (top 50, 100, 200, 500 candidates): positives captured / (positives +
  misses), so labels the chain never proposed count against it, and the same without the misses ("reachable");
  by label kind;
- average precision, labelled positives against all other candidates (a lower bound: unlabelled may be real);
- on Matt's labels (y against n) where a fold has them: ROC AUC and precision at the model's top half.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

DEFAULTS = dict(unlabelled_weight_ratio=1.0, unlabelled_scope_m=None, manual_weight=None, importance=True,
                exclude_features=[], include_features=[],
                budgets=[50, 100, 200, 500], sheet_folds=5, seed=0,
                model=dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
                           early_stopping=True, validation_fraction=0.1, n_iter_no_change=30))


def load_config(path=None) -> dict:
    c = json.loads(json.dumps(DEFAULTS))
    if path:
        import yaml
        d = yaml.safe_load(Path(path).read_text()) or {}
        for k, v in d.items():
            if isinstance(v, dict) and isinstance(c.get(k), dict):
                c[k].update(v)
            else:
                c[k] = v
    return c


def load_dataset(path):
    """(candidates, misses, card); source flags normalised (features v1 and v2 alike)."""
    import pandas as pd

    from .features import normalise_source_flags
    path = Path(path)
    d = add_derived(normalise_source_flags(pd.read_parquet(path / "candidates.parquet")))
    m = pd.read_parquet(path / "misses.parquet") if (path / "misses.parquet").exists() else pd.DataFrame()
    card = json.loads((path / "card.json").read_text())
    return d, m, card


def add_derived(d):
    """Features computed from the dataset's columns: ``ctx_dist_any_way_m``, the distance to the nearest road,
    track, railway or OSM way of any kind (500 m cap): the barrier as physical context, without saying whose
    inventory it belongs to."""
    parts = [c for c in ("ctx_dist_road_m", "ctx_dist_track_m", "ctx_dist_osm_way_m", "ctx_dist_rail_m") if c in d]
    if parts and "ctx_dist_any_way_m" not in d:
        d = d.assign(ctx_dist_any_way_m=d[parts].fillna(500.0).min(axis=1))
    return d


def feature_columns(d, cfg=None) -> list:
    """Numeric, non-empty dataset columns that are not location, identifiers or labels; then the config's
    ``exclude_features`` (names or shell patterns) are removed and ``include_features`` put back."""
    from fnmatch import fnmatch

    from .dataset import NOT_FEATURES
    cols = [c for c in d.columns if c not in NOT_FEATURES and d[c].dtype.kind in "fiub" and d[c].notna().any()]
    cfg = cfg or {}
    ex, inc = cfg.get("exclude_features") or [], cfg.get("include_features") or []
    return [c for c in cols if not any(fnmatch(c, p) for p in ex) or any(fnmatch(c, p) for p in inc)]


def roles(d):
    """Boolean arrays: positive, negative (manual n), unlabelled, excluded (near a positive, unsure)."""
    s = d.struct.to_numpy(float)
    labelled = d.n_labels.fillna(0).to_numpy() > 0
    pos = s == 1
    neg = s == 0
    near = d.near_positive.fillna(False).to_numpy(bool) & ~pos & ~neg
    unl = ~labelled & ~near
    excl = (labelled & np.isnan(s)) | near
    return pos, neg, unl, excl


def design(d, cols, cfg):
    """(X, y, w, keep) for training on ``d``."""
    pos, neg, unl, excl = roles(d)
    X = d[cols].to_numpy(np.float32)
    y = pos.astype(int)
    w = np.zeros(len(d))
    w[pos] = d.weight.to_numpy(float)[pos]
    w[neg] = d.weight.to_numpy(float)[neg]
    if cfg.get("manual_weight") is not None:
        man = (d.label_kind == "manual").to_numpy() & (pos | neg)
        w[man] = cfg["manual_weight"]
    scope = cfg.get("unlabelled_scope_m")
    if scope is not None and "ctx_dist_road_m" in d:
        road = d.ctx_dist_road_m.fillna(np.inf).to_numpy()
        rail = d.ctx_dist_rail_m.fillna(np.inf).to_numpy() if "ctx_dist_rail_m" in d else np.full(len(d), np.inf)
        near = np.minimum(road, rail) <= scope
        unl = unl & near
    if unl.any():
        w[unl] = cfg["unlabelled_weight_ratio"] * w[pos].sum() / unl.sum()
    keep = (pos | neg | unl) & ~excl & (w > 0)
    return X, y, w, keep


def fit(d, cols, cfg, target="structure"):
    from sklearn.ensemble import HistGradientBoostingClassifier
    if target == "bridge":
        sel = (d.struct == 1) & d.struct_class.isin(["culvert", "bridge"])
        X, y = d.loc[sel, cols].to_numpy(np.float32), (d.loc[sel, "struct_class"] == "bridge").astype(int).to_numpy()
        w = d.loc[sel, "weight"].to_numpy(float)
    else:
        X, y, w, keep = design(d, cols, cfg)
        X, y, w = X[keep], y[keep], w[keep]
    if len(np.unique(y)) < 2:
        return None
    params = dict(cfg["model"])
    if params.get("early_stopping") and min(np.bincount(y)) < 20:     # too few of a class for a validation split
        params["early_stopping"] = False
    m = HistGradientBoostingClassifier(random_state=cfg["seed"], **params)
    m.fit(X, y, sample_weight=w)
    return m


def baseline_score(d):
    """The rule-based ordering: tier, then the DEM-only tier, then h_b."""
    t = d.tier.fillna(0).to_numpy(float)
    t2 = d.tier_dem_only.fillna(0).to_numpy(float)
    hb = np.clip(d.dem_h_b.fillna(0).to_numpy(float), -5, 5)
    return t + t2 / 10 + (hb + 5) / 1000


def evaluate(d, score, misses=None, budgets=(50, 100, 200, 500)):
    """Metrics for one test set (``d`` rows with ``score``)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    pos, neg, unl, excl = roles(d)
    out = dict(tiles=int(d.tile_id.nunique()), candidates=int(len(d)), positives=int(pos.sum()),
               misses=int(len(misses)) if misses is not None else 0)
    rank = d.assign(_s=score, _pos=pos).sort_values(["tile_id", "_s"], ascending=[True, False])
    rank["_r"] = rank.groupby("tile_id").cumcount()
    mis_by_tile = misses.groupby("tile_id").size() if misses is not None and len(misses) else None
    denom_all = pos.sum() + (0 if mis_by_tile is None else int(mis_by_tile.reindex(d.tile_id.unique()).fillna(0).sum()))
    for k in budgets:
        cap = int((rank._pos & (rank._r < k)).sum())
        out[f"recall@{k}"] = round(cap / denom_all, 4) if denom_all else None
        out[f"reachable_recall@{k}"] = round(cap / pos.sum(), 4) if pos.sum() else None
    kinds = d.label_kind.fillna("").to_numpy()
    k = budgets[1] if len(budgets) > 1 else budgets[0]
    by_kind = {}
    for kind in sorted(set(kinds[pos])):
        m = pos & (kinds == kind)
        cap = int((rank._pos & (rank._r < k) & (rank.label_kind == kind)).sum())
        miss = int((misses.kind == kind).sum()) if misses is not None and len(misses) else 0
        by_kind[kind] = dict(positives=int(m.sum()), misses=miss, **{f"recall@{k}": round(cap / (m.sum() + miss), 4)})
    out["by_kind"] = by_kind
    ev = ~excl
    if pos[ev].any() and (~pos[ev]).any():
        out["average_precision"] = round(float(average_precision_score(pos[ev], score[ev])), 4)
        out["positive_rate"] = round(float(pos[ev].mean()), 5)
    man = (d.label_kind == "manual").to_numpy() & (pos | neg)
    if man.sum() >= 10 and pos[man].any() and neg[man].any():
        out["manual_auc"] = round(float(roc_auc_score(pos[man], score[man])), 4)
        out["manual_n"] = dict(y=int(pos[man].sum()), n=int(neg[man].sum()))
    return out


def folds(d, cfg):
    """[(name, train mask, test mask)]: leave-one-council-out, sheets within councils, holdout."""
    from sklearn.model_selection import GroupKFold
    hold = d.holdout.fillna(False).to_numpy(bool)
    out = []
    councils = sorted(d.council.dropna().unique())
    if len(councils) > 1:
        for c in councils:
            out.append((f"council:{c}", (d.council != c).to_numpy() & ~hold, (d.council == c).to_numpy()))
    sheets = d.loc[~hold, "sheet"]
    k = min(cfg["sheet_folds"], sheets.nunique())
    if k >= 2:
        idx = np.flatnonzero(~hold)
        for i, (tr, te) in enumerate(GroupKFold(n_splits=k).split(idx, groups=sheets.to_numpy())):
            trm, tem = np.zeros(len(d), bool), np.zeros(len(d), bool)
            trm[idx[tr]], tem[idx[te]] = True, True
            out.append((f"sheets:{i + 1}/{k}", trm, tem))
    if hold.any():
        out.append(("holdout", ~hold, hold))
    return out


def run(dataset, out, cfg=None, name=None, log=print):
    """Cross-validate, train the final models on everything outside the holdout, write the report."""
    import pickle

    import pandas as pd

    from . import code_version
    cfg = cfg or load_config()
    d, misses, card = load_dataset(dataset)
    cols = feature_columns(d, cfg)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results, scores = [], []
    for fname, tr, te in folds(d, cfg):
        test_tiles = set(d.loc[te, "tile_id"])
        mis = misses[misses.tile_id.isin(test_tiles)] if len(misses) else misses
        m = fit(d[tr], cols, cfg)
        if m is None:
            log(f"{fname}: skipped (one class in training)")
            continue
        s = m.predict_proba(d.loc[te, cols].to_numpy(np.float32))[:, 1]
        r = dict(fold=fname, train_tiles=int(d.loc[tr, "tile_id"].nunique()),
                 model=evaluate(d[te], s, mis, cfg["budgets"]), baseline=evaluate(d[te], baseline_score(d[te]), mis, cfg["budgets"]))
        mb = fit(d[tr], cols, cfg, target="bridge")
        sel = te & (d.struct == 1).to_numpy() & d.struct_class.isin(["culvert", "bridge"]).to_numpy()
        if mb is not None and sel.sum() >= 10 and d.loc[sel, "struct_class"].nunique() == 2:
            from sklearn.metrics import roc_auc_score
            yb = (d.loc[sel, "struct_class"] == "bridge").astype(int)
            r["bridge_auc"] = round(float(roc_auc_score(yb, mb.predict_proba(d.loc[sel, cols].to_numpy(np.float32))[:, 1])), 4)
        results.append(r)
        scores.append(pd.DataFrame(dict(cand_id=d.loc[te, "cand_id"].to_numpy(), fold=fname, score=s)))
        k = cfg["budgets"][1]
        log(f"{fname}: recall@{k} model {r['model'][f'recall@{k}']} vs tiers {r['baseline'][f'recall@{k}']}; "
            f"AP {r['model'].get('average_precision')} vs {r['baseline'].get('average_precision')}"
            + (f"; manual AUC {r['model'].get('manual_auc')} vs {r['baseline'].get('manual_auc')}" if "manual_auc" in r["model"] else ""))
    hold = d.holdout.fillna(False).to_numpy(bool)
    final = fit(d[~hold], cols, cfg)
    importance = None
    if cfg.get("importance") and final is not None and hold.any():
        importance = permutation_importance(final, d[hold], cols, cfg)
    final_b = fit(d[~hold], cols, cfg, target="bridge")
    meta = dict(name=name or out.name, built=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                code=code_version(), dataset=str(dataset), dataset_card=card.get("versions"), features=cols, config=cfg,
                trained_on=dict(tiles=int(d.loc[~hold, "tile_id"].nunique()), candidates=int((~hold).sum())))
    # plain pickle (protocol 5) rather than joblib: joblib < 1.6 sets array.shape when loading, deprecated in NumPy
    # 2.5 (384 warnings in the tests on Matt's workstation, 15 Oct 2026)
    with open(out / "model.pkl", "wb") as f:
        pickle.dump(dict(structure=final, bridge=final_b, features=cols, meta=meta), f, protocol=5)
    if scores:
        pd.concat(scores, ignore_index=True).to_parquet(out / "fold_scores.parquet", index=False)
    rep = dict(meta=meta, folds=results, importance=importance)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    (out / "report.md").write_text(report_md(rep, cfg["budgets"]))
    return rep


def permutation_importance(model, d, cols, cfg, n=20000, repeats=3, top=25):
    """Drop in average precision (labelled positives against the rest) when one feature is shuffled, on the
    holdout (at most ``n`` candidates); the ``top`` features."""
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(cfg["seed"])
    pos, _, _, excl = roles(d)
    idx = np.flatnonzero(~excl)
    if len(idx) > n:
        idx = np.sort(rng.choice(idx, n, replace=False))
    X = d.iloc[idx][cols].to_numpy(np.float32)
    y = pos[idx]
    if y.sum() < 5:
        return None
    base = average_precision_score(y, model.predict_proba(X)[:, 1])
    out = []
    for j, c in enumerate(cols):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            drops.append(base - average_precision_score(y, model.predict_proba(Xp)[:, 1]))
        out.append((c, float(np.mean(drops))))
    out.sort(key=lambda t: -t[1])
    return dict(metric="average precision on the holdout", base=round(float(base), 4),
                top=[dict(feature=c, drop=round(v, 4)) for c, v in out[:top]])


def report_md(rep, budgets) -> str:
    m = rep["meta"]
    c = m["config"]
    L = [f"# Candidate scorer {m['name']}", "", f"Built {m['built']}; code {m['code']}; {len(m['features'])} features; "
         f"unlabelled weight ratio {c['unlabelled_weight_ratio']}, scope {c.get('unlabelled_scope_m')} m, manual weight "
         f"{c.get('manual_weight') or 'as labelled'}; "
         f"final model on {m['trained_on']['tiles']} tiles ({m['trained_on']['candidates']:,} candidates, holdout excluded).", "",
         "Recall: positives captured in the top K candidates of each tile, over positives plus misses (labels without a "
         "candidate). Tiers = the rule-based ordering.", "",
         "| fold | test tiles | positives | misses | " + " | ".join(f"recall@{k} model / tiers" for k in budgets)
         + " | AP model / tiers | manual AUC model / tiers | bridge AUC |",
         "|---|---|---|---|" + "---|" * len(budgets) + "---|---|---|"]
    for r in rep["folds"]:
        a, b = r["model"], r["baseline"]
        L.append(f"| {r['fold']} | {a['tiles']} | {a['positives']} | {a['misses']} | "
                 + " | ".join(f"{a[f'recall@{k}']} / {b[f'recall@{k}']}" for k in budgets)
                 + f" | {a.get('average_precision')} / {b.get('average_precision')} | "
                 f"{a.get('manual_auc', '')} / {b.get('manual_auc', '')} | {r.get('bridge_auc', '')} |")
    imp = rep.get("importance")
    if imp:
        L += ["", f"Permutation importance ({imp['metric']}, base {imp['base']}): "
              + ", ".join(f"{t['feature']} {t['drop']}" for t in imp["top"][:15]) + "."]
    L += ["", "Recall by label kind (model, at the second budget):", ""]
    for r in rep["folds"]:
        L.append(f"- {r['fold']}: " + "; ".join(f"{k} {v[next(x for x in v if x.startswith('recall@'))]} "
                                                f"({v['positives']} + {v['misses']} missed)" for k, v in r["model"]["by_kind"].items()))
    return "\n".join(L) + "\n"


def load_model(path):
    """The saved scorer: dict(structure=, bridge=, features=, meta=). ``path``: model.pkl or its folder."""
    import pickle
    path = Path(path)
    f = path / "model.pkl" if path.is_dir() else path
    with open(f, "rb") as fh:
        return pickle.load(fh)
