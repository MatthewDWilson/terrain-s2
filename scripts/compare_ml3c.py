#!/usr/bin/env python
"""ML-3c, the end-to-end check: tile runs without (A) and with (B) the channel U-Net, against the labels.

    python scripts/compare_ml3c.py --manifest ml/manifests/wave3.csv --runs-b ~/terrain_data/runs_ml3c \
        --scorer ~/terrain_data/models/scorer_waves12

Per council and label kind (council registers, bridges, implied crossings, KiwiRail, manual), from each run's
``ml_probes.parquet`` (where the chain stands at every labelled structure) and ``ml_features.parquet`` (the
candidates, scored here with the same scorer for A and B):

    found         share of labels with a candidate within 15 m (the probes' "found")
    no_channel    share with no channel mapped within 10 m (what ML-3 is for: it should fall)
    recall@K      share of labels within ``--radius`` of one of the tile's top ``--top`` candidates by the scorer,
                  pooled over the tiles, and the mean of the per-tile shares (the scorer's reports use the mean)
    candidates    per tile

The design's test: "found" rises (more council culverts reachable) and recall@100 rises or holds. --scorer-b
scores B with a scorer trained on channel-model runs (its ml_* features from --channel-probs). Use tiles the
scorer never trained on (wave 3) for an honest recall@100. Writes --out (CSV, one row per tile, label kind and run).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def recall_at_k(cand_xy, scores, label_xy, k=100, radius=25.0):
    """(labels within ``radius`` of one of the top-``k`` candidates, labels)."""
    import numpy as np
    from scipy.spatial import cKDTree
    n = len(label_xy)
    if not n:
        return 0, 0
    if not len(cand_xy):
        return 0, n
    top = np.argsort(-np.asarray(scores, float))[:k]
    d, _ = cKDTree(np.asarray(cand_xy, float)[top]).query(np.asarray(label_xy, float))
    return int((d <= radius).sum()), n


def score_table(features_path, bundle, model, channel_probs=None):
    """The scorer's p_structure for a run's candidates (as ml.apply builds the features, with the channel-model
    features when the scorer uses them), with x and y."""
    import numpy as np
    import pandas as pd
    from terrain_s2.ml import dataset, features, train
    tab = pd.read_parquet(features_path)
    if not len(tab):
        return tab.assign(p=[])
    t = features.normalise_source_flags(tab.copy())
    t = dataset.context_features(Path(bundle), t)
    if any(f.startswith("ml_") for f in model["features"]):
        t = dataset.channel_features(Path(features_path).parent, t, channel_probs)
    t = train.add_derived(t)
    X = t.reindex(columns=model["features"]).to_numpy(np.float32)
    return pd.DataFrame({"x": tab.x.to_numpy(), "y": tab.y.to_numpy(), "p": model["structure"].predict_proba(X)[:, 1]})


def tile_rows(tile, council, runs, bundles, models, top, radius, channel_probs=None):
    import pandas as pd
    rows = []
    for name, root in runs.items():
        r = Path(root) / tile
        pf, ff = r / "ml_probes.parquet", r / "ml_features.parquet"
        if not (pf.exists() and ff.exists()):
            rows.append(dict(tile_id=tile, council=council, run=name, missing=True))
            continue
        pr = pd.read_parquet(pf)
        sc = score_table(ff, Path(bundles) / tile, models[name], channel_probs)
        for kind, g in (pr.groupby("kind") if len(pr) else []):
            hit, n = recall_at_k(sc[["x", "y"]].to_numpy(), sc.p.to_numpy(), g[["x", "y"]].to_numpy(), top, radius)
            rows.append(dict(tile_id=tile, council=council, run=name, kind=kind, labels=len(g),
                             found=int((g.reason == "found").sum()), no_channel=int((g.reason == "no_channel").sum()),
                             recall_hits=hit, candidates=len(sc), missing=False))
        if not len(pr):
            rows.append(dict(tile_id=tile, council=council, run=name, kind=None, labels=0, found=0, no_channel=0,
                             recall_hits=0, candidates=len(sc), missing=False))
    return rows


def summarise(d, top):
    """Per council and kind, A against B."""
    import numpy as np
    d = d[~d.missing.astype(bool) & d.kind.notna()]
    out = []
    for (council, kind), g in d.groupby(["council", "kind"]):
        rec = dict(council=council, kind=kind)
        for run, h in g.groupby("run"):
            n = h.labels.sum()
            per_tile = (h.recall_hits / h.labels.where(h.labels > 0)).dropna()
            rec.update({f"labels_{run}": int(n), f"found_{run}": round(h.found.sum() / max(n, 1), 3),
                        f"no_channel_{run}": round(h.no_channel.sum() / max(n, 1), 3),
                        f"recall@{top}_{run}": round(h.recall_hits.sum() / max(n, 1), 3),
                        f"recall@{top}_tile_mean_{run}": round(float(per_tile.mean()), 3) if len(per_tile) else np.nan,
                        f"candidates_per_tile_{run}": round(float(h.drop_duplicates("tile_id").candidates.mean()), 0)})
        out.append(rec)
    import pandas as pd
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--runs-a", help="runs without the channel model (default <data>/runs)")
    ap.add_argument("--runs-b", required=True, help="runs with it")
    ap.add_argument("--bundles")
    ap.add_argument("--scorer", required=True, help="the scorer's folder or model.pkl (ml/train.py), for A (and B)")
    ap.add_argument("--scorer-b", help="another scorer for B (one trained on channel-model runs)")
    ap.add_argument("--channel-probs", help="runs folder holding each tile's ml_channel_prob.tif (default <data>/runs)")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--radius", type=float, default=25.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    import pandas as pd
    from terrain_s2.acquire.elevation import default_data_dir
    from terrain_s2.ml.train import load_model
    data = default_data_dir()
    runs = {"A": a.runs_a or str(data / "runs"), "B": a.runs_b}
    bundles = a.bundles or str(data / "tiles")
    models = {"A": load_model(a.scorer)}
    models["B"] = load_model(a.scorer_b) if a.scorer_b else models["A"]
    probs = a.channel_probs or str(data / "runs")
    m = pd.concat([pd.read_csv(f) for f in a.manifest], ignore_index=True).drop_duplicates("tile_id")
    rows = []
    for i, r in enumerate(m.itertuples(), 1):
        rows += tile_rows(r.tile_id, getattr(r, "council", None), runs, bundles, models, a.top, a.radius, probs)
        if i % 25 == 0:
            print(f"  {i}/{len(m)} tiles")
    d = pd.DataFrame(rows)
    miss = d[d.missing.astype(bool)].groupby("run").tile_id.nunique().to_dict() if "missing" in d else {}
    if miss:
        print(f"tiles without probes or features: {miss}")
    s = summarise(d, a.top)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False))
    out = Path(a.out) if a.out else Path(a.runs_b) / "compare_ml3c.csv"
    d.to_csv(out, index=False)
    s.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
