"""ML-1 on a synthetic dataset: folds, PU weighting, recall against the tiers, the saved model."""
import json

import numpy as np
import pandas as pd

from terrain_s2.ml import train as TR


def _dataset(tmp_path, n_tiles=12, per_tile=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tiles):
        council = "a" if t < 6 else "b"
        sheet = f"S{t // 2}"
        for i in range(per_tile):
            real = rng.random() < 0.08
            f1 = rng.normal(2.0 if real else 0.0, 1.0)            # the signal the tiers do not use
            tier = rng.integers(1, 4)                              # tiers: no signal
            labelled = real and rng.random() < 0.6
            manual_n = (not real) and rng.random() < 0.02
            rows.append(dict(tile_id=f"T{t}", council=council, sheet=sheet, holdout=(t == 11), split="train", wave="w",
                             region="r", latest_survey="s", x=float(i), y=float(t), cand_id=f"T{t}:{i}",
                             patch_file="p.npy", patch_row=i, f1=f1, f2=rng.normal(), tier=float(tier),
                             tier_dem_only=float(tier), dem_h_b=rng.normal(),
                             struct=1.0 if labelled else (0.0 if manual_n else np.nan),
                             n_labels=1 if (labelled or manual_n) else 0, near_positive=False,
                             weight=1.0 if labelled else (0.6 if manual_n else 0.0),
                             label_kind="council" if labelled else ("manual" if manual_n else ""),
                             struct_class=("bridge" if f1 > 3 else "culvert") if labelled else "",
                             label_source="", label_id="", label_dist_m=np.nan, label_sources_all="",
                             in_core=True, in_scope=True))
    d = pd.DataFrame(rows)
    p = tmp_path / "ds"
    p.mkdir()
    d.to_parquet(p / "candidates.parquet", index=False)
    pd.DataFrame(dict(kind=["council"] * 5, tile_id=["T0", "T1", "T7", "T8", "T11"])).to_parquet(p / "misses.parquet", index=False)
    (p / "card.json").write_text(json.dumps(dict(versions=dict(features="2"))))
    return p


def test_scorer_beats_the_tiers_and_saves_a_model(tmp_path):
    ds = _dataset(tmp_path)
    cfg = TR.load_config()
    cfg["model"]["max_iter"] = 60
    rep = TR.run(ds, tmp_path / "model", cfg, log=lambda *a: None)
    names = [r["fold"] for r in rep["folds"]]
    assert names[-3:-1] == ["council:a", "council:b"] and names[-1] == "holdout" and any(n.startswith("blocks:") for n in names)
    for r in rep["folds"]:
        assert r["model"]["recall@50"] > r["baseline"]["recall@50"]
        assert r["model"]["average_precision"] > r["baseline"]["average_precision"]
        assert 0 < r["model"]["recall@500"] <= r["model"]["reachable_recall@500"] <= 1
    h = rep["folds"][-1]
    assert h["model"]["tiles"] == 1 and h["model"]["misses"] == 1
    saved = TR.load_model(tmp_path / "model")
    assert saved["features"] == ["f1", "f2", "tier", "tier_dem_only", "dem_h_b"] or set(saved["features"]) >= {"f1", "f2"}
    assert "x" not in saved["features"] and "struct" not in saved["features"]
    assert saved["structure"].predict_proba(np.zeros((1, len(saved["features"])), np.float32)).shape == (1, 2)
    assert (tmp_path / "model" / "report.md").read_text().startswith("# Candidate scorer")


def test_pu_weights_and_roles():
    d = pd.DataFrame(dict(struct=[1, 1, 0, np.nan, np.nan, np.nan], n_labels=[1, 1, 1, 0, 0, 1],
                          near_positive=[False, False, False, False, True, False], weight=[1.0, 0.7, 0.6, 0, 0, 0],
                          f=[1, 2, 3, 4, 5, 6.0], label_kind=["council", "council", "manual", "", "", "manual"]))
    X, y, w, keep = TR.design(d, ["f"], dict(unlabelled_weight_ratio=1.0))
    assert keep.tolist() == [True, True, True, True, False, False]        # near-positive and unsure left out
    assert y.tolist()[:4] == [1, 1, 0, 0] and np.isclose(w[3], 1.7)         # one unlabelled carries the positives' 1.7
    # scope: unlabelled weak negatives only near roads or rail
    d2 = pd.concat([d, d.iloc[[3]]], ignore_index=True).assign(ctx_dist_road_m=[0, 0, 0, 5, 0, 0, 300.0],
                                                               ctx_dist_rail_m=[500.0] * 6 + [10.0])
    _, _, w2, keep2 = TR.design(d2, ["f"], dict(unlabelled_weight_ratio=1.0, unlabelled_scope_m=20))
    assert keep2.tolist() == [True, True, True, True, False, False, True] and np.isclose(w2[3], 0.85)  # rail counts
    d2.loc[6, "ctx_dist_rail_m"] = 400.0
    _, _, w3, keep3 = TR.design(d2, ["f"], dict(unlabelled_weight_ratio=1.0, unlabelled_scope_m=20, manual_weight=1.0))
    assert not keep3[6] and np.isclose(w3[3], 1.7) and w3[2] == 1.0        # far from roads and rail: no weight;
    #                                                                          Matt's label at manual_weight


def test_feature_selection_patterns_and_any_way():
    d = pd.DataFrame(dict(a=[1.0, 2], road_dist_m=[1.0, 2], road_distance_at=[1.0, 2], ctx_dist_road_m=[5.0, 600],
                          ctx_dist_track_m=[50.0, 600], ctx_dist_osm_way_m=[np.nan, 3.0], ctx_dist_rail_m=[500.0, 500],
                          src_road_x_drain=[0.0, 1], src_testA=[1.0, 0]))
    d = TR.add_derived(d)
    assert d.ctx_dist_any_way_m.tolist() == [5.0, 3.0]
    cfg = dict(exclude_features=["road_dist_m", "road_distance_*", "ctx_*", "src_road_x_drain"],
               include_features=["ctx_dist_any_way_m"])
    assert TR.feature_columns(d, cfg) == ["a", "src_testA", "ctx_dist_any_way_m"]


def test_blocks_are_balanced_across_councils():
    assert TR.block_of("BW23_10000_0201") == "BW23:0" and TR.block_of("BW23_10000_0405") == "BW23:1"
    assert TR.block_of("BW23_10000_0502_SE") == "BW23:2"
    rows = []
    for council, sheets in (("a", ["AA01", "AA02", "AA03"]), ("b", ["BB01", "BB02", "BB03", "BB04"])):
        for sh in sheets:
            for r in range(1, 6):
                rows.append(dict(tile_id=f"{sh}_10000_{r:02d}01", council=council, holdout=False))
    d = pd.DataFrame(rows)
    f = TR.folds(d, dict(folds=["blocks"], block_folds=3, seed=0))
    assert len(f) == 3
    for _, tr, te in f:
        assert set(d[te].council) == {"a", "b"} and not (tr & te).any()
        assert not set(d[te].tile_id.map(TR.block_of)) & set(d[tr].tile_id.map(TR.block_of))
