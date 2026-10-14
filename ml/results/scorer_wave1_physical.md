# Candidate scorer scorer_wave1_physical

Built 2026-10-14T04:34:56+00:00; code v0.1.0-27-g262eb49-dirty; 118 features; unlabelled weight ratio 1.0, scope None m, manual weight as labelled; final model on 71 tiles (138,607 candidates, holdout excluded).

Recall: positives captured in the top K candidates of each tile, over positives plus misses (labels without a candidate). Tiers = the rule-based ordering.

| fold | test tiles | positives | misses | recall@50 model / tiers | recall@100 model / tiers | recall@200 model / tiers | recall@500 model / tiers | AP model / tiers | manual AUC model / tiers | bridge AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| council:hbrc | 38 | 611 | 163 | 0.3527 / 0.1176 | 0.4341 / 0.2442 | 0.5336 / 0.3553 | 0.6615 / 0.5013 | 0.2011 / 0.0324 |  /  | 0.8589 |
| council:waimakariri | 45 | 930 | 786 | 0.2448 / 0.0862 | 0.3001 / 0.1259 | 0.3549 / 0.1742 | 0.4219 / 0.3252 | 0.3281 / 0.0553 | 0.6105 / 0.8092 | 0.9141 |
| sheets:1/5 | 22 | 203 | 246 | 0.3318 / 0.1693 | 0.3519 / 0.2094 | 0.3964 / 0.2561 | 0.4388 / 0.3786 | 0.355 / 0.0714 |  /  | 0.929 |
| sheets:2/5 | 11 | 289 | 77 | 0.3934 / 0.0765 | 0.4973 / 0.2213 | 0.5956 / 0.3798 | 0.6831 / 0.5219 | 0.3261 / 0.0415 |  /  | 0.9476 |
| sheets:3/5 | 12 | 143 | 26 | 0.5207 / 0.1775 | 0.5917 / 0.3018 | 0.6746 / 0.3846 | 0.8107 / 0.6095 | 0.3233 / 0.0443 |  /  | 0.8903 |
| sheets:4/5 | 16 | 186 | 148 | 0.3413 / 0.1287 | 0.4192 / 0.2036 | 0.4701 / 0.2455 | 0.509 / 0.3982 | 0.3402 / 0.0416 |  /  | 0.8727 |
| sheets:5/5 | 10 | 102 | 49 | 0.4238 / 0.1258 | 0.4967 / 0.1788 | 0.5629 / 0.2384 | 0.649 / 0.3377 | 0.199 / 0.0161 |  /  | 0.9098 |
| holdout | 12 | 618 | 403 | 0.2116 / 0.0421 | 0.2987 / 0.0823 | 0.3634 / 0.1342 | 0.4388 / 0.2919 | 0.339 / 0.056 | 0.5963 / 0.8092 | 0.95 |

Permutation importance (average precision on the holdout, base 0.326): ctx_dist_any_way_m 0.2746, log10_upa_max5 0.0694, tier 0.0552, dsm_minus_dem_at 0.0457, rem_min5 0.0078, dsm_minus_dem_max5 0.0065, veg_frac 0.0057, channel_mask_at 0.0042, dem_h_b 0.003, rem_max5 0.0028, openness_neg_max5 0.0028, tophat_black41_at 0.0027, tophat_white11_at 0.0026, relief_med21_at 0.002, tophat_black21_at 0.002.

Recall by label kind (model, at the second budget):

- council:hbrc: asset 0.2699 (115 + 48 missed); bridge 0.5964 (140 + 26 missed); council 0.2759 (140 + 34 missed); implied 0.5351 (216 + 55 missed)
- council:waimakariri: asset 0.3433 (43 + 24 missed); bridge 0.6637 (89 + 24 missed); council 0.279 (484 + 473 missed); implied 0.2767 (259 + 265 missed); manual 0.0909 (55 + 0 missed)
- sheets:1/5: bridge 0.8049 (35 + 6 missed); council 0.3413 (120 + 132 missed); implied 0.25 (48 + 108 missed)
- sheets:2/5: asset 0.5217 (21 + 2 missed); bridge 0.5167 (49 + 11 missed); council 0.3377 (60 + 17 missed); implied 0.5485 (159 + 47 missed)
- sheets:3/5: asset 0.2917 (36 + 12 missed); bridge 0.8421 (35 + 3 missed); council 0.4783 (38 + 8 missed); implied 0.8649 (34 + 3 missed)
- sheets:4/5: asset 0.28 (18 + 7 missed); bridge 0.6852 (43 + 11 missed); council 0.4286 (79 + 68 missed); implied 0.3056 (46 + 62 missed)
- sheets:5/5: asset 0.3582 (40 + 27 missed); bridge 0.6667 (30 + 9 missed); council 0.6111 (28 + 8 missed); implied 0.3333 (4 + 5 missed)
- holdout: asset 0.3134 (43 + 24 missed); bridge 0.5745 (37 + 10 missed); council 0.2688 (299 + 274 missed); implied 0.3441 (184 + 95 missed); manual 0.1273 (55 + 0 missed)
