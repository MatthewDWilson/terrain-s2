# Candidate scorer scorer_waves12

Built 2026-10-15T07:36:16+00:00; code v0.1.0-32-gf17191c; 118 features; unlabelled weight ratio 1.0, scope None m, manual weight as labelled; final model on 333 tiles (645,007 candidates, holdout excluded).

Recall: positives captured in the top K candidates of each tile, over positives plus misses (labels without a candidate). Tiers = the rule-based ordering.

| fold | test tiles | positives | misses | recall@50 model / tiers | recall@100 model / tiers | recall@200 model / tiers | recall@500 model / tiers | AP model / tiers | manual AUC model / tiers | bridge AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| council:auckland | 101 | 1256 | 294 | 0.4213 / 0.111 | 0.5077 / 0.209 | 0.5955 / 0.331 | 0.7071 / 0.4465 | 0.3261 / 0.024 |  /  | 0.9586 |
| council:environment_southland | 165 | 1120 | 146 | 0.6643 / 0.3712 | 0.7267 / 0.4455 | 0.7867 / 0.4984 | 0.8436 / 0.6532 | 0.4813 / 0.0748 |  /  | 0.9733 |
| council:hbrc | 38 | 682 | 89 | 0.4163 / 0.1219 | 0.5253 / 0.2542 | 0.6187 / 0.3684 | 0.7237 / 0.5292 | 0.343 / 0.0321 |  /  | 0.9213 |
| council:waimakariri | 41 | 1032 | 556 | 0.2361 / 0.0856 | 0.3029 / 0.1304 | 0.3747 / 0.1883 | 0.4748 / 0.3652 | 0.2852 / 0.0559 | 0.5968 / 0.8092 | 0.9288 |
| sheets:1/5 | 61 | 774 | 130 | 0.5144 / 0.1914 | 0.594 / 0.2987 | 0.6836 / 0.3993 | 0.76 / 0.5442 | 0.4244 / 0.0356 |  /  | 0.9594 |
| sheets:2/5 | 53 | 521 | 78 | 0.5643 / 0.2604 | 0.6511 / 0.3356 | 0.7346 / 0.4224 | 0.818 / 0.5275 | 0.4433 / 0.0371 |  /  | 0.973 |
| sheets:3/5 | 65 | 619 | 136 | 0.502 / 0.1589 | 0.5907 / 0.2464 | 0.6662 / 0.3298 | 0.7629 / 0.457 | 0.4088 / 0.0256 |  /  | 0.9789 |
| sheets:4/5 | 82 | 673 | 191 | 0.5775 / 0.2812 | 0.6308 / 0.3507 | 0.6806 / 0.4167 | 0.7477 / 0.6042 | 0.4464 / 0.0557 |  /  | 0.9405 |
| sheets:5/5 | 72 | 802 | 258 | 0.4047 / 0.1255 | 0.4858 / 0.2274 | 0.5792 / 0.3377 | 0.6783 / 0.4858 | 0.3609 / 0.0296 |  /  | 0.9419 |
| holdout | 12 | 701 | 292 | 0.2226 / 0.0473 | 0.3021 / 0.0906 | 0.3797 / 0.147 | 0.4783 / 0.3192 | 0.3388 / 0.0593 | 0.6461 / 0.8092 | 0.9477 |

Permutation importance (average precision on the holdout, base 0.3279): ctx_dist_any_way_m 0.2822, dsm_minus_dem_at 0.0557, log10_upa_max5 0.0472, breach_depth_max5 0.0156, network_link 0.0123, tier_dem_only 0.0116, rem_mean10 0.0085, relief_med5_min5 0.0075, relief_med5_mean10 0.0074, dsm_minus_dem_max5 0.0071, relief_med21_at 0.0056, rem_at 0.0056, valley_s2_mean10 0.0055, channel_mask_mean10 0.0053, gap_m 0.0049.

Recall by label kind (model, at the second budget):

- council:auckland: asset 0.3635 (455 + 109 missed); bridge 0.6632 (664 + 105 missed); implied 0.3318 (137 + 80 missed)
- council:environment_southland: asset 0.6604 (189 + 23 missed); bridge 0.756 (628 + 81 missed); implied 0.7072 (303 + 42 missed)
- council:hbrc: asset 0.454 (137 + 26 missed); bridge 0.7665 (151 + 16 missed); council 0.3506 (163 + 11 missed); implied 0.5318 (231 + 36 missed)
- council:waimakariri: asset 0.4062 (49 + 15 missed); bridge 0.69 (86 + 14 missed); council 0.2746 (555 + 348 missed); implied 0.2897 (287 + 179 missed); manual 0.0545 (55 + 0 missed)
- sheets:1/5: asset 0.4593 (171 + 38 missed); bridge 0.7025 (394 + 53 missed); council 0.3333 (43 + 2 missed); implied 0.5517 (166 + 37 missed)
- sheets:2/5: asset 0.4699 (159 + 24 missed); bridge 0.7821 (246 + 34 missed); council 0.7647 (17 + 0 missed); implied 0.605 (99 + 20 missed)
- sheets:3/5: asset 0.4833 (239 + 61 missed); bridge 0.6997 (319 + 54 missed); council 0.7391 (23 + 0 missed); implied 0.3898 (38 + 21 missed)
- sheets:4/5: asset 0.7755 (91 + 7 missed); bridge 0.7993 (254 + 25 missed); council 0.36 (136 + 89 missed); implied 0.6298 (192 + 70 missed)
- sheets:5/5: asset 0.4094 (121 + 28 missed); bridge 0.7273 (277 + 42 missed); council 0.3524 (146 + 64 missed); implied 0.3874 (258 + 124 missed)
- holdout: asset 0.3594 (49 + 15 missed); bridge 0.5532 (39 + 8 missed); council 0.2639 (353 + 204 missed); implied 0.3704 (205 + 65 missed); manual 0.0727 (55 + 0 missed)
