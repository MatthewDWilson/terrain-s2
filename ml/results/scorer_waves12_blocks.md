# Scorer on waves 1 + 2, spatial-block folds (17 Oct 2026)

Run in the sandbox on Matt's `waves12` dataset (345 tiles, 686,305 candidates), configuration `ml/config/scorer.yml`
(as `scorer_waves12`), folds `blocks`: a sheet's tiles two 1:10k rows at a time, dealt into 5 folds with every council in
each (StratifiedGroupKFold); the BW24 holdout excluded. Out-of-fold scores for 645,007 candidates.

| council | tiles | recall@100 model / tiers | AP model / tiers | recall@100 over labels with a candidate |
|---|---|---|---|---|
| auckland | 101 | 0.551 / 0.209 | 0.388 / 0.024 | implied 0.50, asset 0.55, bridge 0.80 |
| environment_southland | 165 | 0.746 / 0.446 | 0.537 / 0.075 | implied 0.84, asset 0.73, bridge 0.88 |
| hbrc | 38 | 0.562 / 0.254 | 0.379 / 0.032 | council 0.42, implied 0.73, asset 0.53, bridge 0.81 |
| waimakariri | 29 | 0.385 / 0.197 | 0.307 / 0.062 | council 0.65, implied 0.66, bridge 0.91 |
| all | 333 | 0.588 / 0.287 | 0.419 / 0.033 | council 0.55, implied 0.73, asset 0.59, bridge 0.84 |

Leave-one-council-out (`scorer_waves12.md`) is lower, as expected for transfer to a new kind of area: Waimakariri council
culverts 0.45 over labels with a candidate, against 0.65 here.
