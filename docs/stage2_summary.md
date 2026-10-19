# terrain-s2 Stage 2: summary and status (18 Oct 2026)

For the Stage 1 merge and for the full documentation. Stage 2 is the part of terrain-s2 that finds what a DEM does
not show (drains and streams, the culverts and bridges that carry them under roads and rail, embankments) from 1 m
LiDAR and national vectors, for flood modelling in EDDIE. Stage 1 (Cowork) makes the DEMs. This summary says what
exists, what it achieves, what is open, and **where the detail is** (paths are relative to the repository root;
`DESIGN_NOTES.md` is cited as DN with its section).

## 1. The parts

| Part | What it does | Code | Detail |
|---|---|---|---|
| Physics chain (R1) | DEM breaching and depressions, channel network (reaches, stream/drain classes), crossing candidates (Tests A and B at embankments, road and rail crossings), conditioned hydrology, relative elevation (REM), floodplain; products: `network.gpkg`, rasters | `run.py` (`run.network`), `network.py`, `channels.py`, `drains.py`, `crossings.py`, `conditioning.py`, `hydro.py`, `rem.py`, `floodplain.py`, `features.py`, `products.py`, `bands.py` | DN 3, 4-4l, 7k-7m; benchmarks DN 4b-4g |
| Data acquisition (M1) | Site and tile bundles: LiDAR DEM/DSM (LINZ, COG mirror), council channels, culverts, stopbanks (ArcGIS), LINZ roads/rail/bridges (WFS, key), KiwiRail, OSM (local extract or Overpass); source registry and coverage index | `acquire/` (`build.py`, `site.py`, `harmonise.py`, `arcgis.py`, `linz_wfs.py`, `osm.py`, `elevation.py`, `registry.py`), `sites/sources.yml` | DN 7, 7a-7i; `docs/source notes.md`; `docs/council_data_candidates.md` |
| Partition and tiles | National grid of LINZ 1:10k tiles (7,678 with LiDAR), quadrants, tile bundles as the processing unit | `grid.py`, `scripts/partition.py`, `scripts/tile_pool.py` | DN 7d-7g |
| Batch on the GPU machine | Many tiles in parallel, supervised, memory-sized; saved rasters for ML | `ml/batch.py`, `scripts/run_tiles.py`, `docs/wsl/terrain_hpc.sh` | `docs/wsl/terrain_hpc.md`; DN 11 |
| ML-0: dataset | Tile selection and folds, labels (council registers, KiwiRail, bridges, implied, manual), candidate features (141) and patches, probes of lost labels | `ml/select.py`, `ml/labels.py`, `ml/features.py`, `ml/dataset.py`, `ml/probes.py`, `ml/train.py` (folds) | DN 8, 11; `ml/config/*.yml` |
| ML-1: crossing scorer | Gradient boosting, weighted positive-unlabelled; scores in the run (`p_structure`, `p_bridge`, recorded structures kept) | `ml/train.py`, `ml/apply.py`, `scripts/train_scorer.py` | DN 11; `ml/results/scorer_*.md` |
| ML-1b: patch CNN | Below the scorer; deferred | `ml/cnn.py`, `scripts/train_cnn.py` | DN 11 |
| ML-3a: channel labels | Council lines snapped to the DEM bed, widths, label rasters (positive / negative / ignored, weights), training chips | `ml/snap.py`, `ml/channel_labels.py`, `ml/channel_rasters.py`, `ml/chips.py`; scripts `snap_channels.py`, `build_channel_labels.py`, `build_chips.py` | `docs/ml3_channel_model.md`; DN 11 (17-18 Oct); `ml/config/channels.yml` |
| ML-3b: channel U-Net | U-Net on 23 bands, clDice loss, folds, evaluation against the physics map, whole-tile prediction, review sets | `ml/unet.py`, `ml/channel_predict.py`; scripts `train_unet.py`, `predict_channels.py` | DN 11 (18 Oct); `ml/config/unet.yml`, `unet_v2.yml`; `docs/review_channels.md` |
| Checks and repair tools | Completeness scan, per-source refresh, clips for sharing | `scripts/check_runs.py`, `scripts/refresh_source.py`, `scripts/osm_extract.py`, `scripts/clip_run.py` | DN 11 (16-17 Oct) |

## 2. Data and machines

- **Matt's workstation** (Windows, conda `terrain-s2`, `TERRAIN_DATA_DIR=D:\eddie_data`): development, tests,
  source indexing and tile selection, git. Holds wave 1.
- **The RTX 6000 machine** (WSL distro `terrain`, 64 threads, 96 GB GPU, WSL limited to 150 GB): batches, chips,
  training. Data on their own ext4 disk `E:\WSL\terrain_data.vhdx` (3 TB), attached by a logon task and linked from
  `~/terrain_data`; from Windows `\\wsl.localhost\terrain\mnt\wsl\terrain_data`. Everything is run through
  `docs/wsl/terrain_hpc.sh` (modes: setup, update, check, start / status / stop, channels, chips, unet, predict,
  check-runs, guard-setup, torch, dataset). Lessons (memory, disk, rebuild): `docs/wsl/terrain_hpc.md`, DN 11.
- **Data layout** under the data root: `tiles/<id>` (bundles), `runs/<id>` (network, features, saved rasters,
  `ml_channels.gpkg`, label rasters, probabilities), `datasets/` (scorer datasets, chip datasets), `models/`,
  `labels/manual/` (Matt's reviews), `osm/` (the OSM extract), `cog/`, `stac/`, `snapshots/`, `partition/`.
  `docs/status.md` ("Data").

## 3. Training data

| Wave | Tiles | Councils (labels) | Manifest |
|---|---|---|---|
| 1 | 83 | Waimakariri 45, HBRC 38 (channels and culvert registers) | `ml/manifests/wave1.csv` |
| 2 | 266 | Environment Southland 165, Auckland 101 (channels) | `ml/manifests/wave2.csv` |
| 3 | 220 | Waikato 60, ECan 60, Gisborne 44 (held out), Hauraki 28, Otago 28 (scheme drains; WRC floodgates) | `ml/manifests/wave3.csv` |

Holdout: sheet BW24 (Matt's manual labels) and all of Gisborne. New sources are assessed in
`docs/council_data_candidates.md` (tranche 2 pending licences). Scorer dataset `waves12` (686,305 candidates,
4,090 positives); chip dataset `ml3_waves123` (170,700 chips).

## 4. Results so far

- **Physics chain**: blind benchmarks SH12 4/4, AOI2 2/2 (DN 4b-4c); a full LINZ tile in ~12 min with GPU features,
  ~10 GB memory (DN 7j, 11).
- **Crossing scorer (ML-1)**, waves 1 + 2, spatial blocks: recall@100 per tile 0.59 (rule tiers 0.29), AP 0.42
  (0.03); leave one council out 0.30-0.73; council culverts with a candidate 0.55 found (`ml/results/`).
- **Why ML-3**: of 57 lost council culverts (12 Waimakariri tiles), 43 had no channel mapped within 10 m: small
  drains (~500 m2 catchments) the physics map misses (`docs/ml3_channel_model.md` section 2).
- **Snapping** (ML-3a): council lines moved onto the bed; share of lines on the DEM bed rises for every council
  (e.g. Waimakariri 0.56 -> 0.87, Environment Southland 0.51 -> 0.92, Waikato 0.50 -> 0.92) (DN 11).
- **Channel U-Net v1** (ML-3b, threshold 0.8): fold 1 AP 0.87, centreline recall 0.86 (physics map 0.80); holdout
  AP 0.97, recall 0.96 (physics 0.96). Recall of the council drains the physics map misses: 0.35 (fold 1), 0.20
  (holdout): the gap ML-3 has to close. Learning curve: overall AP flat, small-drain recall still rising with data.
- **First review** (Matt, 164 reaches, 3 tiles): 65 % of the model's detections off the physics map and council
  lines are real drains (Southland 82 %, Hauraki 68 %, Waimakariri 41 %); where the held-out model sees nothing on a
  council line, 18 of 20 pieces have no drain; real drains it missed had probabilities ~0.46 (cut by the 0.8
  threshold) (DN 11, 18 Oct).

## 5. Status and next steps (ML)

1. **Now**: folds 2-5 of unet_v1 training; then out-of-fold probabilities for every tile (`predict ... oof`).
2. **Labels v3**: cleaning with out-of-fold probabilities, the review answers (built, DN 11 18 Oct).
3. **unet_v2**: extra weight and sampling for drains off the physics map, threshold for recall off physics
   (`ml/config/unet_v2.yml`); compare with v1 on the same labels; second review on new tiles.
4. **Aerial imagery** as input bands (LINZ open imagery), as a controlled experiment after v2.
5. **ML-3c**: the U-Net in the run (`run.network(..., channel_model=...)`), the end-to-end check: more council
   culverts found without recall@100 falling. **ML-3d**: the stream/drain head.
6. Retrain the scorer with wave 3 (floodgates, more councils; the OSM-refreshed tiles rerun); ML-1b again; ML-2
   (link decisions) and ML-4 (bed elevation, against Stage 1's GeoFabrics river bathymetry).
7. Tranche 2 sources (BOPRC, Horizons, culvert-rich sources) as wave 4.

`docs/status.md` keeps the short version current.

## 6. Integration with Stage 1 and EDDIE

- Done on the Stage 2 side: S2-1 (`run.network` follows the plan's signature), S2-9 (shared elevation reader with the
  COG mirror), S2-10 (one cache root, `TERRAIN_DATA_DIR`) (DN 9, 11).
- Open: S2-6 (honest and burned DEMs), S2-7 (structures layer), S2-8 (tile ownership, structure ids from location),
  S2-13 (provenance); reserved fields `p_structure`, `p_class`, `model_version` (DN 9).
- The plan: `docs/EDDIE Terrain — Revised Stage 1 and Stage 2 Integration Plan_with_diagrams.md`; Stage 1's own
  account: DN 10, `docs/stage1_workstation.md` and, on `stage1/rivers`, `docs/stage1_status_and_closeout.md`,
  `docs/stage1_rivers.md`.
- **The Stage 1 merge** (18 Oct): `stage1/rivers` (9 commits since 14 Oct) and `main` (Stage 2, 34 commits since)
  change no file in common; Stage 1's changes are in its own areas (`stage1/`, `cli.py`, `client/`, `settings.py`,
  `eddie/`, `docker/`, `environment-worker.yml`, its docs and tests). `integrate/stage1-stage2` is already in `main`.

## 7. Operational lessons (for the documentation)

All in DN 11 and `docs/wsl/terrain_hpc.md`: WSL's memory against Windows (the file cache; page file; the cache
guard), the data disk and its reattachment, the lost system disk and the rebuild, CRLF scripts, the LINZ API key in
`env.sh`, Overpass limits and the OSM extract, survey-edge DSM gaps, PyTorch and numpy's OpenMP on Windows, stale
`failed.json`, GPU starved by data loading.

## 8. Suggested structure for the full documentation

1. Overview (this file, sections 1 and 4); architecture diagram `docs/terrain_architecture.svg`.
2. The physics chain: algorithm (DN 3, 4-4l), parameters (`config.py`), products and their fields (`products.py`).
3. Data: sources and licences (`sites/sources.yml`, `docs/source notes.md`, `docs/council_data_candidates.md`),
   bundles, the partition.
4. Running it: workstation (README, `docs/stage1_workstation.md`), the GPU machine (`docs/wsl/terrain_hpc.md`),
   batches, checks.
5. Machine learning: decisions (DN 8, `docs/ml3_channel_model.md` section 7), datasets and labels, the scorer, the
   channel model, evaluation and reviews (`docs/review_channels.md`), results (`ml/results/`, model reports).
6. Integration with Stage 1 and EDDIE (section 6 above).
7. Operations and lessons (section 7 above); status and roadmap (`docs/status.md`, section 5).
