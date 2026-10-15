# terrain-s2: status (17 Oct 2026, ML-3a step 1)

Where the work stands, who is on what, and what comes next. Detail and evidence are in `DESIGN_NOTES.md`
(section numbers below); the plan is `docs/EDDIE Terrain — Revised Stage 1 and Stage 2 Integration Plan_with_diagrams.md`.

## Done

| Area | What | Notes |
|---|---|---|
| Stage 2 chain (R1) | Network, crossings (Tests A and B, road crossings), stream/drain classes, conditioned hydrology, REM | 7c-7l; benchmarks SH12 4/4, AOI2 2/2 |
| Data acquisition (M1) | Site and tile bundles from LINZ, councils, KiwiRail, OSM; source registry and coverage index | 7a-7i, 7e |
| Partition | LINZ map-sheet grid; 7,678 LiDAR tiles (1:10k), quadrants for testing; tile pool | 7d-7g |
| Performance | Superlinear costs removed; full tile 42.7 -> 15.8 min (CPU), ~12 min with GPU features; ~9-10 GB per tile | 7g-7j, 11 |
| Integration plan, Stage 2 side | S2-1 `run.network` (plan signature), S2-9 shared elevation reader with COG mirror, S2-10 one cache root (`TERRAIN_DATA_DIR`) | 9, 11 |
| Stage 1 (Cowork) | W1-W9 on `main`: layout, contract, keys, raster and GeoFabrics backends, store, client, CLI, EDDIE module, worker image | 10 |
| ML-0 steps 1-3 | Tile selection and folds, batch runner, label builder | 11 |
| ML-0 step 4 | Candidate features (141 columns) and oriented 64 x 64 m patches (14 bands), extracted inside the run | 11 |
| ML-0 step 5 | Dataset assembly: features + labels + folds, misses, card (`build_dataset.py`) | 11 |
| HPC install | `docs/wsl/terrain_hpc.sh` (WSL, no Docker): setup, update, check, batch, dataset | 11 |
| Wave 1 | 80 of 83 tiles run; dataset `wave1` (173,284 candidates, 1,522 positives, 73 negatives, 941 misses); the 3 failures fixed | 11 |
| ML-1 scorer | Gradient boosting, weighted PU; chosen configuration 1.4-3.6 x the tiers (`ml/results/`) | 11 |
| ML-1 step 9 | Scores in the run (`--crossing-model`): `p_structure`, `p_bridge`, recorded structures always included | 11 |
| Miss diagnosis | Probes at every labelled structure; reasons per council (`diagnose_misses.py`) | 11 |
| ML-1b | Patch CNN (`train_cnn.py`): below the scorer; ensemble no gain; deferred behind ML-3 | 11 |
| Waves 1 + 2 | 345 tiles, 686,305 candidates; scorer 2-3.5 x the tiers on every held-out council (`ml/results/`) | 11 |

## In progress

- **ML-3 (a learned channel map)**: design agreed (`docs/ml3_channel_model.md`, decisions in its section 7); the work
  from ML-3a on continues in a new conversation, from `docs/handover_ml3.md`.
- **Stage 1 rivers** (Cowork, branch `stage1/rivers`): GeoFabrics river bathymetry for the ML-4 benchmark; merge when
  tested.
- **ML-3a step 2: the rerun with saved rasters** (`--param rasters=all`, all 349 tiles including the four wave-1 tiles
  whose features and candidates disagreed), on the RTX 6000 machine after the memory fix of 16 Oct (DESIGN_NOTES 11);
  then `waves12` rebuilt with all 349 (`terrain_hpc.sh dataset`).

## Next

1. ML-3a: snapping (step 3) done on all 349 tiles; label rasters (step 4) built, to run on the RTX 6000 machine
   (`terrain_hpc.sh channels`); next the chips (step 5);
   chips from the saved rasters; review sets for Matt. Step 1 (widths) done: no council records widths usably, so
   widths come from the fallbacks in `ml/config/channels.yml` (DESIGN_NOTES 11).
2. ML-3b-d: the U-Net, its integration into the run, the end-to-end check on found labels, the stream/drain head.
3. ML-1b again (the CNN with the scorer's scalar features), after ML-3.
4. ML-2 (link decisions) and ML-4 (bed elevation, against GeoFabrics' river bathymetry).

## Benchmark for bed elevation (ML-4)

Stage 1 runs GeoFabrics without its river stages: "rivers, waterways, stopbanks and patch stages stay out of
Stage 1; new templates can add them later without code changes" (plan section 3). ML-4 needs GeoFabrics' river
bathymetry (`RiverBathymetryGenerator`, `WaterwayBedElevationEstimator`) as its benchmark, so this is a new
GeoFabrics template for the Stage 1 backend, evaluated in W7 (section 12). Inputs it needs that we do not hold
yet: a river network with flows and Manning's n per reach (REC with flood statistics), and the instructions
Smart Ideas used for `8m_geofabric.nc` (run offline, not recorded: 4e).

## Who owns what (section 9 item 1)

- Cowork: `stage1/`, `io/`, `key.py`, `store.py`, `client/`, `cli.py`, `eddie/`, `docker/`, `compose/`.
- Stage 2 agent: the existing modules, `run.py`, `acquire/elevation.py`, `ml/`, and `condition.py`,
  `structures.py`, `tiles.py` when they land.
- Shared, one owner at a time: `pyproject.toml`, `environment.yml`, `sites/sources.yml`, `acquire/registry.py`,
  `README.md`, `DESIGN_NOTES.md`, this file. Deliveries as git patches on the latest `main`.

## Data (Matt's workstation)

`TERRAIN_DATA_DIR = D:\eddie_data`: `stac/` (STAC documents, item indexes), `cog/` (mirrored LINZ tiles),
`snapshots/` (vector snapshots, both stages), `tiles/` (bundles), `runs/` (runs, labels, features),
`labels/manual/` (Matt's review files), `partition/` (`nz_tiles.gpkg`, `nz_quadrants.gpkg`).

## Wave 3 (first tranche, 17 Oct)

Sources enabled, indexed and selected: 220 tiles (`ml/manifests/wave3.csv`, DESIGN_NOTES 11). To run: on the workstation, index the six
sources (`index_sources.py --only ...`) and select (`select_tiles.py --config ml/config/wave3.yml`), commit
`sites/sources_coverage.geojson` and `ml/manifests/wave3.csv`; on the RTX 6000 machine, after the disk move,
`update`, then `start ml/manifests/wave3.csv --param rasters=all`, then `channels` for the manifest.
