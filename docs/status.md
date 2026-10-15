# terrain-s2: status (15 Oct 2026, evening)

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

- **Wave 2 batch** (RTX 6000 machine): 259 of 266 done; 7 Auckland tiles to rerun after the cache fix.
- **ML-1 first results**: `train_scorer.py` on `wave1`.
- **Stage 1 rivers** (Cowork, branch `stage1/rivers`): GeoFabrics river bathymetry for the ML-4 benchmark; merge when tested.
- **Stage 1** (Cowork): remaining W items (W3 acceptance on the shared reader, W5 on PostGIS, W7, W10, W11).

## Next

1. ML-3 (channel detection for small drains; draft in `docs/ml3_channel_model.md`): most lost council culverts
   have no channel mapped.
2. Rerun BW23_10000_0201, 0202, 0302, 0303 (`--rerun`) and rebuild `waves12`.
2. ML-1: gradient-boosting candidate scorer (weighted positive-unlabelled), evaluation on the folds
   (leave-one-council-out; BW24 held out), then the patch CNN (ML-1b, GPU).
3. Wave 2 (Environment Southland 165, Auckland 101 tiles; `ml/config/wave2.yml`) on the RTX 6000 machine,
   installed with `docs/wsl/terrain_hpc.sh`.
4. ML-2 (link decisions), ML-3 (dense U-Net), ML-4 (bed elevation, benchmarked against GeoFabrics river
   bathymetry: Stage 1 work, see below). The second-pass outputs (stream REM, conditioned hydrology) move
   after ML (7m).

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
