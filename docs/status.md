# terrain-s2: status (14 Oct 2026)

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

## In progress

- **Wave 1 batch** (Matt's workstation): 83 LINZ tiles, Waimakariri 45 and HBRC 38 (`ml/manifests/wave1.csv`),
  4 workers, 2 on the GPU. First two tiles: 823 s (GPU) and 1,124 s (CPU), peak 10.1 and 9.5 GB. Tiles run
  before the step 4 patch need one rerun with features (bundles are reused).
- **Stage 1** (Cowork): remaining W items (W3 acceptance on the shared reader, W5 on PostGIS, W7, W10, W11).

## Next

1. ML-0 step 5: dataset assembly (features + labels + folds; Parquet; dataset card).
2. ML-1: gradient-boosting candidate scorer (weighted positive-unlabelled), evaluation on the folds
   (leave-one-council-out; BW24 held out), then the patch CNN (ML-1b, GPU).
3. Wave 2 (Environment Southland, Auckland; channel-only councils) on the RTX 6000 machine.
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
