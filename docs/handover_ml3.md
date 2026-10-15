# Handover: terrain-s2 ML-3 (a learned channel map), from ML-3a on

For the Stage 2 agent picking this up in a new conversation (17 Oct 2026). Read this first. Then read
`docs/ml3_channel_model.md` (the design, with Matt's decisions in section 7), `docs/status.md`, and the end of
`DESIGN_NOTES.md` section 11 (the evidence, in date order).

## 1. The project in two paragraphs

terrain-s2 (https://github.com/MatthewDWilson/terrain-s2, branch `main`) is the terrain library behind the EDDIE terrain
module (GeospatialResearch Digital-Twins). Stage 1 (owned by a Cowork agent) makes DEMs, through a raster backend or
GeoFabrics. Stage 2 (you, the Stage 2 agent) finds what the DEM does not show: drains and streams, the culverts and
bridges that carry them under roads and rail, and embankments. It works from 1 m LINZ LiDAR (DEM and DSM) and national
vectors, and outputs a channel network and crossing candidates for flood modelling.

The chain is physics first (`run.network`: breaching, depressions, channel network, Tests A and B, road crossings,
conditioned hydrology, REM). Then comes machine learning. ML-0 is the dataset builder. ML-1 is the candidate scorer, a
gradient-boosting model, now 2-3.5 times the rule-based tiers. Next is ML-3, a U-Net channel map, because most lost
council culverts sit on small drains the channel map never detects.

## 2. How we work (keep to this)

- **Matt** (the user) applies everything himself. On his Windows workstation he uses PowerShell from the repository
  root `D:\EDDIE_Terrain\terrain-s2`. On the RTX 6000 machine he uses WSL bash in the distro `terrain`.
- **Deliveries are git patches** on the latest `origin/main`: `git format-patch` into `/mnt/user-data/outputs/`. Matt
  runs `git pull`, then `git am \\file\Usersm$\mwi104\Home\Downloads\<patch>`, then `python -m pytest -q`, then
  `git push`. Clone `main` fresh before each change: Matt pushes between turns, and Cowork pushes too.
- **Commands for Matt**: git in PowerShell. Files he downloads land in `\\file\Usersm$\mwi104\Home\Downloads\`.
- **File ownership**, with Cowork, the Stage 1 agent:
  - Cowork owns `stage1/`, `io/`, `key.py`, `store.py`, `client/`, `cli.py`, `eddie/`, `docker/` and `compose/`.
  - You own the rest, including `src/terrain_s2/ml/`, `run.py`, `acquire/elevation.py` and the scripts.
  - Shared files change one owner at a time: `pyproject.toml`, `environment.yml`, `sites/sources.yml`,
    `acquire/registry.py`, `README.md`, `DESIGN_NOTES.md` and `docs/status.md`.
  - Cowork's next work is the branch `stage1/rivers` (GeoFabrics river bathymetry, the ML-4 benchmark). A merge will be
    needed when Matt says it is tested; do it as an integration branch, delivered as a git bundle.
- **Record as you go**: DESIGN_NOTES section 11 (new paragraphs at the end, before "## Tested versions") and
  `docs/status.md`. Tests: `python -m pytest -q` (147 pass, 19 skip); `tests/test_static.py` checks undefined names.
- **The sandbox** has about 4 GB of RAM and 300 s per command. Run long jobs in the background (`nohup ... &`, then
  poll), in pieces (one fold per process). Matt can upload datasets; `waves12/candidates.parquet` was 80 MB+. It cannot
  reach conda-forge or download.pytorch.org (no PyTorch here); PyPI and GitHub work.

## 3. Machines and data

| | Matt's workstation | The RTX 6000 machine |
|---|---|---|
| Hardware | 16 threads, RTX 4000 Ada (20 GB) | 64 threads, RTX PRO 6000 Blackwell (96 GB); WSL sees 173 GB |
| Environment | conda `terrain-s2` (miniforge), with `TERRAIN_DATA_DIR=D:\eddie_data` and `PYTHONNOUSERSITE=1` set as conda env vars (the Windows user site-packages had a CPU PyTorch and an old joblib) | WSL distro `terrain`; micromamba environment; `~/terrain/terrain_hpc.sh` (modes: setup, update, check, start / status / stop, batch, dataset, torch) |
| Data root | `D:\eddie_data` | `~/terrain_data` (about 6 TB and 3.6 TB free on two data SSDs) |
| Holds | wave 1 (83 tiles): bundles, runs, `datasets/wave1`, models | waves 1 + 2 (349 tiles): runs, bundles (wave 1's copied without their `.tif`), `datasets/waves12`, `models/scorer_waves12` |
| Mirror of `main` | (the clone) | `~/terrain/terrain-s2` (never edited; `update` fast-forwards it) |

Under the data root:

- `tiles/<id>` holds bundles: `dem.tif`, `dsm.tif`, `*_source.tif`, `labels.gpkg`, `context.gpkg`, `roads.gpkg`,
  `aoi.geojson` and `site.json`.
- `runs/<id>` holds `network.gpkg`, `network.json`, `done.json`, `ml_features.parquet`, `ml_patches.npy`,
  `ml_labels.gpkg` and (new runs) `ml_probes.parquet`.
- `cog/` (mirrored LINZ tiles), `stac/`, `snapshots/`, `datasets/`, `models/`, `labels/manual/` (Matt's crossing reviews,
  8 GeoJSON files) and `partition/nz_tiles.gpkg`.

## 4. Where things stand (numbers to beat or keep)

**Waves.** Wave 1 has 83 LINZ tiles: Waimakariri 45 and HBRC 38, which have council culvert registers. Wave 2 has 266:
Environment Southland 165 and Auckland 101, which have channels only. `ml/manifests/wave1.csv` and `wave2.csv`; configs
`ml/config/wave1.yml` and `wave2.yml`.

**Dataset `waves12`** (345 tiles; 4 skipped, see section 6):

- 686,305 candidates.
- +4,090 positives: bridges 1,529, implied 958, KiwiRail 830, council 718, manual 55.
- -73 negatives (Matt's, all in the BW24 holdout).
- 1,085 misses: labels with no candidate within 25 m.

**Scorer** (`ml/config/scorer.yml`): HistGradientBoosting, weighted positive-unlabelled.

- *Features:* no road-specific features; one derived distance to any way. That is the key finding.
- *Over spatial blocks, out of fold:* recall@100 per tile 0.59 (tiers 0.29), AP 0.42 (0.03).
- *Over labels with a candidate:* council culverts 0.55, implied 0.73.
- *Leave one council out:* recall@100 0.30 (Waimakariri) to 0.73 (Environment Southland).
- *On Matt's y/n labels:* AUC about 0.6. His "y" are mostly small driveway and side-drain crossings, an acceptable loss.

Reports: `ml/results/scorer_waves12.md` and `scorer_waves12_blocks.md`.

**CNN** (ML-1b, `train_cnn.py`): below the scorer; the ensemble gave no gain. Deferred behind ML-3. Its likely fix: feed
it the scalar features, especially distance to any way.

**Why ML-3** (`diagnose_misses.py`, 12 Waimakariri tiles). Of 57 council culverts lost:

| Reason | Count |
|---|---|
| No channel mapped within 10 m | 43 |
| Network passes, no crossing proposed | 11 |
| Cleaned out of the network | 3 |

Lost labels sit on catchments of about 500 m² (found ones about 13 ha), behind barriers of 0.34 m (found: 0.92 m),
30-50 m from the channel map.

**In the run** (`run.network(..., crossing_model=...)`): the crossings get `p_structure`, `p_bridge` and `recorded`, and
recorded structures with no candidate are added.

## 5. ML-3a: what to do first

The design is `docs/ml3_channel_model.md`. In order:

1. **Width fields.** Probe the four channel sources for a width attribute: `scripts/probe_source.py --fields` on
   `waimakariri_channels`, `hbrc_channels`, `es_drainage` and `auckland_channels` (`sites/sources.yml`). Map any width
   field in `sources.yml` and the bundle harmonisation. Then the fallbacks: physics channel-map width, a default by type
   (drain about 1 m half-width, stream about 2 m), and wide-channel polygons for rivers.
2. **Saved rasters.** Rerun the tiles with `--rasters all` on the RTX 6000 machine. Changing the `rasters` parameter
   changes the done.json hash, so the batch reruns them:
   - `terrain_hpc.sh start <manifest> --param rasters=all`;
   - about 1 TB for 349 tiles;
   - wave 1's bundles there lack their DEM and DSM, so they rebuild from LINZ once, unless Matt copies the `.tif` files;
   - include the four tiles of section 6.

   Check that features, labels and probes are still written alongside.
3. **Snapping.** A new module, for example `ml/snap.py`, moves each council channel line within a ±15 m corridor to the
   least-cost path along the DEM bed. The cost comes from `valley_s2`, `tophat_black*` and relative elevation in the
   saved rasters. Lines with no bed in the corridor are flagged and down-weighted.
   - *Check:* most snapped length lies within 2 m of the bed.
   - The same snapping may later help the crossing labels (the M2 task).
4. **Label rasters.** Positives with a half-width, using the per-council weights: Environment Southland about 1.5,
   Waimakariri and HBRC 1.0, Auckland about 0.7.
   - *Ignored:* a 3 m ring around positives, and physics channels with no council line. In Auckland, every unrecorded
     detection is ignored.
   - *Negatives:* everything else, at a low weight. In Auckland, only flat, un-incised ground counts. Test before trying
     no Auckland negatives at all.
   - Matt's channel reviews: `exists = n` are hard negatives. Files are `network_<site>_review2.gpkg`; only SH12 has
     round-2 edits (41 spurious); all four carry round-1 `prev_exists` / `prev_true_class`.
5. **Chips.** Cut from the saved rasters, mosaicked across tile edges: the rasters are cropped to the tile cores, so
   build a VRT.
   - *Size:* 128 x 128 m.
   - *Sampling:* about half on council channels, a quarter hard (physics channels with no label, lost-label crests from
     the probes), a quarter random.
   - *Bands:* about 22 (the 17-band feature stack, DSM - DEM, REM, log10 upstream area, breach depth, depression depth,
     plus a distance-to-any-way band rasterised from the bundle's vectors).
   - Write a dataset card, as ML-0 did.
6. **Folds.** Use `train.folds`: spatial blocks balanced across councils first, then leave one council out, then the BW24
   holdout.
7. **Review sets** for Matt, 2-3 tiles: where the model and the data disagree, as a GeoPackage of reaches with `exists`
   y / n / unsure. He uses the LINZ basemap himself. Only confident answers train.

Then ML-3b (the U-Net with clDice, on the RTX 6000 machine; PyTorch via `terrain_hpc.sh torch`, the cu128 build for
Blackwell), ML-3c (`run.network(..., channel_model=...)`, the end-to-end check: the share of labels "found" rises without
recall@100 falling), and ML-3d (the stream/drain head; the second-pass stream REM and conditioned hydrology move onto
the merged network, DESIGN_NOTES 7m).

## 6. Known issues and gotchas

- **Four wave-1 tiles need `--rerun`** (BW23_10000_0201, 0202, 0302, 0303). Their candidates and features disagree,
  because an old `diagnose_misses --compute` overwrote `network.gpkg` (now fixed: it uses a scratch folder).
- **CPU and GPU runs can give slightly different candidates** on some tiles, though not on the quadrant compared on
  13 Oct. Never regenerate a run's `network.gpkg` without its features and labels.
- **`TERRAIN_DATA_DIR` must be set**, or the defaults point at the home folder. Scripts now say so on stderr.
- **Shared JSON caches** (`acquire/cachefile.py`) are atomic and self-healing. That fixed seven Auckland tiles failing
  every retry.
- **Overpass** (OSM) falls back to mirrors with waits; its failures are not fatal.
- **The HPC batch:** use `start` (detached, restarted, keeps Windows awake), not `batch`, for long runs.
  - Size workers by memory: about 12 GB per LINZ tile with features, 12 workers at most.
  - The GPU pool per tile is about 4-4.4 GB.
- **WSL copies:** robocopy to `\\wsl.localhost\...` only reaches the local machine's WSL. Matt copies between machines
  with Explorer.
- **`--rasters all` output** is float32, cropped to the AOI (tile core), described in `rasters.json` with band names in
  `bands.py`.
- **`p_structure` is a ranking**, not a probability: biased PU training.
- **REC2** is context only (coarse source DEM), never a stream label.
- **The national stopbank inventory** (Matt is an author; not released) may train but must never be published or
  committed.
- **Integration-plan items still open** on the Stage 2 side: S2-6 (honest and burned DEMs), S2-7 (structures layer),
  S2-8 (tile ownership, structure ids), S2-13 (provenance). See DESIGN_NOTES section 9 for the agreed changes (ids from
  location only, `p_structure` / `p_class` / `model_version` reserved).

## 7. First message to Matt in the new conversation

Confirm you have read this and the design, then start ML-3a step 1 (width fields) and give him the step 2 command (the
rerun with saved rasters, since it runs for hours on the RTX 6000 machine).
