# terrain-s2 — EDDIE terrain Stage 2 prototype (Phase 2a)

DEM-only detection of drainage crossings (culverts kept as ground, unremoved bridges) from a
1 m bare-earth DEM, per `EDDIE_terrain_stage2_design.md`. Research code: the core algorithm
first, production scaffolding later. Package name is a working name pending TR-5.

Input is any previously generated raster DEM in a projected metric CRS: Stage 1 output
(Input A), the LINZ 1 m DEM (Input B), or another national DEM. A DSM on the same grid is
optional.

## Environment

```bash
conda env create -f environment.yml          # CPU
conda env create -f environment-gpu.yml      # GPU workstation (edit cuda-version first)
conda activate terrain-s2                    # or terrain-s2-gpu
pytest                                       # 25 tests; GPU parity is skipped without CuPy + CUDA
```

The package does not need installing: tests and scripts use `src/` directly. If you do install
it, use `pip install --no-deps -e .`, so pip cannot pull binary wheels into the conda environment.

**Windows: install the PATH guard** (once per environment, after creating it):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_path_guard.ps1
conda deactivate; conda activate terrain-s2
```

conda's Python adds PATH folders to the DLL search, and Windows searches them in an unspecified
order, so other GIS software on PATH (LAStools, a standalone GDAL, netCDF) can supply its own,
older `gdal.dll`, `geos_c.dll`, `sqlite3.dll` or `libcurl.dll` and break rasterio. The guard hides
such folders while the environment is active, lists what it hid, and restores PATH on
deactivation; other shells are unaffected. Tools that run Python without activating the
environment bypass it.

**Troubleshooting.** If `tests/test_io.py` fails (or rasterio will not import) while the algorithm
tests pass, the GDAL stack is broken or shadowed. Run `python scripts/check_env.py`. On Windows,
"DLL load failed … The specified procedure could not be found" almost always means a same-named
DLL from another installation on PATH (QGIS, OSGeo4W, PostgreSQL/PostGIS, ArcGIS, R), or a pip
wheel mixed into the environment. `python scripts/diagnose_dll.py` names the exact DLL and the
missing functions. To confirm a PATH conflict, try a minimal PATH in `cmd`:

```bat
set PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\Scripts;C:\Windows\System32;C:\Windows
python -c "import rasterio._warp; print('ok')"
```

For an exact lock (recommended once the repo exists):
`conda-lock -f environment.yml -p linux-64 -p win-64 -p osx-arm64`, then
`conda-lock install -n terrain-s2 conda-lock.yml`.

## Test site (Whirinaki / SH12)

```bash
python scripts/run_aoi.py --dem data/AW27.tif --dsm data/AW27_dsm.tif \
    --aoi examples/aoi_whirinaki_sh12.geojson --buffer 200 --out out/sh12 --device auto
```

Only the AOI + buffer window is read from the 1:50k tile.

**CRS.** Inputs need a projected horizontal CRS in metres; compound CRSs such as NZTM 2000 +
NZVD2016 (`EPSG:2193+7839`) are handled directly. LINZ elevation is native EPSG:2193 (the S3
collections are published under `.../2193/`). The LINZ Data Service *reprojects* on export into the
CRS chosen at download, so a tile downloaded as EPSG:9528 (NZGD2000 + NZVD2016, geographic) has
been resampled to degree pixels: download in EPSG:2193 instead (heights are unchanged, still
NZVD2016; only the vertical label is absent, and `--assume-crs EPSG:2193+7839` restores it). If
only a geographic or foreign-CRS copy exists, `--reproject-to EPSG:2193 [--resolution 1
--resampling bilinear]` warps it (horizontal only; vertical label kept), at the cost of a second
resampling. `--assume-crs` relabels without reprojection and refuses data that really are
geographic; `--aoi-crs` relabels the AOI. Several `--dem` tiles can be
given; they are mosaicked over the window, which is how neighbours are included near tile
edges. A warning is printed if the window extends beyond the supplied tiles.

Outputs in `out/sh12/`:

| File | Content |
|---|---|
| `candidates.gpkg` | `candidates` (Test A, Test B or both, field `test`); `channel_obstructions` (false barriers inside channels, for DEM conditioning); `test_a` (every gap-bridging path, with `kind`, flags and features); `breaches` (every breach ≥ 0.1 m, with `kind`, flags and features); `breach_paths` (full pit → outlet paths) |
| `features.tif` | One band per feature (band descriptions set): median relief, top-hats, Hessian measures, openness, breach depth, depression depth, log10 upstream area, channel mask, channel centreline, DSM − DEM |
| `quicklook.png` | Hillshade with numbered candidates by test; channel map with Test A bridging paths |
| `run.json` | Provenance: inputs, grid, parameters, timings, package versions |

To iterate without running locally, clip the window and share the two small files
(a few MB each):

```bash
python scripts/clip_window.py --aoi examples/aoi_whirinaki_sh12.geojson --buffer 200 \
    --src data/AW27.tif --src data/AW27_dsm.tif --out data/clips
```

## Reviewing candidates

```bash
python scripts/review_candidates.py --run out/sh12 --dem data/AW27.tif --dsm data/AW27_dsm.tif \
    --labels earlier_review.geojson --out out/sh12/review
```

Writes `candidates_for_review.geojson` (one point per candidate, numbered as in the quick-look,
with blank label fields; labels from earlier reviews within 10 m carried across) and galleries of
80 m chips for Test A, Test B and channel obstructions.

Score a run against labelled review files (precision per test, labelled culverts found or
missed, and unlabelled candidates written out for the next review):

```bash
python scripts/score_labels.py --run out/aoi2 --labels aoi2_candidates_for_review.geojson \
    --out out/aoi2/unlabelled.geojson
```

Rank candidates with a learned model (leave-one-site-out evaluation; needs runs made with
`run_aoi.py --no-gates` and a `sites.json` listing runs, label files and inventories):

```bash
python scripts/rank_candidates.py --config sites.json --out out/ranking
```

## Network products (drains, streams, crossings)

```bash
python scripts/run_network.py --dem dem.tif --dsm dsm.tif --aoi aoi.geojson [--roads roads.gpkg] --out out/site
```

Writes `network.gpkg` (layers `channels`, `crossings`, `repairs`, `rivers`, `floodplain`); see the
script's docstring. Processing is restricted to the floodplain (HAND ≤ `--max-hand`, default 10 m;
`--max-hand 0` disables) and the network is cleaned of small isolated pieces (`--no-clean` keeps
them).
`scripts/train_stream_model.py` trains and evaluates an optional stream/drain model on reference
channels (the default is a transparent prior plus smoothing along the network).

The SH12 benchmark runs as a test when the site data are available locally:

```bash
TERRAIN_S2_SH12_DIR=/path/to/sh12 pytest tests/test_benchmark_sh12.py   # AW27_clip.tif, AW27_dsm_clip.tif, roads.gpkg
TERRAIN_S2_AOI2_DIR=/path/to/aoi2 pytest tests/test_benchmark_aoi2.py   # AW27_aoi2_clip.tif, AW27_aoi2_dsm_clip.tif, roads.gpkg
```

## Synthetic scene

`python scripts/make_synthetic.py data/synthetic` writes an SH12-like DEM and DSM with ground
truth: a culvert kept as ground (must be found by both tests), a removed bridge, a farm-track ford
and a round mound (must not be flagged). `make_divide_scene()` reproduces the SH12 configuration:
a culvert on a DEM drainage divide, which only Test A can find, plus two controls. The tests in
`tests/test_synthetic.py` encode these.

## Layout

```
src/terrain_s2/
  backend.py     NumPy/SciPy or CuPy/cupyx, selected at run time
  config.py      parameters, all in metres
  features.py    feature stack (device-agnostic)
  hydro.py       least-cost breaching with path records; fill; retained pond; D8 accumulation (CPU, Numba)
  channels.py    channel map, centrelines, channel ends, gap bridging (CPU, Numba)
  crossings.py   Test A (bridging paths) and Test B (breaches): features, kind, candidates
  pipeline.py    in-memory pipeline with timings
  dataio.py      windowed multi-tile reads, GeoTIFF writer, provenance
  quicklook.py   PNG rendering
  synthetic.py   test scene with ground truth
scripts/         run_aoi.py, run_network.py, review_candidates.py, score_labels.py, rank_candidates.py,
                 train_stream_model.py, clip_window.py, make_synthetic.py, check_env.py, diagnose_dll.py,
                 install_path_guard.ps1, migrate_to_miniforge.ps1
tests/           synthetic exit criteria, unit tests, I/O (GDAL stack), GPU parity, Whitebox cross-check
```

Licence: AGPL-3.0-or-later (TR-12).
