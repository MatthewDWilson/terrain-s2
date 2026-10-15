# Stage 1 and `eddie_terrain`: status and proposed close-out

Cowork, 15 October 2026. Written for the thread that closes out Stage 1 and merges `stage1/rivers` into `main`
once Stage 2 is complete. Suggested place in the repository: `docs/stage1_status_closeout.md`.

Related documents:

- `DESIGN_NOTES.md` §10: Stage 1 design, and decisions 1–16.
- `docs/stage1_workstation.md`: workstation runbook (slot 5, step script).
- `docs/stage1_rivers.md`: GeoFabrics river and waterway bed estimation, the W7 results (§4d) and what is open (§5).
- `docs/w11_amendments_and_issues.md`: document amendments, and drafts of the upstream issues.
- `docs/status.md` (Stage 2 agent): the status of the shared work.
- Project doc `claude/EDDIE_terrain_stage1_implementation_status.md`: running log.

---

## 1. Summary

- **Implemented.** Stage 1 (thin EDDIE implementation, GeoFabrics processing) covers work packages W1–W6, W8 and W9,
  and the Stage 1 part of W11.
- **Live exits met on the workstation (slot 5, core `v5-integration`):** W3, W4, W5 and W9.
- **W7 at 1 m.** Run over the four example AOIs (raster, GeoFabrics, and GeoFabrics with rivers): 12 products, no
  gaps, no bias between backends.
- **Rivers.** GeoFabrics river bathymetry from REC1 flows runs end to end. The burn now takes about 2 min at SH12,
  down from 30.
- **Where the code is.**
  - `main` holds the W1–W9 work (merged from `stage1/w1-w9`).
  - Branch `stage1/rivers` holds everything since: rivers, the CLI API client, the W3 and W5 fixes, the W7
    tooling and results. Matt has applied it to `f58355f` (8 commits ahead of `main`, which has since moved
    with Stage 2 work).
- **Not yet closed:**
  - W8 against core `v4.0.0`;
  - a formal W6 report;
  - the merge, under the one-owner rule;
  - roughness off by default;
  - W11 review by the team;
  - the formal module document;
  - the v0.1.0 tag.
- **Joint track, after Stage 2.** The W7 backend decision (TR-3), because its main measure is culverts connected
  from Stage 2 runs on both DEMs.
- **Dev team.** W10.

---

## 2. What Stage 1 is (as built)

### 2.1 Two ways in, one library

| Route | Entry | Needs EDDIE? | Use |
|---|---|---|---|
| Library / CLI | `terrain dem …` (`src/terrain_s2/cli.py` → `stage1/dem.py`) | No | Scripts, notebooks, how most work runs now |
| API client | `terrain api dem\|task\|product` (`client/api.py`, standard library only) | A running instance | Calling an EDDIE instance's `/terrain` API from anywhere |
| EDDIE module | `eddie_terrain` (`eddie/` distribution): Celery tasks on queue `terrain`, blueprint routes, `TerrainConfig` | Yes (v4 or v5 core) | FReDT and Smart Ideas, through `terrain_s2.client.compat` |

Tidy-up flagged by Matt for a single-agent thread:

- the step script's name and place (`docs/wsl/eddie_step8_terrain.sh` is the slot-5 workstation harness);
- the README section describing the two ways in.

### 2.2 Backends and products

- **`raster`**
  - Reads LINZ 1 m DEM tiles through the Stage 2 shared elevation reader (S2-9):
    - newest survey first, profile priority, national mosaic as fill only;
    - per-cell survey index (`lidar_source`).
  - Every cell has `data_source = 5`.
  - About 10 s per AOI at 1 m.
- **`geofabrics`**
  - GeoFabrics 1.1.30, driven with `runner.from_instructions_dict` from `stage1/templates/*.json`.
  - Point clouds come from OpenTopography (otCatalog, `OTLAS.` identifiers), newest first in `dataset_mapping`.
  - Land from `coverage` (default), `topo50` or `file`.
  - Optional river and waterway bed estimation (§4).
- **Product `auto`** (default): `geofabric` (DEM plus roughness `zo`) with GeoFabrics, else `dem`.
  - Matt's decision: roughness should be off by default (close-out item C3, §8).
  - Roughness can come later from the 1 m DEM and DSM rather than the point cloud.
- **Output contract** (`io/contract.py`)
  - netCDF with `z`, `data_source` and `lidar_source` (and `zo` for `geofabric`), on CRS `EPSG:2193+7839`.
  - A COG copy of `z` at every resolution.
  - Provenance JSON.
- **Cache key** (`key.py`)
  - Covers the request, the source version (checksums, catalogue snapshot, river plan), the profile hash, and the
    code version and commit.
  - Every code build therefore gives new products.
  - Products live in `<product_dir>/<generator_key>/<grid_id>/`.
  - The registry is the `terrain_product` table in PostGIS (`store.py`).

### 2.3 Deployment (workstation slot 5)

- **Core:** pinned to `v5-integration`; `bash $S core v4` re-pins to `v4.0.0`.
  - `eddie_terrain` is installed in the backend and core-worker images.
  - Both images add `libgdal-netcdf`.
- **Terrain worker image:** `docker/terrain-worker.Dockerfile` and `environment-worker.yml`.
  - conda-pack `/venv`; Celery `-P threads`; `TERRAIN_CELERY_LOGS=1`; `PROJ_DATA` and `GDAL_DATA` set.
  - Pins: numpy < 2.4 and SQLAlchemy < 2.1.
  - PDAL plugins with missing libraries are dropped at build time.
- **Volumes:**
  - cache: `~/eddie/data/terrain/cache` = `/terrain_data`;
  - products: `~/eddie/data/terrain/products` = `/terrain_products`, mounted read-only in the core containers.

**Step script modes** (`S=~/eddie/scripts/eddie_step8_terrain.sh`):

- Setup and checks: `setup`, `build`, `check`, `bootstrap`, `branch`, `core v4|v5`, `test`.
- Requests: `e2e`, `task`.
- Products: `products`, `export`.
- Checks for particular packages: `edge` (W3), `shim` (W5), `otcatalog` (W4), `rec1`.
- Diagnostics: `read` (elevation reader for an AOI) and `timeline` (GeoFabrics stage timings).
- Evaluation: `w7` and `compare` (W6 harness).

---

## 3. Work package status

Integration plan §11. Owner in brackets.

| WP | Exit (plan) | Status | Evidence / what remains |
|---|---|---|---|
| W0 (dev team) | Baseline archived | Not seen by Cowork | Needed for W7's option 1 regression and W10 |
| W1 (Cowork + S2) | Both distributions install; tests pass | **Met** | Merged to `main` |
| W2 (Cowork) | Contract round-trip incl. GDAL `netcdf:"…":z`; key determinism | **Met** | Tests on `main` |
| W3 (Cowork; S2-9) | SH12 1 m and 8 m; survey edge with no gaps; checksums in provenance | **Met** (14 Oct) | Edge pair BV24_10000_0104/0204 (canterbury_2018-2019 + hurunui-rivers_2013): 0 gaps, checksums 3/3 |
| W4 (Cowork) | SH12 1 m and 8 m; `zo` present; `dataset_mapping` newest first; otCatalog date check | **Met** | Products 3 (8 m) and 11 (1 m, `zo` present); otCatalog fixture `tests/fixtures/otcatalog_sh12_20261012.json` |
| W5 (Cowork) | Cache hit on identical request; new product on changed settings; shim returns what FReDT expects | **Met** | `bash $S shim` from the core worker (needs `libgdal-netcdf` in the core image) |
| W6 (Cowork) | CLI; harness report for SH12 comparing two runs | **Effectively met**; formal entry pending | `terrain compare`, step-script `compare <a> <b>`. The W7 reports are harness reports. Close-out item C2 records a formal SH12 pair across code versions |
| W7 (Matt + Cowork) | Report; default backend decided (TR-3) | **Partly done** | DEM agreement and runtime at 1 m done (§5). Remaining: Stage 2 scoring on both DEMs, R-c (`topo50` land), 8 m runs, coastal site, option 1 regression |
| W8 (Cowork) | §9.3 against core `v4.0.0` and `v5-integration` | **v5 met live; v4 sandbox only** | Close-out item C1 |
| W9 (Cowork) | Full stack starts; SH12 completes; no host ports | **Met** | Slot 5 since 12 Oct |
| W10 (dev team) | FReDT and Smart Ideas PRs; §10 acceptance | Not started | Consumers need `libgdal-netcdf` (W11 §10) |
| W11 (Cowork) | Drafts reviewed by the team | **Drafted** | `docs/w11_amendments_and_issues.md`: amendments, plus upstream issues GeoFabrics 1–8 and geoapis 1 |
| W12 (Stage 2 agent) | §7 and §8.4 acceptance | In progress (about half way, ML-0 batch) | `docs/status.md` |

---

## 4. Rivers and waterways (branch `stage1/rivers`)

Full detail is in `docs/stage1_rivers.md`.

### 4.1 Settings and network

- **Settings and switches.**
  - Library settings: `rivers` and `waterways`.
  - EDDIE settings: `TERRAIN_GF_RIVERS` and `TERRAIN_GF_WATERWAYS`.
  - Request fields: `rivers=true` and `waterways=true`.
  - CLI: `--rivers` and `--waterways`.
  - Both require the GeoFabrics backend, and both are off by default.
- **Network: NIWA REC1 flood statistics.**
  - ArcGIS item 548b3bfcff8f4728b0f64caf6f815233, layer `Flood_Statistics_2018_REC1`.
  - Field mapping: id `NZREACH`, area `Areakm2` (km²), flow `H_C18_MAF`; Manning's n is constant at 0.035.
  - The layer has no node fields, so `stage1/rivers.py` builds the topology from the geometry: ends snapped to
    1 m, each reach oriented downstream by the areas of its neighbours.
  - Junctions were consistent in every W7 window (23/23, 28/28, 41/41).

### 4.2 How the runs work

- **River selection.**
  - Reaches with a catchment of at least 10 km² (`area_threshold_m2`) that meet the grid; at most 5 rivers.
  - Largest reach first, then tributary mouths.
  - One GeoFabrics `RiverBathymetryGenerator` call per main stem, which follows the largest upstream branch, as
    GeoFabrics does.
  - Burned label: `bed_elevation_Rupp_and_Smart`.
- **OSM.**
  - GeoFabrics has no automatic snap to OSM: alignment needs an explicit `osm:{type,id}` per river (profile
    `rivers.osm`).
  - Waterways use Overpass. The endpoint can be set with `TERRAIN_OVERPASS_URL` or the profile's
    `osm.overpass_url`.
- **Burn performance fix.**
  - The raw DEM is made first in its own GeoFabrics call.
  - The hydrologic stage then runs with 50 m chunks (`rivers.burn_chunk_m`).
  - An optional `rivers.interpolation: idw` is available; the default stays RBF, to match GeoFabrics.

### 4.3 Assessment and decisions

- **SH12 assessment (Matt).**
  - The meander cut-off and the missing side channel come from the REC1 and OSM data. They are for Stage 2 and not
    to be fixed in Stage 1.
  - The depths look too deep for MAF through an ~11 m channel: Rupp & Smart median 3.2 m.
- **For Stage 2.** Implement the Neal et al. and Rupp & Smart depth formulas from the papers rather than reusing
  GeoFabrics' code (GPL-3.0). Then:
  - read the 1 m raw DEM once;
  - burn only the cells inside the channel polygon, vectorised (10,706 cells in 0.3 s in a prototype).
- **Sources.** The network stays in the Stage 1 profile (`profiles/nz.yml`). When ML-4 needs the same flows, move it
  to `sites/sources.yml` in one patch under the one-owner rule, and point the profile at it.

---

## 5. W7 results at 1 m (15 Oct 2026)

Run: `bash $S w7 1`. Code commit `13ddb4c`, GeoFabrics 1.1.30, waterways off (Overpass was down). Output:
`D:\EDDIE\w7_1m_20261015_0902`.

### 5.1 Runtime

Wall clock per request, in seconds.

| AOI | raster | geofabrics | + rivers | River |
|---|---|---|---|---|
| SH12 | 10 | 81 | 311 | 1011466 (43 km², MAF 76–81 m³/s) |
| aoi2 | 10 | 80 | 302 | 1011466 |
| canterbury1 | 10 | 533 | 1,549 | 13040752 (65 km², MAF 14–17 m³/s) |
| canterbury2 | 11 | 321 | 291 | none selected |

GeoFabrics stages (min:s):

| Stage | SH12 before fix | SH12 | aoi2 | canterbury1 |
|---|---|---|---|---|
| River stage | 1:20 | 1:43 | 1:27 | 8:34 |
| Raw DEM | 0:42 | 0:46 | 0:41 | 5:29 |
| Hydrologic DEM | 30:10 | 1:56 | 1:53 | 10:43 |
| Hydrologic DEM, no rivers | 0:18 | 0:18 | 0:18 | 1:59 |

**What the timings show.**

- **What's left of the burn scales with grid area, not river length.** It costs about 0.3–0.4 s per 50 m chunk of
  the whole grid. GeoFabrics schedules every chunk, and still evaluates every cell of each chunk the river touches
  (upstream issue GeoFabrics 8).
- **The river stage reads the LiDAR a second time.** It builds its own ground and vegetation DEMs for the corridor:
  about 6.5 of the 8.5 min at canterbury1, which has two surveys.
- **Canterbury is slower for GeoFabrics in general.** It has two surveys, and each pass over a survey takes about
  90 s.

### 5.2 DEM agreement: geofabrics − raster

| AOI | median | p5 | p95 | max abs | > 0.2 m |
|---|---|---|---|---|---|
| SH12 | 0.001 | −0.095 | 0.129 | 2.41 | 4.4 % |
| aoi2 | 0.002 | −0.103 | 0.118 | 3.63 | 4.4 % |
| canterbury1 | 0.000 | −0.014 | 0.023 | 2.19 | 1.1 % |
| canterbury2 | 0.000 | −0.016 | 0.020 | 3.22 | 0.9 % |

- No bias at any site.
- Canterbury (flat, open) agrees to about ±2 cm. Northland is about ±0.1 m, which fits different ground
  interpolation under vegetation and over water.
- `terrain compare` now also breaks the differences down by B's `data_source`, so they can be split into
  GeoFabrics' interpolated cells (0) and LiDAR cells (1): `bash $S compare <raster id> <gf id>`.

### 5.3 Rivers − geofabrics

- **Changes stay in the channel:** 10,625 cells at SH12, 6,423 at aoi2 and 13,652 at canterbury1. At canterbury1,
  10,819 of the changed cells were interpolated water cells (`data_source` 0 → 3).
- **The bed is lowered by up to:** 5.0 m at SH12, 6.3 m at aoi2 and 2.4 m at canterbury1.
- **canterbury1 main stem:**
  - 10 reaches, 674 bathymetry points;
  - widths 10–33 m (median 14 m);
  - Rupp & Smart depth below bank 1.1–2.3 m (median 1.9 m).
- **canterbury2: no river selected.** It has 34 REC1 reaches, but none with a catchment of 10 km² or more meets the
  grid. The data are identical to the GeoFabrics product.
  - Its key still differs, because the empty river plan is in the source version, so it rebuilt the raw DEM.
  - Option: give an empty plan the plain GeoFabrics key, so the request becomes a cache hit.
  - For small streams and drains, use waterways (OSM) or a lower threshold per request.

---

## 6. Decisions taken while building

DESIGN_NOTES §10 lists decisions 1–16. The later ones, to add to §10 at the merge:

17. **REC1 topology from geometry** (§4.1). Provenance records the counts: reversed, undecided, connections, and
    `connections_downstream_larger`.
18. **One GeoFabrics river run per main stem**, selected in Stage 1 (`rivers.select_rivers`). A failed river is
    recorded and the others carry on. The build fails only if every river fails.
19. **Burn with small chunks after a separate raw-DEM call** (`rivers.burn_chunk_m`, default 50 m). This is a
    processing setting, so it is not in the key.
20. **numpy < 2.4 in the worker.** GeoFabrics' RBF interpolation breaks with numpy 2.4 (upstream issue
    GeoFabrics 7).
21. **Overpass endpoint configurable** (`TERRAIN_OVERPASS_URL`, profile `osm.overpass_url`). The OSMPythonTools
    cache goes in the product folder; the default `./cache` was not writable.
22. **COG at every resolution** (amends plan 4.3). x and y are written without `_FillValue`.
23. **Product `auto`.** Resolved after backend fallback. To change to `dem` by default: close-out item C3.
24. **Raster no-data errors name the cause:** no LiDAR, or the land clip, with the surveys considered.
25. **otCatalog.** An `OT…` identifier takes precedence over the provider name. `discover` lists the datasets it
    dropped.
26. **Compatibility shim checks the GDAL netCDF driver** before `open_rasterio`, and fails with a clear message.
    Consumers need `libgdal-netcdf`.

---

## 7. Known issues and open questions

### 7.1 Upstream drafts (W11 §2)

- **GeoFabrics:**
  1. grid origin snapping;
  2. float64 coordinates;
  3. a list of land files;
  4. area averaging when downsampling;
  5. valid-data extents;
  6. the upstream trace compares reach ids with index labels;
  7. RBF fails with numpy 2.4;
  8. the river and waterway burn evaluates whole chunks.
- **geoapis:** 1. otCatalog query typo, and survey dates.

### 7.2 Core findings to raise with the core team

- **v5 core silences the `celery` logger** (`propagate=False`). A v5 worker shows no task errors. The terrain worker
  works around it with `TERRAIN_CELERY_LOGS`.
- **Tracebacks are stored as a list of lines** by `OnFailureStateTask`.

### 7.3 Open questions for Matt

1. **`src/terrain_s2/settings.py` ownership.** Written by Cowork in W2, but not listed in `docs/status.md`. Confirm
   it is Stage 1's.
2. **Default backend.** It stays `raster` until the W7 decision (proposed rule: GeoFabrics becomes the default only
   if it beats raster on culverts connected at three or more sites, and is no worse on channel coverage).
3. **Empty river plan.** Should it reuse the plain GeoFabrics key (§5.3)?
4. **OSM alignment for the SH12 meander.** Rerun with waterways, and an OSM id in `rivers.osm`, once Overpass is
   reliable.

### 7.4 Minor

- GeoFabrics' own log lines appear in the worker log as `WARNING celery.redirected`, because Celery redirects
  stdout. This is cosmetic: the `timeline` mode filters them.
- Raster provenance has no `data_source_counts`; every cell is 5.

---

## 8. Proposed close-out

Proposed order for the close-out thread, which starts once Stage 2 is complete. Items C1–C7 are Cowork's; C8 and C9
are joint or belong to the dev team.

**C1. W8 on core v4.0.0** (workstation).

```bash
bash $S core v4 && bash $S build && (cd ~/eddie/instances/terrain && docker compose up -d) && bash $S check
bash $S e2e raster 1 2>&1 | tee /mnt/d/EDDIE/eddie_step8_v4_e2e_raster1.txt
bash $S e2e geofabrics 1 dem 2>&1 | tee /mnt/d/EDDIE/eddie_step8_v4_e2e_gf1.txt
bash $S shim 2>&1 | tee /mnt/d/EDDIE/eddie_step8_v4_shim.txt
bash $S core v5 && bash $S build && (cd ~/eddie/instances/terrain && docker compose up -d)
```

Exit: plan §9.3 checks pass on both cores. Record the results in DESIGN_NOTES §10.

**C2. W6 formal report.** Run `bash $S compare <earlier SH12 gf 1 m id> 14`, which compares two code versions on the
same grid. Exit: the report shows differences explained (expected: none, or the river and chunk changes only).
Archive it under `docs/`.

**C3. Roughness off by default.**

- Change `product_for`: `auto` → `dem` for both backends; `geofabric` stays available on request.
- Update `TerrainConfig` docs, `.env` defaults, tests and the integration plan amendment (W11).
- One small patch, Stage 1 files only.

**C4. Merge `stage1/rivers` into `main`.**

- Rebase onto current `main` first. Stage 2 has moved on; expect conflicts only in shared files.
- Shared files touched, which need one owner for the patch: `DESIGN_NOTES.md` §10 (add §§4–6 of this note) and
  `docs/status.md` (Stage 1 rows). Possibly `README.md` (two ways in) and `sites/sources.yml`, if ML-4 needs the
  REC1 entry by then.
- Before merging: Stage 1 tests pass in the sandbox (61 at `f58355f`); `bash $S test` passes in the worker image;
  one `bash $S e2e geofabrics 1 dem rivers=true` on the merged commit.

**C5. W11 to the team.** Send the amendments and upstream drafts for review. File the upstream issues once agreed:
GeoFabrics 6, 7 and 8 matter most.

**C6. Formal module document.** As a living doc. It should cover:

- purpose;
- the two ways in;
- the request and settings reference (`TERRAIN_*`);
- the output contract;
- the cache key and registry;
- backends, rivers and deployment;
- the W7 findings;
- known limits.

Source material: DESIGN_NOTES §10, this note, `stage1_workstation.md` and `stage1_rivers.md`.

**C7. Tag `v0.1.0`** on `main` after C4, so versions read `0.1.x`, and rebuild the worker.

**C8. W7 decision (joint, after Stage 2).**

- Run Stage 2 on R-a (raster) and R-b (GeoFabrics) at the four sites, with the same `Params` and one light retune
  on R-b at SH12.
- Score with `score_site.py`: culverts connected, channel coverage.
- Add R-c (`topo50` land), the 8 m runs, the coastal mangrove site, and the option 1 regression against W0.
- Decide TR-3 by the rule in §7.3, then record the decision and close W11.

**C9. W10 (dev team).** FReDT and Smart Ideas PRs, plan §10. Core images need `libgdal-netcdf`. FReDT can then
switch to `terrain_s2.client.compat`.

**Tidy-up (single-agent thread, Matt).**

- Step script name and location.
- README section on the two ways in.
- Consider moving `docs/wsl/` tooling under `tools/`.

---

## 9. Handover notes for the next thread

- **Branch and patches.** Branch `stage1/rivers` at `f58355f`. Cowork cannot push (HTTP 403), so commits go as
  `git am` mbox patches. Matt applies them in PowerShell in `D:\EDDIE_Terrain\terrain-s2` and pushes.
- **Step script install** (WSL distro `eddie`):
  `tr -d '\r' < /mnt/d/EDDIE/eddie_step8_terrain.sh > ~/eddie/scripts/eddie_step8_terrain.sh`. The mirror follows
  `TERRAIN_S2_BRANCH` in `.env` (`bash $S branch main` after the merge).
- **File ownership** (`docs/status.md`).
  - Cowork: `stage1/`, `io/`, `key.py`, `store.py`, `client/`, `cli.py`, `eddie/`, `docker/`, `compose/` (and
    probably `settings.py`; see §7.3).
  - Stage 2 agent: the existing modules, `run.py`, `acquire/elevation.py`, `ml/`, `condition.py`,
    `structures.py`, `tiles.py`.
  - Shared, one owner at a time: `pyproject.toml`, `environment.yml`, `sites/sources.yml`,
    `acquire/registry.py`, `README.md`, `DESIGN_NOTES.md`, `docs/status.md`.
- **Test traps.**
  - `profile.load()` is cached: deep-copy before mutating.
  - Fixtures live in `tests/fixtures` (`tests/data` is gitignored).
  - River tests need seven-digit reach ids, to avoid GeoFabrics' id-versus-index bug.
  - The ML tests need `pyarrow`.
- **Keys change with every commit.** Products from different builds never share a cache entry; compare across
  builds with `compare`.
- **GeoFabrics skips stages whose outputs exist.** The library deletes partial outputs before a rebuild. Keep this
  in mind when debugging inside a product folder.