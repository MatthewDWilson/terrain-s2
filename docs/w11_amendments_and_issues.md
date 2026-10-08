# W11 drafts: document amendments and upstream issues (integration plan section 13)

These are drafts for the team to review. They close after W7. Amendment text is written so it can be pasted into
each document.

## 1. Document amendments

### `EDDIE_terrain_stage1_geofabrics.md` (Document 2)

- **§0 summary table.** Replace the row "NewZeaLiDAR: one consolidated line under GeospatialResearch" with
  "Retired. A library (`terrain-s2`), a pinned GeoFabrics 1.1.30 driven by instruction templates, a terrain
  worker on Celery queue `terrain`, and a compatibility shim (`terrain_s2.client.compat`) for FReDT and Smart
  Ideas." Replace "Cache: reuse only on matching …" with "Generator key (inputs, code, configuration) and a
  `terrain_product` registry".
- **§1.3, S1.1, S1.2.** Superseded by the integration plan, sections 5, 6 and 9. Keep them as history, marked
  "not done: NewZeaLiDAR retired 9 Oct 2026 (P-4)".
- **S1.5.** Raster mode bypasses GeoFabrics (P-3): the library reads the LINZ COGs and writes the contract
  itself. Keep the snapping to multiples of the resolution and the area averaging. Remove the `coarse_dems`
  step and the placeholder `dataset_mapping`. Implemented in `terrain_s2/stage1/raster.py`.
- **S1.6.** Consumers change as in plan section 10. No geospatial stack upgrade is needed for terrain.
- **S1.7.** The cache is the new `terrain_product` table with `generator_key` and `config_key`, keyed by plan
  6.5, in place of altering `user_dem`. Legacy rows are left unread.
- **S1.8.** Add the Stage 2 measures (plan section 12). Smart Ideas acceptance follows P-11.
- **§5 decisions.** TR-2: answered (b), now. TR-11 (home of `raster.py`): no longer applies; it is
  `terrain_s2.stage1.raster`.

### `EDDIE_v5_change_plan.md`

- **§1.4, D10, §2 minimum.** The minimum for v5.0.0 is:
  - the library;
  - the terrain worker on GeoFabrics 1.1.30 (optional backend) and the raster backend;
  - the generator-key cache;
  - FReDT and Smart Ideas on the shim, with NewZeaLiDAR removed.

  `eddie_terrain` can land with it, since it is thin (tested against `v4.0.0` and `v5-integration` @ `5cb3880`).
- **§6.1, §6.2, §6.5.** The terrain bullets follow integration plan section 10 and the W-list.
- **§10 Q15-Q18.**
  - Q16: one repository, two distributions (`terrain-s2`, `eddie-terrain`); names pending TR-5.
  - Q17: retire now.
  - Q18: answered by W7.
- **Appendix A.** Add "9 Oct 2026: terrain via `terrain-s2` and `eddie_terrain`; NewZeaLiDAR retired
  (integration plan)".
- **Plugin contract note (§5).** `eddie_terrain` resolves its v5 attributes lazily. v4 imports every plugin
  package in every container, so a module must not import its tasks or heavy dependencies at package import.
  Worth stating in the template repository.

### `EDDIE_terrain_plan.md`

- **§5, §7, §8.**
  - The Stage 1 route follows the integration plan.
  - TR-2 is (b).
  - The terrain library is `terrain-s2`.
  - Stage 3 is named as model-grid derivation (P-10).

### `EDDIE_terrain_stage2_design.md`

- **§6.1, §6.2.** Conditioning and the structures layer follow integration plan section 8 (honest and burned
  DEMs, `modification_source` codes 0-5).
- **§6.5.** Moves to Stage 3.
- Carry in the DESIGN_NOTES §5 amendments.

### `EDDIE_v6_roadmap.md`

- **C10, F2.** Stage 2 products are registered per tile with keys (`terrain_product`, `process_id =
  terrain.network`). Delivery follows P-9, pending hosting.

### Integration plan itself (small corrections found while building)

- **6.6 table.** Add `config_key` (text, indexed) and `grid` (geometry). They support spatial reuse when only the
  inputs differ (DESIGN_NOTES §10, decision 2).
- **6.3 template.**
  - `data_paths.local_cache` is the product folder and `downloads` an absolute path, because GeoFabrics
    requires the result folder inside `local_cache`.
  - `datasets` and `dataset_mapping` go in `default`, so the roughness stage sees the point clouds.
  - `chunk_size` scales with the resolution.
- **6.4.** The land polygon is clipped to the product grid + 100 m (not the AOI + 100 m), so the 200 m Stage 2
  buffer keeps its land.
- **9.2.**
  - The worker environment needs `libgdal-netcdf` (conda-forge's separate netCDF driver).
  - The `--no-deps` installs need `beautifulsoup4`, `geojson`, `lxml`, `ujson` and `geoalchemy2` from conda.
  - `TERRAIN_VERSION` and `TERRAIN_COMMIT` are required build arguments.

## 2. Upstream issues (drafts)

Raise them as issues first, and offer pull requests where the team has capacity (T6). None of them blocks the
plan.

**GeoFabrics 1. Snap the grid origin to multiples of the resolution.** `RawDem` rounds the extents to
0.01 × resolution. Aligned raster inputs (LINZ tiles on whole metres) then hit the interpolation path, unless
the caller snaps the extents. Proposal: snap the origin outward to multiples of the resolution, or make the
snapping configurable. We snap before calling GeoFabrics. With snapped extents, GeoFabrics 1.1.30 put the 1 m
and 8 m grids exactly on them (our test `test_geofabrics_runner_coarse_dem`).

**GeoFabrics 2. Use float64 coordinates.** `geometry.RASTER_TYPE = float32` also types the coordinates. Above
northing 8,388,608 m (many southern-hemisphere UTM zones), float32 resolves only 1 m, which breaks 1 m grids;
at NZTM northings it resolves 0.5 m. Proposal: float64 for `x`/`y`, keeping float32 for values.

**GeoFabrics 3. Accept a list of land files.** `create_catchment` raises when more than one land file is given.
Proposal: accept a list and union it (e.g. coastline and mangrove polygons).

**GeoFabrics 4. Area averaging when downsampling.** Coarse DEMs and patches finer than the output are linearly
interpolated at cell centres, which aliases narrow channels and embankments. Proposal: an option for area
averaging (`Resampling.average`). Also make `drop_patch_offshore` configurable for the coarse-DEM path (it is
hard-coded True).

**GeoFabrics 5. Write the valid-data extents again (optional).** Removed in 0.10.24 (PR #183). Downstream
registries used it. Proposal: an optional `data_paths.extents_out` (GeoJSON footprint of `z`). We compute it
ourselves meanwhile.

**geoapis 1. `otCatalog` query typo, and survey dates.** `lidar.S3QueryBase.query_for_datasets_inside_catchment`
sends `inlcude_federated`, so the API ignores it and applies its own default. Fix the spelling. Also allow
`detail=true`, and return the survey dates (`temporalCoverage`), so callers can rank datasets newest first.

Items GeoFabrics 1 and 4 matter only if raster input ever goes back through GeoFabrics; P-3 avoids that.

Not an issue to raise, but worth knowing: the runner skips any stage whose output file already exists, so a
truncated file from an interrupted run is taken as finished. Callers should delete outputs before a fresh run
(we do).
