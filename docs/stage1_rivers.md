# Stage 1: GeoFabrics river and waterway bed estimation

Cowork, 14 Oct 2026, branch `stage1/rivers`. This is the benchmark for ML-4 (bed elevation) and part of W7
(plan section 12). The plan left these stages out of the first Stage 1 templates (section 3) "so new templates
can add them later". These are those templates, plus the network handling around them.

## 1. What GeoFabrics 1.1.30 does

Read from its source (`processor.py`, `bathymetry_estimation.py`, `runner.py`), not from its documentation.

**River stage** (`RiverBathymetryGenerator`). One main channel per run.

- **Trace.** It starts at reach `network_id` in a river network file. It then repeatedly takes the upstream
  reach with the largest upstream area: the reach whose `to_node` is this reach's `from_node`. It stops when the
  largest is below `area_threshold`. Only the main stem is followed, so a large tributary needs its own run.
- **Corridor DEMs.** It builds ground and vegetation DEMs at 1 m along a corridor `river_corridor_width` wide.
- **Alignment.**
  - *Without `osm`:* the network centreline is moved onto the lowest ground within
    `network_alignment_tolerance`.
  - *With `osm: {type, id}`:* that one OpenStreetMap waterway (a way or relation, from Overpass) becomes the
    centreline. It is clipped to the span between the network's two ends, projected onto it.
  - **GeoFabrics does not look for a matching OSM feature.** The OSM id is an input, chosen by whoever writes
    the instructions; GeoFabrics' Waikanae tests give it by hand.
  - So the "snap" is limited to two things: clipping the given OSM line to the network's ends, and taking flow,
    Manning's n and reach id for each cross-section from the nearest network reach.
- **Characterisation.** On cross-sections every `cross_section_spacing` m it measures width, bank height and
  slope.
- **Depths.** From flow, slope, width and Manning's n, by two methods: Neal et al. (uniform flow) and Rupp and
  Smart. The output `river_bathymetry.geojson` holds `bed_elevation_Neal_et_al` and
  `bed_elevation_Rupp_and_Smart`; `river_polygon.geojson` holds the channel extent.
- **Upstream bug.** `_get_up_stream_reaches` compares reach ids with dataframe index labels. With REC ids (seven
  digits) they never collide; with small ids it reports a false circular network. This is upstream issue
  GeoFabrics 6 in the W11 drafts.

**Waterway stage** (`WaterwayBedElevationEstimator`). It takes OSM drains, streams and culverts from Overpass and
sets bed elevations from the DEM, so open waterways flow downhill and culverts are unblocked. It needs no flows.

**Hydrologic DEM.** It burns each river's bathymetry points inside its polygon (`data_source` 3, "rivers and
fans"), then the waterways (`data_source` 4). The column used is `z_labels.rivers`.

## 2. How Stage 1 runs it

| Step | Where | What |
|---|---|---|
| Request | `TERRAIN_GF_RIVERS`, `TERRAIN_GF_WATERWAYS`, or request parameters `rivers`, `waterways` | GeoFabrics backend only; off by default |
| Network | `stage1/rivers.py` `plan()`, at prepare time | Fetched for the grid plus `margin_m` (1 km), from an ArcGIS layer or item, or a local file. Columns mapped to GeoFabrics' names; reaches clipped to the window |
| Rivers | `select_rivers()` | Reaches at or above `area_threshold_m2` (10 km²), largest first. Each one not already on a chosen stem starts a stem: first the largest river's lowest reach in the window, then large tributaries' mouths. Stems that miss the grid are skipped; at most `max_rivers` (5) |
| Key | `source_version.rivers` | The network source, a digest of the mapped reaches, the threshold and the chosen rivers. The templates' sha256 go in `configuration` |
| Runs | `gf.bed_stages()` | One GeoFabrics call per river in `rivers/<network_id>/`, then the waterways in `waterways/`, then the DEM (and roughness), which lists them in `data_paths.rivers` / `waterways` |
| Failures | | A river that fails is logged, recorded in the provenance (`rivers.runs`) and left out of the DEM. If every river fails, the build fails, so no river-less DEM is cached under a rivers key |
| Parameters | `templates/rivers.json`, `templates/waterways.json`; profile `rivers.parameters`, `waterways.parameters` | Starting values from GeoFabrics' Waikanae tests, scaled down for smaller channels (corridor 150 m, width 5-60 m, banks 0.5-2 m) |

The product folder gains:

- `rivers/network.geojson`;
- `rivers/<id>/`: GeoFabrics' outputs, including `aligned_river_centreline_*.geojson`,
  `river_characteristics.geojson`, `river_polygon.geojson`, `river_bathymetry.geojson`, the corridor DEMs, and
  that river's `instructions.json`;
- `waterways/`.

## 3. The river network (REC1 with flows)

REC2 has upstream area but no flows per reach, so the network is REC1. It is the ArcGIS Online item
`548b3bfcff8f4728b0f64caf6f815233` (Matt, 14 Oct 2026). Its reaches are approximate, which is what the
alignment step is for.

**Confirmed by the probe** (`bash $S rec1`, 14 Oct 2026).

- **Layer.** The item is NIWA's public Feature Service "NZ Flood Statistics (Henderson Collins) on REC1", layer
  `Flood_Statistics_2018_REC1` (polylines, 2,000 records per page). Stage 1 queries it per AOI. 136 reaches lie
  within about 4 km of SH12.
- **Fields.**
  - `NZREACH`: the REC1 reach id;
  - `Areakm2`: upstream area in km², converted to m² for GeoFabrics;
  - `H_C18_MAF`, then `H_C18_5_yr` to `H_C18_1000`: Henderson and Collins (2018) flood estimates in m³/s;
  - `HCse_*`: their standard errors;
  - `q100_reach`, `Strm_Order`, `Rivername`.
- **No node fields.** The layer has no `FROM_NODE` or `TO_NODE`, so the topology is built from the geometry
  (`rivers.topology()`):
  - reach ends within 1 m share a node;
  - each reach is turned to run downstream, judged from the upstream areas at its two ends;
  - the counts go into the key record: reversed, undecided, and junctions where the downstream reach is the
    larger. This checks the direction rule on real data.
- **Mapping** (profile): `id: NZREACH`, `area: Areakm2` (`area_units: km2`), `flow: H_C18_MAF`, with
  Manning's n a constant 0.035.

**Which flow.** GeoFabrics' depth estimate is for bank-full conditions. The mean annual flood (`H_C18_MAF`) is
the usual bank-full proxy. GeoFabrics' own Waikanae test network carries one `flow` per reach in the same way
(`rec1_flow_and_friction.geojson`: `NZREACH`, `CATCHAREA` in m², `to_node`, `from_node`, `flow`, `n`). W7 can try
`H_C18_5_yr` as a sensitivity check.

## 4. Running it

```bash
bash $S rec1                                                     # once: fields, then set them in the profile
bash $S e2e geofabrics 1 auto rivers=true                        # SH12, rivers only
bash $S e2e geofabrics 1 auto rivers=true waterways=true         # rivers and OSM waterways
```

**What to check.**

- **Selected rivers:** the worker log shows the rivers selected (`rivers: ... river(s) above 10 km2: [...]`) and
  a time for each.
- **Provenance:** `rivers.runs` lists each river with ok or error.
- **Data source:** `data_source_counts` should show 3 (rivers) and, with waterways, 4.
- **In QGIS:** `rivers/<id>/river_polygon.geojson` over the DEM, and the aligned centreline against the channel.

## 4a. First live run (SH12, 1 m, 14 Oct 2026)

- **Selection:** 28 reaches in the window; one river above 10 km², reach 1011466.
- **River stage:** ok in 106 s.
- **Failure:** the hydrologic DEM then failed while interpolating the river bathymetry:
  `ValueError: setting an array element with a sequence` in `elevation_from_nearest_points`. GeoFabrics'
  default river interpolation (RBF) returns a 1-element array per cell, which NumPy 2.4 no longer accepts as a
  scalar.
- **Fix:** the worker pins `numpy<2.4`, and the build checks it. Drafted as upstream issue GeoFabrics 7.

## 4b. Second run: end to end (SH12, 1 m, geofabric, 14 Oct 2026; product 11)

**Network.** 28 REC1 reaches were fetched. The topology built from the geometry checks out: at all 23 junctions
the downstream reach is the larger, 0 reaches were reversed, and 2 isolated reaches (clipped at the window edge)
were left undecided. One main stem was traced: reaches 1011677, 1011577 and 1011466; 32.5 to 43.0 km²;
`H_C18_MAF` 76 to 81 m³/s.

**River stage, 72 s.** 359 cross-sections, of which 312 have results. Width 7 to 21 m (median 10.8), bank height
median 2.5 m. Depths:

| Method | Median depth | Maximum depth | Bed elevation range |
|---|---|---|---|
| Rupp and Smart | 3.2 m | 7.1 m | −4.5 to 7.0 m |
| Neal et al. | 3.0 m | 9.1 m | −6.3 to 7.9 m |

The river polygon covers 22,600 m² in 3 parts (gaps where cross-sections failed); 10,706 cells in the grid
carry `data_source` 3.

**Assessment (Matt and Cowork).**

- **The method works end to end.** Against the NIWA `8m_geofabric.nc` the result is somewhat better, but the
  same data-driven errors appear:
  - *A meander bend is cut off.* REC1's centreline is a 30 m staircase, and the bend lies further from it than
    `network_alignment_tolerance` (40 m).
  - *The side channel is missing.* It isn't in REC1, or doesn't connect under SH12 in REC1. The waterways stage
    (OSM) may add it if OSM maps it.

  These are what Stage 2 is for. In Stage 1 the remedies are an OSM alignment per river (`rivers.osm`) and the
  waterways stage.
- **Depths are probably too deep for this lowland, tidal reach.** The mean annual flood is routed through an
  11 m channel at a slope of at most 0.0001, whereas floodplain flow carries part of it. W7 sensitivity: a
  smaller flow (e.g. a bank-full fraction), and the `area_adjusted_depth_*` columns, which GeoFabrics also writes
  (median about 2 m).
- **Slow: 1,759 s in GeoFabrics in total, against 72 s for the river stage.** Where the rest went is to be read
  from the stage logs (`bash $S timeline 11`). GeoFabrics grids, burns rivers and computes roughness with
  per-cell Python loops (`dem.elevation_from_points`, `elevation_from_nearest_points`, `roughness_from_points`:
  a KD-tree query, then a loop over output cells).
  - *River burn:* per cell, an RBF fit to up to 200 points (100 bathymetry and 100 edge). Benchmarked here at
    about 2 ms a cell, so about 25 s for 10,706 cells. That alone doesn't explain 28 minutes.
  - *Next candidates:* the 1 m roughness stage, which had not been run at 1 m before, and the edge-height
    interpolation.
  - The dask heartbeat errors in the log come from a worker held by such a loop for minutes; they are harmless.
- **For Stage 2.** The depth formulas themselves (Neal et al. uniform flow; Rupp and Smart) are a few
  vectorised lines over the cross-sections and cost nothing. The cost is in building the corridor DEMs from LAZ
  and in burning points cell by cell; one vectorised interpolation call does 10,706 cells in 0.3 s here.
  GeoFabrics is GPL-3.0, so Stage 2 should implement the formulas from the papers, not copy GeoFabrics' code.

## 4c. Where the 28 minutes went, and the fix (15 Oct 2026)

The stage logs (`bash $S timeline 11`) for product 11:

| Stage | Time |
|---|---|
| River stage (corridor DEMs, alignment, 359 cross-sections, depths) | 70 s |
| Raw DEM, 1 m | 44 s |
| Hydrologic DEM | 26 min 38 s |
| Roughness, 1 m | 46 s |

Inside the hydrologic DEM, almost all of the time is the river burn.
`dem.elevation_over_chunk_from_nearest` evaluates **every cell of every chunk the river polygon touches**, and
only clips to the polygon afterwards. With 800-cell chunks, that is the whole 995 × 725 grid: about 721,000 cells.
Each goes through a per-cell RBF fit to up to 200 points (100 bathymetry and 100 edge), about 2.2 ms a cell.
721,000 × 2.2 ms is about 26 minutes, the time measured. At a full 1:10k tile (35 M cells) this would take hours.

**Fix (Stage 1, no change to GeoFabrics).**

- **Small chunks for the burn.** When rivers or waterways are on, the hydrologic stage runs with small chunks
  (`rivers.burn_chunk_m`, 50 m), so only chunks near the channel are evaluated.
- **Raw DEM made separately.** The raw DEM is first made in its own GeoFabrics call with the usual chunks, because
  it reads the LAZ tiles chunk by chunk. In that call `result_dem` is named as the raw DEM, so the runner skips
  the hydrologic stage.
- **Interpolation option.** `rivers.interpolation` can switch the burn from RBF (GeoFabrics' default) to `idw`,
  about 0.03 ms a cell against about 2 ms, at some cost in smoothness. It stays on RBF to match GeoFabrics.
- **Drafted upstream:** GeoFabrics 8.

Roughness at 1 m took 46 s, so it is not the cost. Leaving it out by default is a design choice for after
Stage 2: roughness is needed at the model's resolution, and could come from the 1 m DEM and DSM rather than the
point cloud (Matt).

**Overpass.** The waterways stage, and roughness, need the Overpass API. While overpass-api.de was down (15 Oct
2026) the waterways run failed: GeoFabrics reports any download failure as "Did not successfully download the OSM
data in 3 attempts". Another instance can be named: `TERRAIN_OVERPASS_URL` in `.env`, or the profile's
`osm.overpass_url`. For example `https://overpass.kumi.systems/api/` or
`https://maps.mail.ru/osm/tools/overpass/api/`.

## 4d. W7 runs: four AOIs at 1 m, with and without rivers (15 Oct 2026)

`bash $S w7 1` on commit `13ddb4c` (`ec5d9b6` as applied): each example AOI as raster, geofabrics and geofabrics
with rivers (product `dem`; waterways off, since Overpass was down). Results in `D:\EDDIE\w7_1m_20261015_0902`.

**Runtime (wall clock per request, s).**

| AOI | raster | geofabrics | + rivers | River (REC1 id, catchment, MAF) |
|---|---|---|---|---|
| SH12 | 10 | 81 | 311 | 1011466, 43 km², 76-81 m³/s |
| aoi2 | 10 | 80 | 302 | 1011466 |
| canterbury1 | 10 | 533 | 1,549 | 13040752, 65 km², 14-17 m³/s |
| canterbury2 | 11 | 321 | 291 | none selected |

GeoFabrics stages (`timeline`):

| Stage | SH12 before the fix (product 12) | SH12 (15) | aoi2 (18) | canterbury1 (21) |
|---|---|---|---|---|
| River stage | 1:20 | 1:43 | 1:27 | 8:34 |
| Waterways | 1:20 | off | off | off |
| Raw DEM | 0:42 | 0:46 | 0:41 | 5:29 |
| Hydrologic DEM | **30:10** | 1:56 | 1:53 | 10:43 |
| Hydrologic DEM without rivers | 0:18 | 0:18 | 0:18 | 1:59 |

- **The burn fix works.** On SH12 the hydrologic stage went from 30 min to 2 min, and the whole river run from
  about 34 min to 5 min. The burn itself is now about 1.6 min at SH12 and aoi2.
- **What is left is per chunk, not per cell.** The burn now costs about 0.3-0.4 s of wall clock per 50 m chunk
  of the whole grid: 300 chunks at SH12 (97 s), 1,316 at canterbury1 (about 8.7 min). GeoFabrics schedules every
  chunk, and evaluates every cell of each chunk the river polygon touches. A larger AOI therefore pays roughly in
  proportion to its area, whatever the river length. Tuning `burn_chunk_m` only moves this between the two
  terms; the real fix is to evaluate only the cells inside the polygon (GeoFabrics issue 8).
- **The river stage re-reads the LiDAR.** It builds its own ground and vegetation DEMs over the river corridor.
  At canterbury1 that is four passes over two surveys (about 6.5 of its 8.5 min), on top of the raw DEM's own
  read. Most of canterbury1's extra 17 minutes is this and the burn.
- **For Stage 2:** both costs go away if the channel estimation reads the 1 m raw DEM once and burns only the cells
  in the channel polygon (vectorised: 10,706 cells in 0.3 s, section 4b). Stage 1 keeps GeoFabrics' behaviour.

**DEM agreement, geofabrics − raster (1 m, all valid cells; 0 gaps in all twelve products).**

| AOI | median | p5 | p95 | max abs | > 0.2 m |
|---|---|---|---|---|---|
| SH12 | 0.001 | −0.095 | 0.129 | 2.41 | 4.4 % |
| aoi2 | 0.002 | −0.103 | 0.118 | 3.63 | 4.4 % |
| canterbury1 | 0.000 | −0.014 | 0.023 | 2.19 | 1.1 % |
| canterbury2 | 0.000 | −0.016 | 0.020 | 3.22 | 0.9 % |

No bias anywhere. In Canterbury (flat, open) the two agree to about 2 cm. In Northland the spread is about ±0.1 m
with 4 % of cells beyond 0.2 m, which is what we would expect from different ground interpolation under
vegetation and over water. The raster product labels every cell `data_source = 5`, so these reports could only
break down by A. `terrain compare` now also breaks down by B's `data_source`, which separates GeoFabrics'
interpolated cells (0) from LiDAR cells (1); rerun with `bash $S compare <raster id> <gf id>`.

**Rivers, rivers − geofabrics.** Changes are confined to the channel: 10,625 cells at SH12 (1.5 %), 6,423 at aoi2,
13,652 at canterbury1. At canterbury1, 10,819 of these were interpolated cells (`data_source` 0, water with no
ground returns) and 2,833 LiDAR cells. Over the interpolated cells the burn lowers the bed by up to 5 m at SH12,
6.3 m at aoi2 and 2.4 m at canterbury1. canterbury1's main stem: 10 reaches, 674 bathymetry points, widths 10-33 m
(median 14 m), Rupp & Smart depth 1.1-2.3 m (median 1.9 m) below bank, bed 0.6 m median elevation.

**canterbury2: no river.** The REC1 window had 34 reaches (28 junctions, all consistent), but none with a
catchment of at least 10 km² (`area_threshold_m2`) meets the grid, so no river was run. Its data are identical
to the geofabrics product. The product still got its own key (the empty river plan is in the source version), so
it rebuilt the raw DEM (291 s) rather than reusing product 23. For small streams and drains, the route is the
waterways stage (OSM), or a lower threshold per request.

## 5. Open

1. **REC1 topology.** Built from the geometry (section 3). Confirmed on the W7 windows: 23/23, 28/28 and 41/41
   junctions consistent, no reaches reversed.
2. **OSM alignment.** GeoFabrics needs the OSM id per river (profile `rivers.osm: {<network_id>: {type, id}}`).
   Choosing it automatically (the OSM waterway that best overlaps the REC stem) is a small addition once we have
   seen the DEM-only alignment on SH12.
3. **Parameters.** Widths, bank heights and corridor are channel-specific. W7 sets them per river class, against
   surveyed cross-sections where any are available.
4. **Smart Ideas' `8m_geofabric.nc`.** Its instructions were not recorded (DESIGN_NOTES 4e), so the comparison is
   of DEMs only, made with these settings. The aim is to improve on it, not to reproduce it (Matt).
5. **Sources.** The network lives in the Stage 1 profile for now, so `sites/sources.yml` (shared) is not touched
   while ML-0 runs. ML-4 needs the same flows: when it starts, move the entry to `sources.yml` in one patch under
   the one-owner rule, and point the profile at it.
