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

## 5. Open

1. **REC1 topology.** Built from the geometry (section 3). On the first live run, check the junction count in
   the provenance (`rivers.network.connections_downstream_larger` against `connections`).
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
