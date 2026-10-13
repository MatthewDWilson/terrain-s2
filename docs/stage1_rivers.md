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

**To be confirmed on the workstation.** Run `bash $S rec1`. It shows:

- whether the item is a service (queried per AOI) or a file (download it once and point
  `rivers.network.source` at `{kind: file, path: /terrain_data/rec1/<file>}`);
- its fields with sample values;
- the reaches around SH12, saved for QGIS.

The profile then needs the field for `flow`, and for Manning's n if the layer has one; otherwise a constant
0.035 is used. Until `flow` is set, a rivers request stops and lists the layer's fields.

**Which flow.** GeoFabrics' depth estimate is for bank-full conditions. Its Waikanae test network
(`rec1_flow_and_friction.geojson`: `NZREACH`, `CATCHAREA`, `to_node`, `from_node`, `flow`, `n`) carries one
`flow` per reach. The mean annual flood is the usual bank-full proxy, so choose that if the layer has it.

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

## 5. Open

1. **REC1 fields and flow** (section 3), from `bash $S rec1`.
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
