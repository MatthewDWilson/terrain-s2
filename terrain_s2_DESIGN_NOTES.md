# terrain-s2 — Design notes (Phase 2a, iteration 2)

**Prepared:** 30 September 2026; **updated** 1 October 2026 (channel network and Test A; SH12 results)
**Relates to:** `EDDIE_terrain_stage2_design.md` (§ refs below), `EDDIE_terrain_plan.md`
**Status:** Tests A and B working on synthetic data and on the SH12 test site (AW27). The SH12
culvert is found by Test A; the Whirinaki bridge is not flagged. Parameters frozen on SH12
(50 labels, two reviews); blind run on a second site (AOI2) scored: Test B 12/28 culverts, Test A
0/1; known culvert found blind. Test A changes for driveway culverts proposed (§6).

---

## 1. Code-base decision

| Component | Choice | Why | Alternatives considered |
|---|---|---|---|
| Raster features (§3) | NumPy/SciPy on CPU; CuPy/cupyx on GPU, behind a two-field backend (`xp`, `ndi`) | `cupyx.scipy.ndimage` mirrors every filter used, so one code path serves both devices. CuPy arrays pass to PyTorch by DLPack for the later CNNs | PyTorch for features: no median filter, and median via `unfold` needs ~2.6 GB per 1.5 M cells at 21 m |
| Depression breaching | Own least-cost breach (Numba), after Lindsay (2016) | Returns each breach **path**; paths become the structures-layer cutlines (§6.2). In-memory, no file round trips | Whitebox `breach_depressions_least_cost` returns only the raster |
| Fill, D8, upstream area | pyflwdir (MIT, Numba) | In-memory; nodata-adjacent cells are outlets, so the sea drains without a land polygon | RichDEM, pysheds (GPL; TR-12 prefers MIT) |
| Cross-check | Whitebox Workflows NG 2.0.6 (MIT/Apache, released June 2026, PyPI wheel) | Independent implementation of the same method | — |
| Channel centrelines | scikit-image `skeletonize`; own spur pruning and gap bridging (Numba) | BSD; conda-forge | — |
| Vectors | shapely 2, geopandas, GeoPackage | As §10 | — |

**Whitebox status (§3 tools list needs updating).** Whitebox has been rewritten as Whitebox
Workflows Next Gen (June 2026, MIT/Apache, soft launch). Its Python API uses category
namespaces with keyword-only calls, e.g.
`env.hydrology.depressions_storage.breach_depressions_least_cost(dem=...)`. It is kept as an
optional cross-check, not a core dependency, until it has matured.

**Cross-check result.** On the synthetic scene both implementations produce the same two
breaches deeper than 0.3 m, at the same places: the culvert (29 vs 28 cells; 5.39 vs 5.28 m)
and the mound (24 cells; 1.13 vs 1.17 m).

## 2. CPU / GPU split and cost

Hydrology (priority queues) is inherently sequential and runs on CPU. Raster features are
data-parallel and run on either device. CPU costs, one core:

| Step | s per M cells |
|---|---|
| Median high-pass 5 / 11 / 21 m | 0.4 / 1.8 / 6.2 |
| Top-hats (any scale; separable min/max) | 0.08 |
| Hessian, σ = 2 / 4 m | 0.09 / 0.14 |
| Openness, 20 m / 50 m, 8 directions | 0.5 / 1.1 |
| Breach + fill + D8 | ≈ 2 |
| Channel network (map, skeleton, ends, gap bridging) | ≈ 3 |
| Test B + Test A evaluation | ≈ 5 |

On the SH12 window (1.5 M cells, Windows workstation CPU) the full run takes 26 s. The default
stack is about 17 s per M cells on one core. A 1:50k tile (≈ 860 M cells) needs
about 4 core-hours, so CPU-only production is feasible with tile parallelism. The GPU mainly
pays off for the median filters and, from Phase 2c, the CNNs. **The GPU path is untested**
(no GPU in the sandbox). Run `pytest tests/test_gpu_parity.py` on the workstation first.

## 3. Algorithm as implemented

1. **Features (§3).** Median relief at 5, 11 and 21 m; white and black top-hats at 21 and 41
   m; Hessian eigenvalue ridge and valley measures; positive and negative openness. Nodata is
   nearest-filled before filtering and re-masked after.
2. **Breach (§3 hydrological).** Pit-flats are found once, then processed in ascending
   elevation. A Dijkstra search runs from each pit to the first cell lower than the pit, or to a
   boundary cell, within 100 m. Cost is the cut cross-section area (m²). Path cells are lowered to
   just below the pit, never raised. Each path is recorded with its maximum cut.
3. **Profile.** Each path with a cut of at least 0.1 m is extended 15 m upstream (main upstream
   flow path) and downstream (D8), on the breached DEM.
4. **Barrier span.** The contiguous path cells with cut > 0.05 m around the deepest cut. This
   gives h_b, L_b, crest elevation, crest width (cells within 0.15 m of the crest), and the
   crossing angle to the raised feature's axis.
5. **Elongation (both tests).** Crest following, after §5.2: from the crest, step 2 m along the
   raised feature's axis, re-centring on the highest top-hat value within ±3 m across it, until
   the feature ends or the crest turns more than 60° from its starting direction. Starting axes
   within ±45° of perpendicular to the path are tried, and the axis is taken from the traced crest
   itself. Elongation = crest run / width at half height; ≥ 3 passes.
6. **Test B.** h_b ≥ 0.3 m, L_b < 60 m, a raised feature (41 m white top-hat ≥ 0.3 m), elongated,
   and a retained pond of depth ≥ 0.3 m and area ≥ 50 m². Each breach also gets a `kind` from the
   channel network (`crossing`: channel both sides; `bank`: channel only downstream, raised axis
   within 30° of it; `overflow`: otherwise) and a vegetation fraction (barrier cells with
   DSM − DEM > 1 m). **`channel_obstruction`** takes precedence: ≥ 95% of the cutline lies within
   5 m of the channel map, i.e. a false barrier inside a channel (typically vegetation over a
   river or drain). These go to their own layer, for DEM conditioning (§6.1). A Test B
   **candidate** excludes `bank`, `channel_obstruction` and vegetation-affected (≥ 50%).
7. **Channel map (§4.1 step 1, unsupervised).** Channel if narrow (valley measure σ 2 m ≥ 0.03
   and 11 m black top-hat ≥ 0.15 m) or wide (valley measure σ 4 m ≥ 0.015, 21 m black top-hat
   ≥ 0.3 m and 11 m ≥ 0.1 m). On slopes > 10%, a channel cell must also have ≥ 1,000 m² upstream
   area. Fragments < 30 m² or with < 25 m of centreline are dropped.
8. **Centrelines and ends (§4.1 step 4).** Skeletonise; prune spurs < 10 m. Segments running
   along a raised feature (within 10 m of it and within 30° of its axis) are cut out when finding
   ends, but remain targets. Each end has a direction (over 8 m) and an approach direction and
   length (over up to 15 m).
9. **Gap bridging (§4.1 step 5).** From each end with a raised feature within 30 m ahead: a
   least-cost search (cost 1 − P_c, floor 0.05) within 60 m, to channel cells more than 150 m
   away along the network and within 60° of the end's direction. A target is accepted only if the
   path crossed a raised feature and rose ≥ 0.3 m above both ends; otherwise the search continues
   through it (so fragments of a broken channel become part of the route).
10. **Test A.** On each bridging path: beds within 5 m of each end; h_b ≥ 0.3 m; barrier span =
    cells above the higher bed + max(0.1 m, 0.25 h_b); 4 m ≤ L_b < 60 m; raised; elongated;
    crossing angle ≥ 45°. `kind` is `crossing` if the channel end approaches the barrier over
    ≥ 10 m at ≥ 45° to its axis; otherwise `bank` (a hop across a bank between parallel channels,
    or a stub). An earlier alternative, "or the channel reached is transverse", admitted five
    SH12 candidates, all labelled "no culvert", and was removed. A Test A **candidate** is a `crossing` and not
    vegetation-affected.
11. **Merge.** Within each test, crests within 10 m are merged; a Test A candidate within 10 m of
    a Test B candidate marks it `A+B`.

## 4. Synthetic validation

The scene is 800 × 600 m at 1 m. It contains a culvert kept as ground under a 4–5 m road
embankment, a removed bridge, a 0.2 m farm-track ford across both channels, a round 2.2 m
mound in the channel, a nodata "sea", and 3 cm noise (12,833 pits). The DSM keeps the bridge
deck.

- **Exactly one candidate:** the culvert, 2 m from truth, passing Tests A and B. h_b 5.4 m,
  L_b 33 m, crest width 8 m (as built), crossing angle 76°, retained pond 14,800 m² / 20,300 m³,
  DSM − DEM 0 on the crest.
- **Not flagged:** the bridge (no breach needed), the fords (cut 0.22–0.27 m), and the mound
  (breach passes around it; not elongated).
- The breached DEM drains completely: 0 unresolved pits; residual fill < 1 µm.
- The culvert is still found with the same scene at 2 m resolution (thresholds in metres).
- With the channel network, the culvert is found by **both** tests (`A+B`): Test A h_b 4.0 m,
  L_b 20 m, crossing angle 88°.

A second scene, `make_divide_scene()`, reproduces the SH12 configuration: a 3 m embankment on a
flat floodplain falling away on both sides, with ditches that start at each toe and drain away,
joined by a culvert. Nothing ponds, so **only Test A** finds it (h_b 3.9 m, crossing angle 90°).
Controls not flagged: a ditch dead-ending at the embankment with nothing opposite, and a ditch
ending in open paddock.

## 4b. Real data: SH12 (AW27 window, 1369 × 1092 m)

**Exit criteria.** The SH12 culvert (1642255.6, 6075854.7) is found by **Test A** 4.2 m away:
h_b 4.9 m, L_b 29 m, crossing angle 88°, elongation 7.8, no vegetation. The Whirinaki bridge
(1642170.4, 6075760.4) is not flagged: no candidate within 26 m.

**Why Test B misses the culvert.** The culvert sits on a DEM drainage divide: the SE ditch
drains south-east along the straight drain and the NW ditch drains north to the river, with
beds within 1 cm of each other. Nothing ponds, so there is nothing to breach. Pipes in flat,
drained floodplains are likely to behave like this generally.

**Labels on the first 18 candidates (Test B, iteration 1; labelled by Matt).** No culverts. Four
classes: stopbank gaps or low points (5), vegetation artefacts (4), slope toes / marshy ground
(4), flow paths or tracks (4); plus #13, a farm track crossing (status open). The vegetation
fraction flags all four vegetation artefacts; `kind = bank` catches three of the five stopbank
gaps. Stopbank gaps are wanted by §5.4 (embankment gaps), not by the structures layer.

**Labels on the 32 candidates of iteration 2.** One culvert (the SH12 culvert, Test A). Classes:
channel hidden by vegetation (7: the main river and drains; not structures, but DEM barriers to
remove), overland flow paths (8), tracks and roads (5), stopbanks and gaps (3), SH12 with drains
both sides but no culvert evident (3, one flagged for field inspection), other (5). First-round
#13 is a farm track passing through a gap in a stopbank (point cloud checked).

**Frozen rules, scored on SH12** (all 50 labels): Test A 4 locations, 1 culvert (precision 1/4);
Test B 14 candidates, 0 culverts; 63 channel obstructions, which absorb all seven "channel hidden
by vegetation" labels. Across both reviews Test B has found no culverts at SH12.

## 4c. Blind test: AOI2 (Whirinaki upper floodplain, AW27 window 1226 × 1247 m)

Run with the parameters frozen on SH12, before the location of the known culvert was given.
29 candidates (Test A 1, Test B 28) and 55 channel obstructions; 23 s on CPU.

**Blind score (labels by Matt).** Test B 12 culverts of 28 (+2 unsure, both stopbank/field-drain
junctions flagged for field inspection): precision 43–50%. Test A 0 of 1. The known culvert (road
over a drainage channel, 1641716.5, 6075208.5) was found blind by Test B. Nine positives are
driveway or field-entrance culverts across the small drains beside the main road (open pipes,
about 20 cm, from Streetview); two are under gravel access roads or drives at drain junctions.
False positives: floodplain micro-relief (elevated areas, slope breaks, vegetated spoil), river
bank trees, a gully.

**Why Test A missed them.** For 7 of the 14 positive/unsure locations a Test A path within 1–6 m
passed every geometric test and failed only the ≥ 10 m approach rule adopted from SH12. Roadside
drains run along the road embankment, so the toe-ditch cut removes them when finding ends; a
driveway culvert joins two pieces of such a ditch, so its ends are stubs. Where the crossing angle
failed (#2 at 43°; #16, #23 at 23°), the crest follower measured the road, not the driveway. The
approach rule overfitted SH12.

**Culvert sizes.** Several NZ councils' vehicle-crossing standards specify a 300 mm minimum
culvert (e.g. Thames-Coromandel, Horowhenua, Rangitikei, Kāpiti); Far North's 2023 Engineering
Standards were not checked. Older field entrances are smaller (≈ 20 cm here). For the structures
layer: diameter unknown from the DEM; a class prior (driveway culvert 0.3 m) with an `assumed`
flag, never a DEM breach (a 1 m breached cell would overstate capacity many times).

## 4d. Test A for ditch-interrupting crossings (attempted, iteration 3)

Changes, all principled and kept (tests pass; no labelled culvert lost at either site):
(a) channel ends from the full centreline as well as the toe-cut one, both kept where they
coincide; (b) barrier axis from the principal direction of the local crest, with a sideways
penalty when re-centring (flat crests), over 1.5 × the half-height width (4–10 m); (c) up to 3
bridging targets per end, one per separate channel, so a cheaper wrong pairing cannot pre-empt
the right one; (d) a target is the end's own channel only if reachable along the network within
5 × the straight-line distance (a paddock-drain loop no longer disqualifies the far side of a
driveway).

**Outcome, after labelling the new candidates.** A real gain: Test A found four more culverts,
all unlabelled when first reported (AOI2: a road crossing over a drain, an industrial-site entrance
and a property entrance over the roadside drain; SH12: a house access track over the roadside
drain). Scores with all labels to date: AOI2 17 of 19 labelled culverts found (two are Matt's added
inventory points), precision 15 / 36 labelled candidates (42%); Test A 3/9, A+B 1/1 (+1 unsure),
Test B 11/26. SH12 2 of 2 culverts found (both Test A), precision 2 / 21 labelled.

**The two misses are caused by the hard exclusion rules.** One was found by Test B (0.78 m) but
removed as `kind = bank` (a drain runs along the road beyond the crossing); the other is a 3 m-thick
crossing that fails Test A's L_b ≥ 4 m and was classed by Test B as `channel_obstruction` (the
channel map reaches within 5 m on both sides). `bank`, `channel_obstruction`, L_b ≥ 4 m and the
approach rule each encode one site's labels; with ~20 positives across two sites they should
become **features for the Phase 2b classifier** rather than gates.

**Earlier report (superseded).** "No measurable gain yet": Test A still co-detected 1 of the 12
AOI2 culverts (#20).
Traced at #6: the driveway crossing *is* now found (end approaching over 15 m, h_b 0.70 m,
kind `crossing`) but fails the crossing angle (39° vs 45°): the barrier axis is still measured
diagonally between driveway and road. Next: measure the barrier orientation at the crest from
the Hessian of the top-hat at the barrier's own scale (ridge direction), not by crest tracing.
New unlabelled candidates: SH12 6, AOI2 11 (`unlabelled.geojson` per site). Test B already finds
every labelled AOI2 culvert, so the combined output misses none of the labelled culverts at
AOI2; at SH12 one "unsure" flow pathway is missed.

## 4e. Stage 1 channel processing (code check, 1 Oct 2026)

- **GeoFabrics** (`rosepearson/GeoFabrics` @ `1ad676f`, Mar 2026): `RiverBathymetryGenerator` takes
  a centreline from a river network (REC: `network_file`, `network_id`, `area_threshold`) or OSM
  (`align_channel_from_osm`), smooths it with a parametric spline, aligns it to the DEM, measures
  widths on cross-sections (`ChannelCharacteristics`), smooths widths/slopes upstream with a
  rolling mean (`upstream_smoothing_factor`), takes flow and Manning's n from the nearest reach,
  and estimates depth by Neal et al. (uniform flow, Manning) and Rupp & Smart (hydraulic
  geometry). The river polygon is built from spline-fitted bank lines of those cross-sections
  (`_create_flat_water_polygon`). Drains and ditches have a separate `WaterwayBedElevationEstimator`.
  At a tight double meander, cross-sections perpendicular to a centreline that cuts the neck sweep
  across it, and the spline-fitted banks enclose both bends: the "pond" in the Smart Ideas DEM.
- **NewZeaLiDAR** (`LukeParky/NewZeaLiDAR@update_sqlalchemy` @ `4b13f20`, GeoFabrics `<=0.10.23`)
  runs only `RawLidarDemGenerator` and `HydrologicDemGenerator`, and `gen_instructions` adds no
  river data. Production DEMs therefore carry no river bathymetry (as the Stage 1 doc says).
- **Smart Ideas** (`FReDT-Smart-Ideas` @ `b18b443`) does not use NewZeaLiDAR's DEM for the
  hydrodynamic model: it reads a pre-generated `$HYDROMT_PATH/river_data/<site>/8m_geofabric.nc`
  (and 4 m roughness/Strahler files). Nothing in the repository produces it. So the channel
  processing was run offline with GeoFabrics, and its instructions (network source, width limits,
  smoothing factor) are not recorded. They are needed to reproduce or replace it.
- **Consequences** (unchanged direction): Stage 2 reads the raw 1 m LiDAR DEM, never a
  bathymetry-burned DEM; a bed estimate, if used, should be driven by Stage 2's channel map
  (centrelines and widths from 1 m, obstructions repaired) instead of a network line; at 8 m,
  narrow channels belong in LISFLOOD-FP sub-grid channels rather than widened burns.

## 5. Proposed amendments to the Stage 2 design

1. **§4.2 Test A elongation window.** A length-to-width ratio > 3 measured within 30 m caps
   the measurable length at 60 m. Embankments wider than 20 m at the threshold height can
   never pass. *Superseded by item 8.*
2. **§4.2 Test B depression.** Fill-based depression labels give any bump inside a dammed pond
   the whole pond's depth and area. In the synthetic scene, the upstream ford inherited the
   culvert pond's 14,650 m². Proposed: measure the **retained pond** of each barrier, i.e. cells
   connected to the pit below min(crest, fill level) (implemented; flagged if truncated).
3. **§3 local relief.** Use morphological top-hats at embankment scales. They are about 250×
   cheaper than a 41 m median on CPU and preserve planar slopes exactly. Keep the median
   high-pass at ≤ 21 m for channels (the Lidberg input).
4. **§3 tools.** Replace the WhiteboxTools entries with Whitebox Workflows NG (as a cross-check),
   and record the own breach implementation and its reason.
5. **§3 tiling.** An overlap of 128 cells is marginal. It must cover the breach search length
   (100 m) plus the widest filter half-width (~25 m). Proposed: 256 m. Two quantities are not
   local, and need a separate design in the data pipeline:
   - **Upstream area.** Options are a tile-graph accumulation, or a coarse global pass.
   - **Large retained ponds.** These are flagged by `pond_truncated`.
6. **§11 runbook.** The LINZ national 1 m composite is tiled on the **1:50k** grid (e.g. `AW27`,
   ≈ 24 × 36 km); survey collections use 1:10k subtiles. The reader uses windowed reads and
   mosaics neighbours.
7. **§4.1 step 5 gap bridging: all channel ends.** Bridge from every channel end at a raised
   feature, not only downstream-dangling ends. At SH12 both channels *start* at the embankment.
8. **§4.2 elongation.** Measure by crest following (as in §5.2), against the barrier's width at
   half height, with a reach proportional to that width. Tall fills across narrow valleys are not
   penalised; a round mound still scores about 1.
9. **§4.2 test site expectation.** "The SH12 crossing should trigger both tests" is wrong: it can
   trigger only Test A (§4b). Culverts on drainage divides are a Test A-only class.
10. **§4.1 gap bridging: barrier-aware targets and toe ditches.** Accept a target only behind a
    barrier (else continue through it), and cut channel segments running along raised features
    when finding ends, keeping them as targets. Both were needed for SH12 and the synthetic scene.
11. **§4.2 kind and false-positive controls.** Record `kind` (`crossing` / `bank` / `overflow`)
    and the vegetation fraction for both tests; candidates exclude `bank` and vegetation-affected.
    Test A also needs L_b ≥ 4 m (narrower barriers were fences, hedges and spoil banks). Test A's
    requirement of a channel on both sides is not sufficient on its own: banks between parallel
    channels satisfy it.
12. **§4.1 step 1 channel map.** Add the slope/upstream-area condition (GeoNet-style) on slopes
    > 10%, and a 25 m minimum centreline per fragment. Without them, hillside roughness gave about
    three times as many channel ends.
13. **§4.2 L_b limit.** At SH12 the embankment is 29 m thick along the path (52 m toe to toe
    between channel heads). Motorway embankments can exceed 60 m, so the limit should become
    context-dependent (e.g. relative to h_b) rather than fixed.
14. **§4.2–4.3 role of Test B.** At SH12 its value is not culverts but *false barriers inside
    channels* (vegetation over rivers and drains: a §6.1 conditioning product) and embankment gaps
    (§5.4). Record these as kinds and route them to those products; report culvert precision for
    Test A and Test B separately, as §4.3 already asks.
15. **§4.3 classes.** Add `channel_obstruction` (false barrier inside a channel; breach it) and
    distinguish `stopbank_gap`, `track` and `flow_path` within `false_positive`: the labels fall
    into these readily, and they route to different products.

## 6. Next steps

1. **Test A barrier orientation** from the Hessian ridge direction at the barrier's scale (§4d);
   re-score both sites; label the new unlabelled candidates (SH12 6, AOI2 11).
2. **Recall.** Candidates give precision only. An exhaustive inventory of culverts within one AOI
   (e.g. AOI2) would let us measure recall too.
3. **Stage 1 channel processing** (§4e): obtain the GeoFabrics instructions used to make Smart
   Ideas' `8m_geofabric.nc`; decide whether bed estimation moves downstream of Stage 2.
4. **Turn the hard rules into classifier features** (§4d): keep `kind`, vegetation fraction, L_b
   and approach as attributes; candidates = Test A or Test B geometry; rank by a classifier
   trained on the SH12 + AOI2 labels; evaluate on the third site.
4b. **A third site elsewhere in the country** with known culverts (§7.1 site list, open item 2 in
   §13): ideally a road culvert that ponds (Test B) and hillside track culverts.
5. **GPU parity** on the workstation, and CPU vs GPU timings from `run.json`.
6. **Embankments (§5).** Crest extraction now exists in part (crest following); the stopbank gaps
   found by Test B are candidate §5.4 gaps.
7. **Channel map quality.** Perona–Malik smoothing and GeoNet's distribution-based curvature
   threshold, to replace the fixed thresholds in item 7 of §3; channel widths for §6.3.
8. **Data pipeline design** (tile index and metadata database; neighbour handling; non-local
   quantities, see §5 item 5), then Dask over tiles. Data live outside the repository (and
   outside OneDrive), located by configuration rather than repo paths.

## 7. Automated test-data acquisition (inputs for a separate thread)

**Purpose.** Assemble labelled test sites reproducibly: given a site definition, fetch the DEM
and DSM for the AOI plus buffer, fetch ground truth, and produce the clips, the label file and a
provenance record that the core (this repository) consumes unchanged.

**Site definition** (one file per site, e.g. `sites/waimakariri_01.yml`): id, AOI polygon, buffer,
role (`train` / `validation` / `test`), elevation source and survey, ground-truth sources with
query parameters, licence notes. Sites, not individual candidates, are the unit of
train/test separation: SH12 and AOI2 are training sites, the Waimakariri AOIs the first test sites.
Spatial blocking matters: never mix candidates of one site across splits.

**Elevation sources**
- *LINZ nz-elevation* (public S3, Cloud-Optimised GeoTIFFs, STAC catalogue; `AWS_NO_SIGN_REQUEST`).
  Use STAC for discovery and windowed COG reads for the AOI only. Prefer the *survey* collections
  (1:10k subtiles; one survey per AOI, survey dates and point density recorded) over the national
  composite (1:50k, mixes surveys, possible seams). Record STAC item IDs and `file:checksum`. The
  sandbox here cannot reach the S3 host (the STAC mirror on GitHub is reachable), so fetching runs
  locally.
- *LINZ Data Service* exports: some are labelled EPSG:9528 (NZGD2000 + NZVD2016, *geographic*)
  while holding NZTM metres. The core now refuses such files with a hint and accepts
  `--assume-crs EPSG:2193+7839` (relabel only). The fetcher should write correct CRS labels so the
  override is not needed downstream.
- *International*: national 1 m LiDAR DEMs where open (e.g. UK Environment Agency; ACT via ELVIS);
  each needs its own CRS and vertical-datum handling. The core needs only a projected horizontal
  CRS in metres; DEM and DSM must share the vertical datum.

**Ground-truth sources**
- *Council asset layers*: e.g. Waimakariri District Council stormwater culverts (ArcGIS REST
  MapServer layer; material, length, diameter; updated daily, accuracy varies by source).
  Query by AOI envelope; filter by asset type (culvert, not service line or main) and status
  (in service). They are *partial* inventories: council-owned assets only. Road culverts may sit in
  roading (RAMM) data instead, and private driveway and farm culverts are usually absent.
- *OpenStreetMap*: `tunnel=culvert` on waterways; useful but patchy.
- *International*: e.g. UK Watercourse Culvert nodes and links (data.gov.uk); ACT culvert assets
  (dimensions, invert levels, number of cells). Useful for the location-agnostic goal.
- *Manual labels* (QGIS review files, as now). These remain needed for private structures and for
  the false-positive classes.

**Design considerations**
1. *Snapshots, not live queries.* Asset layers change daily, so store each query result with its
   URL, parameters, retrieval time, record count and a content hash. Scores must be reproducible.
2. *Line assets to crossing points.* Culverts are polylines (pipes). Convert them to a reference
   point (the line's intersection with the road or embankment centreline where known, else its
   midpoint) and keep the line. Match candidates to the *line*, within a tolerance (e.g. 10 m),
   not point to point; long culverts would otherwise be missed.
3. *A common label schema*, versioned: location, class (culvert, bridge, ford, floodgate,
   `channel_obstruction`, `stopbank_gap`, `track`, `flow_path`, other), attributes where known
   (diameter or width × height, length, material, number of cells, invert levels), source,
   confidence, and labeller. The current review files are its first version
   (`culvert_y_n_unsure`, `what_is_it`).
4. *Recall needs an inventory scope.* Report recall against "council culverts in the AOI" and
   against manual inventories separately. Never treat absence from a partial inventory as a
   negative.
5. *Positional accuracy.* Asset positions can be metres off; record the source accuracy where given
   and use tolerances accordingly. Asset dimensions are the first ground truth for culvert sizes
   (see the 300 mm council minimum vs about 20 cm observed at AOI2).
6. *Licences and privacy.* LINZ data are CC BY 4.0; council terms vary. Labels describing private
   property stay out of public repositories, and so do all data (repository and OneDrive excluded,
   located by configuration).
7. *Storage and cache.* Cache keyed by source, item ID and checksum, outside the repository; this
   is the same tile index and metadata store as the data pipeline (§6 item 8), not a separate one.
8. *Interface to the core.* Clips (GeoTIFF, correct CRS), labels (GeoJSON, schema above),
   `site.json` provenance. The core's `run_aoi.py`, `review_candidates.py` and `score_labels.py`
   then run unchanged, and scoring can loop over all sites of a role.

## Tested versions (pip, sandbox, 30 Sep 2026; scikit-image 0.26.0 added 1 Oct 2026)

Python 3.12.3; numpy 2.4.4; scipy 1.17.1; numba 0.67.0; pyflwdir 0.5.12; rasterio 1.5.1;
geopandas 1.2.0; shapely 2.1.2; pyogrio 0.13.0; matplotlib 3.10.8; pytest 9.1.1;
whitebox-workflows 2.0.6. The conda environment files were **not** solved in the sandbox
(no conda); report any solve issues.
