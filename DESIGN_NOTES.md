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

## 4f. Blind test: Waimakariri (Canterbury plains), canterbury2

AOI 3.1 km² (BW24 tile, EPSG:2193 download), code as tuned on SH12 + AOI2. 72 candidates (Test A
15, Test B 50, A+B 7), 67 channel obstructions; 224k pits (cultivated micro-relief; 7–10× the
Northland sites); 65 s on CPU for 4.7 M cells.

**Council inventory** (Waimakariri stormwater, `CLASSIFI_2 = Culvert`; mains excluded): 4 records,
3 distinct (SW013193 duplicates SW002597). Found 2 of 3, both `A+B` within 0–2 m. The miss
(SW002597, culvert under Jacksons Rd near Mill Rd) *was* detected by Test B (h_b 1.64 m, retained
pond 8,500 m², kind `crossing`, crest 3 m from the asset line) and removed only by the vegetation
rule (70% of the barrier under DSM cover; trees to 9 m over the road). Without the hard gates,
recall 3 of 3. Every hard gate has now cost a real culvert on a site it was not tuned on.

**Inventory data notes.** `DIAMETER_m` holds millimetres (0 = unknown); ACQN_DATE 1974-06-01 looks
like a legacy-load placeholder; SW005886 (SH1 Northern Motorway culvert, canterbury1) is recorded as
66.5 m but drawn as 36.3 m; mains touching the AOI edge were clipped on export.

## 4g. Canterbury1 blind, gates removed, and a first ranking model

**canterbury1 (gated, blind).** 81 candidates; council culverts found 1 of 2. The miss is the SH1
Northern Motorway culvert (SW005886; embankment ~2.2 m high, ~25 m crest): Test A paths across it
at the culvert had h_b 1.1–2.3 m, L_b 25–32 m, crossing 64–84°, approach angles 58–72°, and failed
only the ≥ 10 m approach rule (toe-drain stubs 1–8 m long).

**canterbury2 labels.** 27 y / 13 unsure / 32 n of 72 (38% precision; A+B 5/7, A 4/15, B 18/50).
About 20 of the 27 are driveway or field-entrance culverts over roadside drains, none of them in the
council data. A council asset within 10 m of a cutline is not always the same structure (#17).

**Gates removed** (`run_aoi.py --no-gates`: kind, vegetation, approach and L_b ≥ 4 m become
attributes). Recall of labelled or inventory culverts: SH12 4/4 (gated 3/4), AOI2 18/19 (17/19),
canterbury2 council 3/3 (2/3; labelled 40/40 either way), canterbury1 council 2/2 (1/2). Candidates
grow 2–3× (SH12 22 → 87, AOI2 38 → 93, canterbury2 72 → 190, canterbury1 81 → 205). On every
test site, every miss had been caused by a hard gate, not by detection.

**First ranking model** (`scripts/rank_candidates.py`; gradient boosting on 26 features incl. the
former gates; leave-one-site-out, so each site is scored by a model that never saw it): AUC
0.68–0.76; reaching 80% of labelled culverts takes 27 vs ~35 reviews at AOI2 and 45 vs ~51 at
canterbury2; canterbury1's council culverts rank 19 and 56 (first matching candidate each) of 205.
Most useful features: mean DSM − DEM over the barrier, downstream incision. Ranking is now the
bottleneck and is data-limited (~50 positives, three labelled sites; labels exist only for
candidates of earlier, gated runs).

## 4h. Drain network first (revision agreed Oct 2026; in progress)

**Why.** canterbury1's top-30 ranked candidates were 2 culverts, 3 unsure, 25 not. The ranker was
trained on labels of *gated* candidates, so it had no labelled examples of the classes the gates
removed (8 of the top 30 are `kind = bank`), and its main feature (no vegetation over the barrier)
meant "road" at the training sites but "bare field edge" here. The dominant false positive is
"drain probably connects": a drain continuing through a gap with no crossing. Agreed approach:
the drain/stream network is a deliverable in its own right (streams and drains in one dataset,
labelled, with confidence); culverts are a property of *gaps* in it.

**Validation data.** Waimakariri stormwater channels (district-wide, 1,843 lines, 717 km;
`CLASSIFICATION3` = Network Drain / Receiving Waterway, i.e. drain vs stream; widths and depths
empty here). Council scheme channels only: roadside and private drains are mostly absent.

**Baseline channel map vs council channels** (within 5 m): streams 73% / 90% (canterbury1 / 2),
council drains 45% / 50%; 21–24 channel ends per km. Missed drains: canterbury1 shallow and broad
(bed ~0.15 m below surroundings, low curvature, no trees; or filled/piped); canterbury2 under
shelterbelts (median vegetation 6 m). Border-dyke irrigation striping makes blanket threshold
lowering risky.

**Hysteresis mapping** (weak channel evidence kept only where connected to strong channel cells;
`drains.hysteresis_mask`) and **continuity joins** (facing ends within 30 m): council drains 58% /
56%, streams 94% / 92%, 6–8 ends per km. Centreline 15.2 / 12.5 km, of which 29% / 49% lies within
5 m of a council channel (the rest includes roadside and farm drains, possibly some furrows).

**Culverts from the network: not working yet.** (a) Typed gaps between facing ends found 4 of 27
labelled canterbury2 culverts; the raised-feature test is wrong for driveways (they are at field
level; what rises is the drain *bed*), so the gap test should be bed rise alone. (b) Bed-profile
bumps along drains (`drains.bed_bumps`; bed = lowest ground within 1 m of the centreline, bump ≥
0.3 m above the higher reference bed 15 m either side) found 3 of 27; a 2 m bed radius had eroded
narrow driveway crests. (c) Main cause: at most labelled culverts the roadside drain is not in the
network on both sides. Driveways are every 20–40 m, so the drain pieces between them are short,
and the channel map's minimum fragment length (25 m, added for Northland hillside roughness)
removes them before pairing.

**Join before filtering (done).** The drain network now starts from the *unfiltered* channel
evidence (`channels.channel_mask(..., filter_fragments=False)`), applies hysteresis, prunes spurs,
joins facing ends, and only then drops components shorter than 15 m (`drains.build_drain_network`).
Gaps are typed by **bed rise** along the gap (culvert ≥ 0.3 m, continuity < 0.15 m), not by a
raised-feature test. Results (council channels within 5 m; labels within 10 m):

| | canterbury1 | canterbury2 |
|---|---|---|
| council drains / streams found | 62% / 94% | **83%** / 96% |
| centreline (share near a council channel) | 24.4 km (19%) | 20.9 km (35%) |
| culvert candidates (culvert gaps + bed bumps) | 81 | 123 |
| labelled culverts with a candidate | 1 / 2 | **17 / 27** |
| labelled non-culverts with a candidate | 3 / 25 | 3 / 32 |

At canterbury2, 63% of labelled culverts have a drain-based candidate against 9% of labelled
non-culverts. Most candidates are unlabelled (precision unknown); the extra centreline at
canterbury1 needs a visual check against border-dyke striping. Review files:
`<site>_drain_candidates_for_review.geojson`.

**Roads and other context as optional inputs.** The core stays DEM/DSM-only, but optional context
layers can add strong evidence where they exist:
- *Road centrelines* (LINZ NZ Road Centrelines; OSM worldwide): every road–drain intersection is a
  crossing hypothesis (culvert or bridge) regardless of what the DEM shows; this is where the DEM
  evidence is weakest (wide motorway fills, vegetation over roads, roadside drains unmapped on one
  side). Roads also explain raised barriers (Test A) and orient roadside drains (parallel to a road
  within ~15 m).
- *Driveways* are rarely mapped, but property parcels and address points (LINZ) predict them: a
  roadside drain gap in front of an address point is a likely entrance culvert.
- *Rules*: optional inputs act as features and priors, never as the only evidence; provenance and
  positional accuracy are recorded; results are always also reported DEM-only, so the value of each
  layer is measured and the pipeline stays location-agnostic.

## 4i. Network products: channels, crossings, repairs (Oct 2026)

`scripts/run_network.py` (module `products.py`) produces `network.gpkg` per AOI: **channels**
(network branches as lines, with segment features, stream/drain class, `class_confidence`,
`channel_evidence`), **crossings** (culvert candidates: drain-bed rise at gaps and bumps; with roads
also road–drain intersections and drains stopping at a road with a drain opposite; evidence
`dem_h_b`, `veg_frac`, `road_dist_m`; `tier` high/medium/low/bridge_or_open, and `tier_dem_only`),
**repairs** (continuity joins). Roads: LINZ Topo50 centrelines supplied for all four AOIs. NZIS
stopbanks (canterbury1) held back for testing only.

**Crossings vs known culverts** (labels + council culverts; within 10 m):

| site | known | found | high tier | high-tier hits, with roads (DEM only) |
|---|---|---|---|---|
| AOI2 | 18 | 11 | 11 | 11 / 29 (6 / 22) |
| canterbury2 | 31 | 19 | 15 | 15 / 64 (13 / 27) |
| canterbury1 | 4 | 1 | 0 | – |
| SH12 | 3 | 0 | 0 | – |

Roads roughly double the high-tier hits at AOI2. Labelled non-culverts rarely reach the high tier,
but most high-tier candidates are unlabelled. The network approach does not replace Tests A/B:
SH12's culvert (drainage divide under a large embankment; drain ends 52 m apart, 25 m from the
road centreline) is outside the gap and road-end limits. **Next:** crossings = union of network
crossings and Test A/B candidates, evidence from all feeding one tier. The SH12 bridge appears as a
road crossing with bed rise 0.25 m (tier medium; should be bridge_or_open).

**Stream/drain classification: not working yet.** Segments labelled from Waimakariri
`CLASSIFICATION3` (≥ 60% within 5 m of one class): 107 / 124 segments. Leave-one-site-out AUC:
trained model 0.63 / 0.42, hand-set prior 0.46 / 0.77. Causes: branches split at every junction
(947 segments at canterbury1, mostly tiny), so classification is at the wrong unit; D8 upstream
area is unreliable on flat land; two sites. **Next:** classify *reaches* (chains of branches through
junctions, continuing the straightest path), with reach sinuosity, width and network position
(drains feed streams), then calibrate the confidence.

## 4j. Culverts and network together: reaches, culvert links, scored tiers (Oct 2026)

Following Matt's review of §4i: channel type should follow sinuosity (straight → drain), drains
feed streams (rarely the reverse), type rarely changes along a channel, roadside channels are
likely drains, lines must be smoothed and generalised, and SH12 is the key benchmark.

**Network** (`network.py`): a graph of junction and end nodes with skeleton branches as edges.
*Reaches* chain branches through junctions along the straightest continuation (≤ 45°). Accepted
Test A paths (h_b ≥ 0.3 m, kind crossing) are drawn into the network first as *culvert links*,
so channel type and flow continue through culverts. Reaches are smoothed (moving average, ends
fixed at junction centres, so they stay connected) and generalised (0.5 m): 4–5 vertices per
reach instead of one per cell.

**Stream/drain classification at reach level.** Features: bends per 100 m (turns < 45°, i.e.
field-corner turns discounted), share of length in straight runs (≥ 25 m), roadside fraction
(within 20 m of a road and within 25° of parallel), width, incision, length. A transparent prior
from these, then smoothing along the graph (`smooth_on_graph`: short and low-confidence reaches
pulled towards neighbours; an outflow reach pulled harder by the reaches flowing into it; direction
from the bed trend along each reach). Leave-one-site-out against Waimakariri `CLASSIFICATION3`
(length-weighted): prior accuracy 0.86 / 0.72 (AUC 0.96 / 0.91), prior + smoothing 0.86 / 0.76
(0.97 / 0.91); a logistic model trained on the other site 0.88 / 0.58 (0.99 / 0.84). The prior plus
smoothing is the default; the trained model is optional (`--stream-model`). Model coefficients
agree with the rules: straight runs and roadside → drain, width → stream.

**Crossings** = network crossings (gaps, bed bumps, road–drain intersections, road end pairs) ∪
Test A/B candidates (ungated), merged within 8 m, with context: DSM deck over the crossing,
distance to road, network link. **Tiers** from a points score over the evidence that separated 51
labelled culverts from 123 labelled non-culverts across the four sites: within 15 m of a road 2
(84% vs 25%), network evidence (gap, bump or culvert link) 2 (gap 47% vs 2%), ≥ 2 independent
sources 1 (71% vs 21%), road–drain source 1; high ≥ 3, medium ≥ 2, both requiring a barrier ≥ 0.3 m
and no vegetation flag. "On a stream" did not separate (45% vs 57%) and is not used. A road over a
continuous bed with a DSM deck ≥ 2 m is a *bridge*. DEM-only tiers drop the road terms.

| site | crossings high / medium | known culverts found | in high | in high or medium |
|---|---|---|---|---|
| SH12 | 27 / 26 | 3 / 3 | 3 | 3 |
| AOI2 | 40 / 29 | 18 / 18 | 15 | 17 |
| canterbury2 | 69 / 37 | 31 / 31 | 23 | 27 |
| canterbury1 | 37 / 62 | 4 / 4 | 1 | 3 |

**SH12 benchmark** (`tests/test_benchmark_sh12.py`, runs when `TERRAIN_S2_SH12_DIR` holds the
clips): culvert crossing tier high (4.2 m); the network passes through the culvert as one reach;
streams either side (p 0.99 / 0.95; previously stream and drain); SH12 bridge classed `bridge`. All
pass.

**Caveats.** The tier weights come from labels on earlier candidate sets (biased towards what
earlier versions proposed); the review of these crossings is the first unbiased test. Reaches are
still numerous (630–1,045 per site), many short.

## 4k. Closing R1: floodplain, cleaning, rivers, stream/drain fix (Oct 2026)

Matt's review of 4j (labels on 81 reaches and 134 crossings, floodplain only): the network does
not miss channels, but (1) drains are classified as streams (and less often the reverse),
(2) small spurious segments sit disconnected from the network, (3) wide rivers carry several lines.
Agreed: restrict processing to the floodplain with a DEM-only method (no new data requirement).

**Changes.**
- *Floodplain* (`floodplain.py`): HAND (Nobre et al., 2016, via pyflwdir) ≤ 5 m (10 m from §4l) above drainage =
  wide channels (≥ 8 m) or D8 upstream area ≥ 0.1 km²; islands < 1 ha removed, enclosed holes
  < 1 ha filled (not holes touching the window edge). Network and crossings are restricted to the
  floodplain + 50 m; crossings are judged on the buffered floodplain because an embankment crest
  can stand > 5 m above the drainage (the SH12 culvert crest). Floodplain share: SH12 0.51, AOI2
  0.46, canterbury1/2 1.0. `--max-hand 0` disables.
- *Stream/drain*: bends measured on a 3 m-simplified line in 10 m steps (Matt's drains 7 vs
  streams 36 deg/100 m; previously 53 vs 87, the measure picked up centreline wiggles); prior
  re-weighted so straight runs dominate; reaches inside river polygons are streams.
- *Cleaning* (`network.clean_reaches`): drop network components < 50 m, components < 100 m whose
  median incision < 0.4 m, and dangling reaches < 15 m; reaches between junctions and through
  culvert links are kept. 0.5 m removed more spurious reaches at SH12 (25 vs 18 of 28) but two real
  drains (0.44 and 0.66 m deep, one small network); 0.4 m keeps 3 of 4. The evidence overlaps: the
  learned channel model (R3) is the real fix.
- *Rivers* (`network.river_polygons`, `river_centrelines`): channel cells ≥ 8 m wide grown back to the
  full channel, smoothed, as polygons (layer `rivers`); one centreline per river (skeleton of the
  polygon, spurs < 40 m pruned); tributary ends at the edge reconnected to it.
- Crossings must lie within 10 m of the cleaned network or carry Test A / Test B evidence.

**Results** (Matt's labels; council `CLASSIFICATION3`; reach kept = ≥ 50% within 1.5 m):

| measure | before (4j) | now |
|---|---|---|
| Matt's drain reaches classified drain | 0 / 29 | 13 / 19 |
| council stream/drain accuracy by length, canterbury1 / 2 | 0.86 / 0.76 | 0.91 / 0.89 |
| Matt's spurious reaches removed | – | 28 / 50 |
| Matt's real reaches kept | – | 21 / 22 |
| SH12 benchmark | 4 / 4 | 4 / 4 |
| run time SH12 (single run) | 76 s | 39 s |

**Crossings (unchanged in method; now scored on all labels):** labelled crossings in the high tier
are culverts 58% (AOI2), 54% (canterbury1), 71% (canterbury2), 20% (SH12); 78–100% of labelled
culverts are in the high tier. About 100 labelled culverts and 90 non-culverts now exist: enough to
start the learned crossing classifier (methodology review R4).

**Review round 2:** `<site>_crossings_review2.geojson` and `network_<site>_review2.gpkg` (layers
channels, crossings, repairs, rivers, floodplain), with earlier labels carried as `prev_*` fields.

**Flagged, not dropped:** (a) stopbanks and stopbank-like features as an output to be preserved in
coarser meshes (design §5; NZIS stopbank at canterbury1 for testing only; building blocks: raised
features, crest following, stopbank-gap findings of §4a); (b) national data-acquisition machinery
for R3, starting with Waimakariri (§7).

## 4l. Review round 2: oblique stream crossings, floodplain 10 m (Oct 2026)

**Whirinaki (SH12) review:** much improved; 41 further spurious reaches flagged (median 6.3 m,
median incision 0.59 m). 38 of the 41 are short links *between* junctions (25 in small loops),
not dangling or isolated pieces. Pruning short loop links would remove 22 of the flagged reaches
but also 77 unflagged ones, so no further rule was added: we are in a diminishing-returns cycle
(Matt), and these labels go to R3 (learned channel segmentation) instead.

**AOI2 culvert #2** (a stream under a road, labelled "stream across river") was tier high only
through the road evidence; DEM-only it was low, with no network link and `on_stream` false. Cause:
all five Test A paths there were `kind = bank`. The one with a real approach (18.6 m) met the road
at 42.6°, just under the 45° transverse threshold; the stream beyond is at 47.3°. The same
failure mode as the driveway culverts at 39° (§4d). Fix (Matt's rule: a stream either side of a
road heading the same way): a Test A path is also a crossing when the channel *continues* through
the barrier: approach ≥ 10 m, both channels ≥ 30° off the barrier axis, and within 30° of each
other (`continuation_min_deg`, `continuation_max_dev_deg`; records carry `continuation` and
`continuation_angle`). Bank hops between channels running along a barrier still fail. Result:
culvert #2 is high in both tiers, a culvert link in the network, and the reach through it is a
stream (p 0.95). Regression test: `tests/test_benchmark_aoi2.py` (`TERRAIN_S2_AOI2_DIR`).

**Floodplain** default raised from HAND 5 m to 10 m: at 5 m AOI2's floodplain fell short of the
true floodplain, because the drainage reference (an imperfect channel network) is itself
incomplete. Floodplain share: AOI2 0.46 → 0.58, SH12 0.51 → 0.57; canterbury1/2 unchanged (1.0).

**No regressions:** reach, class and crossing scores against all labels are identical to §4k;
SH12 benchmark 4/4; AOI2 benchmark 2/2; unit tests 42 passed.

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
3b. **Drain network (§4h)**: join before length filtering; gap typing by bed rise; bumps; then
    stream/drain labelling with confidence (trained on council `CLASSIFICATION3`: straightness,
    width, upstream area, alignment with roads and field boundaries); then the Northland sites.
4. **Ranking (§4g).** Done in first form; next: label the ungated candidates in score order (top
   ~60 per site) so training data are no longer biased to gated runs; add features (barrier
   cross-section shape, road/driveway context, DSM structure); calibrate a review threshold. The
   design's CNN classifier (§4.3) is the later step, once a few hundred labels exist.
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
- *LINZ Data Service* exports are **reprojected into the CRS chosen at download**. A tile
  downloaded as EPSG:9528 (NZGD2000 + NZVD2016, geographic) has been resampled to degree pixels
  (1.03e-5°, about 0.83 × 1.14 m at 43° S): an extra resampling, and non-square cells. Fetch in the
  native EPSG:2193 (heights unchanged, NZVD2016) and record the vertical datum in metadata or as
  `EPSG:2193+7839`. The S3 collections are EPSG:2193 (LINZ naming docs); whether their COGs carry the
  vertical CRS in the GeoTIFF is to be checked with `gdalinfo`.

**CRS handling in the core** (Oct 2026): any projected horizontal CRS in metres, compound or not;
tiles may differ only in the vertical label. `--assume-crs` relabels (refusing data that really are
geographic), `--aoi-crs` relabels an AOI (GeoJSON without `crs` is read as WGS84), and
`--reproject-to` (horizontal-only warp onto a `--resolution` grid, vertical label kept, warns about
resampling) is the explicit path for geographic or foreign-CRS inputs. DEM and DSM must share the
vertical datum.
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

### 7a. M1 implemented (Oct 2026): `terrain_s2.acquire`, `scripts/build_site.py`

Implemented as designed above and in the R2/R3 design document (Claude Doc, 7 Oct 2026):
site definitions (`sites/*.yml`), a source registry (`sites/sources.yml`), snapshot store, and
fetchers for LINZ elevation, ArcGIS REST (layer URL or Hub item id; envelope query with paging, or
object-id chunks when paging is unsupported), LINZ Topo50 WFS and OSM Overpass, with a harmoniser
to label schema v1. Decisions:
- LINZ STAC collections list tiles by name only (1,273 for canterbury_2020-2023), so a footprint
  index is built once per collection from the item records and cached; only the window is read
  from the COGs (HTTP range reads). The catalogue is read from S3 (not mirrored on GitHub).
- ArcGIS GeoJSON is requested in WGS84 and reprojected locally; features are built from the parsed
  JSON, not GDAL's reader, which pages through `exceededTransferLimit` on its own.
- LINZ WFS 1.0.0 (easting, northing always), with a check that returned features fall in the
  window; the API key comes from `LINZ_API_KEY` and is redacted from snapshots.
- Unverified or incomplete sources are in the registry but disabled, each with a note.
Tested with a fake HTTP layer (`tests/test_acquire.py`, 8 tests); live endpoints are not reachable
from the analysis sandbox, so the first live run is `--dry-run` on the workstation.

**First live run (Matt, 7 Oct 2026, canterbury1):** LiDAR (1 tile each, DEM and DSM; 2688 × 1767
cells, matching the manual clip), LINZ roads 34, tracks 2, rail 1, OSM roads 146, rail 1,
waterways 7, culverts 2, in 24 s. Waimakariri culverts and channels returned **0 features without
an error**: ArcGIS reports failures inside an HTTP 200 body (`{"error": …}`), which the first
reader treated as an empty result. Fixed: error bodies raise with the server's message; a refused
GeoJSON query falls back to Esri JSON (geometry converted locally); the dry run asks each ArcGIS
source for its count in the window; an enabled source with 0 features is a warning, with a
query URL to paste into a browser; `scripts/probe_source.py` diagnoses one source. Cause still to
confirm from the probe output.

**Buildings** (Matt: evidence for driveway culverts): LINZ NZ Building Outlines (layer 101290,
from LINZ aerial imagery, outlines ≥ 10 m², including garages and sheds) and OSM buildings
(closed `building` ways), both into `context.gpkg`.

**Sources verified 7 Oct 2026** (Gemini's list checked): Waimakariri layer 19 (network main,
facility pipe and culvert; EPSG:2193, GeoJSON, paging, 2,000 per query; `DIAMETER_mm`) and layer
18 (stormwater channel: network drain, receiving waterway, process channel, water race); Auckland
Stormwater Channel (Hub item `6c4196567eae4cd48f1c071911fe4105`; constructed channels only, no
streams); Canterbury Maps regional Three Waters (nodes and pipelines only, no channels, monthly;
Waimakariri not included at release); KiwiRail culverts, track centreline and bridges (via
data.govt.nz; service URLs to resolve); LINZ Topo50 roads 50329, railways 50319, tracks 50364. Not
yet checked: Christchurch, Waikato LASS, Kāpiti, New Plymouth, Gisborne.

### 7b. Waimakariri culverts in the Before U Dig service (8 Oct 2026)

- **Probe, canterbury1 window.** `3Waters/BUD_Query/MapServer/10` (Main_SW) is live: 84 features, all
  `CLASSIFICATION2 = 'Pipe'`, `CLASSIFICATION3 = 'Network Main'`. No culverts under any field.
- **The filter was on the wrong field.** Matt's export of the old `Stormwater_Assets_In_Service` pipes layer
  (`Stormwater_Pipes.shp`, 9,655 rows) has the type in CLASSIFICATION3 (shapefile `CLASSIFI_2`): 8,611 Network
  Main, 980 Culvert, 64 Facility Pipe; CLASSIFICATION2 is 'Pipe' throughout. `sources.yml` now filters on
  `CLASSIFICATION3 = 'Culvert'` (and `<> 'Culvert'` for `waimakariri_pipes`).
- **Culverts appear to be missing from Main_SW.** The export has 86 rows in the window: the same 84 mains plus
  two culverts, SW005887 (Revells Road, council, 29.9 m) and SW005886 (SH1 Northern Motorway, ownership
  'Private', 36.3 m), both 224 mm concrete, 1974. These match the manual reference count of 2.
- **Not in Before U Dig.** `--find "ASSNBRI IN ('SW005886','SW005887')"` (8 Oct, Matt): no layer of
  `BUD_Query` holds either asset, although the Main_SW layer description lists culverts. Likely a fixed
  definition expression on the Before U Dig layers (culverts are not buried services); the probe now prints it.
- **Found: the open-data layer.** Matt's export is the portal's "Stormwater Culvert Facility Pipe and Network
  Main" dataset (openmaps-waimakariri.hub.arcgis.com, item `9ab2c4f660dd4dc4a6b3a1ab3efea019_14`), served from
  `3Waters/Assets_Stormwater/MapServer/14` (13 is the same data styled by criticality). `Assets_Stormwater`
  appears to replace the withdrawn `Stormwater_Assets_In_Service`. `waimakariri_culverts` and `waimakariri_pipes`
  now use layer 14; `waimakariri_structures` points at layer 20 (Stormwater Structure - Point; unchecked).
- **`--find --scope folder` missed it.** It searched the `3Waters` folder but reported no match, so a layer
  that refused the filter was skipped silently. It now reports services and layers searched and lists refusals.
- **Hub items whose URL is the layer.** `resolve_layer` appended the layer index to an item URL that already
  ended in it (`.../MapServer/14/14`); fixed, with a test.
- **The live layer has a newer schema than the export** (8 Oct probe): `ASSET_ID` (e.g. "291404") and
  `AssetType` ("Stormwater-Pipe-Culvert"), coded values with `_DISPLAY` companions (`SERVICE_STATUS` "IN").
  `ASSNBRI` is absent, which is why every `Assets_*` layer "failed to execute" the `--find ASSNBRI IN (...)`
  query. The layer answers normally: canterbury1 has 4 culverts in the window (export: 2), canterbury2 7. The
  `BUD_Query` channels still use the old schema (`ASSNBRI`, "In Service"), so the two services may not be
  equally current; `Assets_Stormwater/MapServer/3` (Stormwater Channel) is the new-schema candidate for channels.
- **Silent schema drift guarded.** `harmonise` used to fall back to row numbers for a missing id field and NaN
  for missing sizes. The probe now checks the fields `sources.yml` relies on against the layer (and `--fields`
  lists them); `build_site` warns in both dry and real runs; `site.json` records the warning.
- **`sites/*.yml` matched `sources.yml`** (KeyError 'bounds' after the last site): the build skips the registry,
  and `load_site` names a file that is not a site definition.
- **Rows keyed on `ASSET_ID`** (new asset system); `Legacy_ID` carries the old SW numbers and stays in the
  `_raw` layer. `id_field` may list alternatives (`[ASSET_ID, ASSNBRI]`), so the old-schema fallback still keys
  rows. Channels moved to `Assets_Stormwater/3` (same 45 features and classes as `BUD_Query/12`, which is now the
  fallback), so culverts and channels share one service, schema and id system.
- **canterbury1 culverts, live (8 Oct):** 4 = the export's two (Legacy SW005887, SW005886) plus SW037568 and
  SW037640 (375 mm and 300 mm, 250 Revells Road), added since the export. One of the four is private (SH1).
- **M1 gate reference from the export** (window = AOI + buffer): canterbury1 2 culverts (SW005886 private,
  SW005887 council); canterbury2 7 in the window, 4 within the AOI (the manual count of 4, 3 distinct).
- **Probe options added:** `--values` (value counts of text fields in the window), `--layers` (service layer
  list), `--find WHERE` with `--scope service|folder`, `--url` and `--where` overrides, `--search` (ArcGIS
  Online items), and the layer's definition expression in the header. Fallback URLs (`fallback_urls`, `arcgis.first_working`) restored:
  the 7 Oct change had not reached the repository.
- **Benchmarks** accept the file names in Matt's clip folders; both pass in the sandbox on the uploaded clips
  (61 passed, 1 skipped: GPU).

### 7c. M1 gate and R1 on the assembled bundles (8 Oct 2026)

**Build** (Matt, all four sites, 20–50 s each): 0 errors except linz_rail at canterbury2.
- *Rasters:* every bundle DEM and DSM matches the manual clip cell for cell (same grid, shape and
  origin; |difference| ≤ 0.001 m in about 0.5 % of cells, LERC rounding).
- *Council labels:* canterbury1 4 culverts, 45 channels; canterbury2 7 culverts, 44 channels. Inside
  the AOI, council-owned channel length is 6.60 km and 9.17 km: the manual references (6.6, 9.2).
  Most culverts have no diameter in the new asset system; lengths are filled from the geometry
  when blank (`length_from_geometry`).
- *linz_rail, canterbury2:* WFS selects by bounding box, so the 346 km Main North Line came back for
  a window it never enters, and the axis-order guard misread that. The guard now raises only when the
  features' extent misses the window; otherwise the result is empty (`bbox_only` in `site.json`).
- *Whole features:* roads, rail and other context came back whole (SH12 roads 39.6 km for a 1.4 km
  window). Context layers and `roads.gpkg` are now clipped to the window; labels keep whole features.
- *KiwiRail culverts* added (national points; `ASSETNUM`); size units to confirm before mapping.

**R1 on the bundles** (sandbox, CPU). A = manual clips + manual roads (the R1 inputs); B = bundle
rasters + bundle roads clipped to the AOI; C = bundle rasters + roads clipped to the window.

| site | A vs B | B vs C (roads beyond the AOI) |
|---|---|---|
| canterbury1, canterbury2 | identical (reaches, classes, crossings, tiers) | identical |
| AOI2 | identical | one extra high crossing (road_end_pair at the AOI's north edge) |
| SH12 | 4 of 231 reaches and one crossing differ (medium → low, moved 5 m), from the ≤ 1 mm DEM differences | the bridge point moves 3 m |

Benchmarks on the bundles: SH12 4/4, AOI2 2/2 (65 passed, 1 skipped). So the automated bundles
reproduce R1; C (window roads) is the new default.

**Scoring against the bundle labels** (`scripts/score_site.py`, new; run C):
- *Culverts in the AOI:* canterbury1 3/3 found within 10 m (high 2, medium 1), including SW037640,
  added to the register after R1 (a blind find); canterbury2 4/4 (high 2, low 2). Culverts in the
  200 m buffer are outside the network and not scored.
- *Channels along the labels* (2 m samples, nearest reach within 5 m): stream/drain accuracy by length
  0.90 / 0.86 (canterbury1 / 2); council-owned 0.88 / 0.86; coverage 54 % / 82 % (council 68 % / 88 %).
  The method differs from §4k (0.91 / 0.89, reach-side), so the figures are not directly comparable;
  low coverage is mostly council lines drawn off the channel bed (M2: snap labels to the bed).

### 7d. Connectivity scoring, the national partition, Environment Southland (8 Oct 2026)

**Found is not enough.** Matt: the canterbury1 culvert under SH1 (OBJECTID 8081, asset 291725,
Legacy SW005886, 36 m) is missing from the prediction, and it is a critical feature. `score_site.py`
had counted it found, because a Test A candidate lay 3.5 m from one end of the culvert line. Culverts
are now scored at the crest (the midpoint of the line), and separately for *connected*: a predicted
reach within 3 m of the crest, i.e. the network passes through. In the AOI: canterbury1 found 3/3,
connected 1/3 (291725 nearest reach 13.3 m; 311935 6.8 m); canterbury2 found 4/4, connected 2/4.
Connectivity through culverts is the target measure for R3/R4: this class of model has reached the
point where fixing one case by parameters breaks others (Matt).

**Partition** (`terrain_s2.grid`, `scripts/partition.py`). LINZ names its elevation tiles on the
map-sheet grid (1:50k sheets 24 x 36 km, 1:10k tiles 4.8 x 7.2 km), so the grid is the partition:
- origin E 1,012,000, N 6,234,000, rows AS, AT, ... (no I or O); fitted to the tile names served to the
  four sites (all five tiles reproduced), and checked against cached footprints with `--check`;
- processing tile: 1:5k (2.4 x 3.6 km, 8.6 km2, about the size of the test sites), each inside exactly
  one LINZ 1:10k COG tile, so a tile reads one COG (plus neighbours for its buffer);
- the national index comes from the collection records alone (they list tile names), one request per
  collection, no item records;
- `split` is assigned per 1:50k sheet (hash of the id; 80/10/10), so neighbouring tiles never straddle
  train and test.
Other regions get a `RegularGrid` or their published tiling through a region profile (T1).

**Environment Southland** (layer pages read 8 Oct, to be confirmed live): `es_drainage` (MapServer/27,
ES Drainage Network: Name, District, Catchment, SmartId; no stream/drain class) and `es_stopbanks`
(MapServer/28, ES Flood Protection Network: ReachID, Scheme, River, Bank, ReachPurpose). Culverts are
not in either layer; `--layers` lists the rest of the service. The probe no longer needs a site: the
window is a tile (`--tile`), a box (`--bbox`) or the layer's extent.

- **KiwiRail sizes** (canterbury1 probe, 5 concrete pipes): `DIAMETER` 450–1300, so mm (mapped to
  `diameter_m`); `DEPTH` 1.9–4.2 m, probably cover or fill depth (kept raw; useful later as embankment
  height); `KRSTARTOFF` is line metreage in km.
- **Implied crossings** (`labels.gpkg`, layer `crossings_implied`): where a labelled channel crosses a road
  or railway, a structure must exist, recorded or not. Environment Southland publishes drainage lines that
  cross highways with no culvert layer (Matt: live, but the culverts are implicit); council registers miss
  private and some road culverts. `recorded` marks a labelled crossing within 15 m. canterbury1: 9 (7 in
  the AOI), none recorded (council channels stop at culvert ends, so recorded culverts do not intersect
  roads); canterbury2: 19 (8 in the AOI), 2 recorded. In the AOI: found 4/7 and 6/8, connected 2/7 and 2/8.
  Positions are only as good as Topo50 centrelines (1:50k) and council lines, so the 3 m connectivity
  test is strict for these; snapping to the DEM road crest is an M2 task.

### 7e. Partition built; source registry (8 Oct 2026)

**Partition, first national run** (Matt): `--check` 3,550 cached footprints, 0 disagree. 213 collections
(108 DEM, 105 DSM), 8,102 LINZ tiles. The first run gave 42,400 processing tiles and 366,336 km2 (more
than NZ), 42,164 of them in region "new-zealand": LINZ's national mosaic (`new-zealand/new-zealand/dem_1m`),
tiled by 1:50k sheet (sea included) and dated by its latest input, won every tile. It no longer chooses
surveys; tiles record `in_national` (useful to fill buffers where a survey ends; Matt: neighbouring tiles
are needed for the buffer, and surveys share the sheet grid, so only survey edges need care).

**Source registry** (`acquire/registry.py`): two files. `sources.yml` stays the hand-edited definition
(mapping, notes, optional `coverage: auto | nz | global | [bbox]`). `sources_coverage.geojson` is generated
by `scripts/index_sources.py`: for ArcGIS layers, a feature count per 1:50k sheet in the service extent
(limited to sheets with LiDAR via `--partition`), the coverage being the sheets with features; LINZ WFS
is national, OSM global. Sites default to `sources: auto` (optional `layers:`, `exclude:`), which gets
every enabled source whose coverage meets the window; zero-feature results are then normal and not
warned. All four sites moved to `auto` (their former lists kept as comments; the resolved set is
recorded in `site.json`). `scripts/tile_pool.py` joins the partition with the index: layer `pool`, the
tiles whose sheet has label features (crossings, channels, stopbanks), with the sources per type.
Re-run index and pool as sources are added.

### 7f. LINZ tile as the processing unit; tile-level coverage (8 Oct 2026)

**Partition rerun** (Matt): 7,678 LINZ tiles, 30,712 1:5k tiles (265,352 km2), 16 regions; national
mosaic on 424 sheets. **Processing unit changed to the LINZ 1:10k tile** (Matt: keep the full LINZ tile,
buffer into neighbours): 4.8 x 7.2 km, one COG, provenance per tile; `--scale 10000` is now the default
(5k / 1k remain for training chips). Each tile records `n_neighbours_lidar`, `n_neighbours_same_survey`
(0-8) and `survey_edge`, since the buffer crosses a survey edge where neighbours differ. With a 200 m
buffer a tile is 5.2 x 7.6 km, ~39.5 M cells at 1 m, ~8 x canterbury1 (141 s on Matt's CPU): about 20 min
per tile on CPU if it scales linearly; memory to be measured on one tile first.

**Coverage was per sheet, too coarse.** The pool claimed `kiwirail_culverts` for tiles far from the rail
(e.g. BV20_5000_0305: 76 KiwiRail culverts in its 24 x 36 km sheet, none near the tile). The index now
counts per 1:10k tile inside the sheets that have features; pool membership and `label_features` are per
tile. Sources with `url: TODO` (kiwirail_track) are recorded as unresolved, not queried.

**Held-out sheets.** BW24, the sheet of canterbury1 and canterbury2, hashed to train; any training
tile in it would neighbour the test sites. Sheets holding a test or validation site now take that role.

**Tiles as sites.** `build_site.py --tile <id> --partition <gpkg>` builds a bundle for a partition tile
(bounds, region, latest survey, split as role, `sources: auto`).

### 7g. Quadrant test units; first profile (8 Oct 2026)

**Full-tile run** (Matt, BW24_10000_0402, CPU): still running after 40 min; CPU 50-60 %, memory ~82 GB
(machine shared with Docker VMs, so not a clean figure). Production target: 192 GB, RTX 6000 Pro, 32 cores.

**Quadrants for testing** (Matt): `partition.py --unit quadrant` splits each LINZ tile into four 2.4 x 3.6 km
units named `BW24_10000_0402_NW/_NE/_SW/_SE` (they coincide with 1:5k tiles). `grid.tile_bounds` and
`build_site.py --tile` accept them. Production stays on whole tiles (`--unit tile`).

**Profile, canterbury1** (4.75 M cells, sandbox, 1 core): 117 s, peak RSS 1.34 GB (~280 bytes per cell,
so a buffered LINZ tile of 39.5 M cells should need ~11 GB; 82 GB is not explained by the arrays). Stages:
features 33 s (of which the 3 median filters, scipy `rank_filter`, 28 s), Test A 19, Test B 16,
floodplain 15 (HAND 10 s; flow accumulation recomputed), channel network 9, fills 8 (priority-flood
run 3 times). `run_network.py` now records stage timings and peak memory in `network.json` (`perf`), and
`--profile` writes a cProfile.

**Quadratic terms removed** (outputs identical on canterbury1: every layer, geometry and attribute):
- Test A copied the whole white top-hat grid (`np.nan_to_num`) once per path: O(paths x cells), 9 s at
  canterbury1, ~64 x at a full tile. Now once per run (Test A 18.6 -> 6.7 s).
- four all-pairs distance loops in `run_network.py` (reaches x culvert links, candidates x links,
  candidates x reaches): now spatial-index queries.
canterbury1 117 -> 97 s; the full-tile gain should be much larger, since these terms grew with area squared.

### 7h. First quadrant run; rasters; more sources (8 Oct 2026)

**BW24_10000_0402_SE** (Matt, CPU, 16 logical cores; 2800 x 4000 cells with buffer): 406 s. Test B 153 s,
features 69, floodplain 44, Test A 40, crossings 25, channel network 23, hydrology ~26. Test B is linear in
the number of breaches (sandbox: 1,525 breaches 7 s, 6,011 breaches 20 s), a per-breach Python loop: the
numba target. Peak memory not recorded (psutil missing; now a dependency). Labels: culverts found 11/15,
connected 4/15; implied crossings found 15/22, connected 8/22; channels covered 58 %, stream/drain 0.91.

**Hydrology once.** Flow directions are routed once (pipeline) and reused for HAND; outputs identical
(SH12, every layer and raster). The fill on the raw DEM (depressions) and on the breached DEM (routing)
are different inputs and both remain.

**Rasters** (`run_network.py --rasters all|hydro|none`, default all), cropped to the AOI so tiles mosaic
without overlap, float32, with `rasters.json` describing every band:
- `features.tif`, 17 bands: relief_med5/11/21, tophat_white and tophat_black 11/21/41, laplacian, ridge and
  valley at sigma 2 and 4 m, openness_pos/neg. (QGIS shows bands 1-3 as RGB by default: the "three layers"
  were the relief_med bands.)
- `hydro.tif`, 9 bands: hand_m, floodplain, log10_upstream_area_m2, breach_depth_m, depression_depth_m,
  channel_mask, channel_centreline, dsm_minus_dem_m, road_distance_m.
Size: SH12 AOI 44 MB; a quadrant ~0.6 GB; a full tile ~2 GB (production: `hydro` or `none`).

**Not used in processing:** buildings (context only so far); OSM roads (in context.gpkg; `roads.gpkg` is
LINZ Topo50 when present). OSM detail is uneven (some dual carriageways and footways, some centrelines
only): LINZ stays the barrier geometry; OSM is for tags (bridge, tunnel=culvert, layer, highway class).

**Sources added** (8 Oct searches): LINZ Topo50 bridges (50244, own label layer `bridges`; implied crossings
count them as recorded), embankments (50266), dams (50260), enabled. Disabled pending probes: ECan bylaw
stopbanks and drains (gis.ecan.govt.nz PlanningZones), BOPRC Defence Against Water (culverts, floodgates,
stopbanks), HBRC asset lines and drains (open data items), NIWA/ESNZ REC2 v2.5 river lines (national,
CUM_Area upstream area: edge seeding for upstream area and natural-stream labels). Not open: NZTA state
highway bridge and culvert locations (OIA 2022, declined for security). Known but not online: the NZ
Inventory of Stopbanks (Crawford-Flett et al. 2022, 5,284 km, via regional councils).

### 7i. Two more grid-scaling costs; probes of new sources (8 Oct 2026)

**Matt: the full tile is slow, the quadrant was fast.** Stage ratios for a quarter of canterbury1 vs the
whole (4 x cells) found terms growing faster than the area; all fixed with outputs identical (canterbury1,
every layer, geometry and attribute):
- Test B pond flood fill (numba) kept a hash set per breach; ponds reach the 2 M-cell cap on large
  windows, and nested breaches each refill the same pond. Now a reused visit-stamp array and queue
  (same visiting order): pond fills 9.0 -> 1.4 s, Test B 17.5 -> 10.0 s at canterbury1.
- Test A built `skeleton & ~toe` (a whole-grid array) once per path: now once per run.
- road x drain candidates: a 3 x 3 minimum filter of the whole DEM per intersection, intersection with
  the union of all roads per drain line, and an all-pairs loop over drain ends near roads with a distance
  to the road union per end. Now: one filter, spatial-index queries, k-d tree pairs (same pair order).
Remaining ratios ~4-5 x (Test A ~7 x, mostly more paths: 108 -> 615). The quarter vs full-tile timing on
Matt's machine is the check.

**Probes** (Matt, 8 Oct): ECan bylaw stopbanks (layer 17: 540 features, 115 LINZ tiles; 2019 amendment is
layer 4, now primary; drains are layers 3 / 16); BOPRC Defence Against Water publishes 6 stopbanks without
geometry (unusable); HBRC Asset Lines has Stopbank 601, Drain 572, Culvert 176, River Channel 125, Bridge -
Road 3 (split into hbrc_stopbanks / _culverts / _channels / _bridges by AssetType); REC2 fields are
CUM_AREA and StreamOrde. **REC2 is not for stream identification** (Matt: coarse source DEM, channels
missing or misplaced; GeoFabrics snaps it to a finer network): context for upstream area and flow
statistics where it snaps.

**National stopbank inventory** (Matt is an author; not released because not all councils agreed): may be
used for training but must not be released. Prefer the original council sources; if used, it needs a
restricted path (kept out of the repository, bundles, index and published products).

### 7j. Timings after the fixes; GeoFabrics bed estimation; ML decisions (8 Oct 2026)

**Matt's machine, CPU, outputs unchanged:** full tile BW24_10000_0402 (39.5 M cells) 2,557 -> 946 s
(15.8 min), peak 8.65 GB; quadrant _SE (11.2 M cells) 406 -> 252 s, peak 2.48 GB (~220 B per cell). Full
tile before -> after: Test B 1,277 -> 250 s, road crossings 364 -> 7, Test A 246 -> 77. Tile / quadrant is
3.53 x the cells and 3.76 x the time, so nearly linear; Test B is still 6.1 x (41 -> 250 s), the next
target (numba, or pond reuse between nested breaches). Largest stages now: features 251 s (median
filters: GPU), Test B 250, HAND/floodplain 124, Test A 77, channel network 70. Sizing for the 32-core /
192 GB machine: memory allows ~20 tiles at once; 7,678 tiles x 15.8 min / ~24 workers ~ 3.5 days CPU-only.

**Conditioned hydrology (Matt).** `hydro.tif` describes the DEM *before* detection: HAND has
discontinuities where drainage crosses undetected barriers. Proposed: after the network is assembled,
a second hydrology pass on a DEM conditioned with the result (culvert links set to the bed interpolated
between their ends, as GeoFabrics sets a tunnel to the minimum around it), giving flow, upstream area
(seeded from REC2 where it snaps), HAND to the mapped network, and residual depressions. Residual ponds
against a barrier with inflow are the "drainage completeness" signal: candidates for missed crossings
and an ML input. Cost about 3.5 min per tile (+22 %). A full second detection pass to be judged on it.

**GeoFabrics 1.1.30** (rosepearson/GeoFabrics, last commit 12 Mar 2026), river bed:
`RiverBathymetryGenerator` takes the main channel from REC (with flow and Manning's n per reach), aligns
it to the DEM with transects at `cross_section_spacing` (bank threshold above the water level; centre
re-estimated from the widths), smooths width, flat-water width, bank height and slope with an upstream
rolling mean of `cross_section_spacing x upstream_smoothing_factor` (km scale), fits the water surface
with a monotone penalised spline (`_unimodal_smoothing`, third-difference penalty lambda 100), and takes
depth from Neal et al. (uniform flow, d = (n Q / (W sqrt S))^(3/5)) or Rupp & Smart (hydraulic geometry,
d = (Q / (6.16 W S^0.305))^(1/1.745)), converted from bank-full to the depth below the LiDAR water surface;
a minimum slope is enforced. `WaterwayBedElevationEstimator`: OSM waterways and tunnels (culverts), a
tunnel's bed = the minimum elevation around it, open waterways forced downhill. `StopbankCrestElevation
Estimator`: crest = maximum around the stopbank. Where to smooth less (Matt): the km-scale rolling means
and the curvature penalty; a monotone fit without a curvature term (isotonic) keeps steps (weirs, riffles),
with width and slope smoothed only within a reach and Q from REC2 where it snaps.

## 8. Machine learning: decisions (8 Oct 2026, with Matt)

- **Target, first:** at every place a channel meets a barrier (road, rail, embankment, stopbank): culvert,
  bridge or no structure, and where there is one, the link through it. Then channels and classes; then bed
  elevation (with less smoothing than GeoFabrics).
- **Approach:** hybrid, building on the physics chain (candidates, evidence) rather than starting blind;
  grounded in the Swedish U-Net approach, alternatives tested later. A wide-area training set matters most.
- **Labels:** council data = reliable (high weight); Matt's labels = partially reliable; unlabelled = the
  target, never a negative. As council data accumulate they take precedence. Post-processing: every
  council-recorded crossing is in the final product even where the model misses it.
- **Evaluation:** primary, automated against council data over a wide sample of tiles; Matt's reviews
  secondary (limited time). Precision against a partial inventory is measured where the inventory claims
  completeness: on council channels at roads and rail (implied crossings), within council asset scope.
- **Compute:** development on Matt's workstation (RTX 4000 Ada), full training and inference on the
  RTX 6000 machine (192 GB, 32 cores).
- **Phases:** ML-0 dataset builder; ML-1 candidate scorer (gradient boosting baseline, then patch CNN);
  ML-2 link decisions and network assembly; ML-3 dense multi-task U-Net (channels with clDice, crests,
  crossing heatmap); ML-4 bed elevation.

## Tested versions (pip, sandbox, 30 Sep 2026; scikit-image 0.26.0 added 1 Oct 2026)

Python 3.12.3; numpy 2.4.4; scipy 1.17.1; numba 0.67.0; pyflwdir 0.5.12; rasterio 1.5.1;
geopandas 1.2.0; shapely 2.1.2; pyogrio 0.13.0; matplotlib 3.10.8; pytest 9.1.1;
whitebox-workflows 2.0.6. The conda environment files were **not** solved in the sandbox
(no conda); report any solve issues.
