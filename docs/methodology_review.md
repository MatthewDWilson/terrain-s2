# terrain-s2: methodology, rationale and relation to other work

**Prepared:** 6 October 2026
**Scope:** the Stage 2 processing as implemented in `terrain-s2` (iteration 4j): channel (stream/drain) network mapping and culvert/crossing detection from a 1 m LiDAR DEM (+ DSM; optional road centrelines).
**Purpose:** to set out what each processing step does, why it was chosen, where similar methods have been used and how well they performed, so the approach can be reviewed against the literature before investing in training.
**Companion documents:** `EDDIE_terrain_stage2_design.md` (design), `DESIGN_NOTES.md` (implementation log, §4a–4j).

---

## 1. Summary

The pipeline is **physics- and geometry-based with a small amount of learning**. It derives terrain features from the DEM, maps channels with thresholds, assembles them into a network, and proposes crossings from several kinds of geometric evidence. It then ranks them with transparent rules. This is the "propose" half of the design's *physics proposes, learning disposes* principle; the "disposes" half (learned classifiers) is still at an early stage.

Against the four test sites (two Northland, two Waimakariri), it now finds every known culvert within 10 m. Of the labelled high-tier crossings, 80–83% are culverts at the two sites with most labels. Stream/drain classification reaches 0.76–0.86 length-weighted accuracy, trained on one site and tested on the other. These results rest on about 50 labelled culverts and about 250 labelled channel reaches.

The closest published work reaches similar or better accuracy with **far more training data**:
- Lidberg et al. (2023) trained on 1,607 km of digitised ditches.
- A Swedish culvert study trained on 24,083 field-surveyed culverts.
- US drainage-crossing classifiers use thousands of labelled image chips.

The main gap is therefore training and validation data, not the choice of features or architecture. §8 sets out recommendations.

**Key comparison**

| Task | Published result | Data used | terrain-s2 (current) |
|---|---|---|---|
| Ditch mapping, forest (Sweden) | 86% of ditch channels found, MCC 0.78 (Lidberg et al., 2023) | 1,607 km digitised ditches, 10 regions | Council drains found 62% / 83% (Waimakariri, within 5 m) |
| Ditch mapping, low-relief farmland (Delmarva, USA) | precision 0.88, recall 0.89 (Du et al., 2024) | DL segmentation, lidar features | – (no complete truth yet) |
| Ditch centrelines, farmland (Flanders) | omission 0.03–0.14, commission 0.07–0.08 (Roelens et al., 2018) | point cloud + RGB, random forest | – |
| Ditch vs natural stream (Sweden) | ditches recall 76%, precision 88%; streams "moderate"; streams often mistaken for ditches (Busarello, 2025) | national, DL + terrain indices | 0.86 / 0.76 accuracy by length, leave-one-site-out |
| Road culvert detection (Sweden) | 87% of test culverts mapped (J. Hydrol. Reg. Stud. 57, 102148, 2025) | 24,083 field-surveyed culverts | all known culverts within 10 m; high tier 15/18, 23/31 |
| Drainage-crossing image classification (US Midwest) | > 90% accuracy (GIScience & Remote Sensing, 2023); EfficientNetV2 (Edidem et al., 2025) | labelled DEM chips | not yet implemented (design §4.3) |
| Breach-based culvert finding (Latvia) | ~30% of predicted connections were culverts (2020) | breach depressions | Test B alone: 0 / 40 at SH12; 43% at AOI2 |

---

## 2. Problem and inputs

**Problem.** In a bare-earth LiDAR DEM, roads, tracks and driveways appear as solid embankments across channels, because culverts are below the surface and bridges are removed or interpolated. Flow models then pond water behind embankments or route it along road toes. This is well documented. Without mapped drainage crossings, modelled flowlines stop short or cross roads in the wrong places (Poppenga & Worstell, 2016, as cited by Edidem et al., 2025). Early LiDAR hydrologic-DEM work made the same point: culverts and bridges behave as flow barriers unless they are burned into the DEM (Tang et al., 2013).

**Inputs and the reason for each.**
- **DEM, 1 m (required).** LINZ elevation (native EPSG:2193, NZVD2016) or the Stage 1 raw DEM. Never a bathymetry-burned DEM (see `DESIGN_NOTES` §4e), because burned channels would erase the evidence we detect. The core is DEM-only so that it stays location-agnostic.
- **DSM (optional).** Used for vegetation and structure evidence: DSM − DEM over a barrier, and bridge decks over continuous beds.
- **Road centrelines (optional).** For example LINZ Topo50. A road crossing a channel is strong evidence wherever roads are mapped, and roads explain roadside drains. Every result is also reported DEM-only, so the value of each optional layer is measured.

---

## 3. Processing chain (as implemented)

```
DEM (+DSM, roads)
 ├─ 3.1 terrain features ............ features.py   (median high-pass, top-hats, Hessian, openness)
 ├─ 3.2 hydrological conditioning ... hydro.py      (priority-flood fill, least-cost breaching, D8)
 ├─ 3.3 channel map ................. channels.py   (thresholds; slope/flow gate) → drains.py (hysteresis)
 ├─ 3.4 network ..................... drains.py, network.py (skeleton, joins, reaches, smoothing)
 ├─ 3.5 stream / drain .............. network.py    (reach features, prior, graph smoothing)
 ├─ 3.6 crossing evidence ........... crossings.py, drains.py, products.py
 │      Test A (channel ends bridged across a raised barrier), Test B (pond behind a barrier),
 │      network gaps and bed bumps, road–drain intersections and road end pairs
 └─ 3.7 confidence tiers ............ products.py   (points score; bridge rule; DEM-only tier)
Outputs: channels (reaches), crossings (points + tiers), repairs (continuity joins)
```

---

## 4. Methods, rationale and prior use

Each step below covers: **what** it does, **why** it was chosen, **prior use** (where it has been used and how well it performed), and **our experience**.

### 4.1 Terrain features

**What.**
- **Median high-pass:** DEM minus a median filter, at 5, 11 and 21 m.
- **White and black morphological top-hats:** at 11, 21 and 41 m.
- **Hessian eigenvalue line measures:** "valley" and "ridge" strength, at σ = 2 and 4 m.
- **Positive and negative openness:** within 20 m.

All parameters are in metres, so the method is independent of resolution.

**Why.**
- Channels and embankments are *narrow linear relief* on top of broader terrain, and each of these filters isolates that scale.
- The median high-pass follows the Swedish ditch work.
- Top-hats were preferred at embankment scale because they are about 250× cheaper than a 41 m median on CPU, and they preserve planar slopes exactly.
- Hessian line measures are the standard tool for curvilinear structures (Frangi et al., 1998).
- Openness is a well-established relief visualisation and terrain index (Yokoyama et al., 2002).

**Prior use and success.**
- The high-pass median filter (HPMF) on a 1 m DEM is the *only* input of the Swedish national ditch model, which kept it cheap enough to run nationally (Lidberg et al., EGU 2022). In an earlier Swedish study, the HPMF and an impoundment index were among the strongest predictors of ditches (Paul, Ågren & Lidberg, EGU 2021).
- Threshold methods on relative elevation (the Relative Elevation Attribute) are used widely in agricultural landscapes (Cazorzi et al., 2013; Rapinel et al., 2015; Roelens et al., 2018), but mostly over small study areas (as noted in a 2022 Latvian review). The Swedish work moved past this with deep learning.
- For US drainage-crossing classification, the most useful layers were simple: the DEM itself or a 21-cell topographic position index on its own did best, and stacking five derived layers gave no improvement (Southern Illinois University MSc, 2024).

**Our experience.**
- The features separate channels from paddocks well on flat land.
- On hillsides, convexity and roughness pass the thresholds, which the slope/flow gate in §4.3 addresses.
- The 21 m black top-hat alone isn't specific: it fires on concave embankment toes and hollows.

### 4.2 Hydrological conditioning (fill, breach, D8)

**What.**
- **Priority-flood depression filling** (Barnes et al., 2014).
- **Least-cost depression breaching** (after Lindsay, 2016), implemented in-house so that each breach *path* is kept.
- **D8 flow directions and upstream area** (O'Callaghan & Mark, 1984), via `pyflwdir`.

**Why.**
- Breach paths through raised features are natural culvert hypotheses (Test B).
- A breached DEM is also the conditioned DEM product.
- We cross-checked against Whitebox Workflows NG's breaching: the same breaches, within about 0.1 m.

**Prior use and success.**
- Breaching in flat drained land has a mixed record as a *culvert finder*. A Latvian study that used breach analysis to locate ditch culverts matched only 23 of 68 modelled culverts; with a second method, only 14 of 47 identified ditch connections (about 30%) were culverts (Engineering for Rural Development, 2020).
- US work notes that neither breaching nor burning removes road barriers reliably unless the crossing locations are known (Lessard et al., 2023, as cited by Edidem et al., 2025).
- A US agricultural comparison found stream burning with D8 or D-Infinity gave the best flowlines from high-resolution DEMs, which presupposes known crossings (NSF award 1951741 outputs).

**Our experience.**
- Test B (a breach through a raised barrier with a retained pond) found no culverts at SH12 (0 of 40 candidates over two reviews).
- At AOI2 it found 43–50%, mostly driveway culverts where a roadside drain ponds behind a driveway.
- Its strongest value turned out to be **channel obstructions**: vegetation over rivers and drains that should be breached (a §6.1 conditioning product), and stopbank gaps (§5.4).
- D8 upstream area is unreliable on flat land and is not used for classification.

### 4.3 Channel map

**What.** A cell is "channel" if it is part of a narrow, linear, incised feature at either of two scales:
- **narrow:** Hessian valley measure at σ = 2 m ≥ 0.03, plus an 11 m black top-hat ≥ 0.15 m;
- **wide:** Hessian valley measure at σ = 4 m, plus 21 m and 11 m top-hats.

On slopes steeper than 10%, a channel cell must also carry at least 1,000 m² of upstream area. For the drain network, weaker evidence is then admitted only where it is **connected to strong channel cells** (hysteresis).

**Why.**
- The thresholds are interpretable and need no training data.
- The slope/flow gate follows GeoNet's coupling of curvature with flow accumulation.
- Hysteresis (Canny, 1986) recovers shallow or tree-covered continuations of real drains without admitting isolated irrigation furrows. Border-dyke striping in the Canterbury1 paddocks made a blanket threshold reduction risky.

**Prior use and success.**
- GeoNet (Passalacqua et al., 2010) smooths the DEM by nonlinear diffusion (Perona & Malik, 1990), then traces channels as minimum-cost (geodesic) paths whose cost combines flow accumulation and contour curvature. It was demonstrated on a natural catchment (Skunk Creek, California), and later extended to flat and engineered landscapes (Passalacqua et al., 2012).
- Threshold methods work well in small open agricultural areas.
- Deep learning now leads at scale. In Delmarva, a CNN clearly outperformed a random forest, LiDAR-derived terrain features mattered most, and the connected ditch network reached precision 0.88 and recall 0.89 (Du et al., 2024).

**Our experience.**
- Against Waimakariri council channels, within 5 m:
  - streams 94–96% found;
  - council scheme drains 62% (Canterbury1) and 83% (Canterbury2).
- Misses are broad shallow swales (about 0.15 m deep; Canterbury1), and drains under shelterbelts, where the DEM is interpolated from sparse ground returns (Canterbury2).
- Over-prediction (spurious channels) is visible on review but **not yet quantified**, because the council layer is partial and excludes roadside and farm drains.

### 4.4 Network assembly

**What.**
1. Skeletonise the channel mask (Zhang & Suen, 1984, via scikit-image).
2. Prune spurs shorter than 10 m.
3. Find channel ends and their approach directions.
4. Join pairs of ends that **face each other** within 30 m (each pointing within 30° along the gap).
5. Only then remove isolated fragments shorter than 15 m.

The joined skeleton becomes a graph of junction and end nodes. Branches are chained into **reaches** by continuing the straightest path through each junction (≤ 45°). Reaches are smoothed (moving average, ends fixed at junction centres) and generalised (Douglas & Peucker, 1973, at 0.5 m tolerance).

**Why.**
- Short drain pieces between driveways (every 20–40 m) were being removed before they could be paired, so joining has to come before filtering.
- Stream-or-drain is a property of a reach, not of a 5 m piece between junctions.
- The facing-ends rule was Matt's proposal: two close ends pointing at each other probably belong together.

**Prior use and success.**
- Roelens et al. (2018) similarly reconstruct "ditch dropout" points in blind zones and assemble ditch objects into centrelines. Their best model had omission and commission errors of 0.03 and 0.08 in grassland, and 0.14 and 0.07 in a peri-urban area.
- Cost-minimising connection of ditch fragments is also used for aerial images in Finland.
- Vectorising raster channel predictions is a recognised quality issue in its own right; a Finnish national-mapping pilot set out to compare vectorisation methods for CNN outputs (ICA Abstracts 3, 158, 2021).

**Our experience.**
- Fragmentation fell from 21–24 to 6–14 channel ends per km.
- Lines now have 4–5 vertices per reach instead of one per cell.
- There are still 630–1,045 reaches per site, many of them short.

### 4.5 Stream / drain classification

**What.** For each reach:
- **bends per 100 m**, with turns of 45° or more (field corners) discounted;
- **share of length in straight runs** (≥ 25 m after 1.5 m simplification);
- **roadside fraction** (within 20 m of a road and within 25° of parallel);
- **width**, **incision** and **length**.

A transparent prior gives p(stream). It is then smoothed along the graph:
- short and low-confidence reaches are pulled towards their neighbours, since type rarely changes along a channel;
- an outflow reach is pulled harder by the reaches flowing into it, since drains feed streams and rarely the reverse;
- flow direction comes from the bed trend along each reach.

**Why.** These rules came from Matt's review: straight means drain, sinuous means stream, roadside means drain, and type continuity holds along a channel. A logistic model trained on council classes agrees: straight runs and roadside fraction point to drain, width to stream.

**Prior use and success.**
- Distinguishing natural streams from ditches is recognised as hard: Swedish models detected ditches well but often labelled natural streams as ditches (Busarello et al.).
- The Swedish national framework reports ditch recall 76% and precision 88%, with only moderate results for natural streams. It describes itself as the first to separate the two across a whole country (Busarello, 2025).
- Its training labels used a separate protocol per type: ditches were digitised from hillshade, HPMF and orthophotos, while streams were traced downstream from mapped channel heads and then edited against orthophotos (SND 2024-57).

**Our experience.**
- Validation is leave-one-site-out against Waimakariri `CLASSIFICATION3`, length-weighted:

  | | Canterbury1 | Canterbury2 |
  |---|---|---|
  | Prior | accuracy 0.86, AUC 0.96 | 0.72, AUC 0.91 |
  | Prior + smoothing | **0.86**, AUC 0.97 | **0.76**, AUC 0.91 |
  | Trained logistic model | 0.88, AUC 0.99 | 0.58, AUC 0.84 |

- With only two labelled sites, the trained model does not generalise as well as the rules. The prior plus smoothing is the default.

### 4.6 Crossing (culvert) evidence

Several independent kinds of evidence are combined, because no single one finds all culvert types.

**(a) Test A: channel ends bridged across a raised barrier (gap bridging).**
- **What:**
  - From each channel end that points at a raised feature, a least-cost search (cost = 1 − channel likelihood) runs within 60 m to channel cells that are not the end's own nearby channel. A target is accepted only once the path has crossed a raised feature and risen ≥ 0.3 m above both beds.
  - The barrier must be ≥ 0.3 m high (h_b), 4–60 m thick (L_b), elongated (traced along its crest relative to its half-height width) and crossed at ≥ 45°.
  - Accepted paths become **culvert links** in the network.
- **Why:** this is the only test that finds culverts on DEM drainage divides, such as the SH12 culvert, where both sides drain away from the road and nothing ponds.
- **Continuation rule (§4l):** a path also counts as a crossing when the same channel continues through the barrier: both sides at least 30° off the barrier axis and within 30° of each other. This catches oblique stream crossings such as AOI2 culvert #2 (42.6° to the road), while bank hops between channels running alongside a barrier still fail.
- **Prior use:** the concept (a channel interrupted by an embankment, with the gap bridged) follows the design's §4.1 step 5. Road–stream intersection approaches (Lindsay & Dhun, 2015) and the use of road centrelines for drainage structures ("Mapping Drainage Structures Using Airborne Laser Scanning by Incorporating Road Centerline Information") are related.

**(b) Test B: ponding behind a barrier** (§4.2).

**(c) Network gaps and bed bumps.**
- **What:**
  - A *gap* is a pair of facing ends whose drain bed rises ≥ 0.3 m across the gap.
  - A *bump* is a section of a continuous drain whose bed (the lowest ground within 1 m of the centreline) rises ≥ 0.3 m above the higher reference bed 15 m either side.
- **Why:** a driveway sits at field level; what rises is the *drain bed*, not the surrounding ground. Gaps with a bed rise of less than 0.15 m are *continuity repairs*.

**(d) Roads (optional).**
- **What:** road–drain intersections, and pairs of drain ends that stop at a road and face each other across it.
- **Why:** a road crossing a mapped channel is strong evidence, and it covers the DEM's weakest cases (wide motorway fills, roads under trees).

**Prior use and success (crossings in general).**
- The most directly comparable work is Swedish. A Residual Attention U-Net trained on field-surveyed culverts mapped 87% of the culverts in the test watersheds, but adding them to DEM preprocessing improved extracted stream networks only slightly (J. Hydrol. Reg. Stud., 2025). The Swedish Forest Agency's survey behind it located 24,083 culverts by GPS, to about 0.3 m.
- In the US Midwest, CNNs classifying DEM image chips (crossing present or absent) exceeded 90% accuracy, with the DEM alone the best input (GIScience & Remote Sensing, 2023). Object detectors in the same programme report average precision above 97% (NSF award 1951741 outputs). These classify or detect *chips around candidate locations*, which is the role the design gives the CNNs in §4.3.

**Our experience.** All known culverts are found within 10 m at all four sites: SH12 3/3, AOI2 18/18, Canterbury2 31/31, Canterbury1 4/4.

### 4.6b Floodplain restriction, cleaning and rivers (R1 iteration, §4k of the design notes)

**What.**
- **Floodplain:** HAND, the height above the nearest drainage along D8 flow paths (Nobre et al., 2016), at most 10 m (5 m fell short of the true floodplain at AOI2, because the drainage reference is itself an imperfect network). Drainage is wide channels plus cells with upstream area ≥ 0.1 km². The network and crossings are restricted to the floodplain plus 50 m.
- **Cleaning:** drop isolated network pieces under 50 m, small pieces under 100 m whose median incision is under 0.4 m, and dangling reaches under 15 m.
- **Rivers:** channels ≥ 8 m wide become polygons, each with a single centreline.

**Why.**
- Matt's review showed the network misses little on the floodplain. The errors were spurious small pieces, drains classified as streams, and multiple lines in rivers.
- The floodplain is where the information matters for flood modelling, and HAND needs no extra data.

**Prior use.** HAND was introduced as a proxy for inundation extent and is widely used to delineate floodplains from DEMs (Nobre et al., 2016).

**Our experience.**
- 28 of 50 labelled spurious reaches removed, and 21 of 22 real ones kept.
- Council stream/drain accuracy by length rose to 0.91 / 0.89.
- Cleaning thresholds trade spurious removal against shallow real drains, so a learned channel model (R3) is the lasting fix.

### 4.7 Confidence tiers

**What.** Bridges are classed first: a road over a continuous bed with a DSM deck ≥ 2 m. Every other crossing gets a points score:

| Evidence | Points |
|---|---|
| Within 15 m of a road | 2 |
| Network evidence (gap, bump or culvert link) | 2 |
| Two or more independent sources | 1 |
| A road–drain source | 1 |

- **High** needs at least 3 points and **medium** at least 2. Both require a barrier ≥ 0.3 m with no vegetation flag.
- A DEM-only tier drops the road terms.

**Why.**
- The weights reflect which evidence separated 51 labelled culverts from 123 non-culverts: within 15 m of a road, 84% vs 25%; a drain gap, 47% vs 2%; two or more sources, 71% vs 21%.
- Earlier hard filtering rules (kind, vegetation, approach length, minimum width) each discarded real culverts on a site they weren't tuned on. Scored evidence degrades more gracefully.

**Prior use.** The design's §4.3 plans a CNN classifier to replace this scoring. The published crossing classifiers above suggest it would work, but only with substantially more labelled examples.

**Our experience.**
- At AOI2 and Canterbury2, 80–83% of the labelled high-tier crossings are culverts; at SH12 and Canterbury1 the share is lower.
- Most high-tier crossings are unlabelled.
- The weights were derived from labels on earlier, filtered candidate sets, so they carry that bias (§6).

---

## 5. Validation design

**What.**
- **Blind tests:** new sites run with frozen parameters before the ground truth is revealed.
- **Leave-one-site-out scoring** for anything learned.
- **Matching:** within 10 m of labels or council asset lines; within 5 m for channels.
- **Reporting:** results with and without optional layers.
- **Benchmark:** the SH12 culvert is an automated test, `tests/test_benchmark_sh12.py`. It checks four things: the culvert is found in the high tier, the network passes through it, the reaches either side are streams, and the bridge is classed as a bridge.

**Why.**
- Each new site broke rules tuned on earlier ones (§6).
- Spatial separation of sites is the honest measure of generalisation.

**Prior use.**
- The Swedish studies use spatially separate test watersheds or regions and report recall and the Matthews correlation coefficient.
- Training and validation data quality is stressed repeatedly. The Finnish pilot, for example, identifies the positional accuracy and completeness of training data as critical to CNN results (ICA Abstracts, 2021).

**Current limitation.** Recall can only be measured against partial inventories (council assets; Matt's additions), and precision only on candidates that were reviewed. Over-prediction of channels is not yet measured.

---

## 6. Findings to date (lessons)

- **F1. Detection is good; ranking and classification are data-limited.** Without hard filtering rules, the candidate generators found every labelled or inventoried culvert at four sites. Every miss in earlier versions was caused by a rule tuned on another site.
- **F2. Each crossing test catches a different configuration.**
  - Test A finds culverts on drainage divides (SH12).
  - Test B finds culverts with ponding (driveways over roadside drains).
  - Network gaps and bumps find crossings within mapped drains.
  - Roads find crossings where the DEM evidence is weak.
- **F3. Breach-based culvert finding has low precision on its own**, as in the Latvian study (about 30%). Its value lies in channel obstructions and embankment gaps.
- **F4. Training on labels from a filtered candidate set biases the model.** The first learned ranking failed on Canterbury1 because whole classes of candidates were absent from its training labels. Labels must come from the same candidate generator that is being ranked.
- **F5. Stream/drain is a reach-scale, network-scale property.** Segment-level classification was no better than chance on one site; reach-level rules with network smoothing reached 0.76–0.86 accuracy.
- **F6. The DEM itself limits detection under vegetation.** Drains under shelterbelts and roads under trees are interpolated from sparse ground returns. The DSM, point-cloud classes (e.g. water returns, class 9) and roads are the available remedies.
- **F7. Our labelled data are about two orders of magnitude smaller than those behind the published high-accuracy results.**
  - Here: about 50 culverts and about 250 reaches.
  - Swedish culvert study: 24,083 culverts.
  - Swedish ditch study: 1,607 km of ditches.

---

## 7. Assessment against the literature

The literature points to the same overall shape: geometric and hydrological methods to propose, and learned models (U-Net segmentation for channels; CNN classification or detection for crossings) to decide. Our physics-first stages match what the published work identifies as the important inputs (local relief or HPMF, DEM-derived curvature, simple layers). Our generators reach high recall on the cases we know about.

Where we fall short is on the learning side. The published models are trained on large, consistent inventories:

| Task | Published training data | Ours |
|---|---|---|
| Ditches | Sweden: manually digitised | Waimakariri council drains (partial) |
| Culverts | Sweden: field-surveyed | ~50 labelled |
| Drainage crossings | US Midwest: labelled image chips | – |

Transfer learning has helped where local data are scarce. An Estonian study pretrained a U-Net on the Swedish labels (Lidberg et al., 2023), then fine-tuned it on a small Estonian sample (Virro et al., 2025).

Two NZ specifics differ from the Swedish case and need local data:
- **Agricultural drains under shelterbelts,** and border-dyke irrigation micro-relief.
- **Driveway and field-entrance culverts across roadside drains,** our most common positive.

---

## 8. Recommendations

- **R1. Build complete ground-truth windows** (`truth_window`, `truth_channels`, `truth_crossings`), e.g. 2–3 windows of 300–500 m square per site. This is the only way to measure channel over- and under-prediction and true culvert recall. Also label existing reaches (`exists`, `true_class`).
- **R2. Use inventories at scale.** Waimakariri's stormwater culverts and channels are an example of council data that can supply hundreds to thousands of labels. Other councils and NZTA (RAMM) may hold more. Record their scope and positional accuracy, and treat absence as "unknown", not as a negative.
- **R3. Learned channel segmentation.** Train a U-Net on the median high-pass and DEM derivatives (as in Lidberg et al., 2023), starting from the published Swedish models and code (open data and code are available) and fine-tuning on NZ windows, following the Estonian transfer-learning example. Keep the current threshold map as a baseline and as an input feature.
- **R4. Learned crossing classification** (design §4.3). Classify chips centred on the current candidates, as in the US Midwest studies, with labels gathered from the *same* candidate generator (F4). The current generators provide the proposals and engineered features; the CNN replaces the points score once a few hundred labels per class exist.
- **R5. Comparable metrics.** Report recall, precision and MCC per site, plus length-weighted precision and recall for channels, so results can be compared with the published figures in §1.
- **R6. Point-cloud evidence.** Ground-return density, and water returns (class 9) in channels, for vegetation-affected locations (F6). This is optional in the design and now looks worthwhile.
- **R7. Keep the physics-based generators.** They provide high recall, explanations (h_b, L_b, approach, bed rise) and the network topology that learned per-pixel models don't provide on their own. Several studies also note that per-pixel predictions need vectorising and connecting afterwards.

---

## References

Findings from other studies are paraphrased from sources located during preparation (6 October 2026). Two entries marked † were not re-verified here (bibliographic details from prior knowledge); please check before external use.

- Barnes, R., Lehman, C. & Mulla, D. (2014). Priority-flood: an optimal depression-filling and watershed-labeling algorithm for digital elevation models. *Computers & Geosciences* 62, 117–127.
- Busarello, M. dos S. T. (2025). *Mapping small streams and ditches with high-resolution topographic data and machine learning* (doctoral thesis, SLU); and Busarello, Lidberg, Ågren & Westphal, Automatic detection of ditches and natural streams from digital elevation models using deep learning (data and code: SND 2024-57, doi:10.5878/jrex-z325).
- Canny, J. (1986). A computational approach to edge detection. *IEEE Trans. Pattern Analysis and Machine Intelligence* 8(6), 679–698.
- Cazorzi, F. et al. (2013). Drainage network detection and assessment of network storage capacity in agrarian landscape. *Hydrological Processes*.
- Classification of drainage crossings on high-resolution digital elevation models: a deep learning approach (2023). *GIScience & Remote Sensing* (NSF grant 1951741).
- Deep learning-enhanced detection of road culverts in high-resolution digital elevation models: improving stream network accuracy in Sweden (2025). *Journal of Hydrology: Regional Studies* 57, 102148 (data: SND 2024-140, doi:10.5878/rjpg-ec44).
- Douglas, D. H. & Peucker, T. K. (1973). Algorithms for the reduction of the number of points required to represent a digitized line or its caricature. *Cartographica* 10(2), 112–122.
- Du, L. et al. (2024). Drainage ditch network extraction from lidar data using deep convolutional neural networks in a low relief landscape. *Journal of Hydrology* 628.
- Edidem, M., Xu, B., Li, R., Wu, D., Rekabdar, B. & Wang, G. (2025). Deep learning classification of drainage crossings based on high-resolution DEM-derived geomorphological information. *Frontiers in Artificial Intelligence* 8, doi:10.3389/frai.2025.1561281.
- Frangi, A. F., Niessen, W. J., Vincken, K. L. & Viergever, M. A. (1998). Multiscale vessel enhancement filtering. *MICCAI 1998*, LNCS 1496, 130–137.
- Identification of possible ditch culvert locations using LiDAR data (2020). *Engineering for Rural Development*, Latvia University of Life Sciences and Technologies.
- Lidberg, W., Paul, S. S., Westphal, F., Richter, K. F., Lavesson, N., Melniks, R., Ivanovs, J., Ciesielski, M., Leinonen, A. & Ågren, A. M. (2023). Mapping drainage ditches in forested landscapes using deep learning and aerial laser scanning. *Journal of Irrigation and Drainage Engineering* 149(3), 04022051, doi:10.1061/JIDEDH.IRENG-9796.
- Lindsay, J. B. (2016). Efficient hybrid breaching-filling sink removal methods for flow path enforcement in digital elevation models. *Hydrological Processes*. †
- Lindsay, J. B. & Dhun, K. (2015). Modelling surface drainage patterns in altered landscapes using LiDAR. *International Journal of Geographical Information Science*. †
- Nobre, A. D., Cuartas, L. A., Momo, M. R., Severo, D. L., Pinheiro, A. & Nobre, C. A. (2016). HAND contour: a new proxy predictor of inundation extent. *Hydrological Processes* 30, 320–333.
- O'Callaghan, J. F. & Mark, D. M. (1984). The extraction of drainage networks from digital elevation data. *Computer Vision, Graphics, and Image Processing* 28(3), 323–344.
- Passalacqua, P., Do Trung, T., Foufoula-Georgiou, E., Sapiro, G. & Dietrich, W. E. (2010). A geometric framework for channel network extraction from lidar: nonlinear diffusion and geodesic paths. *Journal of Geophysical Research: Earth Surface* 115, F01002, doi:10.1029/2009JF001254.
- Passalacqua, P., Belmont, P. & Foufoula-Georgiou, E. (2012). Automatic geomorphic feature extraction from lidar in flat and engineered landscapes. *Water Resources Research* 48.
- Perona, P. & Malik, J. (1990). Scale-space and edge detection using anisotropic diffusion. *IEEE Trans. Pattern Analysis and Machine Intelligence* 12(7), 629–639.
- Poppenga, S. & Worstell, B. (2016), as cited in Edidem et al. (2025).
- Paul, S. S., Ågren, A. & Lidberg, W. (2021). Detection of drainage ditches using high-resolution LIDAR data in the Swedish forest. *EGU General Assembly 2021*, EGU21-2226.
- Lidberg, W., Paul, S., Westphal, F. & Ågren, A. (2022). Mapping Sweden's drainage ditches using deep learning and airborne laser scanning. *EGU General Assembly 2022*, EGU22-4639.
- Piloting the use of machine learning methods for automatic mapping of streams and ditches in Finland (2021). *Abstracts of the ICA* 3, 158.
- Rapinel, S. et al. (2015), as cited in the Latvian and Estonian ditch-mapping studies.
- Roelens, J., Höfle, B., Dondeyne, S., Van Orshoven, J. & Diels, J. (2018). Drainage ditch extraction from airborne LiDAR point clouds. *ISPRS Journal of Photogrammetry and Remote Sensing* 146, 409–420, doi:10.1016/j.isprsjprs.2018.10.014.
- Tang, Z., Li, R., Li, X. & Winter, J. (2013). Drainage structure datasets and effects on LiDAR-derived surface flow modeling. *ISPRS International Journal of Geo-Information* 2(4), 1136.
- Virro, H., Kmoch, A., Lidberg, W., Muru, M., Chan, W. T., Moges, D. M. & Uuemaa, E. (2025). Detection of drainage ditches from LiDAR DTM using U-Net and transfer learning. *Big Earth Data* 9(2), 243–264.
- Yokoyama, R., Shirasawa, M. & Pike, R. J. (2002). Visualizing topography by openness: a new application of image processing to digital elevation models. *Photogrammetric Engineering & Remote Sensing* 68(3), 257–265.
- Zhang, T. Y. & Suen, C. Y. (1984). A fast parallel algorithm for thinning digital patterns. *Communications of the ACM* 27(3), 236–239.
