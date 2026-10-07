# terrain-s2 R2/R3: national training data and a learned channel model

Oct 7, 2026 · @Matt

The next thread builds the machinery to assemble training data from anywhere in New Zealand (R2), then trains a learned channel model on it (R3), starting with Waimakariri and adding other councils' data as it is brought in. It inherits a working, physics-based terrain-s2 pipeline whose remaining errors are now limited by training data, not by method.

## Purpose and starting point

R1 of the methodology review is closed (`DESIGN_NOTES` §4k–4l). The pipeline maps the floodplain channel network and proposes crossings from a 1 m LiDAR DEM, with optional DSM and road centrelines. Validation covers four sites: SH12 and AOI2 (Whirinaki, Northland) and canterbury1 and canterbury2 (Waimakariri).

| Measure (as of R1 close) | Result |
| --- | --- |
| Known culverts found within 10 m | all, at all four sites |
| Labelled high-tier crossings that are culverts | 20% (SH12), 54% (canterbury1), 58% (AOI2), 71% (canterbury2) |
| Council stream/drain accuracy by length | 0.91 (canterbury1), 0.89 (canterbury2) |
| Matt's real reaches kept / spurious removed | 21 of 22 / 28 of 50 |
| Benchmarks | SH12 culvert 4/4 checks; AOI2 culvert #2 2/2 checks |

Labelled data so far: about 100 culverts, 90 non-culverts and 120 reaches labelled by Matt, plus the Waimakariri council culvert and channel layers. The leading published systems trained on two orders of magnitude more: 1,607 km of digitised ditches (Lidberg et al., 2023) and 24,083 surveyed culverts (Sweden, 2025).

## Lessons and outstanding issues from R1

The hand-tuned stages now trade one error for another, so R2/R3 replace tuning with training data.

- **F1. Detection is good; classification and ranking are data-limited.** Unfiltered candidate generators found every labelled or inventoried culvert. Every earlier miss came from a rule tuned on another site.
- **F2. Labels must come from the same candidate generator that is being ranked.** The first learned ranking failed on canterbury1 because whole classes were missing from its training labels.
- **F3. Diminishing returns on rules.** Cleaning thresholds trade spurious removal against shallow real drains (0.4 vs 0.5 m incision). 38 of Matt's 41 round-two SH12 flags are short links between junctions, which no simple rule separates from real ones.
- **F4. Stream/drain is a reach- and network-scale property.** Reach rules plus smoothing along the network reach 0.89–0.91, but drains still read as streams in places.
- **F5. Partial inventories.** Council layers hold council assets only; roadside and private drains and driveway culverts are mostly absent. Absence is never a negative.

Outstanding issues the new work must address:

1. **Drains that do not pass through barriers.** Channels stop at a road, driveway or embankment where a culvert carries them through. This is the main visible error after R1 (Matt, 7 Oct 2026); it belongs to R3 (continuous channel probability through structures) and R4 (crossing classifier).
2. **Crossing precision** of 20–71% in the high tier.
3. **Spurious short links** in the network, especially near junctions.
4. **Remaining stream/drain confusion**, mainly drains classed as streams.
5. **Stopbanks and stopbank-like features** as an output to preserve in coarser meshes (design §5). Not started; the data machinery collects their data now, for testing only.

## Scope and goals

R2 makes training data a repeatable, nationwide product; R3 uses it to replace the threshold channel map with a learned one.

| Item | Goal | Done when |
| --- | --- | --- |
| R2 data machinery | Any NZ site assembled from a site definition: elevation, inventories, labels, provenance | The four current sites rebuild automatically and match the manual clips; a new Waimakariri AOI needs no manual steps |
| R2 inventories at scale | Waimakariri first, then other councils' channel and culvert layers, harmonised to one label schema | At least two councils' data in the training set, with scope and accuracy recorded |
| R3 channel model | Per-pixel probability of drain, stream and background from the DEM (and DSM) | Beats the threshold map on held-out councils, and both benchmarks still pass |
| R4 crossings (linked) | Classifier on chips around candidate crossings | Started once R2 supplies a few hundred labels per class from the same generator (F2) |
| Stopbanks (linked) | Stopbank and stopbank-like features as an output | Data collected by R2 for testing; method in a later pass |

Out of scope here: hill-country tracks (outside the floodplain mask), bathymetry, and the LISFLOOD-FP mesh itself.

## R2: data acquisition machinery

A site definition in, a reproducible labelled site out: every fetch is snapshotted with its query and checksum, so any score can be reproduced later.

&#91;embedded content: R2 data flow · site definition to site bundle and training chips\]

The snapshot store sits between every fetcher and every label, so a score always names the data it was computed from.

**Components**

- **Site registry** (`sites/<id>.yml`): AOI polygon, buffer, role (train / validation / test), council, sources and query parameters, licence notes. Sites are the unit of train/test separation; candidates of one site never cross splits.
- **Elevation fetcher**: LINZ nz-elevation on S3 (Cloud-Optimised GeoTIFFs, STAC catalogue, public, no sign-in). Prefer the *survey* collections (one survey per AOI, survey date and density recorded) over the national composite, which mixes surveys. Windowed reads for the AOI plus buffer; record STAC item ids and file checksums. Native CRS EPSG:2193; whether the files also label NZVD2016 is still to be checked with `gdalinfo` on one file. DEM and DSM must share the vertical datum.
- **Inventory fetchers**, one adapter per source behind a common interface:
  - *Waimakariri District Council* (first): stormwater channels (district-wide, 1,843 lines, 717 km; `CLASSIFICATION3` gives drain or stream) and stormwater culverts (`CLASSIFI_2 = Culvert`; `DIAMETER_m` holds millimetres).
  - *Other councils*: to be surveyed (see Decisions). Many publish through ArcGIS REST services, so one generic ArcGIS adapter should cover most, with a per-council field mapping.
  - *National and open*: LINZ Topo50 road centrelines (context input, already used), OSM waterways and culverts (patchy; weak labels only), NZTA culvert assets if access is granted.
  - *Stopbanks* (testing only): NZIS and regional-council stopbank layers where published.
- **Snapshot store**: each query result stored with URL, parameters, retrieval time, record count and content hash. Council layers change daily, so scores always cite a snapshot.
- **Harmoniser**: maps each source to the common label schema (next section) and records its scope (what the inventory claims to cover) and positional accuracy.
- **Storage**: outside the repository and OneDrive, located by configuration; cache keyed by source, item id and checksum. It is the same tile index and metadata store as the data pipeline design, not a separate one.

**Runs where:** fetching runs locally or on the workstation, since the analysis sandbox cannot reach the S3 or council hosts. Training runs on the C002IT GPU.

## Labels

Positive labels come mostly from council inventories, negatives only from places someone has actually checked, and every label records where it came from.

**Schema** (versioned; Matt's review files are version 0):

| Layer | Geometry | Key fields |
| --- | --- | --- |
| `channels` | line | class (stream / drain / other), exists (y / n / unsure), source, positional accuracy, labeller, date |
| `crossings` | point (+ line for culverts) | type (culvert / bridge / ford / floodgate / none), diameter or width × height, length, cells, invert levels, source |
| `stopbanks` | line | type (stopbank / road embankment / other raised), crest height where known, source |
| `scope` | polygon | what an inventory or review covers: complete, council-owned only, or reviewed candidates only |

**Partial-inventory semantics.**

- A council channel is a positive for its class; absence from a council layer is *unknown*, never background.
- Background (negative) labels come only from reviewed areas: Matt's `exists = n` reaches, `culvert = n` crossings, and any complete windows later.
- Training therefore masks the loss outside known positives and reviewed areas (positive-unlabelled setting), rather than treating unmapped land as "no channel".

**Aligning council lines to the DEM.** Council lines can sit metres off the channel they represent. Before rasterising, each line is snapped to the local channel bed: the lowest ground within ±5 m across the line, smoothed along it. Lines that cannot be snapped are flagged and left out, not forced. The R1 work already matched within 5 m for this reason.

**Chips for training.** Square chips (for example 256 × 256 cells at 1 m) centred on inventory features, plus chips sampled across each council's scope area, so the model sees ordinary paddocks too. Label rasters: drain, stream, background (where known), ignore (elsewhere). The channel label is the snapped centreline buffered by half the channel width where known, else a fixed 2 m.

**Labels for crossings (R4).** Chips around candidates from the *current* generator (F2), labelled from council culverts, Matt's reviews and the bridge rule.

## R3: learned channel model

A U-Net that outputs drain, stream and background probability per cell, trained on Waimakariri first, replaces the threshold channel map and hysteresis; the network, crossing and classification stages downstream stay as they are.

**Task.** Semantic segmentation with three classes plus ignore. The probability is meant to stay continuous *through* structures where a channel continues (outstanding issue 1): labels run the channel line through known culverts, so the model learns that a drain does not end at a driveway.

**Inputs** (stacked per chip, normalised per chip):

- high-pass median filter at 5, 11 and 21 m (the single input of the Swedish national ditch model);
- the existing terrain-s2 features: black and white top-hats, Hessian valley and ridge measures, openness;
- slope, and HAND (floodplain context);
- DSM − DEM, so vegetation over a channel is explicit (drains under shelterbelts).

Start with HPMF alone as a baseline, then add inputs one group at a time; published crossing work found extra layers did not always help.

**Architecture and transfer.** U-Net (Ronneberger et al., 2015), as in Lidberg et al. (2023) and Virro et al. (2025). Two starts, compared:

1. trained from scratch on NZ chips with augmentation (rotation, flips, elevation offsets);
2. initialised from Swedish models and fine-tuned on NZ chips, as Estonia did. The Swedish labelled data and code are published (SND datasets); whether trained weights are, and under what licence, is to be checked.

**Training regime.**

- Loss masked to known positives and reviewed negatives (Labels section); class weights for the rare stream class.
- Splits by site and, once a second council is in, by council (train on one, test on the other).
- Matt's review labels are held out as the clean test set, never used for training.
- Software: PyTorch on the C002IT GPU (CUDA 13), a separate environment from the CPU pipeline.

**Post-processing.** The probability raster feeds the existing network stage: threshold (or hysteresis on probability), skeleton, joins, reaches, smoothing, cleaning. Stream/drain probability from the model joins the reach features and the network smoothing, rather than replacing them.

## Integration with terrain-s2

The new work plugs in through two file contracts, so the existing scripts run unchanged and every result can still be reported with and without the learned model.

| Contract | Producer → consumer | Content |
| --- | --- | --- |
| Site bundle | R2 machinery → `run_network.py`, `score_labels.py`, review scripts | DEM and DSM clips (correct CRS labels, NZVD2016 recorded), roads, label layers in the schema, `site.json` provenance |
| Channel probability | R3 model → `run_network.py --channel-prob` (new option) | GeoTIFF on the DEM grid: p(drain), p(stream), p(channel); replaces the threshold mask and hysteresis when given |

**Stays physics-based:** floodplain (HAND), breaching and Tests A and B, network assembly and reaches, culvert links, crossings evidence and the bridge rule. These give recall, explanations (h\_b, L\_b, bed rise, continuation) and network topology that a per-pixel model does not provide alone. Per-pixel predictions need vectorising and connecting afterwards in any case.

## Evaluation

The learned model is judged against the current threshold pipeline on sites and councils it has never seen, with metrics comparable to the published work.

- **Splits:** by site; with two or more councils, leave-one-council-out. Matt's labelled reaches and crossings stay a held-out test set.
- **Channel metrics:** precision and recall by length within 5 m of snapped reference lines, and MCC per site (as reported by Lidberg et al., 2023, and Du et al., 2024); spurious reaches removed and real reaches kept against Matt's labels; stream/drain accuracy by length against council classes.
- **Through-structure continuity:** the share of known culverts where a channel line passes within 10 m on both sides and through the crossing. This measures outstanding issue 1 directly.
- **Benchmarks (must pass):** SH12 culvert (4 checks) and AOI2 culvert #2 (2 checks), run as tests when the site data are present.
- **Baseline:** the R1 pipeline's scores (Starting point table), reported alongside every new result.

## Milestones

Five milestones, each closed by a gate; no dates are set yet.

&#91;embedded content: milestones M1–M5 · each with its gate · R4 and stopbank data in parallel\]

M1 and M2 are the R2 machinery and can start now. M3 needs the M2 chip set. The R4 crossing classifier starts from M2 because it needs the same label machinery (F2).

## Decisions and open questions

**Decided** (Matt, Oct 2026):

- **D1.** Waimakariri is the starting point; other councils' data join the training as they are brought in.
- **D2.** Floodplain only, by HAND ≤ 10 m from the DEM alone; no extra data requirement.
- **D3.** Roads are an optional input; results are always also reported DEM-only.
- **D4.** Stopbanks are a required output for a later pass; their data are collected now, for testing only.
- **D5.** R2 and R3 run in a separate thread; this document is its starting point.

**Open:**

- [ ] **Q1.** Which councils next? Survey which publish open channel and culvert layers, prioritising contrasting landscapes (for example Northland, Waikato, Hawke's Bay) over more Canterbury plains.
- [ ] **Q2.** NZTA culvert assets: is access possible, and under what terms?
- [ ] **Q3.** Swedish models: are trained weights published, and under what licence?
- [ ] **Q4.** How much further review labelling can Matt give, and in what form (reach flags as now, or complete windows)?
- [ ] **Q5.** Data storage location for snapshots and chips on the workstation, and backup.
- [ ] **Q6.** Does NZVD2016 appear in the LINZ S3 GeoTIFF metadata (`gdalinfo` on one file)?

## References and pointers

**In the terrain-s2 repository**

- `docs/methodology_review.md`: methods, rationale and the literature, with findings F1–F7 and recommendations R1–R7. This document carries out R2 and R3.
- `DESIGN_NOTES.md`: §4a–4l (implementation log and results per iteration), §5 (proposed design amendments, including stopbanks), §7 (automated test-data acquisition, the first design of the R2 machinery).
- Code the new work plugs into: `scripts/run_network.py`, `src/terrain_s2/network.py`, `products.py`, `floodplain.py`; benchmarks `tests/test_benchmark_sh12.py` and `tests/test_benchmark_aoi2.py`.

**Planning documents:** `EDDIE_terrain_stage2_design.md` (§4.3 CNN classifiers, §5 embankments) and `EDDIE_terrain_plan.md`.

**Data in hand** (outside the repository): LINZ 1 m DEM and DSM clips for the four sites; Waimakariri stormwater channels (district-wide) and culverts for both AOIs; LINZ Topo50 roads for the four AOIs; NZIS stopbank at canterbury1 (testing only); Matt's labels (review rounds 1 and 2: reaches and crossings).

**Literature** (full list in the methodology review): Lidberg et al. (2023), ditch mapping with deep learning; Virro et al. (2025), U-Net transfer learning to Estonia; Du et al. (2024), low-relief ditch networks; the Swedish road-culvert study (J. Hydrol. Reg. Stud. 57, 2025); Busarello (2025), streams versus ditches; Nobre et al. (2016), HAND; Ronneberger et al. (2015), U-Net.
