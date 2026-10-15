# ML-3: a learned channel map (design, 17 Oct 2026)

Agreed with Matt on 17 Oct (section 7 records the decisions). Phase names follow DESIGN_NOTES section 8 (ML-3: dense
multi-task U-Net). Evidence is in DESIGN_NOTES section 11. The handover for the work from ML-3a on is
`docs/handover_ml3.md`.

## 1. Why, and what it must do

The scorer (ML-1) now ranks well among the candidates the chain proposes. On waves 1 and 2 (345 tiles), leave one
council out, recall@100 per tile is 0.30 (Waimakariri) to 0.73 (Environment Southland), against 0.13-0.45 for the
tiers. Counted only over labels that have a candidate, Waimakariri council culverts reach 0.45 and Environment
Southland implied crossings 0.81.

The limit is upstream of the scorer: 1,085 labels have no candidate at all, 359 of them council culverts. The miss
diagnosis (12 Waimakariri tiles) says why. Of 57 council culverts lost, 43 have no channel mapped within 10 m; 11 have
the network passing with no crossing proposed; 3 were cleaned away. The lost ones sit on catchments of ~500 m2
(13 ha for those found), behind low barriers (0.34 m against 0.92 m), 30-50 m from anything the channel map holds.
They are small field and roadside drains.

ML-3's job is to put those drains in the channel map, so the existing crossing tests and the scorer can see them.

- **Primary output.** A per-cell channel probability on the 1 m grid, and from it centrelines that join the physics
  network.
- **Secondary output.** Stream or drain, per cell (a second head). It replaces the reach classifier where
  misclassified streams currently distort the stream REM (DESIGN_NOTES 7m).
- **Success measure, end to end.** The share of register culverts and implied crossings that get a candidate
  ("found" in the probes) rises, without recall@100 falling.
- **Intermediate measure.** The length of council channel recovered within 3 m. Its main use is to tune the model.

## 2. Labels

| Source | Content | Status |
|---|---|---|
| Waimakariri council channels | scheme drains, rivers | 2,662 lines in wave 1 |
| HBRC channels (drain, river channel) | scheme drains | 741 |
| Environment Southland drainage | scheme drains (named, long) | 336 |
| Auckland channels | stormwater open channels | 10,833 |
| ECan drains and watercourses (bylaw) | regional drains | not yet enabled (one probe) |
| Matt's channel review (`network_*_review2.gpkg`) | `exists` y/n per reach, class | SH12 (41 spurious), rounds 1 |

None of these registers is complete, especially for private drains, so the labels are positive-unlabelled again.
The proposal:

- **Positive.** Council channel cells, after snapping (below), with a 1-2 m half-width.
- **Ignored.** Physics-detected channel cells that match no council line: they may be real, unrecorded drains. Also a
  3 m ring around every positive, for position error.
- **Negative.** Everything else, at a low weight. Most of the landscape is not channel, so this is mostly true.
- **Matt's reviews.** `exists = n` reaches are hard negatives, and reviewed real reaches are positives.

**Snapping.** Council lines are drawn up to tens of metres off the bed; the probes showed found labels a median
4 m from the channel map. Before rasterising, each council line is moved within a ±15 m corridor to the least-cost
path along the DEM bed (cost from `valley_s2`, `tophat_black` and relative elevation). Lines that cannot be
snapped, because no bed is found in the corridor, stay as they are, flagged and down-weighted. This is the M2 "snap
labels to the bed" task, and it helps the crossing labels too.

REC2 is not a stream label (coarse source DEM; DESIGN_NOTES 7i).

## 3. Inputs and training chips (superseded in part: chips come from saved rasters, section 7)

**Bands** (1 m, as in `features.tif` and `hydro.tif`): the 17-band DEM feature stack, DSM - DEM, REM, log10 upstream
area, breach depth, depression depth, and a rasterised distance to any way (roads, tracks, rail, OSM ways). That is
about 22 bands, all available everywhere.

**Chips** are 128 x 128 m, extracted inside the tile run like the patches (ML-0 step 4), stored as float16 with their
label rasters. That is about 0.7 MB per chip. Per tile:

| Chip type | Share | Centred on |
|---|---|---|
| Positive | ~50 % | council channel segments |
| Hard | ~25 % | physics channel ends and channels that match no label, and the crests of lost labels from the probes |
| Background | ~25 % | random |

About 300 chips per tile comes to ~200 MB per tile, or ~70 GB for waves 1 and 2. This needs one more run of the
tiles (chips are made where the arrays are in memory): about 5-6 h on the RTX 6000 machine for all 349.

## 4. Model and training

- **Network.** A U-Net, following the Swedish ditch-mapping work: a ResNet-34-sized encoder, 22 input channels, four
  down and four up stages.
- **Heads.** Channel probability, and stream/drain probability on channel cells.
- **Loss.** Weighted binary cross-entropy plus clDice on the channel head. clDice rewards connected centrelines,
  because a gap in a drain is what loses a crossing. A masked loss on the class head.
- **Augmentation.** Flips and 90-degree rotations; this is a plain map, not an oriented patch.
- **Folds.** The scorer's folds: leave one council out, sheets within councils, and the BW24 holdout.
- **Training.** On the RTX 6000 (96 GB): batch 32-64, about 30 epochs, mixed precision.

## 5. Integration

`run.network(..., channel_model=...)` would add one step after the physics channel map:

1. Infer the channel probability over the whole tile, in overlapping 128 m windows on the GPU, about 10-20 s per
   LINZ tile.
2. Threshold it, skeletonise it, and merge it with the physics network (union; a reach seen only by the model is
   flagged `source = ml`).
3. The existing crossing tests then run on the merged network: Test A, Test B, road crossings and road-end pairs.
4. The scorer, retrained with a "found by ML channel" flag, ranks the candidates as before.

The stream/drain head can then replace the reach classifier, and the second-pass outputs (stream REM, conditioned
hydrology) run on the merged network (7m).

## 6. Phases and checks

1. **ML-3a: labels and chips.** Snapping, rasterising, sampling; chip extraction in the batch; the dataset card.
   - *Check:* snapped council lines lie within 2 m of the DEM bed for most of their length; the card has counts per
     council and chip type.
2. **ML-3b: the U-Net.** Training and evaluation on the folds: council channel length recovered at 3 m, precision on
   reviewed reaches, and the stream/drain accuracy.
   - *Check:* recall of council channel length well above the physics map's, especially below 1 ha upstream.
3. **ML-3c: integration and the end-to-end check.** The tiles rerun with the channel model.
   - *Check:* the share of labels "found" rises (probes), and the scorer's recall@100 rises or holds.
4. **ML-3d: the class head.** The class head in the network, and the second-pass outputs on it.

## 7. Decisions (Matt, 17 Oct 2026)

1. **Positives and weights, per council.** All four councils' channel data are positives, weighted by how completely
   they record what the DEM can show:
   - Environment Southland (rural scheme drains): higher, about 1.5;
   - Waimakariri and HBRC: 1.0;
   - Auckland (urban stormwater; long stretches run underground, so the network looks fragmented and many pieces are
     missing): about 0.7.
     Its swales count at reduced weight (0.5), provided they do not harm the model: ML-3b compares that with
     swales ignored (Matt, 17 Oct).

   In Auckland, a detected channel with no council line is not an error: those cells are ignored, and only clearly
   flat, un-incised ground is a (weak) negative there. Auckland gets no negatives at all only if that is tested first:
   with none, the model can learn "urban means channel" (as the 20 m scope rule did for the scorer). A gap in a
   network is usually a pipe, which the DEM cannot show; the model is not asked to find pipes. The weights live in the
   ML-3 configuration (one table per council: positive weight, negative weight, ignore rules).
2. **Evaluation folds.** The primary measure is spatial blocks: a 1:50k sheet's tiles two 1:10k rows at a time, dealt
   into 5 folds with every council in every fold (`train.folds`, `blocks`; now also the scorer's and the CNN's).
   Single tiles would be too optimistic: neighbours share drains and surveys. Leave one council out stays as the second
   measure: transfer to a new kind of area, the worst case. The BW24 holdout stays.
   - On waves 1 + 2 the scorer's out-of-fold recall@100 over blocks is 0.59 against the tiers' 0.29
     (`ml/results/scorer_waves12_blocks.md`).
   - Waimakariri has few blocks and spreads unevenly across folds, so its numbers are the noisiest.
3. **Saved rasters, not chips made in the run.** The tiles are rerun with `--rasters all` (features.tif, hydro.tif,
   conditioned.tif: ~2-3 GB per tile, ~1 TB for 349 tiles; the RTX 6000 machine has ~6 TB and 3.6 TB free on two data
   SSDs).
   - "Rerun" is processing, not downloading: the DEM and DSM are in the bundles and the COG mirror. The exception is
     wave 1 on that machine, whose bundles were copied without their rasters.
   - Chips are then cut from the saved rasters, mosaicked across tile edges (the rasters are cropped to tile cores), so
     size and sampling can change without another run.
   - The distance-to-any-way band is rasterised from the bundles' vectors when chips are made.
4. **Channel width.** The council's width attribute where there is one, with fallbacks in this order:
   - the physics channel map's width where it overlaps the snapped line;
   - a default by type (drain ~1 m half-width, stream ~2 m);
   - for rivers, the wide-channel polygons.

   None of the four sources' field mappings carries a width yet: probing them is the first ML-3a task. A LiDAR-derived
   width (as GeoFabrics does) does not work for channels this small.
5. **Reviews.** Matt reviews 2-3 tiles of channels, `exists` y / n / unsure, marking "unsure" freely: only confident
   answers train. The review sets come from where the model and the data disagree (channels the model sees that no
   council records; council lines the model misses). They are delivered as GeoPackages; Matt views them over the LINZ
   basemap, so no imagery is packaged.
