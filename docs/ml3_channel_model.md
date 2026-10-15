# ML-3: a learned channel map (draft, 16 Oct 2026)

Draft for discussion. Phase names follow DESIGN_NOTES section 8 (ML-3: dense multi-task U-Net). Evidence is in
DESIGN_NOTES section 11.

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

## 3. Inputs and training chips

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

## 7. Decisions for Matt

1. **Positives.** Which channel sources count as positives: all four councils' channel data? Auckland's are urban
   stormwater channels; Environment Southland's are long scheme drains.
2. **Another run of the tiles** for the chips (~6 h on the RTX 6000 machine), or chips from saved rasters instead
   (`--rasters all`: ~2-3 GB per tile, ~1 TB for waves 1 and 2).
3. **Channel width** for the positive raster: one fixed half-width (1.5 m), or from the council's width attribute where
   there is one.
4. **More reviews.** Whether more of your time goes on channel reviews, even 2-3 tiles of `exists` y/n. They are the
   only true negatives for small drains.
