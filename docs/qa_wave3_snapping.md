# Brief: visual check of wave 3's channel snapping (for the council-data thread)

From the Stage 2 (ML) thread, 17 Oct 2026. Use once wave 3 has run and `terrain_hpc.sh channels ml/manifests/wave3.csv`
has written `ml_channels.gpkg` and `channels_summary.csv` for its tiles. Report back findings; do not change code or
configuration (`sources.yml`, `ml/`): the ML thread turns them into per-council settings in `ml/config/channels.yml`,
as it did for Environment Southland (DESIGN_NOTES 11, 17 Oct).

## Why

Council lines are moved onto the channel bed in the DEM within a corridor (default +-15 m). Two failures matter:

- **False snaps.** The bed is outside the corridor, so the line is "snapped" onto a weaker feature nearby (paddock
  furrows, road edges). Environment Southland did this: lines drawn ~25 m off the drains, snapped onto furrows.
- **Missed snaps.** The line is on a real drain the DEM shows, but is left unsnapped (`snapped = 0`).

## What to clip

For each council (waikato, hauraki, otago, gisborne, ecan), from `channels_summary.csv` pick three tiles:
the one with most km, one with a low `snapped_share`, one with a large `offset_med_m` or low `physics_frac`. On the
RTX 6000 machine:

```bash
cd ~/terrain/terrain-s2
terrain-hpc python scripts/clip_run.py --tile <tile_id> --out /mnt/b/terrain-s2
```

## What to look at (QGIS, over the LINZ basemap or a hillshade of `dem.tif`)

Layers: `ml_channels.gpkg` `council_channels` (snapped geometry; `snapped`, `bed_frac`, `offset_med_m`,
`physics_frac`, `width_source`), its `geometry_council` column as WKT (the council line: load the layer a second
time with that column as geometry, or use "Geometry by expression" `geom_from_wkt("geometry_council")`),
`council_footprints` (the label polygons), and `network.gpkg` `channels` (the physics network).

For each council, record:

1. **Typical offset** of council lines from the visible drain (a few measurements), and whether it is a shift, coarse
   vertices, or old plans.
2. **False snaps**: snapped lines not on the drain. How far was the drain, and what did the line follow instead?
3. **Missed snaps**: unsnapped lines on a drain that is clearly visible. How deep and wide does it look?
4. **Footprints**: are the label polygons about the width of the drain (too wide, too narrow, river polygons that
   spread up side branches)?
5. Anything else odd (lines along roads, piped stretches, lines on stopbanks).

One short table per council (tile, line id, finding, metres) is enough.
