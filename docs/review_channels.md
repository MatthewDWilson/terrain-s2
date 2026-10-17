# Reviewing the channel U-Net (for Matt)

`terrain_hpc.sh predict ... --review` writes `review.gpkg` beside the model
(`<data>/models/<name>/<fold>/review.gpkg`, layer `reaches`) and a probability raster per tile
(`<data>/runs/<tile>/ml_channel_prob.tif`). The tiles are test tiles of that fold: the model has not seen them.

Each reach is one of:

- `model_only`: the model draws a channel here (at its threshold) that is neither on the physics map nor on a
  council line. An unrecorded drain, or a false detection?
- `council_missed`: a piece (~50 m) of a snapped council line where the model detects nothing.
  `physics_share` says whether the physics map has it.

Fields: `tile`, `kind`, `length_m`, `mean_prob`, `physics_share`, `council_share` (share of the reach within 3 m
of a council channel), `council` and `source_id` (council_missed), and the two to fill:

- `exists`: `y` (there is a channel / drain here), `n` (there is not), `unsure` (leave it if not confident; only
  confident answers are used, decision 5 of docs/ml3_channel_model.md);
- `notes`: anything useful (piped, filled in, a road edge, a fence line, a stopbank toe...).

In QGIS: the LINZ basemap (or aerial imagery), a hillshade of the tile's `dem.tif` (`<data>/tiles/<tile>/`), the
probability raster (pseudocolour 0-1, transparent below ~0.3), `review.gpkg`, and for context the physics network
(`<data>/runs/<tile>/network.gpkg`, layer `channels`) and the council lines (`ml_channels.gpkg`,
`council_channels`). Style `reaches` by `kind`. Edit `exists` in the attribute table; save the GeoPackage and send it
back. A few dozen reaches of each kind per tile are enough for a first precision estimate.
