"""ML-3a step 4: the channel label raster of a tile, on the grid of its saved rasters (the tile core).

Two bands in ``ml_channel_labels.tif`` beside the run (float32, deflate):

    label    1 channel, 0 not channel, 255 ignored
    weight   the loss weight of the cell: the council line's weight on positives, ``negative_weight`` on negatives,
             0 where ignored

From the tile's ``ml_channels.gpkg`` (snapping, step 3) and saved rasters, in order (a later rule never turns a
positive into anything else):

1. *Positives*: the council footprints of snapped lines (``council_footprints``), each with its line's weight; where
   footprints overlap, the larger weight.
2. *Unsnapped lines* (``unsnapped: ignore``, the default): their corridor (+-``snap.corridor_m`` of the council
   geometry, per council) is ignored rather than labelled. A line the snapping could not put on a bed is either drawn
   further off than the corridor or not visible in the DEM (piped, filled): a positive at the council position would
   be wrong either way. ``unsnapped: positive`` keeps their footprints as positives at their (halved) weight.
3. *The ring*: cells within ``ignore_ring_m`` of a positive are ignored (position and width error).
4. *Unrecorded channels*: physics channel cells (``channel_mask``) matching no council line are ignored where the
   council's ``ignore_unrecorded`` is set (councils record scheme drains, not every drain).
5. *Negatives*: everything else, at the council's ``negative_weight`` (default: the global one). Where the council's
   ``negatives`` is ``flat`` (Auckland), only un-incised ground (11 m black top-hat < ``flat_depth_m`` and 21 m one <
   twice that) is negative; the rest is ignored.
6. *No data* (no features: off the DEM) is ignored.

The tile's council comes from the manifest (else the council with most line length in the tile).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

POS, NEG, IGNORE = 1, 0, 255
LABEL_VERSION = 1
DEFAULTS = dict(unsnapped="ignore", flat_depth_m=0.05)


def council_rules(cfg: dict, council: str) -> dict:
    """The label rules of one council: its entry in ``councils`` over the global defaults."""
    c = (cfg.get("councils") or {}).get(council) or {}
    snap = dict((cfg.get("snap") or {}), **(c.get("snap") or {}))
    return dict(negative_weight=float(c.get("negative_weight", cfg.get("negative_weight", 0.05))),
                negatives=c.get("negatives", "all"), ignore_unrecorded=bool(c.get("ignore_unrecorded", True)),
                corridor_m=float(snap.get("corridor_m", 15.0)))


def label_raster(shape, transform, footprints, lines, mask, depth11, depth21, valid, cfg: dict, council: str):
    """(label, weight, counts) arrays for one tile. ``footprints``/``lines``: GeoDataFrames from ml_channels.gpkg
    (lines with ``snapped``, ``council`` and the council geometry as WKT in ``geometry_council``); ``mask``: the
    physics channel mask; ``depth11``/``depth21``: the 11 m and 21 m black top-hats; ``valid``: cells with data."""
    from rasterio.features import rasterize
    from scipy import ndimage as ndi
    from shapely import wkt
    D = dict(DEFAULTS, **(cfg.get("labels") or {}))
    R = council_rules(cfg, council)
    cs = abs(transform.a)
    label = np.full(shape, NEG, np.uint8)
    weight = np.zeros(shape, np.float32)

    use = footprints
    if footprints is not None and len(footprints) and D["unsnapped"] == "ignore":
        use = footprints[footprints.snapped.astype(bool)]
    pos_w = np.zeros(shape, np.float32)
    if use is not None and len(use):
        order = np.argsort(use.weight.to_numpy(float))                 # larger weights burnt last: the max wins
        shapes = [(g, float(w)) for g, w in zip(use.geometry.iloc[order], use.weight.to_numpy(float)[order])
                  if g is not None and not g.is_empty and w > 0]
        if shapes:
            pos_w = rasterize(shapes, out_shape=shape, transform=transform, fill=0.0, dtype="float32",
                              all_touched=False)
    pos = pos_w > 0

    ignore = np.zeros(shape, bool)
    if lines is not None and len(lines) and D["unsnapped"] == "ignore":
        un = lines[~lines.snapped.astype(bool)]
        if len(un):
            corr = []
            for g, c in zip(un.geometry_council, un.council):
                geom = wkt.loads(g) if isinstance(g, str) else g
                corr.append((geom.buffer(council_rules(cfg, c)["corridor_m"]), 1))
            ignore |= rasterize(corr, out_shape=shape, transform=transform, fill=0, dtype="uint8") > 0
    ring = (ndi.distance_transform_edt(~pos) * cs <= D.get("ignore_ring_m", cfg.get("ignore_ring_m", 3.0))) & ~pos
    ignore |= ring
    if R["ignore_unrecorded"] and mask is not None:
        ignore |= np.asarray(mask, bool)
    if R["negatives"] == "flat":
        f = float(D["flat_depth_m"])
        incised = (np.nan_to_num(depth11, nan=np.inf) >= f) | (np.nan_to_num(depth21, nan=np.inf) >= 2 * f)
        ignore |= incised
    ignore |= ~np.asarray(valid, bool)

    label[ignore] = IGNORE
    weight[~ignore] = R["negative_weight"]
    label[pos] = POS
    weight[pos] = pos_w[pos]
    weight[label == IGNORE] = 0.0
    counts = dict(positive=int(pos.sum()), negative=int((label == NEG).sum()), ignored=int((label == IGNORE).sum()),
                  positive_weight_sum=round(float(weight[pos].sum()), 1),
                  negative_weight_sum=round(float(weight[label == NEG].sum()), 1))
    return label, weight, counts


def tile_labels(run, cfg: dict, council=None, write=True):
    """The label raster of one run (needs features.tif, hydro.tif and ml_channels.gpkg); returns its counts."""
    import geopandas as gpd
    import pyogrio
    import rasterio

    from .. import dataio
    from .channel_labels import _horizontal
    run = Path(run)
    with rasterio.open(run / "features.tif") as s:
        T, crs, shape = s.transform, _horizontal(s.crs), (s.height, s.width)
        names = list(s.descriptions)
        b11 = s.read(names.index("tophat_black11") + 1)
        b21 = s.read(names.index("tophat_black21") + 1) if "tophat_black21" in names else b11
        valid = np.isfinite(b11)
    with rasterio.open(run / "hydro.tif") as s:
        mask = s.read(list(s.descriptions).index("channel_mask") + 1) > 0
    f = run / "ml_channels.gpkg"
    have = {n for n, _ in pyogrio.list_layers(f)} if f.exists() else set()
    lines = gpd.read_file(f, layer="council_channels") if "council_channels" in have else None
    foot = gpd.read_file(f, layer="council_footprints") if "council_footprints" in have else None
    if lines is not None:
        lines = lines.set_crs(crs, allow_override=True)
    if foot is not None:
        foot = foot.set_crs(crs, allow_override=True)
    if not council:
        council = (lines.assign(L=lines.length).groupby("council").L.sum().idxmax()
                   if lines is not None and len(lines) else None)
    label, weight, counts = label_raster(shape, T, foot, lines, mask, b11, b21, valid, cfg, council or "")
    counts.update(tile=run.name, council=council, label_version=LABEL_VERSION,
                  unsnapped=dict(DEFAULTS, **(cfg.get("labels") or {}))["unsnapped"])
    if write:
        dataio.write_stack(run / "ml_channel_labels.tif", {"label": label, "weight": weight}, T, crs)
        (run / "ml_channel_labels.json").write_text(json.dumps(counts, indent=1))
    return counts
