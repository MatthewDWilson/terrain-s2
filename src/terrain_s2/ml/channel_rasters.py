"""ML-3a step 4: the channel label raster of a tile, on the grid of its saved rasters (the tile core).

Two bands in ``ml_channel_labels.tif`` beside the run (float32, deflate):

    label    1 channel, 0 not channel, 255 ignored
    weight   the loss weight of the cell: the council line's weight on positives, ``negative_weight`` on negatives,
             0 where ignored

From the tile's ``ml_channels.gpkg`` (snapping, step 3) and saved rasters, in order (a later rule never turns a
positive into anything else):

1. *Positives*: the council footprints (rebuilt here from the lines and the run's wide-channel polygons, and written
   back to ``council_footprints``) of snapped lines and of river lines, snapped or not (a river line's label is the
   polygon, wherever the line was drawn: Gisborne's long river lines on wide flat beds failed the bed test and had
   lost their labels, 17 Oct), each with its line's weight; where footprints overlap, the larger weight.
2. *Unsnapped lines* other than river lines (``unsnapped: ignore``, the default): their corridor (+-``snap.corridor_m`` of the council
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

Then the corrections (LABEL_VERSION 3, ``corrections``), after the first review (18 Oct):

7. *Cleaning*: a piece (``clean_piece_m``) of a snapped, non-river council line whose out-of-fold probability
   (``ml_channel_prob.tif`` written with ``held_out``: a model that never trained on the tile) averages at most
   ``clean_prob`` and which lies on the physics map for at most ``clean_physics_share`` of its length is ignored
   rather than positive: the review found nothing there for 18 of 20 such pieces (piped, filled, misplaced).
8. *Review answers* (``reviews``: GeoPackages of reviewed reaches under the data folder, ``exists`` y / n /
   unsure), overriding the rest: a council piece answered n is ignored, one answered y is positive again; a
   model-only reach answered y becomes positive (``review_half_width_m``, ``review_positive_weight``), one answered
   n a hard negative (within ``review_buffer_m``, ``review_negative_weight``); unsure is ignored.

The tile's council comes from the manifest (else the council with most line length in the tile).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

POS, NEG, IGNORE = 1, 0, 255
LABEL_VERSION = 3              # 2: river lines positive whatever the snapping; 3: corrections (review, cleaning)
DEFAULTS = dict(unsnapped="ignore", flat_depth_m=0.05, clean_prob=0.05, clean_physics_share=0.1, clean_piece_m=50.0,
                reviews="labels/manual/review_*.gpkg", review_positive_weight=1.0, review_negative_weight=1.0,
                review_half_width_m=1.0, review_buffer_m=3.0)


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
        river = footprints["width_source"].eq("river") if "width_source" in footprints else False
        use = footprints[footprints.snapped.astype(bool).to_numpy() | np.asarray(river, bool)]
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
        river = lines["width_source"].eq("river").to_numpy() if "width_source" in lines else np.zeros(len(lines), bool)
        un = lines[~lines.snapped.astype(bool).to_numpy() & ~river]
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
    foot = None
    if lines is not None:
        from .channel_labels import footprints
        lines = lines.set_crs(crs, allow_override=True)
        rivers = None
        net = run / "network.gpkg"
        if net.exists() and "rivers" in {n for n, _ in pyogrio.list_layers(net)}:
            rivers = gpd.read_file(net, layer="rivers").to_crs(crs)
        foot = footprints(lines, rivers, cfg) if len(lines) else None
        if write and foot is not None and len(foot):
            foot.to_file(f, layer="council_footprints", driver="GPKG")
    if not council:
        council = (lines.assign(L=lines.length).groupby("council").L.sum().idxmax()
                   if lines is not None and len(lines) else None)
    label, weight, counts = label_raster(shape, T, foot, lines, mask, b11, b21, valid, cfg, council or "")
    counts.update(corrections(run, label, weight, T, crs, lines, foot, mask, valid, cfg))
    counts.update(tile=run.name, council=council, label_version=LABEL_VERSION,
                  unsnapped=dict(DEFAULTS, **(cfg.get("labels") or {}))["unsnapped"])
    if write:
        dataio.write_stack(run / "ml_channel_labels.tif", {"label": label, "weight": weight}, T, crs)
        (run / "ml_channel_labels.json").write_text(json.dumps(counts, indent=1))
    return counts


def _review_rows(run, D):
    """Reviewed reaches of this tile from every review GeoPackage under the data folder."""
    import glob

    import geopandas as gpd
    import pandas as pd
    from ..acquire.elevation import default_data_dir
    pat = D["reviews"]
    pat = pat if Path(pat).is_absolute() else str(default_data_dir() / pat)
    parts = []
    for f in sorted(glob.glob(pat)):
        try:
            g = gpd.read_file(f, layer="reaches")
        except Exception:  # noqa: BLE001 - a malformed review file is skipped, not fatal
            continue
        if "tile" in g and "exists" in g:
            g = g[g["tile"] == Path(run).name]
            if len(g):
                parts.append(g.to_crs(2193) if g.crs else g.set_crs(2193))
    if not parts:
        return None
    r = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=2193)
    r["answer"] = r["exists"].fillna("").astype(str).str.strip().str.lower()
    return r[r.answer.isin(["y", "n", "unsure"])]


def corrections(run, label, weight, T, crs, lines, foot, mask, valid, cfg):
    """Apply the cleaning and the review answers to ``label`` and ``weight`` in place; returns their counts."""
    from rasterio.features import rasterize
    from scipy import ndimage as ndi
    from shapely.ops import substring
    D = dict(DEFAULTS, **(cfg.get("labels") or {}))
    run = Path(run)
    shape = label.shape
    cs = abs(T.a)
    ring = float(cfg.get("ignore_ring_m", 3.0))
    out = dict(cleaned_positive_cells=0, cleaned_pieces=0, review_rows=0, review_ignored_cells=0,
               review_positive_cells=0, review_negative_cells=0)

    def burn(geoms):
        geoms = [g for g in geoms if g is not None and not g.is_empty]
        if not geoms:
            return np.zeros(shape, bool)
        return rasterize([(g, 1) for g in geoms], out_shape=shape, transform=T, fill=0, dtype="uint8") > 0

    # 7. cleaning with out-of-fold probabilities
    pj, pf = run / "ml_channel_prob.json", run / "ml_channel_prob.tif"
    if lines is not None and len(lines) and pj.exists() and pf.exists():
        try:
            held = json.loads(pj.read_text()).get("held_out")
        except ValueError:
            held = False
        if held:
            import rasterio
            with rasterio.open(pf) as s:
                prob = s.read(1)
            near_phys = ndi.distance_transform_edt(~np.asarray(mask, bool)) * cs <= 3.0
            hw = dict(zip(foot.source_id, foot.width_source)) if foot is not None and "source_id" in foot else {}
            use = lines[lines.snapped.astype(bool) & (lines.get("width_source", "") != "river")]
            pieces = []
            for g, h in zip(use.geometry, use.get("half_width_m", [1.0] * len(use))):
                n = max(1, int(round(g.length / D["clean_piece_m"])))
                for k in range(n):
                    pc = substring(g, k * g.length / n, (k + 1) * g.length / n)
                    if pc.length < 5:
                        continue
                    pts = [pc.interpolate(d) for d in np.arange(0, pc.length, 2.0)]
                    cc = np.floor((np.array([q.x for q in pts]) - T.c) / T.a).astype(int)
                    rr = np.floor((np.array([q.y for q in pts]) - T.f) / T.e).astype(int)
                    ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
                    if ok.sum() < 3:
                        continue
                    pv = prob[rr[ok], cc[ok]]
                    if np.isfinite(pv).mean() < 0.5:
                        continue
                    if np.nanmean(pv) <= D["clean_prob"] and near_phys[rr[ok], cc[ok]].mean() <= D["clean_physics_share"]:
                        hh = float(h) if h is not None and np.isfinite(h) else 1.0
                        pieces.append(pc.buffer(hh + ring))
            if pieces:
                m = burn(pieces) & (label == POS)
                out.update(cleaned_positive_cells=int(m.sum()), cleaned_pieces=len(pieces))
                label[m], weight[m] = IGNORE, 0.0
    # 8. review answers
    r = _review_rows(run, D)
    if r is not None and len(r):
        out["review_rows"] = int(len(r))
        cm, mo = r[r.kind == "council_missed"], r[r.kind == "model_only"]
        ign = burn(list(cm[cm.answer == "n"].geometry.buffer(D["review_buffer_m"] + ring)) +
                   list(r[r.answer == "unsure"].geometry.buffer(D["review_buffer_m"])))
        label[ign], weight[ign] = IGNORE, 0.0
        out["review_ignored_cells"] = int(ign.sum())
        pos = burn(list(cm[cm.answer == "y"].geometry.buffer(D["review_half_width_m"])) +
                   list(mo[mo.answer == "y"].geometry.buffer(D["review_half_width_m"])))
        ringm = (ndi.distance_transform_edt(~pos) * cs <= ring) & ~pos & (label == NEG)
        label[ringm], weight[ringm] = IGNORE, 0.0
        label[pos], weight[pos] = POS, float(D["review_positive_weight"])
        out["review_positive_cells"] = int(pos.sum())
        neg = burn(list(mo[mo.answer == "n"].geometry.buffer(D["review_buffer_m"]))) & ~pos
        label[neg], weight[neg] = NEG, float(D["review_negative_weight"])
        out["review_negative_cells"] = int(neg.sum())
    nod = ~np.asarray(valid, bool)
    label[nod], weight[nod] = IGNORE, 0.0
    return out
