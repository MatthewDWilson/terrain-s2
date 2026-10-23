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

Labels v5 (``labels: osm: true``; channel network v2, for unet_v4; ``LABEL_VERSION_OSM``), between steps 6 and 7:

6b. *OSM rivers and waterways as positives* (``osm_labels``): the river extents of the tile's OSM river areas
   refined on the DEM (``guidance.river_extent``), at ``osm_river_weight``; and OSM waterway lines of
   ``osm_kinds`` snapped onto the bed (``guidance.snap_waterways``), as footprints of ``osm_half_width_m`` by kind,
   at ``osm_line_weight``. They overrule ignored and negative cells (an unrecorded physics channel that OSM maps
   becomes positive); cells within the ring of them that were negative become ignored; the corridor of an OSM line
   the snapping could not place is ignored where it was negative. The review answers (step 8) still overrule them.
6c. *The class band* (``class_band``): for positive cells, 1 river, 2 stream, 3 drain, 255 unknown (ignored by
   the class loss). River extents and council river footprints are rivers; otherwise the OSM tag of a snapped
   line (river 1, stream 2, canal / drain / ditch 3), else the council class (stream 2, drain and race 3; other
   unknown). A council class is a guide to kind only, never to size (Matt, 23 Oct).

The OSM waterways are also an input band of unet_v4 (judged on council labels, since OSM is then a label source too).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

POS, NEG, IGNORE = 1, 0, 255
LABEL_VERSION = 3              # 2: river lines positive whatever the snapping; 3: corrections (review, cleaning)
LABEL_VERSION_OSM = 5          # 3 + OSM rivers and waterways as positives, and the class band (chips "labels v5";
                               # the chip set "labels v4" was version 3 with both reviews applied)
CLASS = dict(river=1, stream=2, drain=3)
CLASS_UNKNOWN = 255
OSM_CLASS = dict(river=1, stream=2, canal=3, drain=3, ditch=3)
COUNCIL_CLASS = dict(stream=2, drain=3, race=3)
DEFAULTS = dict(unsnapped="ignore", flat_depth_m=0.05, clean_prob=0.05, clean_physics_share=0.1, clean_piece_m=50.0,
                reviews="labels/manual/review_*.gpkg", review_positive_weight=1.0, review_negative_weight=1.0,
                review_half_width_m=1.0, review_buffer_m=3.0, osm=False, osm_kinds=("river", "stream", "canal", "drain"),
                osm_river_weight=1.0, osm_line_weight=0.7,
                osm_half_width_m=dict(river=3.0, stream=1.5, canal=2.5, drain=1.0, ditch=0.75))


def label_version(cfg: dict) -> int:
    """The label version a configuration builds: 5 with OSM labels, else 3."""
    return LABEL_VERSION_OSM if (cfg.get("labels") or {}).get("osm") else LABEL_VERSION


def label_file(cfg: dict) -> str:
    """The label raster's file name: ``ml_channel_labels.tif``, or ``ml_channel_labels_v5.tif`` for labels v5, so
    building v5 never overwrites the labels earlier chip sets were made from."""
    return "ml_channel_labels_v5.tif" if label_version(cfg) == LABEL_VERSION_OSM else "ml_channel_labels.tif"


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


def _read_dem(bundle, T, shape):
    """The bundle's DEM on the run's grid (NaN outside it), or None."""
    import rasterio
    from rasterio.windows import from_bounds
    f = Path(bundle) / "dem.tif"
    if not f.exists():
        return None
    x0, y1 = T.c, T.f
    x1, y0 = x0 + shape[1] * T.a, y1 + shape[0] * T.e
    with rasterio.open(f) as s:
        win = from_bounds(x0, y0, x1, y1, s.transform)
        z = s.read(1, window=win, out_shape=shape, boundless=True, fill_value=np.nan, masked=True)
    return np.ma.filled(z.astype(np.float32), np.nan)


def osm_labels(run, bundle, label, weight, T, valid, b11, b21, cfg):
    """Step 6b (labels v5): OSM river extents and snapped OSM waterways as positives, in place. Returns (river
    extent mask, OSM footprint class codes (0 none), counts)."""
    from rasterio.features import rasterize
    from scipy import ndimage as ndi

    from .. import guidance as G
    D = dict(DEFAULTS, **(cfg.get("labels") or {}))
    hw = dict(DEFAULTS["osm_half_width_m"], **(D.get("osm_half_width_m") or {}))
    shape = label.shape
    cs = abs(T.a)
    bounds = (T.c, T.f + shape[0] * T.e, T.c + shape[1] * T.a, T.f)
    g = G.load(Path(bundle) / "context.gpkg", bounds)
    z = _read_dem(bundle, T, shape)
    rx = G.river_extent(g["areas"], z, T) & valid if z is not None else np.zeros(shape, bool)
    S = G.snap_waterways(g["lines"], {"tophat_black11": b11, "tophat_black21": b21}, T, kinds=tuple(D["osm_kinds"]),
                         river_mask=rx)
    code = np.zeros(shape, np.uint8)
    foot = np.zeros(shape, bool)
    unplaced = np.zeros(shape, bool)
    if len(S):
        ok = S.snapped.astype(bool).to_numpy()
        for kind, grp in S[ok].groupby("waterway"):
            shapes = [(x.buffer(hw.get(kind, 1.0)), 1) for x in grp.geometry if x is not None and not x.is_empty]
            if shapes:
                m = rasterize(shapes, out_shape=shape, transform=T, fill=0, dtype="uint8") > 0
                code[m & (code == 0)] = OSM_CLASS.get(kind, 3)
                foot |= m
        if (~ok).any():
            corr = float(cfg.get("snap", {}).get("corridor_m", 15.0))
            shapes = [(x.buffer(corr), 1) for x in S[~ok].geometry if x is not None and not x.is_empty]
            if shapes:
                unplaced = rasterize(shapes, out_shape=shape, transform=T, fill=0, dtype="uint8") > 0
    foot &= valid
    new_pos = (rx | foot) & valid
    before_pos = label == POS
    w_new = np.where(rx, D["osm_river_weight"], np.where(foot, D["osm_line_weight"], 0.0)).astype(np.float32)
    label[new_pos] = POS
    weight[new_pos] = np.maximum(np.where(before_pos[new_pos], weight[new_pos], 0.0), w_new[new_pos])
    ring = ((ndi.distance_transform_edt(~new_pos) * cs <= float(cfg.get("ignore_ring_m", 3.0))) & ~new_pos
            if new_pos.any() else np.zeros(shape, bool))          # (no OSM positive: no ring; edt has no zero then)
    turn = (ring | unplaced) & (label == NEG)
    label[turn] = IGNORE
    weight[turn] = 0.0
    counts = dict(osm_river_cells=int(rx.sum()), osm_line_cells=int((foot & ~rx).sum()),
                  osm_new_positive_cells=int((new_pos & ~before_pos).sum()), osm_lines=int(len(S)),
                  osm_lines_snapped=int(S.snapped.astype(bool).sum()) if len(S) else 0, osm_source=g["source"])
    return rx, code, counts


def class_band(label, T, foot, lines, rx, osm_code):
    """Step 6c (labels v5): the class of each positive cell (``CLASS``; ``CLASS_UNKNOWN`` elsewhere)."""
    from rasterio.features import rasterize
    shape = label.shape
    cls = np.full(shape, CLASS_UNKNOWN, np.uint8)
    pos = label == POS
    if foot is not None and len(foot):
        f = foot.copy()
        kind = {}
        if lines is not None and len(lines) and "class" in lines and "source_id" in lines:
            kind = dict(zip(lines.source_id.astype(str), lines["class"].astype(str)))
        river = f["width_source"].eq("river").to_numpy() if "width_source" in f else np.zeros(len(f), bool)
        codes = [CLASS["river"] if r else COUNCIL_CLASS.get(kind.get(str(sid), ""), 0)
                 for r, sid in zip(river, f.get("source_id", [None] * len(f)))]
        shapes = [(g, c) for g, c in zip(f.geometry, codes) if c and g is not None and not g.is_empty]
        if shapes:
            council = rasterize(shapes, out_shape=shape, transform=T, fill=0, dtype="uint8")
            cls[pos & (council > 0)] = council[pos & (council > 0)]
    o = pos & (osm_code > 0)
    cls[o] = osm_code[o]                                     # an OSM tag over a council class
    cls[pos & rx] = CLASS["river"]                           # a river extent over everything
    return cls


def tile_labels(run, cfg: dict, council=None, write=True, bundle=None):
    """The label raster of one run (needs features.tif, hydro.tif and ml_channels.gpkg); returns its counts. With
    ``labels: osm`` (labels v5) also the OSM positives and the class band, from the tile's bundle (``bundle``,
    default ``<TERRAIN_DATA_DIR>/tiles/<tile>``)."""
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
    use_osm = bool((cfg.get("labels") or {}).get("osm"))
    if use_osm:
        if bundle is None:
            from ..acquire.elevation import default_data_dir
            bundle = default_data_dir() / "tiles" / run.name
        rx, osm_code, oc = osm_labels(run, bundle, label, weight, T, valid, b11, b21, cfg)
        counts.update(oc)
    counts.update(corrections(run, label, weight, T, crs, lines, foot, mask, valid, cfg))
    bands = {"label": label, "weight": weight}
    if use_osm:
        cls = class_band(label, T, foot, lines, rx, osm_code)
        bands["class"] = cls
        counts["class_cells"] = {k: int((cls == v).sum()) for k, v in CLASS.items()}
        counts["class_cells"]["unknown"] = int(((cls == CLASS_UNKNOWN) & (label == POS)).sum())
    counts.update(tile=run.name, council=council, label_version=label_version(cfg),
                  unsnapped=dict(DEFAULTS, **(cfg.get("labels") or {}))["unsnapped"])
    if write:
        f = label_file(cfg)
        dataio.write_stack(run / f, bands, T, crs)
        (run / f.replace(".tif", ".json")).write_text(json.dumps(counts, indent=1, default=str))
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
