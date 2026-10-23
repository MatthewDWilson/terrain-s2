"""ML-3a step 5: training chips for the channel model, cut from the saved rasters (docs/ml3_channel_model.md).

A chip is a ``size`` x ``size`` window (128 m on the 1 m grid) of the input bands with the label raster's label and
weight. Windows are read across tile edges from the neighbouring runs (the rasters are cropped to tile cores), so a
chip near an edge is whole when its neighbour was run; where no run covers it, the inputs are 0 and the label is
ignored.

Inputs (``BANDS``, 23): the 17 bands of ``features.tif``; from ``hydro.tif`` the REM, log10 upstream area, breach and
depression depth and DSM - DEM; and the distance to any way (roads, rail, tracks, OSM ways: the scorer's
``ctx_dist_any_way_m``, capped at 500 m), rasterised once per tile from the bundle's ``context.gpkg`` into
``ml_ways.tif``. The physics channel mask is not an input: the model is meant to find what it misses.

Chip centres lie in the tile core, so each chip belongs to one tile (and so to one fold). Per tile (``per_tile``):

    positive    ~50 %  on council channel cells (label 1)
    hard        ~25 %  physics channel cells no council records (ignored in the labels), and the crests of register,
                       bridge and implied labels the chain lost (``ml_probes.parquet``: no_channel, not_in_network)
    random      ~25 %  anywhere with data (the rest when a tile has too few positives or hard places)

Centres are jittered by up to ``jitter_m`` (a channel is not always in the middle) and kept ``min_sep_m`` apart.

Chips version 3 (unet_v4, channel network v2): optional ``osm_band`` (a 24th input, ``dist_osm_waterway_m``: the
distance to any OSM waterway line or river area, capped at 500 m, rasterised once per tile into ``ml_osm.tif``
by ``osm_raster``; ``BANDS_V4``); ``label_file`` (``ml_channel_labels_v5.tif``: labels v5, with the class band, kept
per chip as ``cls``: 1 river, 2 stream, 3 drain, 255 unknown or not positive); and ``positive_class_shares``, the
positive chips split by the class of their centre cell (e.g. river 0.2, stream 0.4, drain 0.4; any shortfall filled
from the other positives), so wide rivers do not take the positive chips from the drains.
Written per tile as ``<dataset>/chips/<tile>.npz`` (``x`` float16 (n, bands, size, size), ``label`` uint8, ``weight``
float16, ``physics`` uint8 (the physics channel mask, for the baseline), ``cx``, ``cy``, ``kind``) with an index of all chips (``chips.parquet``: tile, council, wave, holdout and the
chip's centre, kind, positive and ignored shares) and a card (``card.json``, ``card.md``: counts per council and
kind, bands, per-band mean and standard deviation over the chips' valid cells for normalisation).
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

FEATURE_BANDS = ("relief_med5", "relief_med11", "relief_med21", "tophat_white11", "tophat_black11", "tophat_white21",
                 "tophat_black21", "tophat_white41", "tophat_black41", "laplacian_s2", "ridge_s2", "valley_s2",
                 "laplacian_s4", "ridge_s4", "valley_s4", "openness_pos", "openness_neg")
HYDRO_BANDS = ("rem_m", "log10_upstream_area_m2", "breach_depth_m", "depression_depth_m", "dsm_minus_dem_m")
WAY_BAND = "dist_any_way_m"
BANDS = FEATURE_BANDS + HYDRO_BANDS + (WAY_BAND,)
OSM_BAND = "dist_osm_waterway_m"
BANDS_V4 = BANDS + (OSM_BAND,)
WAY_CAP_M = 500.0
OSM_CAP_M = 500.0
CLASS_CODES = dict(river=1, stream=2, drain=3)
WAY_LAYERS = ("roads", "rail", "tracks", "osm_roads")
CHIPS_VERSION = 3              # 2: the physics channel mask kept per chip (not an input: the baseline in ML-3b);
                               # 3: optional OSM band, the class array (labels v5), positives by class
DEFAULTS = dict(size=128, per_tile=300, shares=dict(positive=0.5, hard=0.25), jitter_m=32.0, min_sep_m=40.0,
                hard_reasons=["no_channel", "not_in_network"], seed=0, label_file="ml_channel_labels.tif",
                osm_band=False, positive_class_shares=None)


def bands_for(cfg) -> tuple:
    """The input bands of a chip configuration: ``BANDS_V4`` with ``osm_band``, else ``BANDS``."""
    return BANDS_V4 if (cfg or {}).get("osm_band") else BANDS


def osm_distance(shape, transform, lines=None, areas=None, cap_m=OSM_CAP_M):
    """Distance (m, capped) to any OSM waterway line or river area on a grid (0 on and inside them)."""
    from rasterio.features import rasterize
    from scipy import ndimage as ndi
    shapes = []
    for g in (lines.geometry if lines is not None and len(lines) else []):
        if g is not None and not g.is_empty:
            shapes.append((g, 1))
    if areas is not None and len(areas):
        a = areas[areas.water_kind == "river_area"] if "water_kind" in areas else areas
        shapes += [(g, 1) for g in a.geometry if g is not None and not g.is_empty]
    if not shapes:
        return np.full(shape, cap_m, np.float32)
    on = rasterize(shapes, out_shape=shape, transform=transform, fill=0, all_touched=True, dtype="uint8") > 0
    if not on.any():
        return np.full(shape, cap_m, np.float32)
    return np.minimum(ndi.distance_transform_edt(~on) * abs(transform.a), cap_m).astype(np.float32)


def osm_raster(run, bundle, write=True):
    """``ml_osm.tif``: the distance to any OSM waterway or river area (``osm_distance``) on the run's grid, with the
    OSM within ``OSM_CAP_M`` of the core counted so the distance is right at the tile's edge."""
    import rasterio
    from rasterio.transform import from_origin

    from .. import dataio
    from .. import guidance as G
    from .channel_labels import _horizontal
    run, bundle = Path(run), Path(bundle)
    with rasterio.open(run / "features.tif") as s:
        T, crs, (h, w) = s.transform, _horizontal(s.crs), (s.height, s.width)
    cs = abs(T.a)
    m = int(np.ceil(OSM_CAP_M / cs))
    Tb = from_origin(T.c - m * cs, T.f + m * cs, cs, cs)
    shape = (h + 2 * m, w + 2 * m)
    bounds = (Tb.c, Tb.f - shape[0] * cs, Tb.c + shape[1] * cs, Tb.f)
    g = G.load(bundle / "context.gpkg", bounds)
    d = osm_distance(shape, Tb, g["lines"], g["areas"])[m:m + h, m:m + w]
    if write:
        dataio.write_stack(run / "ml_osm.tif", {OSM_BAND: d}, T, crs)
    return d


def ways_raster(run, bundle, write=True):
    """Distance (m, capped) to any road, rail line, track or OSM way, on the run's grid: ``ml_ways.tif``.
    Ways within ``WAY_CAP_M`` of the core count, so the distance is right at the tile's edge."""
    import geopandas as gpd
    import pyogrio
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from scipy import ndimage as ndi

    from .. import dataio
    from .channel_labels import _horizontal
    run, bundle = Path(run), Path(bundle)
    with rasterio.open(run / "features.tif") as s:
        T, crs, (h, w) = s.transform, _horizontal(s.crs), (s.height, s.width)
    cs = abs(T.a)
    m = int(np.ceil(WAY_CAP_M / cs))
    Tb = from_origin(T.c - m * cs, T.f + m * cs, cs, cs)
    geoms = []
    ctx = bundle / "context.gpkg"
    if ctx.exists():
        have = {n for n, _ in pyogrio.list_layers(ctx)}
        for name in WAY_LAYERS:
            if name in have:
                g = gpd.read_file(ctx, layer=name)
                if len(g):
                    geoms += [x for x in g.to_crs(crs).geometry if x is not None and not x.is_empty]
    shape = (h + 2 * m, w + 2 * m)
    if geoms:
        on = rasterize([(g, 1) for g in geoms], out_shape=shape, transform=Tb, fill=0, all_touched=True,
                       dtype="uint8") > 0
        d = np.minimum(ndi.distance_transform_edt(~on) * cs, WAY_CAP_M).astype(np.float32) if on.any() \
            else np.full(shape, WAY_CAP_M, np.float32)
    else:
        d = np.full(shape, WAY_CAP_M, np.float32)
    d = d[m:m + h, m:m + w]
    if write:
        dataio.write_stack(run / "ml_ways.tif", {WAY_BAND: d}, T, crs)
    return d


class Mosaic:
    """Windows of the saved rasters across tile edges. ``tiles``: {tile_id: run folder}; each run's extent is its
    features.tif's (the tile core). Datasets are opened lazily and a few kept open. ``bands``: the input bands
    (``BANDS`` or ``BANDS_V4``); ``label_file``: the label raster (its ``class`` band read when it has one)."""

    INPUTS = (("features.tif", FEATURE_BANDS), ("hydro.tif", HYDRO_BANDS), ("ml_ways.tif", (WAY_BAND,)),
              ("ml_osm.tif", (OSM_BAND,)))

    def __init__(self, tiles: dict, keep_open=24, bands=BANDS, label_file="ml_channel_labels.tif"):
        import rasterio
        self.bands = tuple(bands)
        inputs = [(f, names) for f, names in self.INPUTS if any(n in self.bands for n in names)]
        if tuple(n for _, names in inputs for n in names) != self.bands:
            raise ValueError(f"bands {self.bands} are not features, hydrology, ways[, osm] in that order")
        self.inputs = tuple(inputs)
        self.FILES = self.inputs + ((label_file, ("label", "weight", "class")), ("hydro.tif", ("channel_mask",)))
        self.runs, self.bounds = {}, {}
        for t, r in tiles.items():
            f = Path(r) / "features.tif"
            if f.exists():
                with rasterio.open(f) as ds:
                    self.runs[t], self.bounds[t] = Path(r), tuple(ds.bounds)
        self._open = OrderedDict()
        self.keep = keep_open

    def _ds(self, path):
        import rasterio
        path = str(path)
        if path in self._open:
            self._open.move_to_end(path)
            return self._open[path]
        ds = rasterio.open(path)
        idx = {d: i + 1 for i, d in enumerate(ds.descriptions)}
        self._open[path] = (ds, idx)
        while len(self._open) > self.keep:
            self._open.popitem(last=False)[1][0].close()
        return self._open[path]

    def close(self):
        for ds, _ in self._open.values():
            ds.close()
        self._open.clear()

    def read_inputs(self, x0, y1, width, height, cs=1.0):
        """The input bands only, (len(bands), height, width) float32 with NaN where no run, for a region of any
        size with top-left corner (x0, y1): whole-tile prediction reads its tile and margin once with this."""
        return self._read(x0, y1, width, height, cs, self.inputs)

    def read(self, x0, y1, size, cs=1.0):
        """(inputs (len(bands), size, size) float32 with NaN where no run, label uint8, weight float32, physics
        channel mask uint8, class uint8 (255 unknown, not positive or no class band)) for the window with top-left
        corner (x0, y1)."""
        out = self._read(x0, y1, size, size, cs, self.FILES)
        nb = len(self.bands)
        x, lab, wt, cl = out[:nb], out[nb], out[nb + 1], out[nb + 2]
        bad = ~np.isfinite(x).all(axis=0) | ~np.isfinite(lab)
        label = np.where(bad, 255, np.nan_to_num(lab, nan=255)).astype(np.uint8)
        weight = np.where(bad | (label == 255), 0.0, np.nan_to_num(wt, nan=0.0)).astype(np.float32)
        physics = (np.nan_to_num(out[nb + 3], nan=0.0) > 0).astype(np.uint8)
        cls = np.where(label == 1, np.nan_to_num(cl, nan=255), 255).astype(np.uint8)
        return x, label, weight, physics, cls

    def _read(self, x0, y1, width, height, cs, files):
        import rasterio.windows as W
        x1, y0 = x0 + width * cs, y1 - height * cs
        out = np.full((sum(len(n) for _, n in files), height, width), np.nan, np.float32)
        for t, (bx0, by0, bx1, by1) in self.bounds.items():
            if bx0 >= x1 or bx1 <= x0 or by0 >= y1 or by1 <= y0:
                continue
            ix0, ix1 = max(x0, bx0), min(x1, bx1)
            iy0, iy1 = max(y0, by0), min(y1, by1)
            c0, c1 = int(round((ix0 - x0) / cs)), int(round((ix1 - x0) / cs))
            r0, r1 = int(round((y1 - iy1) / cs)), int(round((y1 - iy0) / cs))
            if c1 <= c0 or r1 <= r0:
                continue
            k = 0
            for fname, names in files:
                f = self.runs[t] / fname
                if not f.exists():
                    k += len(names)
                    continue
                ds, idx = self._ds(f)
                win = W.from_bounds(ix0, iy0, ix1, iy1, transform=ds.transform).round_offsets().round_lengths()
                bands = [idx.get(n) for n in names]
                got = [b for b in bands if b]
                if got:
                    a = ds.read(got, window=win, boundless=True, fill_value=np.nan, out_dtype="float32")
                    hh, ww = min(a.shape[1], r1 - r0), min(a.shape[2], c1 - c0)
                    j = 0
                    for i, b in enumerate(bands):
                        if b:
                            out[k + i, r0:r0 + hh, c0:c0 + ww] = a[j][:hh, :ww]
                            j += 1
                k += len(names)
        return out


def _thin(cands, n, sep, taken, rng):
    """Up to n of the candidate points (k, 2), at least ``sep`` from each other and from ``taken``."""
    if not len(cands) or n <= 0:
        return np.zeros((0, 2))
    order = rng.permutation(len(cands))[: max(n * 50, 500)]
    got = list(taken)
    out = []
    for i in order:
        p = cands[i]
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= sep * sep for q in got):
            got.append(p)
            out.append(p)
            if len(out) >= n:
                break
    return np.array(out).reshape(-1, 2)


def sample_centres(run, cfg=None, seed=0):
    """Chip centres of one tile: [(x, y, kind)], from its label raster, channel mask and probes."""
    import pandas as pd
    import rasterio
    P = dict(DEFAULTS, **(cfg or {}))
    rng = np.random.default_rng(seed)
    run = Path(run)
    with rasterio.open(run / P["label_file"]) as s:
        lab, T = s.read(1), s.transform
        cls = s.read(list(s.descriptions).index("class") + 1) if "class" in s.descriptions else None
    with rasterio.open(run / "hydro.tif") as s:
        mask = s.read(list(s.descriptions).index("channel_mask") + 1) > 0
    cs = abs(T.a)
    h, w = lab.shape

    def xy(rows, cols):
        return np.column_stack([T.c + (cols + 0.5) * cs, T.f - (rows + 0.5) * cs])

    def sub(mask_, k=20000):
        r, c = np.nonzero(mask_)
        if len(r) > k:
            i = rng.choice(len(r), k, replace=False)
            r, c = r[i], c[i]
        return xy(r, c)

    n = int(P["per_tile"])
    n_pos = int(round(n * P["shares"]["positive"]))
    n_hard = int(round(n * P["shares"]["hard"]))
    taken = []
    out = []
    shares = P.get("positive_class_shares")
    if shares and cls is not None:                   # positives by class, so wide rivers leave chips for the drains
        pos = np.zeros((0, 2))
        for name, sh in shares.items():
            code = CLASS_CODES.get(name)
            got = _thin(sub((lab == 1) & (cls == code)), int(round(n_pos * float(sh))), P["min_sep_m"],
                        taken + list(pos), rng) if code else np.zeros((0, 2))
            pos = np.vstack([pos, got])
        if len(pos) < n_pos:
            pos = np.vstack([pos, _thin(sub(lab == 1), n_pos - len(pos), P["min_sep_m"], taken + list(pos), rng)])
    else:
        pos = _thin(sub(lab == 1), n_pos, P["min_sep_m"], taken, rng)
    taken += list(pos)
    out += [(p[0], p[1], "positive") for p in pos]
    hard_c = sub(mask & (lab == 255), 10000)
    pf = run / "ml_probes.parquet"
    if pf.exists():
        pr = pd.read_parquet(pf)
        if len(pr) and {"x", "y", "reason"} <= set(pr.columns):
            lost = pr[pr.reason.isin(P["hard_reasons"])]
            if len(lost):
                hard_c = np.vstack([np.column_stack([lost.x, lost.y]), hard_c]) if len(hard_c) else \
                    np.column_stack([lost.x, lost.y])
    hard = _thin(hard_c, n_hard, P["min_sep_m"], taken, rng)
    taken += list(hard)
    out += [(p[0], p[1], "hard") for p in hard]
    rest = n - len(out)
    rand = _thin(sub(lab != 255, 20000) if (lab != 255).any() else np.zeros((0, 2)), rest, P["min_sep_m"], taken, rng)
    out += [(p[0], p[1], "random") for p in rand]
    j = rng.uniform(-P["jitter_m"], P["jitter_m"], size=(len(out), 2))
    x0, x1 = T.c, T.c + w * cs
    y0, y1 = T.f - h * cs, T.f
    return [(float(np.clip(x + dx, x0, x1 - cs)), float(np.clip(y + dy, y0 + cs, y1)), k)
            for (x, y, k), (dx, dy) in zip(out, j)]


def tile_chips(tile_id, run, mosaic, out_dir, cfg=None, seed=0):
    """Cut and write one tile's chips; returns (index rows, band sums for the card)."""
    P = dict(DEFAULTS, **(cfg or {}))
    size = int(P["size"])
    centres = sample_centres(run, P, seed)
    nb = len(mosaic.bands)
    X = np.zeros((len(centres), nb, size, size), np.float16)
    Y = np.full((len(centres), size, size), 255, np.uint8)
    Wt = np.zeros((len(centres), size, size), np.float16)
    Ph = np.zeros((len(centres), size, size), np.uint8)
    Cl = np.full((len(centres), size, size), 255, np.uint8)
    s1 = np.zeros(nb)
    s2 = np.zeros(nb)
    cnt = np.zeros(nb)
    ccount = np.zeros(4)
    rows = []
    for i, (cx, cy, kind) in enumerate(centres):
        x0, y1 = np.floor(cx - size / 2), np.ceil(cy + size / 2)
        x, lab, wt, ph, cl = mosaic.read(x0, y1, size)
        Cl[i] = cl
        ccount += np.bincount(np.where(cl == 255, 0, cl).ravel(), minlength=4)[:4] * [0, 1, 1, 1]
        ok = np.isfinite(x)
        xv = np.where(ok, x, 0.0)
        s1 += xv.sum(axis=(1, 2))
        s2 += (xv.astype(np.float64) ** 2).sum(axis=(1, 2))
        cnt += ok.sum(axis=(1, 2))
        X[i] = np.clip(xv, -65000, 65000).astype(np.float16)
        Y[i], Wt[i], Ph[i] = lab, wt.astype(np.float16), ph
        rows.append(dict(tile_id=tile_id, i=i, cx=cx, cy=cy, kind=kind, pos_frac=float((lab == 1).mean()),
                         ignored_frac=float((lab == 255).mean())))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{tile_id}.npz", x=X, label=Y, weight=Wt, physics=Ph, cls=Cl,
                        cx=np.array([c[0] for c in centres]), cy=np.array([c[1] for c in centres]),
                        kind=np.array([c[2] for c in centres]), bands=np.array(mosaic.bands))
    return rows, dict(s1=s1.tolist(), s2=s2.tolist(), n=cnt.tolist(), cls=ccount[1:].tolist())


def card(index, sums, name, cfg, bands=BANDS) -> dict:
    """The dataset card: counts per council and kind, bands and their normalisation statistics, and (chips 3)
    the positive cells by class."""
    s1 = np.sum([s["s1"] for s in sums], axis=0)
    s2 = np.sum([s["s2"] for s in sums], axis=0)
    n = np.maximum(np.sum([s["n"] for s in sums], axis=0), 1)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0.0))
    by = index.groupby(["council", "kind"]).size().unstack(fill_value=0)
    return dict(name=name, chips_version=CHIPS_VERSION, config=cfg, chips=int(len(index)), tiles=int(index.tile_id.nunique()),
                per_council_kind={c: {k: int(v) for k, v in r.items()} for c, r in by.iterrows()},
                positive_share_mean=round(float(index.pos_frac.mean()), 4),
                ignored_share_mean=round(float(index.ignored_frac.mean()), 4),
                bands=list(bands), mean=[round(float(v), 5) for v in mean], std=[round(float(v), 5) for v in std],
                label_file=(cfg or {}).get("label_file", "ml_channel_labels.tif"),
                class_cells=dict(zip(CLASS_CODES, np.sum([s.get("cls", [0, 0, 0]) for s in sums], axis=0).astype(int)
                                     .tolist())))


def card_markdown(c) -> str:
    lines = [f"# Chip dataset {c['name']}", "", f"{c['chips']:,} chips from {c['tiles']} tiles (chips version "
             f"{c['chips_version']}); mean positive share {c['positive_share_mean']:.2%}, ignored "
             f"{c['ignored_share_mean']:.1%}.", "", "| Council | positive | hard | random |", "|---|---|---|---|"]
    for k, v in c["per_council_kind"].items():
        lines.append(f"| {k} | {v.get('positive', 0)} | {v.get('hard', 0)} | {v.get('random', 0)} |")
    if any(c.get("class_cells", {}).values()):
        lines += ["", "Positive cells by class: " + ", ".join(f"{k} {v:,}" for k, v in c["class_cells"].items())
                  + f" (labels {c.get('label_file')})."]
    lines += ["", "| Band | mean | std |", "|---|---|---|"]
    lines += [f"| {b} | {m:.4g} | {s:.4g} |" for b, m, s in zip(c["bands"], c["mean"], c["std"])]
    return "\n".join(lines) + "\n"
