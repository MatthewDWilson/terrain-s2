"""ML-3b: the channel U-Net over whole tiles, and review sets for Matt.

``predict_tile``: the model's channel probability on a run's grid (``ml_channel_prob.tif``), from windows of the
chips' size read across tile edges like the chips (``chips.Mosaic``), overlapping by half and blended with a smooth
taper. The inputs are prepared as for the chips (no data -> 0, then standardised with the training card's
statistics).

``review_reaches``: where the model and the councils disagree, as reaches for review:

    model_only       a reach of the model's centreline (probability >= threshold) at least ``min_len_m`` long and
                     mostly (> half its cells) more than ``tol_m`` from both the physics map and a council channel:
                     an unrecorded drain, or a false detection
    council_missed   a piece (``piece_m``) of a snapped council line along which the model detects nothing within
                     ``tol_m`` for most of its length (with whether the physics map has it)

Each with ``exists`` (to fill: y / n / unsure) and ``notes``; only confident answers train (decision 5).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _taper(window):
    t = np.sin(np.pi * (np.arange(window) + 0.5) / window) ** 2
    return np.outer(t, t).astype(np.float32)


def predict_windows(get, h, w, model, device, window=128, step=64, batch=256):
    """Blended probability over an h x w grid from windows of the chips' size, overlapping by half, with a smooth
    taper: ``get(r, c)`` returns the standardised inputs (bands, window, window) and the valid mask of the window
    whose top-left cell is (r, c) (which may lie outside the grid). NaN where no window had valid input."""
    import torch
    wt = _taper(window)
    off = (window - step) // 2
    acc = np.zeros((h + 2 * window, w + 2 * window), np.float32)
    den = np.zeros_like(acc)
    good = np.zeros((h, w), bool)
    jobs = [(r, c) for r in range(-off, h, step) for c in range(-off, w, step)]
    for k in range(0, len(jobs), batch):
        part = jobs[k:k + batch]
        got = [get(r, c) for r, c in part]
        with torch.no_grad():
            xt = torch.from_numpy(np.stack([g[0] for g in got])).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pr = torch.sigmoid(model(xt).float()).cpu().numpy()
        for (r, c), p, (_, ok) in zip(part, pr, got):
            ww = wt * ok
            acc[r + window:r + 2 * window, c + window:c + 2 * window] += ww * p
            den[r + window:r + 2 * window, c + window:c + 2 * window] += ww
            r0, c0 = max(r, 0), max(c, 0)
            r1, c1 = min(r + window, h), min(c + window, w)
            if r1 > r0 and c1 > c0:
                good[r0:r1, c0:c1] |= ok[r0 - r:r1 - r, c0 - c:c1 - c]
    d = den[window:window + h, window:window + w]
    out = acc[window:window + h, window:window + w] / np.maximum(d, 1e-6)
    return np.where(good & (d > 0), out, np.nan).astype(np.float32)


def _norm(meta):
    mean = np.asarray(meta["mean"], np.float32)[:, None, None]
    std = np.asarray(meta["std"], np.float32)
    return mean, np.where(std > 1e-6, std, 1.0).astype(np.float32)[:, None, None]


def predict_tile(run, mosaic, model, meta, device, window=128, step=64, batch=256):
    """(probability float32 on the run's grid, NaN where no input; transform; crs). Windows of the training chips'
    size, overlapping by half, blended with a smooth taper: the model's group norm makes its output depend on the
    window it sees, so larger windows, or abutting ones, leave seams (seen on BW24_10000_0402, 18 Oct). The tile and
    a window's margin are read once into memory and the windows sliced from it: reading each window from the files
    kept the first run on one CPU core with the GPU idle (18 Oct)."""
    import rasterio

    from .channel_labels import _horizontal
    from .chips import BANDS
    if list(meta["bands"]) != list(BANDS):
        raise ValueError("the model's bands differ from the chips' (BANDS)")
    run = Path(run)
    with rasterio.open(run / "features.tif") as s:
        T, crs, (h, w) = s.transform, _horizontal(s.crs), (s.height, s.width)
    cs = abs(T.a)
    mean, std = _norm(meta)
    pad = window                                  # the tile and a window's margin, read once (~4 GB for a tile)
    region = mosaic.read_inputs(T.c - pad * cs, T.f + pad * cs, w + 2 * pad, h + 2 * pad, cs)
    okr = np.isfinite(region).all(axis=0)
    region = np.clip((np.where(np.isfinite(region), region, 0.0) - mean) / std, -10, 10).astype(np.float32)

    def get(r, c):
        return (region[:, r + pad:r + pad + window, c + pad:c + pad + window],
                okr[r + pad:r + pad + window, c + pad:c + pad + window])
    return predict_windows(get, h, w, model, device, window, step, batch), T, crs


# ---------------------------------------------------------------- in the run (ML-3c)

WAY_LAYERS = ("roads", "rail", "tracks", "osm_roads")


def ways_distance(shape, transform, crs, roads=None, context=None, cap_m=500.0):
    """Distance (m, capped) to any road, rail line, track or OSM way on a grid, as ``chips.ways_raster`` makes it
    for the chips (here from the run's roads and the bundle's context.gpkg, on the run's window)."""
    import geopandas as gpd
    import pyogrio
    from rasterio.features import rasterize
    from scipy import ndimage as ndi
    geoms = []
    if roads is not None and len(roads):
        geoms += [g for g in roads.to_crs(crs).geometry if g is not None and not g.is_empty]
    if context is not None and Path(context).exists():
        have = {n for n, _ in pyogrio.list_layers(context)}
        for name in WAY_LAYERS:
            if name in have and not (name == "roads" and roads is not None):
                g = gpd.read_file(context, layer=name)
                if len(g):
                    geoms += [x for x in g.to_crs(crs).geometry if x is not None and not x.is_empty]
    if not geoms:
        return np.full(shape, cap_m, np.float32)
    on = rasterize([(g, 1) for g in geoms], out_shape=shape, transform=transform, fill=0, all_touched=True,
                   dtype="uint8") > 0
    if not on.any():
        return np.full(shape, cap_m, np.float32)
    return np.minimum(ndi.distance_transform_edt(~on) * abs(transform.a), cap_m).astype(np.float32)


def run_inputs(feats, z, upa, z_breached, dep_depth, rem, dsm, ways):
    """The 23 input bands on the run's window, named as the chips' (``chips.BANDS``): the features, the hydrology
    bands as ``run._write_rasters`` writes them, and the distance to any way."""
    from .chips import BANDS, FEATURE_BANDS
    valid = np.isfinite(z)
    b = {k: feats[k] for k in FEATURE_BANDS}
    b["rem_m"] = rem if rem is not None else np.full(z.shape, np.nan, np.float32)
    b["log10_upstream_area_m2"] = np.where(valid, np.log10(np.maximum(upa, 1.0)), np.nan)
    b["breach_depth_m"] = np.where(valid, z - z_breached, np.nan)
    b["depression_depth_m"] = np.where(valid, dep_depth, np.nan)
    b["dsm_minus_dem_m"] = dsm - z if dsm is not None else np.full(z.shape, np.nan, np.float32)
    b["dist_any_way_m"] = ways
    return [np.asarray(b[n], np.float32) for n in BANDS]


def channel_mask_in_run(model_dir, bands, device="auto", threshold=None, min_cells=30, model=None, meta=None):
    """(probability, channel mask, info) on the run's window: the U-Net over the in-memory bands (``run_inputs``),
    thresholded (the model's threshold unless given), with fragments of fewer than ``min_cells`` cells dropped.
    ``model``/``meta`` may be given directly (tests); otherwise loaded from ``model_dir``."""
    import time

    import torch
    from scipy import ndimage as ndi

    from .chips import BANDS
    from .unet import load_model
    t0 = time.perf_counter()
    dev = torch.device("cuda" if device != "cpu" and torch.cuda.is_available() else "cpu")
    if model is None:
        model, meta = load_model(model_dir, dev)
    if list(meta["bands"]) != list(BANDS):
        raise ValueError("the model's bands differ from the chips' (BANDS)")
    mean, std = _norm(meta)
    h, w = bands[0].shape
    win = 128

    def get(r, c):
        r0, c0, r1, c1 = max(r, 0), max(c, 0), min(r + win, h), min(c + win, w)
        x = np.full((len(bands), win, win), np.nan, np.float32)
        if r1 > r0 and c1 > c0:
            for k, a in enumerate(bands):
                x[k, r0 - r:r1 - r, c0 - c:c1 - c] = a[r0:r1, c0:c1]
        ok = np.isfinite(x).all(axis=0)
        return np.clip((np.where(np.isfinite(x), x, 0.0) - mean) / std, -10, 10).astype(np.float32), ok
    prob = predict_windows(get, h, w, model, dev, window=win, step=64)
    thr = float(threshold if threshold is not None else meta.get("threshold", 0.5))
    m = np.nan_to_num(prob) >= thr
    if min_cells > 1 and m.any():
        lab, n = ndi.label(m, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())
        keep = sizes >= min_cells
        keep[0] = False
        m = keep[lab]
    info = dict(model=str(model_dir), threshold=thr, device=dev.type, cells=int(m.sum()),
                seconds=round(time.perf_counter() - t0, 1), dataset=meta.get("dataset"))
    return prob, m, info


def write_prob(run, prob, T, crs, model_dir, threshold, fold=None, held_out=None):
    """``ml_channel_prob.tif`` and its record; ``held_out`` says the tile was in the fold's test set (the model never
    trained on it), which the label cleaning (labels v3) requires."""
    from .. import dataio
    run = Path(run)
    dataio.write_stack(run / "ml_channel_prob.tif", {"channel_prob": prob}, T, crs)
    (run / "ml_channel_prob.json").write_text(json.dumps(dict(model=str(model_dir), threshold=threshold, fold=fold,
                                                              held_out=held_out), indent=1))


def _near(mask, radius_px):
    from scipy import ndimage as ndi
    if not mask.any():
        return np.zeros_like(mask, bool)
    return ndi.distance_transform_edt(~mask) <= radius_px


def review_reaches(run, prob, T, crs, threshold, tol_m=3.0, min_len_m=20.0, piece_m=50.0, max_per_kind=40, seed=0):
    """The review reaches of one tile (GeoDataFrame), sampled to at most ``max_per_kind`` of each kind."""
    import geopandas as gpd
    import pyogrio
    import rasterio
    from shapely.geometry import LineString
    from shapely.ops import substring
    from skimage.morphology import skeletonize

    from ..network import build_reaches
    run = Path(run)
    cs = abs(T.a)
    tol = tol_m / cs
    with rasterio.open(run / "hydro.tif") as s:
        phys = s.read(list(s.descriptions).index("channel_mask") + 1) > 0
    with rasterio.open(run / "ml_channel_labels.tif") as s:
        lab = s.read(1)
    valid = np.isfinite(prob)
    pred = valid & (np.nan_to_num(prob) >= threshold)
    near_phys, near_pos, near_pred = _near(phys, tol), _near(lab == 1, tol), _near(pred, tol)
    rows = []
    _, _, reaches = build_reaches(skeletonize(pred), T)
    for rch in reaches:
        cells = rch.cells
        if len(rch.coords) < 2:
            continue
        g = LineString(rch.coords)
        if g.length < min_len_m:
            continue
        r, c = cells[:, 0], cells[:, 1]
        ph, co = float(near_phys[r, c].mean()), float(near_pos[r, c].mean())
        if ph <= 0.5 and co <= 0.5:
            rows.append(dict(kind="model_only", length_m=round(g.length, 1), mean_prob=round(float(np.nanmean(prob[r, c])), 3),
                             physics_share=round(ph, 2), council_share=round(co, 2), geometry=g))
    f = run / "ml_channels.gpkg"
    if f.exists() and "council_channels" in {n for n, _ in pyogrio.list_layers(f)}:
        lines = gpd.read_file(f, layer="council_channels")
        lines = lines[lines.snapped.astype(bool)]
        for g, sid, council in zip(lines.geometry, lines.source_id, lines.council):
            n = max(1, int(round(g.length / piece_m)))
            for k in range(n):
                piece = substring(g, k * g.length / n, (k + 1) * g.length / n)
                if piece.length < min_len_m:
                    continue
                pts = [piece.interpolate(d) for d in np.arange(0, piece.length, 2.0)]
                cc = np.floor((np.array([q.x for q in pts]) - T.c) / T.a).astype(int)
                rr = np.floor((np.array([q.y for q in pts]) - T.f) / T.e).astype(int)
                ok = (rr >= 0) & (rr < prob.shape[0]) & (cc >= 0) & (cc < prob.shape[1])
                if ok.sum() < 3:
                    continue
                hit = float(near_pred[rr[ok], cc[ok]].mean())
                if hit < 0.5:
                    rows.append(dict(kind="council_missed", length_m=round(piece.length, 1),
                                     mean_prob=round(float(np.nanmean(prob[rr[ok], cc[ok]])), 3),
                                     physics_share=round(float(near_phys[rr[ok], cc[ok]].mean()), 2),
                                     council_share=1.0, council=council, source_id=str(sid), geometry=piece))
    if not rows:
        return gpd.GeoDataFrame(columns=["kind", "geometry"], geometry=[], crs=crs)
    d = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    rng = np.random.default_rng(seed)
    keep = []
    for kind, g in d.groupby("kind"):
        keep += list(rng.permutation(g.index.to_numpy())[:max_per_kind])
    d = d.loc[sorted(keep)].reset_index(drop=True)
    d.insert(0, "tile", run.name)
    d["exists"] = ""
    d["notes"] = ""
    return d
