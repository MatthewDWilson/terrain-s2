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


def predict_tile(run, mosaic, model, meta, device, window=128, step=64, batch=128):
    """(probability float32 on the run's grid, NaN where no input; transform; crs). Windows of the training chips'
    size, overlapping by half, blended with a smooth taper: the model's group norm makes its output depend on the
    window it sees, so larger windows, or abutting ones, leave seams (seen on BW24_10000_0402, 18 Oct)."""
    import rasterio
    import torch

    from .channel_labels import _horizontal
    from .chips import BANDS
    if list(meta["bands"]) != list(BANDS):
        raise ValueError("the model's bands differ from the chips' (BANDS)")
    run = Path(run)
    with rasterio.open(run / "features.tif") as s:
        T, crs, (h, w) = s.transform, _horizontal(s.crs), (s.height, s.width)
    cs = abs(T.a)
    mean = np.asarray(meta["mean"], np.float32)[:, None, None]
    std = np.asarray(meta["std"], np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)[:, None, None]
    taper = np.sin(np.pi * (np.arange(window) + 0.5) / window) ** 2
    wt = np.outer(taper, taper).astype(np.float32)
    off = (window - step) // 2
    acc = np.zeros((h + 2 * window, w + 2 * window), np.float32)
    den = np.zeros_like(acc)
    good = np.zeros((h, w), bool)
    jobs = [(r, c) for r in range(-off, h, step) for c in range(-off, w, step)]
    for k in range(0, len(jobs), batch):
        part = jobs[k:k + batch]
        xs, oks = [], []
        for r, c in part:
            x, _, _, _ = mosaic.read(T.c + c * cs, T.f - r * cs, window, cs)
            oks.append(np.isfinite(x).all(axis=0))
            xs.append(np.clip((np.where(np.isfinite(x), x, 0.0) - mean) / std, -10, 10).astype(np.float32))
        with torch.no_grad():
            xt = torch.from_numpy(np.stack(xs)).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pr = torch.sigmoid(model(xt).float()).cpu().numpy()
        for (r, c), p, ok in zip(part, pr, oks):
            ww = wt * ok
            acc[r + window:r + 2 * window, c + window:c + 2 * window] += ww * p
            den[r + window:r + 2 * window, c + window:c + 2 * window] += ww
            r0, c0 = max(r, 0), max(c, 0)
            r1, c1 = min(r + window, h), min(c + window, w)
            if r1 > r0 and c1 > c0:
                good[r0:r1, c0:c1] |= ok[r0 - r:r1 - r, c0 - c:c1 - c]
    out = acc[window:window + h, window:window + w] / np.maximum(den[window:window + h, window:window + w], 1e-6)
    return np.where(good & (den[window:window + h, window:window + w] > 0), out, np.nan).astype(np.float32), T, crs


def write_prob(run, prob, T, crs, model_dir, threshold):
    from .. import dataio
    run = Path(run)
    dataio.write_stack(run / "ml_channel_prob.tif", {"channel_prob": prob}, T, crs)
    (run / "ml_channel_prob.json").write_text(json.dumps(dict(model=str(model_dir), threshold=threshold), indent=1))


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
