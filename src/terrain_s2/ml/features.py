"""ML-0 step 4: features and oriented patches for every crossing candidate, inside the run.

``extract(res, out)`` takes a ``run.network(..., keep=True)`` result and writes, beside the run:

- ``ml_features.parquet``: one row per written candidate (the ``crossings`` layer, same order, keyed by ``x``,
  ``y``): the candidate's own evidence (h_b, gap, deck, vegetation, road distance, tiers, which tests and
  sources proposed it), and for every raster band (the 17-band feature stack and the hydrology bands) four
  statistics around the crest: the value at it, the maximum and minimum within 5 m, and the mean within
  10 m; plus the nearest residual-pond outlet (distance, volume, inflow) from the conditioned pass;
- ``ml_patches.npy`` (float16, candidates x bands x 64 x 64, 1 m) and ``ml_patches.json`` (band names, size):
  patches centred on the crest and rotated so the local channel axis runs along the patch's x axis (the
  barrier then lies across it), for the patch CNN (ML-1b). The direction along the axis is arbitrary (flip
  in training). ``z_rel`` is the DEM relative to the crest cell.

Only quantities available everywhere are used: DEM, DSM, roads (LINZ / OSM) and what the chain derives from
them; never council layers (they are the labels). ``FEATURES_VERSION`` is part of the batch runner's
parameter hash, so a change here reruns the tiles.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

FEATURES_VERSION = "2"     # 2: one flag per proposing test (v1 had one per combination, e.g. src_bump+gap+testA)
PATCH_BANDS = ["z_rel", "dsm_minus_dem", "rem", "log10_upa", "breach_depth", "depression_depth",
               "residual_depression", "channel_mask", "centreline", "road_distance", "relief_med11",
               "tophat_white21", "tophat_black21", "openness_pos"]
MASK_BANDS = {"channel_mask", "centreline"}
TIERS = {"low": 1, "medium": 2, "high": 3, "bridge_or_open": 4, "bridge": 5}
# the tests and sources that propose candidates (products.crossings, crossings.candidates); a fixed set, so every
# tile has the same columns
SOURCE_FLAGS = ["testA", "testB", "bump", "gap", "road_x_drain", "road_end_pair"]


def split_sources(v) -> list:
    """'bump+gap+testA' or 'testA,testB' -> the individual names."""
    import re
    return [s for s in re.split(r"[+,;]", str(v or "")) if s]


def normalise_source_flags(df):
    """Features v1 (one column per combination, ``src_bump+gap+testA``) -> v2 flags (``src_bump``, ``src_gap``,
    ``src_testA``, ...); v2 frames are returned unchanged. Works on a frame or a dataset."""
    combos = [c for c in df.columns if c.startswith("src_") and "+" in c]
    singles_v1 = [c for c in df.columns if c.startswith("src_") and c[4:] in SOURCE_FLAGS]
    if not combos:
        return df
    out = df.drop(columns=combos + singles_v1)
    flags = {s: np.zeros(len(df)) for s in SOURCE_FLAGS}
    for c in combos + singles_v1:
        v = np.nan_to_num(df[c].to_numpy(float))
        for s in split_sources(c[4:]):
            if s in flags:
                flags[s] = np.maximum(flags[s], v)
    import pandas as pd
    return pd.concat([out, pd.DataFrame({f"src_{s}": flags[s] for s in SOURCE_FLAGS}, index=df.index)], axis=1)


def _bands(state) -> dict:
    """The rasters on the run's window grid (float32, NaN outside the DEM)."""
    w, res = state["w"], state["res"]
    z = w.z.astype(np.float32)
    valid = np.isfinite(z)
    b = {}
    for k, v in res.feats.items():
        b[k] = np.asarray(v, np.float32)
    dsm = state.get("dsm")
    b["dsm_minus_dem"] = (np.asarray(dsm.z if hasattr(dsm, "z") else dsm, np.float32) - z) if dsm is not None \
        else np.full(z.shape, np.nan, np.float32)
    b["rem"] = np.asarray(state["rem"], np.float32) if state.get("rem") is not None else np.full(z.shape, np.nan, np.float32)
    b["log10_upa"] = np.where(valid, np.log10(np.maximum(np.nan_to_num(res.upa), 1.0)), np.nan).astype(np.float32)
    b["breach_depth"] = np.where(valid, z - res.breach.z_breached, np.nan).astype(np.float32)
    b["depression_depth"] = np.where(valid, res.dep.depth, np.nan).astype(np.float32)
    cond = state.get("cond")
    b["residual_depression"] = (np.asarray(cond.residual, np.float32) if cond is not None
                                else np.full(z.shape, np.nan, np.float32))
    b["channel_mask"] = np.asarray(state["net"]["mask"], np.float32)
    b["centreline"] = np.asarray(state["sk"], np.float32)
    rd = state.get("rd")
    b["road_distance"] = (np.minimum(np.asarray(rd, np.float32), 200.0) if rd is not None
                          else np.full(z.shape, np.nan, np.float32))
    b["z"] = z
    return b


def _rc(T, x, y):
    c = np.floor((np.asarray(x) - T.c) / T.a).astype(int)
    r = np.floor((np.asarray(y) - T.f) / T.e).astype(int)
    return r, c


def _window_stats(a, r, c, rad):
    """max, min, mean of ``a`` in a disk of radius ``rad`` cells around each (r, c) (NaN ignored)."""
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    disk = (yy ** 2 + xx ** 2) <= rad ** 2
    dr, dc = yy[disk], xx[disk]
    rr = np.clip(r[:, None] + dr[None], 0, a.shape[0] - 1)
    cc = np.clip(c[:, None] + dc[None], 0, a.shape[1] - 1)
    v = a[rr, cc]
    with np.errstate(all="ignore"), _quiet():
        return np.nanmax(v, 1), np.nanmin(v, 1), np.nanmean(v, 1)


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def tabular(cands, bands, T, cs, outlets=None):
    """One row of features per candidate (a DataFrame, same order as ``cands``)."""
    import pandas as pd
    r, c = _rc(T, cands.x.to_numpy(), cands.y.to_numpy())
    H, W = bands["z"].shape
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    r, c = np.clip(r, 0, H - 1), np.clip(c, 0, W - 1)
    f = {"x": cands.x.to_numpy(), "y": cands.y.to_numpy()}            # columns, framed once at the end
    for k in ("dem_h_b", "veg_frac", "gap_m", "deck_m", "road_dist_m"):
        f[k] = pd.to_numeric(cands[k], errors="coerce").to_numpy() if k in cands else np.nan
    for k in ("on_stream", "network_link"):
        f[k] = cands[k].astype(float).to_numpy() if k in cands else np.nan
    for k in ("tier", "tier_dem_only"):
        f[k] = cands[k].map(TIERS).astype(float).to_numpy() if k in cands else np.nan
    srcs = cands["sources"].fillna("").astype(str) if "sources" in cands else pd.Series([""] * len(cands))
    prim = cands["source"].fillna("").astype(str) if "source" in cands else pd.Series([""] * len(cands))
    sets = [set(split_sources(a)) | set(split_sources(b)) for a, b in zip(srcs, prim)]
    for s in SOURCE_FLAGS:
        f[f"src_{s}"] = np.array([float(s in st) for st in sets])
    f["n_sources"] = np.array([float(len(st)) for st in sets])
    r5, r10 = max(int(round(5 / cs)), 1), max(int(round(10 / cs)), 1)
    for k, a in bands.items():
        if k == "z":
            continue
        f[f"{k}_at"] = np.where(ok, a[r, c], np.nan)
        mx, mn, _ = _window_stats(a, r, c, r5)
        _, _, me = _window_stats(a, r, c, r10)
        f[f"{k}_max5"], f[f"{k}_min5"], f[f"{k}_mean10"] = mx, mn, me
    if outlets is not None and len(outlets):
        from scipy.spatial import cKDTree
        xy = np.column_stack([outlets.geometry.x, outlets.geometry.y])
        d, i = cKDTree(xy).query(np.column_stack([f["x"], f["y"]]))
        f["outlet_dist_m"] = d
        f["outlet_volume_m3"] = outlets.volume_m3.to_numpy()[i]
        f["outlet_inflow"] = outlets.inflow.astype(float).to_numpy()[i]
        f["outlet_max_depth_m"] = outlets.max_depth_m.to_numpy()[i]
    else:
        for k in ("outlet_dist_m", "outlet_volume_m3", "outlet_inflow", "outlet_max_depth_m"):
            f[k] = np.nan
    return pd.DataFrame(f)


def axis_angles(sk, T, x, y, cs, search_m=10.0, radius_m=8.0):
    """Local channel axis (rad, map frame) at the centreline cell nearest each point (within ``search_m``);
    0 where there is none."""
    from .. import channels
    ang = channels.skeleton_orientation(np.asarray(sk, bool), max(int(round(radius_m / cs)), 2))
    rs, cs_ = np.nonzero(sk)
    out = np.zeros(len(x))
    if not len(rs):
        return out
    from scipy.spatial import cKDTree
    px = T.c + (cs_ + 0.5) * T.a
    py = T.f + (rs + 0.5) * T.e
    d, i = cKDTree(np.column_stack([px, py])).query(np.column_stack([x, y]))
    near = d <= search_m
    out[near] = ang[rs[i[near]], cs_[i[near]]]
    return out


def patches(cands, bands, T, sk, cs, size=64, step_m=1.0, chunk=256):
    """(N, len(PATCH_BANDS), size, size) float16, rotated to the channel axis, centred on each crest."""
    N = len(cands)
    out = np.full((N, len(PATCH_BANDS), size, size), np.nan, np.float16)
    if not N:
        return out
    x, y = cands.x.to_numpy(float), cands.y.to_numpy(float)
    a = axis_angles(sk, T, x, y, cs)
    u = (np.arange(size) - (size - 1) / 2) * step_m
    U, V = np.meshgrid(u, -u)                                  # patch row 0 at the top (+v)
    r0, c0 = _rc(T, x, y)
    zc = bands["z"][np.clip(r0, 0, bands["z"].shape[0] - 1), np.clip(c0, 0, bands["z"].shape[1] - 1)]
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        ca, sa = np.cos(a[s:e])[:, None, None], np.sin(a[s:e])[:, None, None]
        X = x[s:e, None, None] + U * ca - V * sa
        Y = y[s:e, None, None] + U * sa + V * ca
        cc = (X - T.c) / T.a - 0.5
        rr = (Y - T.f) / T.e - 0.5
        coords = np.stack([rr.ravel(), cc.ravel()])
        for bi, name in enumerate(PATCH_BANDS):
            src = bands["z"] if name == "z_rel" else bands.get(name)
            if src is None:
                continue
            v = ndi.map_coordinates(np.asarray(src, np.float32), coords, order=0 if name in MASK_BANDS else 1,
                                    mode="constant", cval=np.nan).reshape(e - s, size, size)
            if name == "z_rel":
                v = v - zc[s:e, None, None]
            out[s:e, bi] = v.astype(np.float16)
    return out


def extract(res, out_dir, size=64, write=True, with_patches=True):
    """Features and patches for the run's written candidates (``res`` from run.network(..., keep=True)).
    ``with_patches=False``: the table only (scoring inside a run)."""
    if res.state is None:
        raise ValueError("features need the run's arrays: run.network(..., keep=True)")
    st = res.state
    T = st["w"].transform
    cs = abs(T.a)
    cands = res.layers.get("crossings")
    if cands is None or not len(cands):
        import pandas as pd
        cands = pd.DataFrame(columns=["x", "y"])
    bands = _bands(st)
    cond = st.get("cond")
    tab = tabular(cands, bands, T, cs, outlets=getattr(cond, "outlets", None) if cond is not None else None)
    pat = patches(cands, bands, T, st["sk"], cs, size=size) if with_patches else None
    meta = dict(version=FEATURES_VERSION, bands=PATCH_BANDS, size=size, cell_m=1.0, n=int(len(tab)),
                columns=list(tab.columns), dtype="float16", order="rows of ml_features.parquet")
    if write:
        out_dir = Path(out_dir)
        tab.to_parquet(out_dir / "ml_features.parquet", index=False)
        if pat is not None:
            np.save(out_dir / "ml_patches.npy", pat)
            (out_dir / "ml_patches.json").write_text(json.dumps(meta, indent=1))
    return tab, pat, meta
