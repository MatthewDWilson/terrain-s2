"""Relative elevation model (REM): height above a smooth surface interpolated from channel elevations.

Replaces HAND (height above *nearest* drainage along the flow path) as the height-above-drainage used
for the floodplain mask and in the rasters (Matt, 8 Oct 2026): HAND takes one drainage elevation per
cell, so it jumps wherever neighbouring cells drain to different channel cells, and it is undefined
where flow leaves the window without meeting the drainage. The REM, as in OpenTopography's RiverREM
(Larrieu 2022; https://github.com/OpenTopography/RiverREM), samples elevations along the channels,
interpolates them across the DEM by inverse distance weighting (IDW, k nearest samples, power 1) and
subtracts the result from the DEM. It is continuous and defined everywhere.

Differences from RiverREM, which is not used as a dependency (it pins numpy < 2, GDAL < 3.9 and
osmnx < 2, works through files, and takes one river from OpenStreetMap):
- samples come from our drainage (first pass: wide channels and cells with large upstream area;
  conditioned pass: the mapped network), one per ``sample_m`` block, at a low quantile of the channel
  cells' elevations in the block (the water surface or bed rather than the banks);
- the interpolation runs on a coarse grid (``coarse_m``, default 20 m: the surface is smooth by design)
  and is upsampled bilinearly to the DEM grid, which makes it cheap (a LINZ tile: ~10^5 coarse cells).

Edge bias: within about k x sample_m / 2 (~60 m by default) of the window edge the k nearest samples lie
on one side, so along a sloping channel the surface is biased (up to ~1 m on a 2 % valley in the tests).
That band is inside the 200 m buffer of sites and tiles, so the core is not affected.

Caveat: IDW is planimetric, not hydrological. Across a stopbank or terrace the surface blends the river
and the landside drains; HAND follows flow instead. For the floodplain mask this smoothness is the
point; a river-only REM (wide channels only) is the variant to use where that blending matters.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def samples(z, T, drain, sample_m=10.0, q=0.25):
    """(x, y, elevation) per ``sample_m`` block containing drainage cells: the q-quantile of their
    elevations, at their mean position."""
    import pandas as pd
    r, c = np.nonzero(drain & np.isfinite(z))
    if not len(r):
        return np.zeros((0, 2)), np.zeros(0)
    cs = abs(T.a)
    b = max(int(round(sample_m / cs)), 1)
    nb = z.shape[1] // b + 1
    df = pd.DataFrame({"blk": (r // b) * nb + c // b, "r": r, "c": c, "z": z[r, c]})
    g = df.groupby("blk", sort=False)
    zq = g.z.quantile(q).to_numpy()
    rm, cm = g.r.mean().to_numpy(), g.c.mean().to_numpy()
    x = T.c + (cm + 0.5) * T.a
    y = T.f + (rm + 0.5) * T.e
    return np.column_stack([x, y]), zq


def base_surface(shape, T, xy, zs, coarse_m=20.0, k=12, power=1.0):
    """Channel-elevation surface on the DEM grid: IDW of the samples on a coarse grid, upsampled
    bilinearly (cell centres aligned)."""
    cs = abs(T.a)
    f = max(int(round(coarse_m / cs)), 1)
    nr, nc = -(-shape[0] // f) + 1, -(-shape[1] // f) + 1
    gx = T.c + (np.arange(nc) * f + 0.5 * f) * T.a
    gy = T.f + (np.arange(nr) * f + 0.5 * f) * T.e
    X, Y = np.meshgrid(gx, gy)
    k = int(min(k, len(zs)))
    d, i = cKDTree(xy).query(np.column_stack([X.ravel(), Y.ravel()]), k=k)
    if k == 1:
        d, i = d[:, None], i[:, None]
    d = np.maximum(d, 1e-6)
    w = 1.0 / d ** power
    coarse = ((w * zs[i]).sum(1) / w.sum(1)).reshape(nr, nc)
    # DEM cell (row, col) centre sits at coarse coordinate ((row + 0.5) / f - 0.5)
    rr = (np.arange(shape[0]) + 0.5) / f - 0.5
    cc = (np.arange(shape[1]) + 0.5) / f - 0.5
    R, C = np.meshgrid(rr, cc, indexing="ij")
    return ndi.map_coordinates(coarse, [R, C], order=1, mode="nearest").astype(np.float32)


def rem(z, T, drain, sample_m=10.0, coarse_m=20.0, k=12, power=1.0, q=0.25):
    """(REM, base surface, number of samples). NaN outside the DEM; all NaN if there is no drainage."""
    xy, zs = samples(z, T, drain, sample_m, q)
    if not len(zs):
        nan = np.full(z.shape, np.nan, np.float32)
        return nan, nan.copy(), 0
    base = base_surface(z.shape, T, xy, zs, coarse_m, k, power)
    out = np.where(np.isfinite(z), z - base, np.nan).astype(np.float32)
    return out, np.where(np.isfinite(z), base, np.nan).astype(np.float32), len(zs)
