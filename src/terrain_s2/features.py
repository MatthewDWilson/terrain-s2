"""Feature stack (design §3), written once for NumPy and CuPy.

Every function takes arrays on the backend's device and returns arrays on the
same device. Scales are in metres and converted to cells here, so the code is
resolution-independent (T1).

Cost notes (CPU, one core, 1.4 M cells): median 11 m ≈ 2.6 s, 21 m ≈ 8 s,
41 m ≈ 29 s; grey opening at any size ≈ 0.06 s (separable min/max). Median
high-pass is therefore used only at channel scales; embankment-scale relief
uses morphological top-hats, which remove raised (white) or incised (black)
features narrower than the structuring element and preserve planar slopes.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .backend import Backend
from .config import FeatureParams


def odd_cells(metres: float, cellsize: float, minimum: int = 3) -> int:
    n = int(round(metres / cellsize))
    n = max(n, minimum)
    return n if n % 2 == 1 else n + 1


def fill_nodata_nearest(z, valid, be: Backend):
    """Replace invalid cells by the nearest valid value (so filters do not see NaN)."""
    if bool(be.xp.all(valid)):
        return z
    idx = be.ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return z[idx[0], idx[1]]


def median_relief(z, scale_m: float, cellsize: float, be: Backend):
    """DEM minus median filter: negative in channels/ditches, positive on ridges."""
    n = odd_cells(scale_m, cellsize)
    return z - be.ndi.median_filter(z, size=n, mode="nearest")


def tophats(z, scale_m: float, cellsize: float, be: Backend):
    """(white, black) top-hat: raised and incised features narrower than ``scale_m``."""
    n = odd_cells(scale_m, cellsize)
    white = z - be.ndi.grey_opening(z, size=(n, n), mode="nearest")
    black = be.ndi.grey_closing(z, size=(n, n), mode="nearest") - z
    return white, black


def hessian_eigen(z, sigma_m: float, cellsize: float, be: Backend):
    """Eigenvalues (l1 <= l2) of the Gaussian-smoothed Hessian, in 1/m.

    Ridge (e.g. embankment crest): l1 << 0, |l1| >> |l2|.
    Valley/channel: l2 >> 0, |l2| >> |l1|.
    Eigenvalues are invariant to the row-axis sign convention.
    """
    s = sigma_m / cellsize
    zyy = be.ndi.gaussian_filter(z, s, order=(2, 0), mode="nearest") / cellsize**2
    zxx = be.ndi.gaussian_filter(z, s, order=(0, 2), mode="nearest") / cellsize**2
    zxy = be.ndi.gaussian_filter(z, s, order=(1, 1), mode="nearest") / cellsize**2
    half_tr = 0.5 * (zxx + zyy)
    disc = be.xp.sqrt((0.5 * (zxx - zyy)) ** 2 + zxy**2)
    return half_tr - disc, half_tr + disc


def ridge_valley_measures(l1, l2, be: Backend):
    """Line-likeness from Hessian eigenvalues, each in [0, 1] x strength (1/m)."""
    xp = be.xp
    eps = 1e-12
    a1, a2 = xp.abs(l1), xp.abs(l2)
    ridge = xp.where(l1 < 0, (a1 - a2).clip(0) / (a1 + a2 + eps), 0.0) * a1
    valley = xp.where(l2 > 0, (a2 - a1).clip(0) / (a1 + a2 + eps), 0.0) * a2
    return ridge, valley


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIRS16_EXTRA = [(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)]


def openness(z, valid, radius_m: float, cellsize: float, be: Backend, n_directions: int = 8):
    """Positive and negative topographic openness (degrees), by shift-and-scan.

    For each direction, cells along the ray are visited by shifting the padded
    array, so the whole computation is vectorised array arithmetic and runs on
    GPU without a custom kernel. Cost: n_directions x (radius / step) array ops.
    Rays stop at nodata (treated as NaN and ignored by fmax/fmin).
    """
    xp = be.xp
    dirs = list(_DIRS8) + (list(_DIRS16_EXTRA) if n_directions == 16 else [])
    kmax = int(math.ceil(radius_m / cellsize))
    pad = 2 * kmax
    zn = xp.where(valid, z, xp.nan).astype(xp.float32)
    zp = xp.pad(zn, pad, mode="constant", constant_values=np.nan)
    nr, nc = z.shape
    pos = xp.zeros_like(zn)
    neg = xp.zeros_like(zn)
    for di, dj in dirs:
        step = cellsize * math.hypot(di, dj)
        nsteps = int(radius_m // step)
        tmax = xp.full_like(zn, -np.inf)
        tmin = xp.full_like(zn, np.inf)
        for k in range(1, nsteps + 1):
            r0, c0 = pad + k * di, pad + k * dj
            t = (zp[r0:r0 + nr, c0:c0 + nc] - zn) / (k * step)
            tmax = xp.fmax(tmax, t)
            tmin = xp.fmin(tmin, t)
        tmax = xp.where(xp.isfinite(tmax), tmax, 0.0)
        tmin = xp.where(xp.isfinite(tmin), tmin, 0.0)
        pos += 90.0 - xp.degrees(xp.arctan(tmax))
        neg += 90.0 + xp.degrees(xp.arctan(tmin))
    return pos / len(dirs), neg / len(dirs)


def feature_stack(z_np: np.ndarray, cellsize: float, be: Backend,
                  p: FeatureParams | None = None) -> Dict[str, np.ndarray]:
    """Compute the §3 raster features. Input and outputs are host NumPy float32.

    Hydrological features (breaching, depressions, flow accumulation) are in
    ``hydro.py`` because they are CPU-only.
    """
    p = p or FeatureParams()
    valid_np = np.isfinite(z_np)
    z = be.asarray(z_np, dtype=be.xp.float32)
    valid = be.asarray(valid_np)
    zf = fill_nodata_nearest(z, valid, be)
    out = {}
    for s in p.relief_median_m:
        out[f"relief_med{s:g}"] = median_relief(zf, s, cellsize, be)
    for s in p.tophat_m:
        w, b = tophats(zf, s, cellsize, be)
        out[f"tophat_white{s:g}"] = w
        out[f"tophat_black{s:g}"] = b
    for s in p.hessian_sigma_m:
        l1, l2 = hessian_eigen(zf, s, cellsize, be)
        r, v = ridge_valley_measures(l1, l2, be)
        out[f"laplacian_s{s:g}"] = l1 + l2
        out[f"ridge_s{s:g}"] = r
        out[f"valley_s{s:g}"] = v
    if p.openness_radius_m > 0:
        po, no = openness(zf, valid, p.openness_radius_m, cellsize, be, p.openness_directions)
        out["openness_pos"] = po
        out["openness_neg"] = no
    be.sync()
    res = {}
    for k, v in out.items():
        a = be.to_numpy(v).astype(np.float32)
        a[~valid_np] = np.nan
        res[k] = a
    return res
