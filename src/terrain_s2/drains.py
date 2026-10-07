"""Drain network refinement and typed gaps (Oct 2026 revision; see DESIGN_NOTES §4h).

The channel network is the foundation; culverts are a property of *gaps* in it.

1. ``hysteresis_mask``: weaker channel evidence is kept only where it connects to strong channel
   cells, recovering shallow or vegetated continuations of real drains without admitting isolated
   furrows (border-dyke irrigation, cultivation).
2. ``facing_pairs``: pairs of channel ends that face each other: within ``max_gap_m``, each end's
   approach direction pointing along the gap within ``max_angle_deg``.
3. ``type_gaps``: for each facing pair, what lies across the gap: a raised crossing (rise above both
   beds ≥ ``min_rise_m`` and a raised feature present) is a *culvert gap*; otherwise a *continuity*
   gap, which is joined into the network (a DEM-conditioning repair).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

_EIGHT = np.ones((3, 3), bool)


def _to_cr(T, x, y):
    """Map coordinates -> fractional (col, row) for a north-up grid (arrays or scalars)."""
    return (np.asarray(x) - T.c) / T.a, (np.asarray(y) - T.f) / T.e


def _to_xy(T, col, row):
    return T.c + np.asarray(col) * T.a, T.f + np.asarray(row) * T.e


@dataclass
class DrainParams:
    weak_narrow_valley: float = 0.012     # Hessian valley measure (sigma 2 m), 1/m
    weak_narrow_incision: float = 0.08    # black top-hat 11 m, m
    weak_wide_valley: float = 0.006       # sigma 4 m
    weak_wide_incision: float = 0.15      # black top-hat 21 m
    max_gap_m: float = 30.0
    max_angle_deg: float = 30.0           # each end's approach vs the gap direction
    min_rise_m: float = 0.3               # gap profile above the higher bed for a raised crossing
    raised_min_m: float = 0.3             # white top-hat (41 m) for a raised crossing
    continuity_max_rise_m: float = 0.15   # join only if the gap rises less than this


def hysteresis_mask(strong: np.ndarray, feats: dict, p: DrainParams) -> np.ndarray:
    weak = (((feats["valley_s2"] >= p.weak_narrow_valley) & (feats["tophat_black11"] >= p.weak_narrow_incision))
            | ((feats["valley_s4"] >= p.weak_wide_valley) & (feats["tophat_black21"] >= p.weak_wide_incision)))
    cand = strong | np.nan_to_num(weak).astype(bool)
    lab, n = ndi.label(cand, _EIGHT)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False
    return keep[lab]


@dataclass
class Gap:
    i: int
    j: int
    xy0: tuple
    xy1: tuple
    length: float
    rise: float
    raised: float
    veg: float
    kind: str                             # "culvert" or "continuity"
    extra: dict = field(default_factory=dict)


def facing_pairs(ends_xy: np.ndarray, dirs: np.ndarray, p: DrainParams):
    cos = math.cos(math.radians(p.max_angle_deg))
    out = []
    for i, j in cKDTree(ends_xy).query_pairs(p.max_gap_m):
        d = ends_xy[j] - ends_xy[i]
        L = float(np.hypot(*d))
        if L < 1.0:
            continue
        u = d / L
        if dirs[i] @ u >= cos and dirs[j] @ (-u) >= cos:
            out.append((i, j, L))
    return out


def _profile(arr, T, a, b, step=0.5):
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step) + 1)
    s = np.linspace(0.0, 1.0, n)
    x = a[0] + s * (b[0] - a[0])
    y = a[1] + s * (b[1] - a[1])
    cc, rr = _to_cr(T, x, y)
    return ndi.map_coordinates(np.nan_to_num(arr), [rr - 0.5, cc - 0.5], order=1, mode="nearest")


def type_gaps(pairs, ends_xy, z, white, T, p: DrainParams, dsm=None):
    gaps = []
    for i, j, L in pairs:
        a, b = ends_xy[i], ends_xy[j]
        zp = _profile(z, T, a, b)
        beds = max(zp[0], zp[-1])
        rise = float(zp.max() - beds)
        raised = float(_profile(white, T, a, b).max())
        veg = float(np.mean(_profile(dsm, T, a, b) - zp > 1.0)) if dsm is not None else float("nan")
        if rise >= p.min_rise_m and raised >= p.raised_min_m:
            kind = "culvert"
        elif rise < p.continuity_max_rise_m:
            kind = "continuity"
        else:
            kind = "uncertain"
        gaps.append(Gap(i, j, tuple(a), tuple(b), L, rise, raised, veg, kind))
    # one gap per end: keep the shortest (most plausible) pairing for each end
    gaps.sort(key=lambda g: g.length)
    used, out = set(), []
    for g in gaps:
        if g.i in used or g.j in used:
            continue
        used.update((g.i, g.j))
        out.append(g)
    return out


def burn_joins(sk: np.ndarray, gaps, T) -> np.ndarray:
    """Draw continuity joins into the skeleton (8-connected straight lines)."""
    from skimage.draw import line
    out = sk.copy()
    for g in gaps:
        if g.kind != "continuity":
            continue
        c0, r0 = _to_cr(T, *g.xy0)
        c1, r1 = _to_cr(T, *g.xy1)
        rr, cc = line(int(r0), int(c0), int(r1), int(c1))
        ok = (rr >= 0) & (rr < sk.shape[0]) & (cc >= 0) & (cc < sk.shape[1])
        out[rr[ok], cc[ok]] = True
    return out


# --------------------------------------------------------------------------- bed-profile bumps
@dataclass
class BumpParams:
    bed_radius_m: float = 1.0             # bed = lowest ground within this distance of the centreline (2 m erodes narrow driveway crests)
    reference_m: float = 15.0             # beds compared this far either side along the drain
    min_height_m: float = 0.3             # h_b: bump above the higher of the two reference beds
    min_length_m: float = 2.0
    max_length_m: float = 60.0
    min_branch_m: float = 10.0


def join_all(sk, pairs, ends_xy, T):
    """Join every facing pair into the skeleton (8-connected straight lines); returns (sk, joins)."""
    from skimage.draw import line
    out = sk.copy()
    joins = []
    used = set()
    for i, j, L in sorted(pairs, key=lambda t: t[2]):
        if i in used or j in used:
            continue
        used.update((i, j))
        c0, r0 = _to_cr(T, *ends_xy[i])
        c1, r1 = _to_cr(T, *ends_xy[j])
        rr, cc = line(int(r0), int(c0), int(r1), int(c1))
        ok = (rr >= 0) & (rr < sk.shape[0]) & (cc >= 0) & (cc < sk.shape[1])
        out[rr[ok], cc[ok]] = True
        joins.append((tuple(ends_xy[i]), tuple(ends_xy[j]), L))
    return out, joins


def branches(sk):
    """Ordered cell sequences of the skeleton between junctions and ends."""
    k = np.ones((3, 3)); k[1, 1] = 0
    nb = ndi.convolve(sk.astype(np.int16), k, mode="constant")
    core = sk & (nb <= 2)                          # remove junction cells
    lab, n = ndi.label(core, _EIGHT)
    out = []
    objs = ndi.find_objects(lab)
    for idx, sl in enumerate(objs, 1):
        if sl is None:
            continue
        cells = np.argwhere(lab[sl] == idx) + [sl[0].start, sl[1].start]
        if len(cells) < 3:
            out.append(cells)
            continue
        S = {tuple(c) for c in cells}
        deg = {c: sum((c[0] + dr, c[1] + dc) in S for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc) for c in S}
        start = min(S, key=lambda c: deg[c])       # an end of the branch (or any cell of a loop)
        order, seen, cur = [start], {start}, start
        while True:
            nxt = [(cur[0] + dr, cur[1] + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                   if (dr or dc) and (cur[0] + dr, cur[1] + dc) in S and (cur[0] + dr, cur[1] + dc) not in seen]
            if not nxt:
                break
            nxt.sort(key=lambda c: abs(c[0] - cur[0]) + abs(c[1] - cur[1]))   # prefer 4-neighbours
            cur = nxt[0]
            seen.add(cur)
            order.append(cur)
        out.append(np.array(order))
    return out


def bed_bumps(sk, z, T, p: BumpParams | None = None, dsm=None):
    """Bumps in the bed profile along each branch: candidate crossings (culverts) of the drain."""
    p = p or BumpParams()
    cs = abs(T.a)
    rad = max(1, int(round(p.bed_radius_m / cs)))
    bed = ndi.minimum_filter(np.where(np.isfinite(z), z, np.inf), size=2 * rad + 1)
    ref_n = max(2, int(round(p.reference_m / cs)))
    out = []
    for br in branches(sk):
        if len(br) * cs < p.min_branch_m:
            continue
        b = bed[br[:, 0], br[:, 1]]
        d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(br, axis=0).T))]) * cs
        n = len(b)
        h = np.full(n, np.nan)
        for i in range(n):
            lo, hi = b[max(0, i - ref_n):i], b[i + 1:i + 1 + ref_n]
            if len(lo) == 0 or len(hi) == 0:
                continue
            h[i] = b[i] - max(lo.min(), hi.min())
        on = np.nan_to_num(h, nan=-1) >= p.min_height_m
        lab, k = ndi.label(on)
        for m in range(1, k + 1):
            idx = np.nonzero(lab == m)[0]
            L = d[idx[-1]] - d[idx[0]] + cs
            if not (p.min_length_m <= L <= p.max_length_m):
                continue
            top = idx[np.argmax(h[idx])]
            r, c = br[top]
            x, y = (float(v) for v in _to_xy(T, c + 0.5, r + 0.5))
            rec = dict(x=x, y=y, h_b=float(h[top]), length=float(L), branch_len=float(d[-1]))
            if dsm is not None:
                seg = br[idx]
                rec["veg_frac"] = float(np.mean(dsm[seg[:, 0], seg[:, 1]] - z[seg[:, 0], seg[:, 1]] > 1.0))
            out.append(rec)
    return out


# --------------------------------------------------------------------------- network assembly
@dataclass
class NetworkParams:
    spur_m: float = 10.0
    min_component_m: float = 15.0         # applied *after* joining, so short drain pieces between
                                          # driveways survive if they join the network
    gap_bed_radius_m: float = 1.0         # bed along a gap = lowest ground within this distance
    culvert_rise_m: float = 0.3           # gap bed rise for a culvert gap (h_b)
    continuity_rise_m: float = 0.15       # below this the gap is a continuity repair


def gap_bed_rise(z, T, a, b, radius_m=1.0):
    """Rise of the bed along the straight gap a -> b above the higher end bed (m)."""
    rad = max(1, int(round(radius_m / abs(T.a))))
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1])) + 1)
    s = np.linspace(0.0, 1.0, n)
    cc, rr = _to_cr(T, a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]))
    rr, cc = rr.astype(int), cc.astype(int)
    bed = np.array([np.nanmin(z[max(r - rad, 0):r + rad + 1, max(c - rad, 0):c + rad + 1]) for r, c in zip(rr, cc)])
    ends = max(bed[: max(1, n // 6)].min(), bed[-max(1, n // 6):].min())
    return float(bed.max() - ends), (rr, cc)


def build_drain_network(strong_mask, feats, z, T, dsm=None, dp: DrainParams | None = None,
                        npar: NetworkParams | None = None):
    """Hysteresis mapping -> skeleton -> spur pruning -> join facing ends -> length filter ->
    typed gaps (by bed rise) and bed-profile bumps. Returns a dict."""
    from . import channels
    from .config import ChannelParams
    dp = dp or DrainParams()
    npar = npar or NetworkParams()
    cs = abs(T.a)
    mask = hysteresis_mask(strong_mask, feats, dp)
    sk = channels._prune(channels.skeletonize(mask), max(1, int(round(npar.spur_m / cs))))
    E = channels.channel_ends(sk, cs, ChannelParams())
    xy = np.column_stack([T.c + (E.rc[:, 1] + 0.5) * cs, T.f - (E.rc[:, 0] + 0.5) * cs])
    pairs = facing_pairs(xy, E.approach, dp)
    sk_j, joins = join_all(sk, pairs, xy, T)
    lab, n = ndi.label(sk_j, _EIGHT)
    ln = np.bincount(lab.ravel(), minlength=n + 1) * cs
    keep = ln >= npar.min_component_m
    keep[0] = False
    sk_f = keep[lab]
    gaps = []
    for a, b, L in joins:
        rise, (rr, cc) = gap_bed_rise(z, T, a, b, npar.gap_bed_radius_m)
        if not sk_f[rr[len(rr) // 2], cc[len(cc) // 2]]:
            continue                                   # the joined component was filtered out
        kind = ("culvert" if rise >= npar.culvert_rise_m else
                "continuity" if rise < npar.continuity_rise_m else "uncertain")
        veg = float(np.mean(dsm[rr, cc] - z[rr, cc] > 1.0)) if dsm is not None else float("nan")
        gaps.append(dict(x=(a[0] + b[0]) / 2, y=(a[1] + b[1]) / 2, gap_m=L, rise=rise, veg_frac=veg,
                         kind=kind, x0=a[0], y0=a[1], x1=b[0], y1=b[1]))
    bumps = bed_bumps(sk_f, z, T, dsm=dsm)
    return dict(mask=mask, skeleton=sk_f, joins=joins, gaps=gaps, bumps=bumps, n_ends_raw=len(xy))
