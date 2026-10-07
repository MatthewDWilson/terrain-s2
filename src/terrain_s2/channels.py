"""Channel network (design §4.1): unsupervised channel map, centrelines, channel ends, gap bridging.

1. **Channel map (P_c,u).** A cell is channel if it is part of a narrow, linear, incised
   feature at either of two scales:
     narrow: Hessian valley measure (σ = 2 m) and black top-hat at 11 m (ditches, streams)
     wide:   Hessian valley measure (σ = 4 m) and black top-hat at 21 m (larger channels)
   Fragments smaller than ``min_area_m2`` or with less than ``min_length_m`` of centreline are
   dropped (hillslope roughness). The supervised map (§4.1 step 2) replaces or joins this later.
2. **Vectorise.** Skeletonise; prune spurs shorter than ``spur_m``.
3. **Channel ends.** Skeleton end points, with the outward direction measured over
   ``end_direction_m``. Every end is used, whatever the DEM flow direction: at SH12 both
   channels *start* at the embankment (the culvert sits on a DEM drainage divide), so the design's
   "downstream-dangling ends" alone would miss it.
4. **Gap bridging.** From each end that points at a raised feature, a least-cost search over
   cost = 1 − P_c (+ a small floor), within ``bridge_max_m``, to network cells that are not the
   end's own nearby channel (within ``geodesic_exclusion_m`` along the skeleton). The accepted
   target must lie within ``max_turn_deg`` of the end's outward direction.

The Test A rules on each bridging path are applied in ``crossings.test_a``.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy import ndimage as ndi

from .config import ChannelParams

_DR = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
_DC = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
_EIGHT = np.ones((3, 3), bool)


# --------------------------------------------------------------------------- channel map
def channel_mask(feats: dict, cellsize: float, p: ChannelParams, z: np.ndarray | None = None,
                 upa: np.ndarray | None = None, filter_fragments: bool = True) -> np.ndarray:
    narrow = (feats["valley_s2"] >= p.narrow_valley) & (feats["tophat_black11"] >= p.narrow_incision)
    wide = ((feats["valley_s4"] >= p.wide_valley) & (feats["tophat_black21"] >= p.wide_incision)
            & (feats["tophat_black11"] >= p.wide_min_local_incision))
    m = np.nan_to_num(narrow | wide).astype(bool)
    if z is not None and upa is not None and p.slope_gate > 0:
        # On slopes, a channel must also concentrate flow (GeoNet-style), which removes hillside
        # roughness; on flat floodplains D8 accumulation is unreliable, so it is not required there.
        zs = ndi.gaussian_filter(np.nan_to_num(z, nan=float(np.nanmean(z))), 2.0 / cellsize)
        gr, gc = np.gradient(zs, cellsize)
        steep = np.hypot(gr, gc) > p.slope_gate
        m &= ~steep | (np.nan_to_num(upa) >= p.slope_min_upstream_m2)
    if not filter_fragments:                       # the drain network joins first, filters later
        return m
    lab, n = ndi.label(m, _EIGHT)
    if n == 0:
        return m
    area = np.bincount(lab.ravel(), minlength=n + 1) * cellsize**2
    keep = area >= p.min_area_m2
    keep[0] = False
    m = keep[lab]
    # centreline length per component
    sk = skeletonize(m)
    lab, n = ndi.label(m, _EIGHT)
    sk_len = np.bincount(lab[sk], minlength=n + 1) * cellsize
    keep = sk_len >= p.min_length_m
    keep[0] = False
    return keep[lab]


def skeletonize(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize as _sk
    return _sk(mask)


# --------------------------------------------------------------------------- skeleton graph
@njit(cache=True)
def _nbcount(sk):
    nr, nc = sk.shape
    out = np.zeros((nr, nc), np.int8)
    for r in range(nr):
        for c in range(nc):
            if not sk[r, c]:
                continue
            k = 0
            for d in range(8):
                rr, cc = r + _DR[d], c + _DC[d]
                if 0 <= rr < nr and 0 <= cc < nc and sk[rr, cc]:
                    k += 1
            out[r, c] = k
    return out


@njit(cache=True)
def _walk(sk, nb, r0, c0, max_cells, buf):
    """Walk from an end point along degree-2 cells. Returns (n, stop_kind): 1 end, 2 junction, 0 limit."""
    nr, nc = sk.shape
    pr, pc = -1, -1
    r, c = r0, c0
    n = 0
    while n < max_cells:
        buf[n, 0] = r
        buf[n, 1] = c
        n += 1
        if n > 1 and nb[r, c] != 2:
            return n, (2 if nb[r, c] >= 3 else 1)
        nxt_r, nxt_c = -1, -1
        for d in range(8):
            rr, cc = r + _DR[d], c + _DC[d]
            if 0 <= rr < nr and 0 <= cc < nc and sk[rr, cc] and not (rr == pr and cc == pc):
                seen = False
                for k in range(n):
                    if buf[k, 0] == rr and buf[k, 1] == cc:
                        seen = True
                        break
                if not seen:
                    nxt_r, nxt_c = rr, cc
                    break
        if nxt_r < 0:
            return n, 1
        pr, pc = r, c
        r, c = nxt_r, nxt_c
    return n, 0


@njit(cache=True)
def _prune(sk, max_cells):
    """Remove spurs (end point -> junction) shorter than max_cells, twice."""
    nr, nc = sk.shape
    buf = np.empty((max_cells + 2, 2), np.int64)
    for _ in range(2):
        nb = _nbcount(sk)
        for r in range(nr):
            for c in range(nc):
                if sk[r, c] and nb[r, c] == 1:
                    n, kind = _walk(sk, nb, r, c, max_cells + 1, buf)
                    if kind == 2 and n - 1 < max_cells:
                        for k in range(n - 1):          # keep the junction cell
                            sk[buf[k, 0], buf[k, 1]] = False
    return sk


@dataclass
class ChannelEnds:
    rc: np.ndarray         # (n, 2) end cells
    direction: np.ndarray  # (n, 2) unit outward direction in map coordinates (dx, dy), over end_direction_m
    approach: np.ndarray   # (n, 2) unit outward direction over up to approach_m (the approaching channel)
    approach_len: np.ndarray  # (n,) length of unbranched channel behind the end, capped at approach_m


def channel_ends(sk: np.ndarray, cellsize: float, p: ChannelParams) -> ChannelEnds:
    nb = _nbcount(sk)
    ends = np.argwhere(sk & (nb == 1))
    back = max(2, int(round(p.end_direction_m / cellsize)))
    far = max(back, int(round(p.approach_m / cellsize)))
    buf = np.empty((far + 2, 2), np.int64)
    dirs = np.zeros((len(ends), 2))
    appr = np.zeros((len(ends), 2))
    alen = np.zeros(len(ends))

    def unit(r, c, rb, cb):
        dx, dy = (c - cb), -(r - rb)                     # map x east, y north
        norm = math.hypot(dx, dy)
        return (dx / norm, dy / norm) if norm > 0 else (0.0, 0.0)

    for i, (r, c) in enumerate(ends):
        n, _ = _walk(sk, nb, r, c, far + 1, buf)
        rb, cb = buf[min(n, back + 1) - 1]
        dirs[i] = unit(r, c, rb, cb)
        rf, cf = buf[n - 1]
        appr[i] = unit(r, c, rf, cf)
        alen[i] = math.hypot(rf - r, cf - c) * cellsize
    return ChannelEnds(ends, dirs, appr, alen)


def skeleton_orientation(sk: np.ndarray, radius: int) -> np.ndarray:
    """Local axis angle (rad, map coordinates) of the skeleton from second moments within radius."""
    v, u = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(float)
    y = -v                                             # map y north
    disk = (u**2 + y**2 <= radius**2).astype(float)
    s = sk.astype(float)
    conv = lambda k: ndi.convolve(s, k[::-1, ::-1] * disk, mode="constant")  # noqa: E731
    n = np.maximum(conv(np.ones_like(u)), 1.0)
    mx, my = conv(u) / n, conv(y) / n
    cxx = conv(u * u) / n - mx**2
    cyy = conv(y * y) / n - my**2
    cxy = conv(u * y) / n - mx * my
    ang = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    return np.where(sk, ang, np.nan)


def toe_parallel(sk: np.ndarray, white: np.ndarray, cellsize: float, p: ChannelParams) -> np.ndarray:
    """Skeleton cells that run along a raised feature (toe ditches, drains beside banks)."""
    raised = np.nan_to_num(white) >= p.raised_min_m
    if not raised.any():
        return np.zeros_like(sk)
    near = ndi.distance_transform_edt(~raised) * cellsize <= p.toe_distance_m
    wsm = ndi.gaussian_filter(np.nan_to_num(white), 2.0 / cellsize)
    gr, gc = np.gradient(wsm, cellsize)
    gang = np.arctan2(-gr, gc)                        # direction across the raised feature (map coords)
    gmag = np.hypot(gr, gc)
    axis = skeleton_orientation(sk, max(2, int(round(p.orientation_radius_m / cellsize))))
    diff = np.degrees(np.abs(np.remainder(axis - gang + np.pi / 2, np.pi) - np.pi / 2))
    return sk & near & (gmag >= p.toe_min_gradient) & (np.nan_to_num(diff) >= 90.0 - p.toe_parallel_deg)


def merge_ends(a: ChannelEnds, b: ChannelEnds, radius_cells: int) -> ChannelEnds:
    """All ends of ``a`` and ``b``. Coinciding ends are both kept: the toe-cut one may be a stub
    while the full-centreline one has the real approach; duplicate paths merge by crest later.
    Exact duplicates (same cell, same approach) are dropped."""
    if len(b.rc) == 0:
        return a
    keep = np.ones(len(b.rc), bool)
    if len(a.rc):
        same = (np.abs(b.rc[:, None, :] - a.rc[None, :, :]).max(axis=2) == 0)
        same &= np.abs(b.approach_len[:, None] - a.approach_len[None, :]) < 1e-6
        keep = ~same.any(axis=1)
    return ChannelEnds(np.vstack([a.rc, b.rc[keep]]), np.vstack([a.direction, b.direction[keep]]),
                       np.vstack([a.approach, b.approach[keep]]),
                       np.concatenate([a.approach_len, b.approach_len[keep]]))


# --------------------------------------------------------------------------- gap bridging
@njit(cache=True)
def _mark_near(sk, r0, c0, max_len, stamp, tag, cellsize, gdist):
    """Mark skeleton cells within max_len (geodesic, along the skeleton) of (r0, c0), recording
    their geodesic distance in gdist."""
    nr, nc = sk.shape
    diag = math.sqrt(2.0) * cellsize
    dist = {}
    heap = [(0.0, r0 * nc + c0)]
    dist[r0 * nc + c0] = 0.0
    while len(heap) > 0:
        d, i = heapq.heappop(heap)
        if d > dist.get(i, 1e30):
            continue
        stamp[i] = tag
        gdist[i] = d
        r, c = i // nc, i % nc
        for k in range(8):
            rr, cc = r + _DR[k], c + _DC[k]
            if 0 <= rr < nr and 0 <= cc < nc and sk[rr, cc]:
                nd = d + (diag if k % 2 == 1 else cellsize)
                j = rr * nc + cc
                if nd <= max_len and nd < dist.get(j, 1e30):
                    dist[j] = nd
                    heapq.heappush(heap, (nd, j))


@njit(cache=True)
def _bridge(pc, valid, nearest_sk, skcomp, stamp, gdist, geo_ratio, tag, r0, c0, dx, dy, cellsize,
            max_len, max_turn_cos, floor, z, white, raised_min, h_min, max_targets):
    """Least-cost paths from (r0, c0) to channel cells behind a barrier.

    Cost per metre = max(1 - pc, floor). Along each search path the highest raised-feature value
    and ground level are carried; a target is accepted if it is not the end's own nearby channel,
    lies within the turn limit of the end's direction, and the path crossed a raised feature
    (>= raised_min) and rose >= h_min above both ends. The search continues past accepted targets
    and keeps up to ``max_targets``, at most one per skeleton component, so a cheaper wrong pairing
    (e.g. a hop across a road to the opposite roadside drain) cannot pre-empt the right one; Test A
    then judges each. Returns (flat path cells, offsets) with paths ordered start -> target.
    """
    nr, nc = pc.shape
    diag = math.sqrt(2.0)
    start = r0 * nc + c0
    cost = {}
    dist = {}
    prev = {}
    cost[start] = 0.0
    dist[start] = 0.0
    prev[start] = -1
    maxw = {}
    maxz = {}
    maxw[start] = white[r0, c0]
    maxz[start] = z[r0, c0]
    z0 = z[r0, c0]
    heap = [(0.0, start)]
    found = np.full(max_targets, -1, np.int64)
    found_comp = np.full(max_targets, -1, np.int64)
    nfound = 0
    while len(heap) > 0 and nfound < max_targets:
        cst, i = heapq.heappop(heap)
        if cst > cost.get(i, 1e30):
            continue
        r, c = i // nc, i % nc
        s = nearest_sk[i]
        # own channel = reachable along the network within geo_ratio x the straight-line distance
        sr, sc_ = s // nc, s % nc
        own = s >= 0 and stamp[s] == tag and gdist[s] <= geo_ratio * math.hypot(sr - r0, sc_ - c0) * cellsize
        if i != start and s >= 0 and not own:
            vx, vy = (c - c0), -(r - r0)
            norm = math.hypot(vx, vy)
            barrier = maxz[i] - max(z0, z[r, c])
            comp = skcomp[s]
            if (norm > 0 and (vx * dx + vy * dy) / norm >= max_turn_cos
                    and maxw[i] >= raised_min and barrier >= h_min):
                dup = False
                for k in range(nfound):
                    if found_comp[k] == comp:
                        dup = True
                        break
                if not dup:
                    found[nfound] = i
                    found_comp[nfound] = comp
                    nfound += 1
                    continue                       # do not expand beyond an accepted target
        for k in range(8):
            rr, cc = r + _DR[k], c + _DC[k]
            if rr < 0 or cc < 0 or rr >= nr or cc >= nc or not valid[rr, cc]:
                continue
            step = (diag if k % 2 == 1 else 1.0) * cellsize
            nd = dist[i] + step
            if nd > max_len:
                continue
            w = 1.0 - pc[rr, cc]
            if w < floor:
                w = floor
            ncst = cst + w * step
            j = rr * nc + cc
            if ncst < cost.get(j, 1e30):
                cost[j] = ncst
                dist[j] = nd
                prev[j] = i
                maxw[j] = max(maxw[i], white[rr, cc])
                maxz[j] = max(maxz[i], z[rr, cc])
                heapq.heappush(heap, (ncst, j))
    total = 0
    for k in range(nfound):
        i = found[k]
        while i >= 0:
            total += 1
            i = prev[i]
    cells = np.empty(total, np.int64)
    offsets = np.zeros(nfound + 1, np.int64)
    pos = 0
    for k in range(nfound):
        n = 0
        i = found[k]
        while i >= 0:
            n += 1
            i = prev[i]
        i = found[k]
        for m in range(pos + n - 1, pos - 1, -1):
            cells[m] = i
            i = prev[i]
        pos += n
        offsets[k + 1] = pos
    return cells, offsets


@dataclass
class Network:
    mask: np.ndarray
    skeleton: np.ndarray
    ends: ChannelEnds
    paths: list            # bridging paths (flat index arrays), one per end that found a target
    path_end: list         # index into ends for each path
    toe: np.ndarray = None # skeleton cells running along raised features


def build_network(z: np.ndarray, feats: dict, cellsize: float, p: ChannelParams | None = None,
                  upa: np.ndarray | None = None) -> Network:
    p = p or ChannelParams()
    mask = channel_mask(feats, cellsize, p, z, upa)
    sk = _prune(skeletonize(mask), max(1, int(round(p.spur_m / cellsize))))
    white = np.nan_to_num(feats[f"tophat_white{p.raised_scale_m:g}"])
    # Channels running along a raised feature are cut out when finding ends (so a stream that
    # joins a toe ditch still ends at the embankment) but remain bridging targets.
    toe = toe_parallel(sk, white, cellsize, p)
    # Ends from the toe-cut centreline (a stream that joins a toe ditch ends at the embankment) and
    # from the full centreline (a roadside drain interrupted by a driveway ends at the driveway).
    ends = merge_ends(channel_ends(sk & ~toe, cellsize, p), channel_ends(sk, cellsize, p),
                      max(1, int(round(3.0 / cellsize))))
    valid = np.isfinite(z)
    nr, nc = z.shape
    # nearest skeleton cell for every channel cell (so targets are recognised anywhere in the channel)
    if sk.any():
        _, idx = ndi.distance_transform_edt(~sk, return_indices=True)
        nearest = (idx[0] * nc + idx[1]).ravel()
        nearest = np.where(mask.ravel(), nearest, -1).astype(np.int64)
    else:
        nearest = np.full(nr * nc, -1, np.int64)
    pc = mask.astype(np.float64)
    zz = np.where(valid, z, -1e9).astype(np.float64)
    skcomp = ndi.label(sk, _EIGHT)[0].ravel().astype(np.int64)     # skeleton component per cell
    stamp = np.full(nr * nc, -1, np.int64)
    gdist = np.zeros(nr * nc, np.float64)
    look = int(round(p.lookahead_m / cellsize))
    paths, which = [], []
    cos_turn = math.cos(math.radians(p.max_turn_deg))
    for k, ((r, c), (dx, dy)) in enumerate(zip(ends.rc, ends.direction)):
        if dx == 0 and dy == 0:
            continue
        # only ends that point at a raised feature
        t = np.arange(1, look + 1)
        rr = np.clip(np.round(r - t * dy).astype(int), 0, nr - 1)
        cc = np.clip(np.round(c + t * dx).astype(int), 0, nc - 1)
        if white[rr, cc].max() < p.raised_min_m:
            continue
        _mark_near(sk, int(r), int(c), p.geodesic_exclusion_m, stamp, k, cellsize, gdist)
        cells, offs = _bridge(pc, valid, nearest, skcomp, stamp, gdist, float(p.geodesic_ratio), k,
                              int(r), int(c), float(dx), float(dy),
                              cellsize, p.bridge_max_m, cos_turn, p.cost_floor, zz, white,
                              p.raised_min_m, p.min_barrier_m, p.max_targets)
        for m in range(len(offs) - 1):
            paths.append(cells[offs[m]:offs[m + 1]])
            which.append(k)
    return Network(mask, sk, ends, paths, which, toe)
