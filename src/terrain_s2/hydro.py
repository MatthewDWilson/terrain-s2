"""Hydrological features (design §3 'Hydrological'), CPU only.

* ``breach_least_cost`` — least-cost depression breaching (after Lindsay 2016)
  that also returns every breach **path**. Paths that cut through a raised
  feature are the raw material for crossing candidates and, later, culvert
  cutlines in the structures layer (§6.2). Whitebox's tool returns only the
  breached raster, which is why this is implemented here; Whitebox Workflows
  NG is used as an optional cross-check in the tests.
* ``depressions`` — priority-flood fill (pyflwdir), depth, labelled depressions.
* ``flow_accumulation`` — D8 upstream area on the breached-then-filled DEM.

Boundary convention (matches pyflwdir ``outlets='edge'``): valid cells on the
array edge or 8-adjacent to nodata are outlets. Nodata is where the DEM is NaN
(e.g. sea), so coastal drainage is handled without a land polygon.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
import pyflwdir
from numba import njit, types
from numba.typed import List
from scipy import ndimage as ndi

from .config import HydroParams

NODATA = -9999.0
_DR = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
_DC = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)


@njit(cache=True)
def boundary_mask(valid):
    nr, nc = valid.shape
    b = np.zeros_like(valid)
    for r in range(nr):
        for c in range(nc):
            if not valid[r, c]:
                continue
            if r == 0 or c == 0 or r == nr - 1 or c == nc - 1:
                b[r, c] = True
                continue
            for k in range(8):
                if not valid[r + _DR[k], c + _DC[k]]:
                    b[r, c] = True
                    break
    return b


@njit(cache=True)
def _gather_flat(z, valid, boundary, start, tag, flat_tag, buf):
    """Collect the equal-elevation flat containing ``start``.

    Returns (n_cells, drains) where drains means some member is a boundary cell
    or has a strictly lower valid neighbour. Members are written to ``buf``.
    """
    nr, nc = z.shape
    zs = z.flat[start]
    buf[0] = start
    flat_tag[start] = tag
    n, head = 1, 0
    drains = False
    while head < n:
        i = buf[head]
        head += 1
        if boundary.flat[i]:
            drains = True
        r, c = i // nc, i % nc
        for k in range(8):
            rr, cc = r + _DR[k], c + _DC[k]
            if rr < 0 or cc < 0 or rr >= nr or cc >= nc or not valid[rr, cc]:
                continue
            zj = z[rr, cc]
            j = rr * nc + cc
            if zj < zs:
                drains = True
            elif zj == zs and flat_tag[j] != tag:
                flat_tag[j] = tag
                buf[n] = j
                n += 1
    return n, drains


@njit(cache=True)
def _find_pits(z, valid, boundary):
    """One representative cell per pit-flat (a flat with no outlet), any order."""
    nr, nc = z.shape
    n = nr * nc
    flat_tag = np.full(n, -1, np.int64)
    buf = np.empty(n, np.int64)
    reps = List.empty_list(types.int64)
    tag = 0
    for r in range(nr):
        for c in range(nc):
            i = r * nc + c
            if not valid[r, c] or boundary[r, c] or flat_tag[i] >= 0:
                continue
            lower = False
            for k in range(8):
                if valid[r + _DR[k], c + _DC[k]] and z[r + _DR[k], c + _DC[k]] < z[r, c]:
                    lower = True
                    break
            if lower:
                continue
            m, drains = _gather_flat(z, valid, boundary, i, tag, flat_tag, buf)
            tag += 1
            if not drains:
                reps.append(i)
    out = np.empty(len(reps), np.int64)
    for k in range(len(reps)):
        out[k] = reps[k]
    return out


@njit(cache=True)
def _breach(z0, valid, boundary, pits, cellsize, max_len, max_cost, flat_cost, store_min_cut):
    nr, nc = z0.shape
    n = nr * nc
    zb = z0.copy()
    stamp = np.full(n, -1, np.int64)
    cost = np.empty(n, np.float64)
    dist = np.empty(n, np.float64)
    prev = np.empty(n, np.int64)
    flat_tag = np.full(n, -1, np.int64)
    buf = np.empty(n, np.int64)
    pbuf = np.empty(n, np.int64)
    diag = math.sqrt(2.0)

    rec_seed = List.empty_list(types.int64)
    rec_outlet = List.empty_list(types.int64)
    rec_len = List.empty_list(types.float64)
    rec_cut = List.empty_list(types.float64)
    rec_cost = List.empty_list(types.float64)
    rec_ptr = List.empty_list(types.int64)
    path_cells = List.empty_list(types.int64)
    path_znew = List.empty_list(types.float64)
    rec_ptr.append(0)
    n_unresolved = 0

    for k in range(pits.shape[0]):
        p = pits[k]
        zp = zb.flat[p]
        nf, drains = _gather_flat(zb, valid, boundary, p, k, flat_tag, buf)
        if drains:
            continue  # resolved by an earlier breach
        heap = [(0.0, p)]
        heapq.heappop(heap)
        for f in range(nf):
            q = buf[f]
            stamp[q] = k
            cost[q] = 0.0
            dist[q] = 0.0
            prev[q] = -1
            heapq.heappush(heap, (0.0, q))
        found = -1
        while len(heap) > 0:
            c_i, i = heapq.heappop(heap)
            if c_i > cost[i]:
                continue
            if prev[i] >= 0 and (zb.flat[i] < zp or boundary.flat[i]):
                found = i
                break
            r, c = i // nc, i % nc
            for d in range(8):
                rr, cc = r + _DR[d], c + _DC[d]
                if rr < 0 or cc < 0 or rr >= nr or cc >= nc or not valid[rr, cc]:
                    continue
                j = rr * nc + cc
                step = cellsize * (diag if (d % 2) == 1 else 1.0)
                nd = dist[i] + step
                if nd > max_len:
                    continue
                rise = zb[rr, cc] - zp
                if rise < 0.0:
                    rise = 0.0
                ncost = c_i + (rise + flat_cost) * step
                if max_cost >= 0.0 and ncost > max_cost:
                    continue
                if stamp[j] != k or ncost < cost[j]:
                    stamp[j] = k
                    cost[j] = ncost
                    dist[j] = nd
                    prev[j] = i
                    heapq.heappush(heap, (ncost, j))
        if found < 0:
            n_unresolved += 1
            continue
        # trace back: pbuf[0] = outlet ... pbuf[m-1] = seed
        m = 0
        i = found
        while i >= 0:
            pbuf[m] = i
            m += 1
            i = prev[i]
        total = dist[found]
        z_out = zb.flat[found]
        # Lower path cells only to just below the pit (Lindsay 2016), with a
        # decrement small enough that the outlet stays below the last path cell.
        delta = 1e-4
        if z_out < zp and (zp - z_out) / m < delta:
            delta = (zp - z_out) / m
        max_cut = 0.0
        store_from = len(path_cells)
        for t in range(m - 1, -1, -1):  # seed (t = m-1) -> outlet (t = 0)
            q = pbuf[t]
            target = zp - delta * (m - 1 - t)
            if zb.flat[q] > target:
                zb.flat[q] = target
            cut = z0.flat[q] - zb.flat[q]
            if cut > max_cut:
                max_cut = cut
            path_cells.append(q)
            path_znew.append(zb.flat[q])
        if max_cut >= store_min_cut:
            rec_seed.append(pbuf[m - 1])
            rec_outlet.append(found)
            rec_len.append(total)
            rec_cut.append(max_cut)
            rec_cost.append(cost[found])
            rec_ptr.append(len(path_cells))
        else:
            while len(path_cells) > store_from:  # discard geometry of trivial breaches
                path_cells.pop()
                path_znew.pop()
    return (zb, rec_seed, rec_outlet, rec_len, rec_cut, rec_cost, rec_ptr,
            path_cells, path_znew, n_unresolved)


@dataclass
class BreachResult:
    z_breached: np.ndarray      # float64, NaN outside valid; unresolved pits remain
    seed: np.ndarray            # flat index of the pit cell where each stored path starts
    outlet: np.ndarray          # flat index of the path's outlet cell
    length_m: np.ndarray
    max_cut_m: np.ndarray
    cost: np.ndarray            # m² (cut cross-section integrated along path)
    ptr: np.ndarray             # CSR pointers into cells/znew, len = n_paths + 1
    cells: np.ndarray           # flat indices, each path ordered pit -> outlet
    znew: np.ndarray            # breached elevation along each path
    n_pits: int
    n_unresolved: int

    def path(self, k: int):
        s, e = self.ptr[k], self.ptr[k + 1]
        return self.cells[s:e], self.znew[s:e]


def breach_least_cost(z: np.ndarray, cellsize: float, p: HydroParams | None = None) -> BreachResult:
    p = p or HydroParams()
    valid = np.isfinite(z)
    z0 = np.where(valid, z, NODATA).astype(np.float64)
    bnd = boundary_mask(valid)
    pits = _find_pits(z0, valid, bnd)
    pits = pits[np.argsort(z0.ravel()[pits], kind="stable")]
    (zb, seed, outlet, length, cut, cost, ptr, cells, znew, nun) = _breach(
        z0, valid, bnd, pits, float(cellsize), float(p.breach_max_length_m),
        float(p.breach_max_cost), float(p.breach_flat_step_cost), float(p.breach_store_min_cut_m))
    # Paths were appended for every breach and truncated for trivial ones, so
    # ``ptr`` indexes the stored paths contiguously.
    zb = np.where(valid, zb, np.nan)
    return BreachResult(zb, np.asarray(seed, np.int64), np.asarray(outlet, np.int64),
                        np.asarray(length), np.asarray(cut), np.asarray(cost),
                        np.asarray(ptr, np.int64), np.asarray(cells, np.int64),
                        np.asarray(znew), int(pits.size), int(nun))


def fill(z: np.ndarray):
    """Priority-flood fill with edge/nodata outlets. Returns (filled, d8)."""
    valid = np.isfinite(z)
    zf, d8 = pyflwdir.dem.fill_depressions(np.where(valid, z, NODATA).astype(np.float64),
                                           outlets="edge", nodata=NODATA)
    return np.where(valid, zf, np.nan), d8


@dataclass
class Depressions:
    depth: np.ndarray        # fill - DEM (m)
    labels: np.ndarray       # int32, 0 = none
    max_depth: np.ndarray    # per label (index = label), m
    area_m2: np.ndarray
    volume_m3: np.ndarray


def depressions(z: np.ndarray, cellsize: float, min_depth: float = 0.02) -> Depressions:
    zf, _ = fill(z)
    depth = np.nan_to_num(zf - z, nan=0.0)
    lab, n = ndi.label(depth > min_depth, structure=np.ones((3, 3), bool))
    idx = lab.ravel()
    a = cellsize * cellsize
    cnt = np.bincount(idx, minlength=n + 1).astype(float)
    vol = np.bincount(idx, weights=depth.ravel(), minlength=n + 1) * a
    mx = np.zeros(n + 1)
    np.maximum.at(mx, idx, depth.ravel())
    cnt[0] = vol[0] = mx[0] = 0
    return Depressions(depth.astype(np.float32), lab.astype(np.int32), mx, cnt * a, vol)


@njit(cache=True)
def pond_stats(z, valid, seed, level, cellsize, max_cells):
    """Pond retained behind one barrier: cells connected to ``seed`` with z < level.

    Returns (area_m2, volume_m3, depth_m, truncated). ``level`` should be
    min(barrier crest, fill level at the seed) so the pond cannot leak past a
    lower rim, and nested bumps inside a larger pond get their own small pond.
    """
    nr, nc = z.shape
    if z.flat[seed] >= level:
        return 0.0, 0.0, 0.0, False
    seen = {}
    q = np.empty(max_cells, np.int64)
    q[0] = seed
    seen[seed] = True
    n, head = 1, 0
    vol = 0.0
    zmin = z.flat[seed]
    truncated = False
    while head < n:
        i = q[head]
        head += 1
        vol += level - z.flat[i]
        if z.flat[i] < zmin:
            zmin = z.flat[i]
        r, c = i // nc, i % nc
        for k in range(8):
            rr, cc = r + _DR[k], c + _DC[k]
            if rr < 0 or cc < 0 or rr >= nr or cc >= nc or not valid[rr, cc]:
                continue
            j = rr * nc + cc
            if z[rr, cc] < level and j not in seen:
                if n >= max_cells:
                    truncated = True
                    continue
                seen[j] = True
                q[n] = j
                n += 1
    a = cellsize * cellsize
    return n * a, vol * a, level - zmin, truncated


def flow_accumulation(z_conditioned: np.ndarray, transform) -> tuple[np.ndarray, pyflwdir.FlwdirRaster]:
    """D8 upstream area (m²) on a conditioned DEM (fills any remaining pits)."""
    zf, d8 = fill(z_conditioned)
    flw = pyflwdir.from_array(d8, ftype="d8", transform=transform, latlon=False)
    cell_area = abs(transform.a * transform.e)
    upa = flw.upstream_area(unit="cell").astype(np.float64) * cell_area
    upa[~np.isfinite(z_conditioned)] = np.nan
    return upa, flw
