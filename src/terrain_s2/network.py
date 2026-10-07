"""Channel network as a graph of reaches (DESIGN_NOTES §4j).

Nodes are junctions (connected junction cells of the skeleton) and channel ends; edges are skeleton
branches. *Reaches* chain edges through junctions by continuing the straightest path (within
``max_turn_deg``): stream-or-drain is a property of a reach, not of a 5 m piece between junctions.
Each reach is smoothed (moving average, end points fixed at the junction centres so reaches stay
connected) and generalised (Douglas-Peucker). Flow direction comes from the bed trend along the
whole reach, which is more reliable on flat land than D8 accumulation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from . import drains

_EIGHT = np.ones((3, 3), bool)


@dataclass
class Reach:
    edges: list                    # edge indices, in order
    cells: np.ndarray              # (n, 2) skeleton cells, in order
    coords: np.ndarray             # (m, 2) smoothed, generalised map coordinates
    node_start: int
    node_end: int
    attrs: dict = field(default_factory=dict)


def build_graph(sk, T):
    """Junction nodes, end nodes and branch edges. Returns (nodes_xy, edges) where each edge is
    (cells ordered start->end, node_start, node_end)."""
    k = np.ones((3, 3)); k[1, 1] = 0
    nb = ndi.convolve(sk.astype(np.int16), k, mode="constant")
    junction = sk & (nb >= 3)
    jlab, nj = ndi.label(junction, _EIGHT)
    nodes = []
    for i, sl in enumerate(ndi.find_objects(jlab), 1):
        rr, cc = np.nonzero(jlab[sl] == i)
        r, c = rr.mean() + sl[0].start, cc.mean() + sl[1].start
        nodes.append(drains._to_xy(T, c + 0.5, r + 0.5))
    edges = []
    for br in drains.branches(sk):
        if len(br) == 0:
            continue
        ends = []
        for cell in (br[0], br[-1]):
            nid = None
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    r, c = cell[0] + dr, cell[1] + dc
                    if (dr or dc) and 0 <= r < sk.shape[0] and 0 <= c < sk.shape[1] and jlab[r, c]:
                        nid = int(jlab[r, c]) - 1
                        break
                if nid is not None:
                    break
            if nid is None:                                  # a channel end: its own node
                nodes.append(drains._to_xy(T, cell[1] + 0.5, cell[0] + 0.5))
                nid = len(nodes) - 1
            ends.append(nid)
        edges.append((br, ends[0], ends[1]))
    return np.array([(float(x), float(y)) for x, y in nodes]), edges


def _leaving_dir(cells, T, at_start, n=6):
    a = cells[0] if at_start else cells[-1]
    b = cells[min(n, len(cells) - 1)] if at_start else cells[max(-n - 1, -len(cells))]
    v = np.array([b[1] - a[1], -(b[0] - a[0])], float)        # map x east, y north
    n_ = np.hypot(*v)
    return v / n_ if n_ > 0 else v


def build_reaches(sk, T, max_turn_deg=45.0, smooth_window=7, simplify_m=0.5):
    nodes, edges = build_graph(sk, T)
    # at each node, pair incident edge-ends that continue straight through
    inc = {}
    for e, (cells, a, b) in enumerate(edges):
        inc.setdefault(a, []).append((e, 0))
        if b != a or len(cells) > 1:
            inc.setdefault(b, []).append((e, 1))
    pair = {}
    cos_lim = math.cos(math.radians(max_turn_deg))
    for nid, ends in inc.items():
        if len(ends) < 2:
            continue
        dirs = {(e, s): _leaving_dir(edges[e][0], T, s == 0) for e, s in ends}
        cand = []
        for i in range(len(ends)):
            for j in range(i + 1, len(ends)):
                if ends[i][0] == ends[j][0]:
                    continue
                c = float(dirs[ends[i]] @ (-dirs[ends[j]]))
                if c >= cos_lim:
                    cand.append((c, ends[i], ends[j]))
        for c, x, y in sorted(cand, reverse=True):
            if x not in pair and y not in pair:
                pair[x], pair[y] = y, x
    reaches, used = [], set()
    starts = [(e, s) for e in range(len(edges)) for s in (0, 1) if (e, s) not in pair]
    for start in starts + [(e, 0) for e in range(len(edges))]:
        e, s = start
        if e in used:
            continue
        seq, cells_all, node_seq = [], [], []
        cur, ent = e, s
        n0 = edges[e][1] if s == 0 else edges[e][2]
        while cur is not None and cur not in used:
            used.add(cur)
            cells, a, b = edges[cur]
            oc = cells if ent == 0 else cells[::-1]
            seq.append(cur)
            cells_all.append(oc)
            exit_end = 1 - ent
            node_seq.append(b if ent == 0 else a)
            nxt = pair.get((cur, exit_end))
            cur, ent = (nxt if nxt else (None, None))
        cells = np.vstack(cells_all)
        n1 = node_seq[-1]
        reaches.append(Reach(seq, cells, _smooth_coords(cells, T, nodes, n0, n1, smooth_window, simplify_m), n0, n1))
    return nodes, edges, reaches


def _smooth_coords(cells, T, nodes, n0, n1, window, tol):
    from shapely.geometry import LineString
    x, y = drains._to_xy(T, cells[:, 1] + 0.5, cells[:, 0] + 0.5)
    P = np.column_stack([x, y]).astype(float)
    P = np.vstack([nodes[n0], P, nodes[n1]])                 # end at the node centres
    if len(P) > window:
        k = np.ones(window) / window
        pad = window // 2
        Q = np.vstack([np.repeat(P[:1], pad, 0), P, np.repeat(P[-1:], pad, 0)])
        S = np.column_stack([np.convolve(Q[:, 0], k, "valid"), np.convolve(Q[:, 1], k, "valid")])
        S[0], S[-1] = P[0], P[-1]                            # fixed end points
        P = S
    g = LineString(P).simplify(tol) if len(P) >= 2 else None
    return np.array(g.coords) if g is not None else P


def reach_features(reaches, mask, feats, z, T, road_dist=None, road_dir=None, riv=None):
    """Reach-scale features. Bendiness discounts sharp corners (field-boundary turns of drains)."""
    import pandas as pd
    from shapely.geometry import LineString
    cs = abs(T.a)
    half_w = ndi.distance_transform_edt(mask) * cs
    bed = ndi.minimum_filter(np.nan_to_num(z, nan=np.inf), size=3)
    rows = []
    for R in reaches:
        g = LineString(R.coords) if len(R.coords) >= 2 else None
        L = g.length if g is not None else 0.0
        r, c = R.cells[:, 0], R.cells[:, 1]
        bend = straight = 0.0
        corners = 0
        if L >= 10:
            # bends measured on a line simplified at 3 m, in 10 m steps: small wiggles left by the
            # raster centreline no longer count (drains 7 vs streams 36 deg/100 m in Matt's labels,
            # against 53 vs 87 when measured on the 0.5 m line in 5 m steps)
            gb = g.simplify(3.0)
            s = np.arange(0, gb.length, 10.0)
            P = np.array([gb.interpolate(v).coords[0] for v in s] + [gb.coords[-1]])
            h = np.arctan2(np.diff(P[:, 1]), np.diff(P[:, 0]))
            t = np.degrees(np.abs(np.angle(np.exp(1j * np.diff(h))))) if len(h) > 1 else np.zeros(1)
            corners = int((t >= 45).sum())
            bend = float(t[t < 45].sum() / L * 100)
            gs = g.simplify(1.5)
            seg = np.hypot(*np.diff(np.array(gs.coords), axis=0).T)
            straight = float(seg[seg >= 25].sum() / L)
        row = dict(length_m=L, bend_deg_per_100m=bend, corners=corners, straight_frac=straight,
                   width_m=float(np.median(2 * half_w[r, c])),
                   incision_m=float(np.nanmedian(np.fmax(feats["tophat_black11"][r, c], feats["tophat_black21"][r, c]))),
                   bed_start=float(bed[r[0], c[0]]), bed_end=float(bed[r[-1], c[-1]]))
        if road_dist is not None:
            main = math.atan2(R.coords[-1][1] - R.coords[0][1], R.coords[-1][0] - R.coords[0][0])
            near = road_dist[r, c] <= 20
            par = np.abs(np.cos(road_dir[r, c] - main)) >= math.cos(math.radians(25))
            row["roadside_frac"] = float(np.mean(near & par))
        else:
            row["roadside_frac"] = np.nan
        row["river"] = bool(riv is not None and np.mean(riv[r, c]) > 0.5)
        rows.append(row)
    return pd.DataFrame(rows)


REACH_FEATS = ["bend_deg_per_100m", "straight_frac", "roadside_frac", "width_m", "incision_m", "log_length"]


def stream_prior(F):
    """Transparent prior: sinuous -> stream; straight runs or roadside -> drain."""
    rs = F.roadside_frac.fillna(0)
    z = (0.10 * (F.bend_deg_per_100m - 15) - 3.0 * (F.straight_frac - 0.6) - 3.0 * rs
         + 0.3 * (F.width_m - 4) + 0.3 * (np.log10(F.length_m.clip(lower=1)) - 1.7))
    p = 1 / (1 + np.exp(-z))
    if "river" in F:                                         # wide channels are rivers (streams)
        p = np.where(F.river.fillna(False).astype(bool), np.maximum(p, 0.95), p)
    return p


def design_matrix(F):
    X = F.copy()
    X["log_length"] = np.log10(F.length_m.clip(lower=1))
    X["roadside_frac"] = F.roadside_frac.fillna(0)
    return X[REACH_FEATS].fillna(0)


def smooth_on_graph(reaches, F, p0, iters=20, flat_m=0.05):
    """Propagate stream probability along the network. A reach is pulled towards its neighbours
    (type rarely changes), short reaches more than long ones; an outflow reach is pulled more by
    the reaches flowing into it (drains feed streams, rarely the reverse)."""
    n = len(reaches)
    at = {}
    for i, R in enumerate(reaches):
        at.setdefault(R.node_start, []).append(i)
        at.setdefault(R.node_end, []).append(i)
    down = {}                                                 # downstream node of each reach
    for i, R in enumerate(reaches):
        d = F.bed_start[i] - F.bed_end[i]
        down[i] = R.node_end if d > flat_m else (R.node_start if d < -flat_m else None)
    conf0 = np.abs(p0 - 0.5) * 2
    # neighbours move a reach more if it is short and less if it is already confident
    alpha = np.where(F.length_m.values < 50, 0.6, 0.3) * (1 - 0.8 * conf0)
    p = p0.copy()
    for _ in range(iters):
        q = p.copy()
        for i, R in enumerate(reaches):
            num = den = 0.0
            for node in (R.node_start, R.node_end):
                for j in at.get(node, []):
                    if j == i:
                        continue
                    w = (0.2 + conf0[j]) * min(F.length_m.values[j], 200) / 200
                    if down.get(j) == node and down.get(i) != node:   # j flows into i
                        w *= 2.0
                    num += w * p[j]
                    den += w
            if den > 0:
                q[i] = (1 - alpha[i]) * p0[i] + alpha[i] * num / den
        p = q
    return p



def clean_reaches(reaches, min_component_m=50.0, min_dangling_m=15.0, keep=None, incision=None,
                  weak_component_m=100.0, weak_incision_m=0.4):
    """Indices of reaches to keep: drop network components shorter than ``min_component_m`` in
    total, and reaches shorter than ``min_dangling_m`` with a free end (a channel end) and the other
    end at a junction. Reaches *between* junctions are kept so the network stays connected.
    ``keep``: indices never dropped (e.g. reaches through culvert links). In Matt's labels this
    removed 35 of 50 spurious reaches and none of 22 real ones."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    keep = set(keep or [])
    n = len(reaches)
    L = np.array([np.hypot(*np.diff(R.coords, axis=0).T).sum() if len(R.coords) > 1 else 0.0 for R in reaches])
    deg = {}
    for R in reaches:
        for nd in (R.node_start, R.node_end):
            deg[nd] = deg.get(nd, 0) + 1
    rows, cols = [], []
    at = {}
    for i, R in enumerate(reaches):
        for nd in (R.node_start, R.node_end):
            at.setdefault(nd, []).append(i)
    for idx in at.values():
        for a in idx:
            for b in idx:
                rows.append(a); cols.append(b)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, comp = connected_components(A, directed=False)
    comp_len = np.bincount(comp, weights=L)
    weak = np.zeros(comp.max() + 1, bool)
    if incision is not None:                     # small *and* shallow components are dropped too
        inc = np.asarray(incision, float)
        for k in range(comp.max() + 1):
            sel = comp == k
            if comp_len[k] < weak_component_m and np.nanmedian(inc[sel]) < weak_incision_m:
                weak[k] = True
    out = []
    for i, R in enumerate(reaches):
        if i in keep:
            out.append(i)
            continue
        if comp_len[comp[i]] < min_component_m or weak[comp[i]]:
            continue
        free = (deg[R.node_start] == 1) + (deg[R.node_end] == 1)
        if free == 1 and L[i] < min_dangling_m:
            continue
        out.append(i)
    return out


def river_polygons(mask, T, min_width_m=8.0, min_area_m2=500.0, smooth_m=3.0):
    """Wide channels (rivers) as polygons, from the channel mask: cells whose channel width exceeds
    ``min_width_m``, grown back to the full channel within that river. Returns (polygon GeoSeries
    rows as dicts, raster mask)."""
    from rasterio.features import shapes
    from shapely.geometry import shape
    cs = abs(T.a)
    half_w = ndi.distance_transform_edt(mask) * cs
    core = half_w >= min_width_m / 2
    if not core.any():
        return [], np.zeros_like(mask)
    grown = ndi.binary_dilation(core, iterations=max(1, int(round(min_width_m / cs)))) & mask
    lab, n = ndi.label(grown, _EIGHT)
    area = np.bincount(lab.ravel(), minlength=n + 1) * cs * cs
    keep = area >= min_area_m2
    keep[0] = False
    riv = keep[lab]
    riv = ndi.gaussian_filter(riv.astype(float), smooth_m / cs) > 0.5   # smooth the outline
    polys = [dict(geometry=shape(g).simplify(1.0)) for g, v in shapes(riv.astype(np.uint8), mask=riv, transform=T) if v == 1]
    return polys, riv


def river_centrelines(riv, sk, T, spur_m=40.0):
    """Replace the skeleton inside river polygons by one centreline per river (skeleton of the
    smoothed polygon, long spurs pruned) and reconnect tributaries that end at the river edge."""
    from skimage.draw import line
    from . import channels
    cs = abs(T.a)
    cl = channels._prune(channels.skeletonize(riv), max(1, int(round(spur_m / cs))))
    out = (sk & ~riv) | cl
    # tributary ends next to the river: connect them to the nearest centreline cell
    if cl.any():
        _, (ir, ic) = ndi.distance_transform_edt(~cl, return_indices=True)
        k = np.ones((3, 3)); k[1, 1] = 0
        nb = ndi.convolve(out.astype(np.int16), k, mode="constant")
        near_riv = ndi.binary_dilation(riv, iterations=2)
        for r, c in np.argwhere(out & ~cl & (nb == 1) & near_riv):
            rr, cc = line(int(r), int(c), int(ir[r, c]), int(ic[r, c]))
            out[rr, cc] = True
    return out
