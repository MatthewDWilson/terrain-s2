"""Conditioned hydrology: a second hydrology pass on the DEM conditioned with the mapped network.

The first pass (pipeline.run) routes flow on the *breached* DEM, before anything is detected: HAND
is measured to drainage defined by an upstream-area threshold and wide channels, and is
discontinuous where drainage crosses barriers the breaching handled differently from the final
network. This pass starts again from the DEM and imposes the result:

1. channel cells (the cleaned reaches, culvert links, gap repairs and river polygons) are set to
   their local bed (3 x 3 minimum) less ``burn_m``, so flow follows the mapped network across
   microtopography;
2. culvert links and gap repairs (lines through a barrier) are set to the bed interpolated between
   their two ends, as a culvert invert (GeoFabrics sets a tunnel to the minimum around it), and only
   ever lowered;
3. the conditioned DEM is filled and routed (D8); upstream area is accumulated, seeded at the window
   edge from REC2 where a REC2 river enters the window (its upstream area, approximately CUM_AREA less
   the segment's own catchment, added at the mapped channel cell nearest to where it enters);
4. height above the streams: a relative elevation model (terrain_s2.rem; default) on the identified
   streams (reaches classed stream, including the small ones missing from mapped data) and river
   polygons, not the drains: adding or removing a drain must not change it, and a stopbank is then
   judged against the river, not the landside drains (Matt, 8 Oct 2026). Culvert links are not
   sampled either (their original DEM cells are the barrier crest). HAND (to the whole network, along
   the conditioned flow) is the alternative;
5. residual depressions (fill of the conditioned DEM minus the original DEM, so the burn itself never
   counts) are what the mapped network still fails to drain. Each pond's outlet is the cell where its
   water leaves (D8 on the filled conditioned DEM): where a pond a mapped channel flows into spills,
   a crossing is missed or not linked (or the pond is a genuine wetland or closed basin).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from . import hydro


@dataclass
class Conditioned:
    z_conditioned: np.ndarray      # m, float32
    drain: np.ndarray              # bool, the mapped network as rasterised
    upa: np.ndarray                # m2, with REC2 inflow where seeded
    upa_inflow: np.ndarray         # m2 added at each seeded cell (0 elsewhere)
    hand: np.ndarray               # m, height above the streams (REM), or above the network (HAND: NaN where flow misses it)
    residual: np.ndarray           # m, fill - conditioned DEM
    ponds: object                  # GeoDataFrame of residual ponds (polygons, attributes)
    outlets: object = None         # GeoDataFrame: one spill point per pond
    n_seeded: int = 0
    rem_basis: str = ""            # what the REM was sampled on
    rem_samples: int = 0


def _line_cells(geom, T, shape):
    """Cells of a line, 8-connected, in order from its start (rows, cols)."""
    from skimage.draw import line
    from .drains import _to_cr
    rows, cols = [], []
    for part in getattr(geom, "geoms", [geom]):
        pts = np.asarray(part.coords)
        if len(pts) < 2:
            continue
        cc, rr = _to_cr(T, pts[:, 0], pts[:, 1])
        for i in range(len(pts) - 1):
            r_, c_ = line(int(rr[i]), int(cc[i]), int(rr[i + 1]), int(cc[i + 1]))
            rows.append(r_)
            cols.append(c_)
    if not rows:
        return np.zeros(0, int), np.zeros(0, int)
    r, c = np.concatenate(rows), np.concatenate(cols)
    ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    r, c = r[ok], c[ok]
    keep = np.ones(len(r), bool)                       # drop repeats where segments join
    keep[1:] = (r[1:] != r[:-1]) | (c[1:] != c[:-1])
    return r[keep], c[keep]


def condition(z, T, channel_lines, link_lines, river_mask=None, burn_m=0.25):
    """(conditioned DEM, drain mask). ``channel_lines``: mapped reaches; ``link_lines``: culvert links
    and gap repairs, conditioned to the bed interpolated between their ends."""
    valid = np.isfinite(z)
    bed = ndi.minimum_filter(np.where(valid, z, np.inf), size=3)
    bed = np.where(np.isfinite(bed), bed, z)
    zc = z.astype(np.float64).copy()
    drain = np.zeros(z.shape, bool)
    for g in channel_lines:
        r, c = _line_cells(g, T, z.shape)
        drain[r, c] = True
    if river_mask is not None:
        drain |= river_mask
    drain &= valid
    zc[drain] = np.minimum(zc[drain], bed[drain] - burn_m)
    for g in link_lines:
        r, c = _line_cells(g, T, z.shape)
        if len(r) < 2:
            continue
        ok = valid[r, c]
        if ok.sum() < 2:
            continue
        r, c = r[ok], c[ok]
        z0, z1 = bed[r[0], c[0]], bed[r[-1], c[-1]]
        d = np.r_[0.0, np.cumsum(np.hypot(np.diff(r), np.diff(c)))]
        t = d / d[-1] if d[-1] > 0 else np.zeros_like(d)
        invert = z0 + (z1 - z0) * t - burn_m
        zc[r, c] = np.minimum(zc[r, c], invert)
        drain[r, c] = True
    zc[~valid] = np.nan
    return zc, drain


def rec2_inflow(rec2, T, shape, drain, window_bounds, snap_m=30.0):
    """[(flat cell, m2)]: REC2 rivers entering the window across its edge, at the mapped channel cell
    nearest the entry point (within ``snap_m``). Needs up_x/up_y/down_x/down_y and upstream_area_m2
    (and catchment_area_m2) from the REC2 source mapping. Segments wholly inside or leaving are skipped."""
    from shapely.geometry import Point, box
    if rec2 is None or not len(rec2) or not drain.any():
        return []
    need = {"up_x", "up_y", "down_x", "down_y", "upstream_area_m2"}
    if not need <= set(rec2.columns):
        return []
    win = box(*window_bounds)
    edge = win.exterior
    dr, dc = np.nonzero(drain)
    from scipy.spatial import cKDTree
    xs = T.c + (dc + 0.5) * T.a
    ys = T.f + (dr + 0.5) * T.e
    tree = cKDTree(np.column_stack([xs, ys]))
    out = []
    for _, s in rec2.iterrows():
        up_in = win.contains(Point(s.up_x, s.up_y))
        dn_in = win.contains(Point(s.down_x, s.down_y))
        if up_in or not dn_in:
            continue                                     # starts inside, or does not end inside
        cross = s.geometry.intersection(edge)
        pts = [cross] if cross.geom_type == "Point" else list(getattr(cross, "geoms", []))
        pts = [p for p in pts if p.geom_type == "Point"]
        if not pts:
            continue
        p = min(pts, key=lambda q: q.distance(Point(s.up_x, s.up_y)))
        dist, k = tree.query([p.x, p.y])
        if dist > snap_m:
            continue
        area = float(s.upstream_area_m2) - float(s.get("catchment_area_m2", 0.0) or 0.0)
        if area > 0:
            out.append((int(dr[k]) * shape[1] + int(dc[k]), area))
    return out


def residual_ponds(residual, drain, T, crs, idxs_ds, upa, candidates=None, rd=None, cs=1.0, min_depth=0.05,
                   min_area_m2=25.0, outlet_match_m=15.0):
    """(ponds, outlets). Ponds: residual depressions >= min_area_m2 (cells deeper than min_depth) with
    area, volume, maximum depth, mapped channel inside or beside (inflow), crossing candidates on the rim
    (10 m) and road distance. Outlets: per pond, the pond cell whose D8 downstream cell is outside it
    (largest upstream area), with the nearest candidate and road distance there."""
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape
    empty = (gpd.GeoDataFrame(geometry=[], crs=crs), gpd.GeoDataFrame(geometry=[], crs=crs))
    r_ = np.nan_to_num(residual)
    lab, n = ndi.label(r_ > min_depth, structure=np.ones((3, 3), bool))
    if n == 0:
        return empty
    idx = lab.ravel()
    a = cs * cs
    area = np.bincount(idx, minlength=n + 1) * a
    vol = np.bincount(idx, weights=r_.ravel(), minlength=n + 1) * a
    mx = np.zeros(n + 1)
    np.maximum.at(mx, idx, r_.ravel())
    near_drain = ndi.binary_dilation(drain, iterations=1)
    chan = np.bincount(idx, weights=near_drain.ravel().astype(float), minlength=n + 1) * cs
    keep = area >= min_area_m2
    keep[0] = False
    if not keep.any():
        return empty
    lab_k = np.where(keep[lab], lab, 0).astype(np.int32)
    geoms, ids = [], []
    for g, v in shapes(lab_k, mask=lab_k > 0, transform=T):
        geoms.append(shape(g))
        ids.append(int(v))
    P = gpd.GeoDataFrame({"pond": ids}, geometry=geoms, crs=crs).dissolve("pond").reset_index()
    P["area_m2"] = area[P.pond]
    P["volume_m3"] = vol[P.pond]
    P["max_depth_m"] = mx[P.pond]
    P["channel_m"] = chan[P.pond]                        # mapped channel in or beside the pond
    P["inflow"] = P.channel_m > 0
    if rd is not None:
        P["road_dist_m"] = np.asarray(ndi.minimum(rd, lab_k, index=P.pond.values), float)
    # outlets: pond cells draining out of their pond; per pond the one with the largest upstream area
    lf = lab_k.ravel()
    ds = np.asarray(idxs_ds).ravel()
    cand = np.flatnonzero((lf > 0) & (lf[ds] != lf))
    out_cell = {}
    if len(cand):
        u = np.nan_to_num(np.asarray(upa).ravel()[cand])
        order = np.lexsort((-u, lf[cand]))
        first = np.r_[True, lf[cand][order][1:] != lf[cand][order][:-1]]
        for k in cand[order][first]:
            out_cell[int(lf[k])] = int(k)
    nc = residual.shape[1]
    ox = [T.c + (out_cell[p] % nc + 0.5) * T.a if p in out_cell else np.nan for p in P.pond]
    oy = [T.f + (out_cell[p] // nc + 0.5) * T.e if p in out_cell else np.nan for p in P.pond]
    Q = gpd.GeoDataFrame(P[["pond", "area_m2", "volume_m3", "max_depth_m", "channel_m", "inflow"]].copy(),
                         geometry=gpd.points_from_xy(ox, oy), crs=crs)
    Q = Q[np.isfinite(ox)].reset_index(drop=True)
    if rd is not None and len(Q):
        Q["road_dist_m"] = [float(rd.flat[out_cell[p]]) for p in Q.pond]
    if candidates is not None and len(candidates):
        rim = P.geometry.buffer(10.0)
        j = gpd.sjoin(gpd.GeoDataFrame(geometry=rim, crs=crs).reset_index(names="i"),
                      candidates[["tier", "geometry"]], predicate="intersects")
        P["n_candidates"] = j.groupby("i").size().reindex(range(len(P)), fill_value=0).values
        if len(Q):
            nn = gpd.sjoin_nearest(Q[["geometry"]].reset_index(names="i"), candidates[["tier", "geometry"]],
                                   distance_col="cand_dist_m").drop_duplicates("i").set_index("i")
            Q["nearest_candidate_m"] = nn.cand_dist_m.reindex(range(len(Q))).values
            Q["nearest_candidate_tier"] = nn.tier.reindex(range(len(Q))).values
    else:
        P["n_candidates"] = 0
    if len(Q):
        if "nearest_candidate_m" not in Q:
            Q["nearest_candidate_m"], Q["nearest_candidate_tier"] = np.nan, ""
        at = np.nan_to_num(Q.nearest_candidate_m, nan=np.inf) <= outlet_match_m
        Q["status"] = np.where(~Q.inflow, "closed", np.where(at, "inflow_candidate_at_outlet", "inflow_no_candidate"))
        P = P.merge(Q[["pond", "status", "nearest_candidate_m", "nearest_candidate_tier"]], on="pond", how="left")
    P["status"] = P.get("status", "closed")
    P["status"] = P.status.fillna("closed")
    return P, Q


def run(z, T, crs, channel_lines, link_lines, river_mask=None, rec2=None, window_bounds=None,
        candidates=None, rd=None, burn_m=0.25, height_model="hand", rem_lines=None):
    """``rem_lines``: the lines the REM is sampled on (the identified streams), with ``river_mask``;
    None samples the whole network. If they give no samples, the network is used and ``rem_basis`` says so."""
    cs = abs(T.a)
    zc, drain = condition(z, T, channel_lines, link_lines, river_mask, burn_m)
    zf, d8 = hydro.fill(zc)
    import pyflwdir
    flw = pyflwdir.from_array(d8, ftype="d8", transform=T, latlon=False)
    w = np.full(z.shape, cs * cs, np.float64)
    inflow = np.zeros(z.shape, np.float64)
    seeds = rec2_inflow(rec2, T, z.shape, drain, window_bounds or _bounds(T, z.shape)) if rec2 is not None else []
    for k, a in seeds:
        inflow.flat[k] += a
    upa = flw.accuflux(w + inflow)
    upa = np.where(np.isfinite(z), upa, np.nan)
    rem_basis, n_rem = "", 0
    if height_model == "rem":
        from .rem import rem
        basis = drain
        rem_basis = "network"
        if rem_lines is not None:
            basis = np.zeros(z.shape, bool)
            for g in rem_lines:
                r, c = _line_cells(g, T, z.shape)
                basis[r, c] = True
            if river_mask is not None:
                basis |= river_mask
            basis &= np.isfinite(z)
            rem_basis = "streams and rivers"
            if not basis.any():
                basis, rem_basis = drain, "network (no streams identified)"
        hand, _, n_rem = rem(z, T, basis)
    else:
        hand = flw.hand(drain=drain, elevtn=np.nan_to_num(z, nan=-9999.0).astype(np.float32))
        reaches = flw.accuflux(drain.astype(np.float64), direction="down") > 0   # downstream path meets the network
        hand = np.where(np.isfinite(z) & reaches, hand, np.nan)                  # else undefined (drains elsewhere)
    residual = np.where(np.isfinite(z), np.maximum(zf - z, 0.0), np.nan)       # against the original DEM
    ponds, outlets = residual_ponds(residual, drain, T, crs, flw.idxs_ds, upa, candidates, rd, cs)
    return Conditioned(zc.astype(np.float32), drain, upa, inflow, hand.astype(np.float32),
                       residual.astype(np.float32), ponds, outlets, len(seeds), rem_basis, n_rem)


def _bounds(T, shape):
    x0, y1 = T.c, T.f
    return (x0, y1 + shape[0] * T.e, x0 + shape[1] * T.a, y1)
