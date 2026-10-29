"""Stage A, version 2: the catchment's network routed from one outlet (channel network v2; Matt, 27 Oct 2026).

Version 1 (``catchment_mode``) tried to join the tiles' detected pieces into a network and found 5 % of SH12's channel
length reaching an outlet, with 158 outlets. Here the network's connectivity comes from flow routing instead, and the
detected channels decide where it runs and how far up it goes:

1. **A coarse, conditioned DEM** (``res_m``, 5 m), tile by tile from the 1 m DEMs: each tile's stage B reaches lower
   the 1 m surface by ``burn_max_m`` x their evidence (physics channels 1; the U-Net's by their mean probability;
   joins 0.5), each crossing candidate's cutline lowers it by ``carve_max_m`` x the candidate's ``p_structure`` (a
   culvert the scorer believes opens the embankment; one it doubts barely does), and each 5 m cell takes the lowest
   1 m cell in it (channels survive the coarsening, banks and noise do not). OSM tidal flats and coastal water inside
   the catchment are sea. Depressions are then breached by least cost (an embankment no candidate opens is cut
   where it is thinnest, not filled into a lake).
2. **One outlet** per sea-draining catchment: where the catchment's main OSM waterway leaves it (``osm_outlet``: the
   connected OSM river / canal / stream system with the most rank-weighted length inside the catchment, at the
   lowest of its crossings of the boundary or its ends at the sea; Matt, 28 Oct), snapped to the nearest edge cell;
   ``outlet_xy`` overrides it; failing both, the edge cell beside the sea (coastline or tidal flats within
   ``coast_m``) with the largest upstream area in a first routing with every edge cell an outlet. Routed again with that cell alone as the outlet, so every cell of the
   catchment drains to it (pyflwdir: depressions filled towards it). The other mouths the first routing found are
   reported (``outlet_report``): a reason to allow a second outlet would show there.
3. **The routed network**: streams where the upstream area is at least ``a_min_m2``, as segments between
   confluences, drawn downstream.
4. **How far up it goes, as a probability**: each segment's evidence ``e`` (along it, the stage B reaches within
   ``evid_dist_m`` by their evidence, or the U-Net probability, whichever is larger) and a prior from its upstream
   area, ``p_area = logistic((log10 A - log10 a50_m2) / prior_scale)``; ``p_channel = 1 - (1 - e)(1 - p_area)``.
   Segments are removed from the heads down while ``p_channel`` < ``keep_p``, and each head left is trimmed to
   where, coming down it, the smoothed p_channel first reaches ``keep_p``; a doubtful segment between two believed
   ones stays (the network stays connected). (Hand-set for now; to be learned from council and OSM lines.)
5. **On the 1 m channels**: the segments' ends moved to the cheapest cell within ``node_snap_m`` (shared ends move
   together), and each segment redrawn between them as the least-cost path through a corridor of
   ``refine_corridor_m`` (``cost_window``: 1 + 9 (1 - p), p the larger of the U-Net probability and the stage B
   channels' evidence, so a detected channel is followed; plus ``bed_weight`` for height above the local bed, so the
   path keeps to the bed elsewhere). ``offsets`` in the summary: the network's distance from the stage B channels and
   the council lines, before and after.
6. **Drains join it**: stage B drains away from the routed network (too small to route) are joined where a
   least-cost path to it is probable enough, ``p_join = p_evidence x p_barrier x p_rise >= keep_p``: the evidence of a
   channel along the path, ``1 - (1 - mean U-Net probability)(1 - exp(-length / gap_scale_m))`` (a short gap is
   likely the same channel); the crossing candidate's ``p_structure`` where the path crosses a road (``road_prior``
   where there is none); and ``exp(-rise / rise_sigma_m)`` for how far it climbs above its start (not counted
   through a road a candidate opens). No fixed rise, gap or crossing distance.
7. **Classes in order**: each segment's class from its evidence (OSM river areas and tags, the U-Net's class, the
   stage B class), then made consistent downstream: nothing below a stream is a drain, nothing below a river less than
   a river. Joined drains stay drains.
8. **Crossings** wherever the network meets a road, each with the nearest crossing candidate (its ``p_structure``,
   tier and distance) or none (``needs_structure``: the network passes a road no candidate explains).

``trace`` reports, for given points (e.g. the SH12 culvert), what each stage has near them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULTS = dict(res_m=5.0, burn_max_m=2.0, carve_max_m=5.0, breach_max_m=200.0, gap_scale_m=10.0, coast_m=20.0,
                a_min_m2=5000.0, a50_m2=1.0e5,
                prior_scale=0.5, evid_dist_m=6.0, keep_p=0.5, refine_corridor_m=10.0, rise_sigma_m=0.3,
                node_snap_m=5.0, bed_weight=5.0, bed_scale_m=0.5, bed_window_m=11.0, outlet_snap_m=100.0,
                outlet_xy=None, waterway_rank=dict(river=3.0, canal=2.0, stream=1.0),
                burn_mapped_m=5.0, mapped_corridor_m=20.0, mapped_piece_m=400.0,
                mapped_kinds=("river", "stream", "canal", "drain", "ditch"), evidence_share=0.5, split_min_m=20.0,
                presence_model=None, xs_step_m=20.0, xs_half_m=15.0,
                road_prior=0.05, join_max_m=60.0, min_piece_m=20.0, river_min_m2=5.0e7, tier_p=dict(high=0.8,
                medium=0.5, low=0.2), crossing_search_m=25.0, overlap_m=500.0, max_outlets=1)
NODATA = -9999.0
RANK = dict(drain=0, stream=1, river=2)


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def _p_structure(cr, P):
    """A crossing candidate's probability: the scorer's p_structure, else by tier."""
    if "p_structure" in cr:
        p = cr.p_structure.to_numpy(float)
    else:
        p = np.full(len(cr), np.nan)
    if "tier" in cr:
        t = cr.tier.astype(str).map(P["tier_p"]).to_numpy(float)
        p = np.where(np.isfinite(p), p, t)
    return np.nan_to_num(p, nan=0.0)


def reach_evidence(R, prob_files):
    """Evidence per stage B reach: physics 1, U-Net-only its mean probability (0.7 when unknown), joins 0.5."""
    src = R.channel_source.astype(str).to_numpy() if "channel_source" in R else np.full(len(R), "physics")
    e = np.where(src == "link", 0.5, 1.0)
    ml = np.flatnonzero(src == "ml")
    if len(ml):
        pm = sample_prob(prob_files, [g for g in R.geometry.iloc[ml]])
        e[ml] = np.where(np.isfinite(pm), pm, 0.7)
    return e


def sample_prob(prob_files, lines, step=2.0):
    """Mean stored U-Net probability (0-1) along each line (NaN where no file covers it)."""
    import rasterio
    out = np.full(len(lines), np.nan)
    pts, owner = [], []
    for i, g in enumerate(lines):
        n = max(int(g.length // step), 1)
        for t in np.linspace(0, g.length, n + 1):
            p = g.interpolate(t)
            pts.append((p.x, p.y))
            owner.append(i)
    if not pts:
        return out
    pts = np.array(pts)
    owner = np.array(owner)
    val = np.full(len(pts), np.nan)
    for f in prob_files:
        if not Path(f).exists():
            continue
        with rasterio.open(f) as s:
            b = s.bounds
            sel = np.flatnonzero(np.isnan(val) & (pts[:, 0] >= b.left) & (pts[:, 0] < b.right) &
                                 (pts[:, 1] > b.bottom) & (pts[:, 1] <= b.top))
            if not len(sel):
                continue
            v = np.array([x[0] for x in s.sample([tuple(p) for p in pts[sel]])], float)
            v[v > 250] = np.nan
            val[sel] = v / 250.0
    for i in range(len(lines)):
        v = val[owner == i]
        if np.isfinite(v).any():
            out[i] = float(np.nanmean(v))
    return out


# ---------------------------------------------------------------- the coarse DEM, the outlet, the routing

def grid_for(geom, res):
    """(transform, shape) of a ``res`` grid covering ``geom``'s bounds, aligned to multiples of ``res``."""
    from affine import Affine
    x0, y0, x1, y1 = geom.bounds
    x0, y0 = np.floor(x0 / res) * res - res, np.floor(y0 / res) * res - res
    x1, y1 = np.ceil(x1 / res) * res + res, np.ceil(y1 / res) * res + res
    return Affine(res, 0, x0, 0, -res, y1), (int(round((y1 - y0) / res)), int(round((x1 - x0) / res)))


def coarse_dem(tiles, cores, bundles, reaches, evidence, crossings, T5, shape5, P):
    """The conditioned coarse DEM (NODATA where no tile): see the module docstring, step 1."""
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import from_bounds
    from shapely import STRtree
    res = P["res_m"]
    k = int(round(res))
    z5 = np.full(shape5, NODATA, np.float32)
    rt = STRtree(reaches.geometry.to_numpy()) if len(reaches) else None
    ct = STRtree(crossings.geometry.to_numpy()) if crossings is not None and len(crossings) else None
    pstr = _p_structure(crossings, P) if ct is not None else None
    for t in tiles:
        f = Path(bundles) / t / "dem.tif"
        if not f.exists():
            continue
        c = cores[t]
        x0, y0, x1, y1 = c.bounds
        gx0, gy1 = T5.c, T5.f
        x0 = max(x0, gx0)
        y1 = min(y1, gy1)
        x1 = min(x1, gx0 + shape5[1] * res)
        y0 = max(y0, gy1 - shape5[0] * res)
        x0, y0 = np.floor((x0 - gx0) / res) * res + gx0, gy1 - np.ceil((gy1 - y0) / res) * res
        x1, y1 = np.ceil((x1 - gx0) / res) * res + gx0, gy1 - np.floor((gy1 - y1) / res) * res
        if x1 <= x0 or y1 <= y0:
            continue
        h, w = int(round((y1 - y0))), int(round((x1 - x0)))
        with rasterio.open(f) as s:
            z = s.read(1, window=from_bounds(x0, y0, x1, y1, s.transform), out_shape=(h, w), boundless=True,
                       fill_value=np.nan, masked=True).astype(np.float32)
            z = np.ma.filled(z, np.nan)
            if s.nodata is not None:
                z[z == s.nodata] = np.nan
        from affine import Affine
        T1 = Affine(1.0, 0, x0, 0, -1.0, y1)
        from shapely.geometry import box
        win = box(x0, y0, x1, y1)
        burn = np.zeros((h, w), np.float32)
        if rt is not None:
            idx = rt.query(win, predicate="intersects")
            shp = sorted(((reaches.geometry.iloc[i], float(P["burn_max_m"] * evidence[i])) for i in idx),
                         key=lambda s: s[1])
            if shp:
                burn = np.maximum(burn, rasterize(shp, out_shape=(h, w), transform=T1, fill=0, all_touched=True,
                                                  dtype="float32"))
        if ct is not None:
            idx = ct.query(win, predicate="intersects")
            shp = sorted(((crossings.geometry.iloc[i].buffer(1.0), float(P["carve_max_m"] * pstr[i])) for i in idx),
                         key=lambda s: s[1])
            if shp:
                burn = np.maximum(burn, rasterize(shp, out_shape=(h, w), transform=T1, fill=0, all_touched=True,
                                                  dtype="float32"))
        zb = z - burn
        zb = zb.reshape(h // k, k, w // k, k)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                zc = np.nanmin(np.nanmin(zb, axis=3), axis=1)
        r0 = int(round((gy1 - y1) / res))
        c0 = int(round((x0 - gx0) / res))
        sub = z5[r0:r0 + zc.shape[0], c0:c0 + zc.shape[1]]
        ok = np.isfinite(zc)
        sub[ok] = zc[ok]
    return z5


def breach(z5, P):
    """Depressions on the coarse DEM breached by least cost (``hydro.breach_least_cost``, paths up to
    ``breach_max_m``) before routing: an embankment no candidate opens is cut where it is thinnest, rather than
    filled into a lake the routing crosses anywhere (filling still handles what breaching leaves)."""
    from .config import HydroParams
    from .hydro import breach_least_cost
    z = np.where(z5 == NODATA, np.nan, z5).astype(np.float64)
    if not np.isfinite(z).any():
        return z5
    br = breach_least_cost(z, P["res_m"], HydroParams(breach_max_length_m=P["breach_max_m"]))
    zb = br.z_breached
    return np.where(np.isfinite(zb), zb, NODATA).astype(np.float32)


def choose_outlet(z5, T5, sea_mask, P, osm_xy=None, osm_rec=None):
    """(outlet cell index, report). In order: ``outlet_xy`` (given); the main OSM waterway's exit (``osm_outlet``),
    snapped to the nearest edge cell; else the mouth the first routing finds beside the sea with the largest upstream
    area (else the edge cell with the largest). The report holds the mouths that routing found either way."""
    import pyflwdir
    f = pyflwdir.from_dem(z5, nodata=NODATA, transform=T5, latlon=False, outlets="edge")
    upa = f.upstream_area("m2")
    pits = np.asarray(f.idxs_pit, int)
    from scipy import ndimage as ndi
    near = ndi.binary_dilation(sea_mask, iterations=max(1, int(np.ceil(P["coast_m"] / P["res_m"])))) if sea_mask.any() \
        else np.zeros(z5.shape, bool)
    a = upa.ravel()[pits]
    coast = near.ravel()[pits]
    order = np.argsort(-a)
    rep = [dict(x=float(T5.c + (p % z5.shape[1] + 0.5) * T5.a), y=float(T5.f + (p // z5.shape[1] + 0.5) * T5.e),
                area_km2=round(float(a[j]) / 1e6, 4), coast=bool(coast[j])) for j, p in zip(order[:5], pits[order[:5]])]
    cand = order[coast[order]] if coast.any() else order
    out = int(pits[cand[0]])
    method = dict(method="routing")
    xy = tuple(P["outlet_xy"]) if P.get("outlet_xy") else osm_xy
    if xy is not None:
        rr, cc = np.nonzero(edge_cells(z5))
        if len(rr):
            ex = T5.c + (cc + 0.5) * T5.a
            ey = T5.f + (rr + 0.5) * T5.e
            d = np.hypot(ex - xy[0], ey - xy[1])
            j = int(np.argmin(d))
            if d[j] <= P["outlet_snap_m"]:
                out = int(rr[j] * z5.shape[1] + cc[j])
                method = dict(method="given" if P.get("outlet_xy") else "osm", snapped_m=round(float(d[j]), 1),
                              **({k: v for k, v in (osm_rec or {}).items() if k != "method"} if not P.get("outlet_xy")
                                 else {}))
            else:
                method = dict(method="routing", osm_point_too_far_m=round(float(d[j]), 1))
    elif osm_rec is not None:
        method = dict(method="routing", osm=osm_rec.get("reason"))
    chosen = dict(x=float(T5.c + (out % z5.shape[1] + 0.5) * T5.a), y=float(T5.f + (out // z5.shape[1] + 0.5) * T5.e),
                  area_km2=round(float(upa.ravel()[out]) / 1e6, 4), coast=bool(near.ravel()[out]), **method)
    return out, dict(mouths=rep, chosen=chosen)


def osm_outlet(catchment, lines, sea, z5, T5, P):
    """Where the catchment's main OSM waterway leaves it (Matt, 28 Oct): the waterways (``waterway_rank``: river over
    canal over stream) are joined into connected systems; the main one has the most rank-weighted length inside the
    catchment, weighted again by the square of its share inside (so a river along the boundary does not win); its
    outlet is the lowest (on the coarse DEM) of the points where it crosses the
    catchment's boundary or ends at the sea. Returns ((x, y), record) or (None, record)."""
    import shapely
    from shapely.geometry import Point
    from shapely.ops import linemerge, unary_union
    rec = dict(method="osm")
    if lines is None or not len(lines) or "waterway" not in lines:
        return None, dict(rec, reason="no OSM waterways")
    W = lines[lines.waterway.isin(list(P["waterway_rank"]))]
    W = W[W.intersects(catchment)]
    if not len(W):
        return None, dict(rec, reason="no OSM river, canal or stream in the catchment")
    merged = unary_union(list(W.geometry))
    parts = list(getattr(merged, "geoms", [merged]))
    from .connectivity import _UF
    uf = _UF(len(parts))
    tree = shapely.STRtree(np.asarray(parts, object))
    a, b = tree.query(np.asarray(parts, object), predicate="dwithin", distance=1.0)
    for i, j in zip(a, b):
        if i != j:
            uf.union(i, j)
    lab = uf.labels()
    # a part's rank: the highest of the OSM ways it overlaps
    pr = []
    wtree = shapely.STRtree(W.geometry.to_numpy())
    for g in parts:
        idx = wtree.query(g, predicate="intersects")
        pr.append(max([P["waterway_rank"].get(W.waterway.iloc[i], 1.0) for i in idx], default=1.0))
    # a system's score: rank-weighted length inside the catchment x (its share inside)^2, so a large river that only
    # runs along the boundary (the Ashley beside catchment 1589) loses to the catchment's own waterway
    band = catchment.buffer(P["overlap_m"])
    inside, near_ = {}, {}
    for k, g in enumerate(parts):
        inside[lab[k]] = inside.get(lab[k], 0.0) + pr[k] * g.intersection(catchment).length
        near_[lab[k]] = near_.get(lab[k], 0.0) + pr[k] * g.intersection(band).length
    score = {k: inside[k] * (inside[k] / max(near_[k], 1e-9)) ** 2 for k in inside}
    main = max(score, key=score.get)
    sys_lines = linemerge([parts[k] for k in range(len(parts)) if lab[k] == main])
    geom = sys_lines
    cand = []
    x = geom.intersection(catchment.boundary)
    for q in getattr(x, "geoms", [x]):
        if q.is_empty:
            continue
        cand.append(q if q.geom_type == "Point" else q.centroid)
    if sea:
        seau = unary_union(sea)
        for g in getattr(geom, "geoms", [geom]):
            for c in (g.coords[0], g.coords[-1]):
                q = Point(c)
                if seau.distance(q) <= P["coast_m"]:
                    cand.append(q)
    if not cand:
        return None, dict(rec, reason="the main OSM waterway does not leave the catchment")
    h, w = z5.shape
    zs = []
    for q in cand:
        r = int(min(max((T5.f - q.y) // abs(T5.e), 0), h - 1))
        c = int(min(max((q.x - T5.c) // T5.a, 0), w - 1))
        r0, r1, c0, c1 = max(r - 3, 0), min(r + 4, h), max(c - 3, 0), min(c + 4, w)
        win = z5[r0:r1, c0:c1]
        v = win[win != NODATA]
        zs.append(float(v.min()) if v.size else np.inf)
    k = int(np.argmin(zs))
    names = W.get("name")
    nm = None
    if names is not None:
        inside = W[W.intersects(geom.buffer(1.0))]
        nn = inside.name.dropna()
        nm = str(nn.mode().iloc[0]) if len(nn) else None
    return (cand[k].x, cand[k].y), dict(rec, waterway=nm, candidates=len(cand), z=round(zs[k], 2),
                                        main_length_km=round(geom.intersection(catchment).length / 1000, 3))


def edge_cells(z5):
    """The coarse DEM's valid cells beside an invalid one (where an outlet can be)."""
    from scipy import ndimage as ndi
    valid = z5 != NODATA
    return valid & ndi.binary_dilation(~valid, structure=np.ones((3, 3), bool), border_value=1)


def route(z5, T5, outlet):
    """The flow directions with ``outlet`` as the only outlet (every valid cell drains to it)."""
    import pyflwdir
    z = z5.copy()
    valid = z != NODATA
    z.ravel()[outlet] = float(z[valid].min()) - 100.0
    return pyflwdir.from_dem(z, nodata=NODATA, transform=T5, latlon=False, outlets="min")


def segments(flw, T5, catchment_mask, P):
    """The routed network's segments (GeoDataFrame, drawn downstream) with idx, idx_ds, upstream area at their
    downstream end, Strahler order; only where the upstream area reaches ``a_min_m2`` inside the catchment."""
    import geopandas as gpd
    from shapely.geometry import shape
    upa = flw.upstream_area("m2")
    m = (upa >= P["a_min_m2"]) & catchment_mask
    if not m.any():
        return gpd.GeoDataFrame(geometry=[], crs=2193)
    sto = flw.stream_order(mask=m)
    feats = flw.streams(mask=m, direction="down")
    rows = []
    h, w = upa.shape
    for f in feats:
        g = shape(f["geometry"])
        if g.length <= 0:
            continue
        pr = f["properties"]
        i, j = int(pr["idx"]), int(pr["idx_ds"])
        # the segment's own upstream area: at its last cell before the confluence it flows into
        x, y = g.coords[-2] if len(g.coords) >= 2 else g.coords[-1]
        r, c = int((T5.f - y) // abs(T5.e)), int((x - T5.c) // T5.a)
        a_end = float(upa[min(max(r, 0), h - 1), min(max(c, 0), w - 1)])
        rows.append(dict(geometry=g, idx=i, idx_ds=j, upa_m2=a_end, upa_top_m2=float(upa.ravel()[i]),
                         strahler=int(sto.ravel()[i])))
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=2193)


# ---------------------------------------------------------------- presence, refinement, joins

def segment_samples(S, reaches, evidence, prob_files, upa, T5, P, step=2.0):
    """Per segment, samples along it from its top: (distance from the top, evidence, upstream area). Evidence: the
    stage B reaches within ``evid_dist_m`` (their evidence) or the U-Net probability, whichever is larger."""
    import rasterio
    from shapely import STRtree, points
    pts, owner, dist = [], [], []
    for i, g in enumerate(S.geometry):
        n = max(int(g.length // step), 1)
        for t in np.linspace(0, g.length, n + 1):
            q = g.interpolate(t)
            pts.append((q.x, q.y))
            owner.append(i)
            dist.append(t)
    out = [(np.zeros(0), np.zeros(0), np.zeros(0)) for _ in range(len(S))]
    if not pts:
        return out
    pts = np.array(pts)
    owner = np.array(owner)
    dist = np.array(dist)
    e = np.zeros(len(pts))
    if len(reaches):
        tree = STRtree(reaches.geometry.to_numpy())
        pi, ri = tree.query(points(pts), predicate="dwithin", distance=P["evid_dist_m"])
        np.maximum.at(e, pi, evidence[ri])
    for f in prob_files:
        if not Path(f).exists():
            continue
        with rasterio.open(f) as src:
            b = src.bounds
            sel = np.flatnonzero((pts[:, 0] >= b.left) & (pts[:, 0] < b.right) & (pts[:, 1] > b.bottom) &
                                 (pts[:, 1] <= b.top))
            if len(sel):
                v = np.array([x[0] for x in src.sample([tuple(p) for p in pts[sel]])], float)
                v = np.where(v > 250, 0.0, v / 250.0)
                e[sel] = np.maximum(e[sel], v)
    h, w = upa.shape
    r = ((T5.f - pts[:, 1]) // abs(T5.e)).astype(int).clip(0, h - 1)
    c = ((pts[:, 0] - T5.c) // T5.a).astype(int).clip(0, w - 1)
    a = upa[r, c]
    for i in range(len(S)):
        k = owner == i
        out[i] = (dist[k], e[k], a[k])
    return out


def point_p(e, a, P):
    """p_channel at points: ``1 - (1 - e)(1 - p_area)``."""
    pa = _logistic((np.log10(np.maximum(a, 1.0)) - np.log10(P["a50_m2"])) / P["prior_scale"])
    return 1.0 - (1.0 - e) * (1.0 - pa)


def trim_head(g, samples, P, window_m=25.0):
    """A head segment cut where, coming down from its top, the smoothed p_channel first reaches ``keep_p``
    (None when it never does)."""
    from shapely.ops import substring
    d, e, a = samples
    if not len(d):
        return g
    p = point_p(e, a, P)
    k = max(1, int(round(window_m / max(np.diff(d).mean() if len(d) > 1 else 1.0, 1e-6))))
    ps = np.convolve(p, np.ones(k) / k, mode="same")
    ok = np.flatnonzero(ps >= P["keep_p"])
    if not len(ok):
        return None
    t0 = float(d[ok[0]])
    return g if t0 <= 0 else substring(g, t0, g.length)


def p_channel(S, e, P):
    pa = _logistic((np.log10(np.maximum(S.upa_m2.to_numpy(float), 1.0)) - np.log10(P["a50_m2"])) / P["prior_scale"])
    return 1.0 - (1.0 - e) * (1.0 - pa), pa


def prune_heads(S, p, P):
    """Keep mask: segments removed from the heads down while their p_channel is below ``keep_p``."""
    keep = np.ones(len(S), bool)
    up = {}
    for k, (i, j) in enumerate(zip(S.idx, S.idx_ds)):
        up.setdefault(int(j), []).append(k)
    by_idx = {int(i): k for k, i in enumerate(S.idx)}
    n_up = np.array([len(up.get(int(i), [])) for i in S.idx])
    queue = [k for k in range(len(S)) if n_up[k] == 0]
    while queue:
        k = queue.pop()
        if not keep[k] or p[k] >= P["keep_p"]:
            continue
        keep[k] = False
        d = by_idx.get(int(S.idx_ds.iloc[k]))
        if d is not None:
            n_up[d] -= 1
            if n_up[d] == 0:
                queue.append(d)
    return keep


def cost_window(bounds, prob_files, dem_files, reaches, evidence, P):
    """(cost, transform) on a 1 m window: ``1 + 9 (1 - p) + bed_weight x clip(relz / bed_scale_m, 0, 1)``, with ``p``
    the larger of the U-Net probability and the stage B channels' evidence (so a detected channel is followed where
    there is one) and ``relz`` the height above the lowest cell within ``bed_window_m`` (so the path keeps to the
    bed). None when no probability or DEM covers the window."""
    from rasterio.features import rasterize
    from scipy import ndimage as ndi
    from shapely.geometry import box

    from .catchment_mode import _mosaic
    pr, T = _mosaic(prob_files, bounds)
    zz, Tz = _mosaic(dem_files, bounds)
    if pr is None and zz is None:
        return None, None
    if pr is None:
        pr, T = np.full(zz.shape, 0.0, np.float32), Tz
    p = np.where(np.isfinite(pr) & (pr <= 250), pr / 250.0, 0.0)
    if reaches is not None and len(reaches):
        win = box(*bounds)
        idx = np.flatnonzero(reaches.intersects(win).to_numpy())
        if len(idx):
            shp = sorted(((reaches.geometry.iloc[i], float(evidence[i])) for i in idx), key=lambda t: t[1])
            b = rasterize(shp, out_shape=p.shape, transform=T, fill=0, all_touched=True, dtype="float32")
            p = np.maximum(p, b)
    cost = 1.0 + 9.0 * (1.0 - p)
    if zz is not None and zz.shape == p.shape and np.isfinite(zz).any():
        zf = np.where(np.isfinite(zz), zz, np.nanmax(zz))
        k = max(3, int(round(P["bed_window_m"])) | 1)
        relz = zf - ndi.minimum_filter(zf, size=k)
        cost = cost + P["bed_weight"] * np.clip(relz / P["bed_scale_m"], 0.0, 1.0)
    return cost, T


def snap_nodes(S, prob_files, dem_files, reaches, evidence, P):
    """{node key: (x, y)}: each segment end moved to the cheapest cell within ``node_snap_m`` (a coarse cell's centre
    is up to half a cell off the channel), the same for every segment sharing it, so the network stays joined."""
    out = {}
    r = P["node_snap_m"]
    for g in S.geometry:
        for c in (g.coords[0], g.coords[-1]):
            key = (round(c[0], 2), round(c[1], 2))
            if key in out:
                continue
            b = (np.floor(c[0] - r - 1), np.floor(c[1] - r - 1), np.ceil(c[0] + r + 1), np.ceil(c[1] + r + 1))
            cost, T = cost_window(b, prob_files, dem_files, reaches, evidence, P)
            if cost is None:
                out[key] = c
                continue
            ys, xs = np.mgrid[0:cost.shape[0], 0:cost.shape[1]]
            cx = T.c + (xs + 0.5) * T.a
            cy = T.f + (ys + 0.5) * T.e
            ok = np.hypot(cx - c[0], cy - c[1]) <= r
            if not ok.any():
                out[key] = c
                continue
            cc = np.where(ok, cost, np.inf)
            j = np.unravel_index(int(np.argmin(cc)), cc.shape)
            out[key] = (float(cx[j]), float(cy[j]))
    return out


def refine(line, prob_files, P, dem_files=(), reaches=None, evidence=None, ends=None):
    """The least-cost path between the line's ends (moved to ``ends`` when given) through a corridor of
    ``refine_corridor_m`` around it, over ``cost_window``: on a detected channel where there is one, on the bed
    elsewhere."""
    from rasterio.features import rasterize
    from shapely.geometry import LineString
    from skimage.graph import MCP_Geometric
    a = ends[0] if ends else line.coords[0]
    z = ends[1] if ends else line.coords[-1]
    c = P["refine_corridor_m"]
    x0, y0, x1, y1 = line.bounds
    x0, y0 = min(x0, a[0], z[0]), min(y0, a[1], z[1])
    x1, y1 = max(x1, a[0], z[0]), max(y1, a[1], z[1])
    b = (np.floor(x0 - c - 2), np.floor(y0 - c - 2), np.ceil(x1 + c + 2), np.ceil(y1 + c + 2))
    cost, T = cost_window(b, prob_files, dem_files, reaches, evidence, P)
    if cost is None:
        return LineString([a] + list(line.coords[1:-1]) + [z])
    corr = rasterize([(LineString([a] + list(line.coords) + [z]).buffer(c), 1)], out_shape=cost.shape, transform=T,
                     fill=0) > 0
    cost = np.where(corr, cost, -1.0)                                     # negative: not passable
    rc = lambda xy: (int((T.f - xy[1]) // abs(T.e)), int((xy[0] - T.c) // T.a))
    s_, e_ = rc(a), rc(z)
    inside = all(0 <= v < n for v, n in zip(s_ + e_, cost.shape * 2))
    if not inside or cost[s_] < 0 or cost[e_] < 0:
        return LineString([a] + list(line.coords[1:-1]) + [z])
    mcp = MCP_Geometric(cost)
    mcp.find_costs([s_], ends=[e_])
    path = np.array(mcp.traceback(e_))
    if len(path) < 2:
        return LineString([a, z])
    xs = T.c + (path[:, 1] + 0.5) * T.a
    ys = T.f + (path[:, 0] + 0.5) * T.e
    coords = [a] + list(zip(xs[1:-1].tolist(), ys[1:-1].tolist())) + [z]
    return LineString(coords).simplify(0.5)


def offset_stats(lines, ref, max_m=10.0, step=5.0):
    """How far the network lies from reference lines (stage B channels, council lines): over points every ``step`` m
    along the network within ``max_m`` of a reference line, the median distance and the share within 3 m; and the
    share of the network's points with a reference line within ``max_m``."""
    from shapely import STRtree, points
    if ref is None or not len(ref) or not len(lines):
        return None
    pts = []
    for g in lines.geometry:
        n = max(int(g.length // step), 1)
        pts += [g.interpolate(t).coords[0] for t in np.linspace(0, g.length, n + 1)]
    pts = points(np.array(pts))
    tree = STRtree(ref.geometry.to_numpy())
    j = tree.nearest(pts)
    d = np.array([ref.geometry.iloc[k].distance(q) for k, q in zip(j, pts)])
    near = d <= max_m
    if not near.any():
        return dict(points=int(len(d)), near_share=0.0)
    return dict(points=int(len(d)), near_share=round(float(near.mean()), 3),
                median_m=round(float(np.median(d[near])), 2), within_3m=round(float((d[near] <= 3.0).mean()), 3))


def join_path(start_xy, z_start, targets, prob_files, dem_files, roads, crossings, P):
    """The least-cost path from a drain to the network and its probability: (line or None, record)."""
    from rasterio.features import rasterize
    from shapely.geometry import LineString, Point
    from skimage.graph import MCP_Geometric

    from .catchment_mode import _mosaic
    m = P["join_max_m"] + 10.0
    x, y = start_xy
    rec = dict(start_x=float(x), start_y=float(y))
    sp = Point(x, y)
    tg = [g for g in targets if g.distance(sp) <= m]
    if not tg:
        return None, dict(rec, reason="no_network_near", p_join=0.0)
    b = (np.floor(x - m), np.floor(y - m), np.ceil(x + m), np.ceil(y + m))
    pr, T = _mosaic(prob_files, b)
    if pr is None:
        return None, dict(rec, reason="no_probability", p_join=0.0)
    p = np.where(np.isfinite(pr) & (pr <= 250), pr / 250.0, 0.0)
    shape = p.shape
    target = rasterize([(g, 1) for g in tg], out_shape=shape, transform=T, fill=0, all_touched=True) > 0
    road = np.zeros(shape, bool)
    if roads is not None and len(roads):
        road = rasterize([(g, 1) for g in roads.geometry if g is not None and not g.is_empty], out_shape=shape,
                         transform=T, fill=0, all_touched=True) > 0
    pcr = np.zeros(shape, np.float32)                      # a crossing candidate's probability where it crosses
    if crossings is not None and len(crossings):
        ps = _p_structure(crossings, P)
        shp = sorted(((g.buffer(2.0), float(v)) for g, v in zip(crossings.geometry, ps) if g.intersects(
            LineString([(b[0], b[1]), (b[2], b[3])]).envelope)), key=lambda s: s[1])
        if shp:
            pcr = rasterize(shp, out_shape=shape, transform=T, fill=0, dtype="float32")
    p_road = np.where(pcr > 0, pcr, P["road_prior"])
    cost = 1.0 + 9.0 * (1.0 - p) + np.where(road, -np.log(np.clip(p_road, 1e-3, 1.0)) * 10.0, 0.0)
    r0, c0 = int((T.f - y) // abs(T.e)), int((x - T.c) // T.a)
    tr, tc = np.nonzero(target)
    if not len(tr) or not (0 <= r0 < shape[0] and 0 <= c0 < shape[1]):
        return None, dict(rec, reason="no_network_near", p_join=0.0)
    mcp = MCP_Geometric(cost)
    cum, _ = mcp.find_costs([(r0, c0)], ends=list(zip(tr, tc)), find_all_ends=False)
    k = int(np.argmin(cum[tr, tc]))
    end = (int(tr[k]), int(tc[k]))
    path = np.array(mcp.traceback(end))
    xs = T.c + (path[:, 1] + 0.5) * T.a
    ys = T.f + (path[:, 0] + 0.5) * T.e
    ep = Point(xs[-1], ys[-1])
    near = min(tg, key=lambda g: g.distance(ep))
    q = near.interpolate(near.project(ep))
    coords = [(x, y)] + list(zip(xs[1:-1].tolist(), ys[1:-1].tolist())) + [(q.x, q.y)]
    line = LineString(coords if len(coords) >= 2 else [(x, y), (q.x, q.y)]).simplify(0.5)
    pm = float(p[path[:, 0], path[:, 1]].mean())
    on = road[path[:, 0], path[:, 1]]
    pb = float(p_road[path[:, 0], path[:, 1]][on].max()) if on.any() else 1.0
    zz, Tz = _mosaic(dem_files, b)
    rise = 0.0
    if zz is not None and np.isfinite(z_start):
        rr = ((Tz.f - ys) // abs(Tz.e)).astype(int).clip(0, zz.shape[0] - 1)
        cc = ((xs - Tz.c) // Tz.a).astype(int).clip(0, zz.shape[1] - 1)
        prof = zz[rr, cc]
        if np.isfinite(prof).any():
            rise = max(0.0, float(np.nanmax(prof) - z_start))
    if on.any() and pb > P["road_prior"]:
        rise = 0.0                                          # the rise is the embankment the structure passes
    pr_ = float(np.exp(-rise / P["rise_sigma_m"]))
    # channel evidence along the path: the U-Net's, or the gap's shortness (a gap of a few metres is most likely
    # the same channel, broken by noise; one of tens of metres needs the U-Net to see a channel there)
    pg = float(np.exp(-line.length / P["gap_scale_m"]))
    pe = 1.0 - (1.0 - pm) * (1.0 - pg)
    pj = pe * pb * pr_
    rec.update(length_m=round(line.length, 2), prob_mean=round(pm, 3), p_gap=round(pg, 3), p_evidence=round(pe, 3),
               p_barrier=round(pb, 3), rise_m=round(rise, 3),
               p_rise=round(pr_, 3), p_join=round(pj, 3), crosses_road=bool(on.any()),
               too_long=bool(line.length > P["join_max_m"]))
    if line.length > P["join_max_m"]:
        return None, dict(rec, reason="too_far")
    if pj < P["keep_p"]:
        return None, dict(rec, reason="improbable")
    return line, dict(rec, reason="", accepted=True)


# ---------------------------------------------------------------- classes and crossings

def classes(S, reaches, osm_areas, ml_cls_src, P, propagate=True):
    """Class per segment from its evidence (OSM river areas or tags, the U-Net's class, the stage B class nearby,
    a very large upstream area), then made consistent downstream (``RANK``)."""
    from shapely import STRtree
    n = len(S)
    base = np.array(["stream"] * n, object)
    src = np.array(["routed"] * n, object)
    if len(reaches):
        tree = STRtree(reaches.geometry.to_numpy())
        for k, g in enumerate(S.geometry):
            idx = tree.query(g, predicate="dwithin", distance=P["evid_dist_m"])
            if not len(idx):
                continue
            sub = reaches.iloc[idx]
            ln = sub.geometry.intersection(g.buffer(P["evid_dist_m"])).length.to_numpy()
            def vote(col, vals):
                if col not in sub:
                    return None
                v = sub[col].astype(object).to_numpy()
                best, bl = None, 0.0
                for c in vals:
                    l_ = ln[v == c].sum()
                    if l_ > bl:
                        best, bl = c, l_
                return best if bl >= 0.5 * g.length else None
            o = vote("osm_waterway", ("river", "stream", "canal", "drain"))
            mc = vote("ml_class", ("river", "stream", "drain"))
            bc = vote("class", ("stream", "drain"))
            if o == "river":
                base[k], src[k] = "river", "osm"
            elif o == "stream":
                base[k], src[k] = "stream", "osm"
            elif o in ("canal", "drain"):
                base[k], src[k] = "drain", "osm"
            elif mc is not None:
                base[k], src[k] = mc, "u-net class"
            elif bc is not None:
                base[k], src[k] = bc, "stage B class"
    if osm_areas is not None and len(osm_areas):
        t2 = STRtree(osm_areas.geometry.to_numpy())
        for k, g in enumerate(S.geometry):
            idx = t2.query(g, predicate="intersects")
            if len(idx) and sum(g.intersection(osm_areas.geometry.iloc[i]).length for i in idx) >= 0.5 * g.length:
                base[k], src[k] = "river", "osm area"
    big = S.upa_m2.to_numpy(float) >= P["river_min_m2"]
    base[big & (base != "river")] = "river"
    src[big] = np.where(src[big] == "routed", "upstream area", src[big])
    if not propagate:
        return list(base), list(src)
    # downstream consistency: a segment is at least the highest class flowing into it
    rank = np.array([RANK[c] for c in base])
    by_idx = {int(i): k for k, i in enumerate(S.idx)}
    order = np.argsort(S.upa_m2.to_numpy(float))                        # upstream first
    for k in order:
        d = by_idx.get(int(S.idx_ds.iloc[k]))
        if d is not None and rank[d] < rank[k]:
            rank[d] = rank[k]
            src[d] = "below a " + {v: n for n, v in RANK.items()}[rank[k]]
    inv = {v: k for k, v in RANK.items()}
    return [inv[r] for r in rank], list(src)


def network_crossings(lines, roads, crossings, P):
    """Points where the network meets a road, with the nearest crossing candidate (p_structure, tier, distance) or
    ``needs_structure``."""
    import geopandas as gpd
    from shapely import STRtree
    from shapely.geometry import MultiPoint, Point
    if roads is None or not len(roads) or not len(lines):
        return gpd.GeoDataFrame(geometry=[], crs=2193)
    rt = STRtree(roads.geometry.to_numpy())
    rows = []
    for k, g in enumerate(lines.geometry):
        for j in rt.query(g, predicate="intersects"):
            x = g.intersection(roads.geometry.iloc[j])
            pts = [x] if isinstance(x, Point) else list(getattr(x, "geoms", [])) if isinstance(x, MultiPoint) else \
                [x.centroid] if not x.is_empty else []
            for p in pts:
                rows.append(dict(geometry=p, line=k, cls=lines["class_v2"].iloc[k] if "class_v2" in lines else None))
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs=2193)
    X = gpd.GeoDataFrame(rows, geometry="geometry", crs=2193)
    X["p_structure"] = np.nan
    X["tier"] = None
    X["candidate_m"] = np.nan
    if crossings is not None and len(crossings):
        ct = STRtree(crossings.geometry.to_numpy())
        ps = _p_structure(crossings, P)
        for i, p in enumerate(X.geometry):
            j = ct.nearest(p)
            d = crossings.geometry.iloc[j].distance(p)
            if d <= P["crossing_search_m"]:
                X.loc[X.index[i], ["p_structure", "candidate_m"]] = [round(float(ps[j]), 3), round(float(d), 1)]
                X.loc[X.index[i], "tier"] = str(crossings.tier.iloc[j]) if "tier" in crossings else None
    X["needs_structure"] = X.p_structure.isna()
    return X


def trace(points, layers, P):
    """For each (x, y): the nearest feature of each named layer, its distance and a few attributes."""
    from shapely.geometry import Point
    out = []
    for x, y in points:
        p = Point(x, y)
        rec = dict(x=x, y=y)
        for name, g in layers.items():
            if g is None or not len(g):
                rec[name] = None
                continue
            d = g.geometry.distance(p)
            i = int(np.argmin(d.to_numpy()))
            r = g.iloc[i]
            keep = {k: (None if (isinstance(r[k], float) and np.isnan(r[k])) else r[k]) for k in
                    ("tier", "p_structure", "class", "class_v2", "channel_source", "needs_structure", "strahler")
                    if k in g.columns}
            rec[name] = dict(distance_m=round(float(d.iloc[i]), 1), **{k: (v.item() if hasattr(v, "item") else v)
                                                                       for k, v in keep.items()})
        out.append(rec)
    return out


# ---------------------------------------------------------------- version 3: mapped lines, evidence, flow paths

OSM_CLASS = dict(river="river", stream="stream", canal="drain", drain="drain", ditch="drain")


def mapped_lines(W, council, catchment_band, P):
    """The OSM waterways (``mapped_kinds``) and council channel lines meeting the band, as one table with
    ``mapped_source`` (osm / council), ``mapped_class`` (river / stream / drain; a council class is a guide to kind) and
    ``name``; long lines cut into pieces of ``mapped_piece_m`` for the redrawing."""
    import geopandas as gpd
    import pandas as pd
    from shapely.ops import substring
    rows = []
    if W is not None and len(W) and "waterway" in W:
        for g, ww, nm in zip(W.geometry, W.waterway, W.get("name", pd.Series([None] * len(W), index=W.index))):
            if ww in P["mapped_kinds"] and g is not None and g.intersects(catchment_band):
                rows.append(dict(geometry=g, mapped_source="osm", mapped_class=OSM_CLASS.get(ww, "drain"),
                                 waterway=ww, name=nm))
    if council is not None and len(council):
        cls = council["class"].astype(str) if "class" in council else pd.Series(["drain"] * len(council))
        for g, c in zip(council.geometry, cls):
            if g is not None and g.intersects(catchment_band):
                rows.append(dict(geometry=g, mapped_source="council",
                                 mapped_class="stream" if c == "stream" else "drain", waterway=None, name=None))
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs=2193)
    M = gpd.GeoDataFrame(rows, geometry="geometry", crs=2193).explode(index_parts=False)
    M = M[M.geometry.geom_type == "LineString"].reset_index(drop=True)
    out = []
    for r in M.to_dict("records"):
        g = r.pop("geometry")
        n = max(1, int(np.ceil(g.length / P["mapped_piece_m"])))
        for k in range(n):
            piece = substring(g, k * g.length / n, (k + 1) * g.length / n)
            if piece.length > 0.5:
                out.append(dict(r, geometry=piece))
    return gpd.GeoDataFrame(out, geometry="geometry", crs=2193)


def presence_evidence(R, runs, P, log=print):
    """Evidence per stage B reach and the threshold it is judged against: the channel presence model's probability
    (``presence_model``, else ``<data>/models/channel_presence/model.pkl`` when it exists) and its calibrated threshold;
    else the hand-set evidence (``reach_evidence``) and ``keep_p``."""
    import pickle
    from .acquire.elevation import default_data_dir
    f = Path(P["presence_model"]) if P.get("presence_model") else default_data_dir() / "models" / "channel_presence" / \
        "model.pkl"
    prob_files = sorted({str(Path(runs) / t / "channel_prob.tif") for t in R.tile.dropna().unique()}) if len(R) else []
    if f.exists() and len(R):
        from .ml import channel_presence as CPm
        with open(f, "rb") as h:
            M = pickle.load(h)
        p = np.zeros(len(R))
        for t, idx in R.groupby("tile", dropna=False).groups.items():
            sub = R.loc[idx]
            X = CPm.features(sub, Path(runs) / str(t) / "channel_prob.tif" if t is not None else None)
            p[R.index.get_indexer(idx)] = CPm.predict(M, X)
        return p, float(M.get("threshold", 0.5)), dict(source="model", model=str(f), threshold=M.get("threshold"))
    return reach_evidence(R, prob_files), P["keep_p"], dict(source="hand-set", threshold=P["keep_p"])


def evidence_runs(d, ind, P):
    """Pieces of a line's samples with and without evidence: [(start m, end m, has evidence)], runs shorter than
    ``split_min_m`` merged into their neighbours."""
    if not len(d):
        return []
    runs = []
    s0 = 0
    for k in range(1, len(d) + 1):
        if k == len(d) or ind[k] != ind[s0]:
            runs.append([float(d[s0]), float(d[k - 1]) if k < len(d) else float(d[-1]), bool(ind[s0])])
            s0 = k
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for j, r in enumerate(runs):
            if r[1] - r[0] < P["split_min_m"]:
                nb = j - 1 if j > 0 else j + 1
                a, b = sorted((j, nb))
                runs[a] = [runs[a][0], runs[b][1], runs[nb][2]]
                del runs[b]
                changed = True
                break
    for j in range(len(runs) - 1):                                  # contiguous
        runs[j + 1][0] = runs[j][1]
    return [tuple(r) for r in runs]


def _xs_one(g, zz, T, P):
    """(width, depth) of one line from a DEM array covering it (see ``cross_sections``)."""
    h = P["xs_half_m"]
    ws, ds = [], []
    off = np.arange(-h, h + 0.5, 1.0)
    for t in np.arange(min(P["xs_step_m"] / 2, g.length / 2), g.length, P["xs_step_m"]):
        a = g.interpolate(max(t - 1, 0)).coords[0]
        b = g.interpolate(min(t + 1, g.length)).coords[0]
        c = g.interpolate(t).coords[0]
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = np.hypot(dx, dy) or 1.0
        px, py = c[0] - dy / n * off, c[1] + dx / n * off
        r = ((T.f - py) // abs(T.e)).astype(int)
        cc = ((px - T.c) // T.a).astype(int)
        if not ((r >= 0) & (r < zz.shape[0]) & (cc >= 0) & (cc < zz.shape[1])).all():
            continue
        prof = zz[r, cc]
        if not np.isfinite(prof).all():
            continue
        mid = len(off) // 2
        bed = prof[mid - 2:mid + 3].min()
        dep = min(prof[:mid - 1].max(), prof[mid + 2:].max()) - bed
        if dep <= 0.05:
            ws.append(0.0)
            ds.append(max(dep, 0.0))
            continue
        half = bed + dep / 2
        li = np.flatnonzero(prof[:mid] >= half)
        ri = np.flatnonzero(prof[mid:] >= half)
        ws.append(float(((mid - li[-1]) if len(li) else mid) + (ri[0] if len(ri) else len(off) - mid)))
        ds.append(float(dep))
    return (float(np.median(ws)) if ws else np.nan, float(np.median(ds)) if ds else np.nan)


def cross_sections(lines, tiles, cores, bundles, P):
    """(width m, depth m) per line from the 1 m DEM: profiles across it every ``xs_step_m``, ``xs_half_m`` either side;
    depth = the lower bank's top above the bed, width = between where the profile rises halfway up; medians. Each
    tile's DEM (core and buffer) is read once, for the lines whose middle lies in its core."""
    import rasterio
    from affine import Affine  # noqa: F401
    from shapely import STRtree, points
    out = [(np.nan, np.nan)] * len(lines)
    if not len(lines):
        return out
    names = list(tiles)
    tree = STRtree([cores[t].buffer(0.01) for t in names])
    mids = points([g.interpolate(0.5, normalized=True).coords[0] for g in lines])
    li, ti = tree.query(mids, predicate="within")
    own = {}
    for a_, b_ in zip(li, ti):
        own.setdefault(int(b_), []).append(int(a_))
    for k, idx in own.items():
        f = Path(bundles) / names[k] / "dem.tif"
        if not f.exists():
            continue
        with rasterio.open(f) as s:
            zz = s.read(1).astype(np.float32)
            if s.nodata is not None:
                zz[zz == s.nodata] = np.nan
            T = s.transform
        for i in dict.fromkeys(idx):
            out[i] = _xs_one(lines[i], zz, T, P) if lines[i] is not None and lines[i].length >= 2 else (np.nan, np.nan)
        del zz
    return out


def classes_v3(N, seg_down, P):
    """Classes made consistent downstream, through flow paths: a channel is at least the highest channel class that
    flows into it, directly or through flow paths; flow paths keep their class. ``seg_down``: index of each row's
    downstream row (-1: none). Returns (classes, sources)."""
    cls = list(N.class_v2)
    src = list(N.class_source)
    order = np.argsort(N.upa_m2.fillna(0).to_numpy(float))
    carry = np.full(len(N), -1)
    rank = np.array([RANK.get(c, -1) for c in cls])
    for k in order:
        eff = rank[k] if cls[k] != "flow path" else carry[k]
        eff = max(eff, carry[k]) if cls[k] != "flow path" else eff
        d = seg_down[k]
        if d < 0:
            continue
        carry[d] = max(carry[d], eff)
        if cls[d] != "flow path" and carry[d] > rank[d]:
            inv = {v: n for n, v in RANK.items()}
            src[d] = "below a " + inv[carry[d]]
            rank[d] = carry[d]
            cls[d] = inv[carry[d]]
    return cls, src


def run_catchment(catchment_dir, runs, bundles, tiles=None, params=None, trace_points=None, write=True, log=print):
    """Stage A (version 3, 29 Oct) on one catchment folder (``catchment.gpkg``); returns the summary (``stage_a.json``).

    Additive and refining (Matt, 29 Oct): every OSM waterway and council channel line is represented (redrawn on the
    LiDAR, burned into the routing, added where the routing misses it, never removed); detected channels (stage B) are
    added where their evidence reaches the threshold (the channel presence model's, calibrated on the reviews; else
    hand-set); routing gives connectivity, direction, upstream area, order and the one outlet, and makes no channel of
    its own: a routed path without evidence is kept only to carry evidence-backed network to the outlet, as a ``flow
    path``. Heads end where the evidence ends. Every reach gets width and depth (stage B's where it follows a detected
    reach, else from cross-sections of the 1 m DEM)."""
    import geopandas as gpd
    import pandas as pd
    from rasterio.features import rasterize
    from shapely import STRtree, points
    from shapely.ops import substring

    from . import guidance as G
    from .catchment_mode import _layer, assemble, sample_z
    P = dict(DEFAULTS, **(params or {}))
    cd = Path(catchment_dir)
    cat = gpd.read_file(cd / "catchment.gpkg", layer="catchment").to_crs(2193)
    catchment = cat.geometry.iloc[0]
    if tiles is None:
        tiles = list(gpd.read_file(cd / "catchment.gpkg", layer="tiles").tile_id)
    tiles = [t for t in tiles if (Path(runs) / t / "network.gpkg").exists()]
    band = catchment.buffer(P["overlap_m"])
    R, cores, cross, rivers, roads = assemble(tiles, runs, bundles, band, dict(min_piece_m=0.5))
    prob_files = [Path(runs) / t / "channel_prob.tif" for t in tiles]
    dem_files = [Path(bundles) / t / "dem.tif" for t in tiles]
    ev, thr, ev_info = presence_evidence(R, runs, P) if len(R) else (np.zeros(0), P["keep_p"], dict(source="none"))
    log(f"{len(tiles)} tiles; {len(R)} stage B pieces ({R.length.sum() / 1000:.1f} km); {len(cross)} crossing candidates;"
        f" evidence {ev_info['source']}, threshold {thr:.3f}")
    sea, areas, wlines, council = [], [], [], []
    for t in tiles:
        g = G.load(Path(bundles) / t / "context.gpkg", tuple(cores[t].buffer(500).bounds))
        if len(g["lines"]):
            wlines.append(g["lines"][g["lines"].intersects(cores[t])])
        lab = _layer(Path(bundles) / t / "labels.gpkg", "channels")
        if lab is not None and len(lab):
            keep = [c for c in ("class",) if c in lab]
            council.append(lab[lab.intersects(cores[t])][keep + ["geometry"]])
        a = g["areas"]
        if len(a) and "water_kind" in a:
            sea += [x for x in a[a.water_kind == "tidal"].geometry if x is not None]
            areas += [x for x in a[a.water_kind == "river_area"].geometry if x is not None]
        sea += [x.buffer(1.0) for x in g["coastline"].geometry if x is not None]
    osm_areas = gpd.GeoDataFrame(geometry=areas, crs=2193)
    W = gpd.GeoDataFrame(pd.concat(wlines, ignore_index=True), geometry="geometry", crs=2193) if wlines else None
    CL = gpd.GeoDataFrame(pd.concat(council, ignore_index=True), geometry="geometry", crs=2193) if council else None
    # 1. the mapped lines, redrawn on the LiDAR (towards detected channels and the bed)
    Mp = mapped_lines(W, CL, band, P)
    if len(Mp):
        Pm = dict(P, refine_corridor_m=P["mapped_corridor_m"])
        Mp["geometry"] = [refine(g, prob_files, Pm, dem_files, R, ev) for g in Mp.geometry]
    log(f"mapped lines: {len(Mp)} pieces, {Mp.length.sum() / 1000:.1f} km (OSM "
        f"{Mp[Mp.mapped_source == 'osm'].length.sum() / 1000 if len(Mp) else 0:.1f}, council "
        f"{Mp[Mp.mapped_source == 'council'].length.sum() / 1000 if len(Mp) else 0:.1f})")
    # evidence lines for the DEM, the samples and the redrawing: mapped lines (1) and detected reaches (their evidence)
    E = gpd.GeoDataFrame(pd.concat([R[["geometry"]], Mp[["geometry"]]], ignore_index=True), geometry="geometry",
                         crs=2193) if len(Mp) else R[["geometry"]].copy()
    ev_all = np.r_[ev, np.ones(len(Mp))]
    ev_burn = np.r_[ev, np.full(len(Mp), P["burn_mapped_m"] / P["burn_max_m"])]
    # 2. the conditioned coarse DEM, the outlet, the routing
    T5, shape5 = grid_for(catchment, P["res_m"])
    z5 = coarse_dem(tiles, cores, bundles, E, ev_burn, cross, T5, shape5, P)
    cmask = rasterize([(catchment, 1)], out_shape=shape5, transform=T5, fill=0) > 0
    sea_mask = rasterize([(g, 1) for g in sea], out_shape=shape5, transform=T5, fill=0, all_touched=True) > 0 if sea \
        else np.zeros(shape5, bool)
    z5[~cmask | sea_mask] = NODATA
    if not (z5 != NODATA).any():
        raise RuntimeError("no DEM inside the catchment")
    z5 = breach(z5, P)
    oxy, orec = osm_outlet(catchment, W, sea, z5, T5, P)
    outlet, orep = choose_outlet(z5, T5, sea_mask, P, oxy, orec)
    log(f"outlet ({orep['chosen']['method']}) at {orep['chosen']['x']:.0f}, {orep['chosen']['y']:.0f} "
        f"({orep['chosen'].get('waterway') or ''}; beside the sea: {orep['chosen']['coast']})")
    flw = route(z5, T5, outlet)
    S = segments(flw, T5, cmask, P)
    upa = flw.upstream_area("m2")
    # 3. evidence along each segment: a mapped line or a detected reach at or above the threshold within evid_dist
    Pe = dict(P, keep_p=thr)
    smp = segment_samples(S, E, np.where(ev_all >= thr, 1.0, 0.0), [], upa, T5, P) if len(S) else []
    share = np.array([float((x[1] >= 0.5).mean()) if len(x[1]) else 0.0 for x in smp])
    keep = prune_heads(S, np.where(share >= P["evidence_share"], 1.0, 0.0), dict(P, keep_p=0.5)) if len(S) else \
        np.zeros(0, bool)
    S = S.assign(evidence_share=np.round(share, 3))
    # heads trimmed to where the evidence starts
    has_up = set(int(j) for j, k_ in zip(S.idx_ds, keep) if k_)
    geoms = list(S.geometry)
    pruned = [S.iloc[np.flatnonzero(~keep)][["geometry"]]] if (~keep).any() else []
    for k in np.flatnonzero(keep):
        if int(S.idx.iloc[k]) in has_up:
            continue
        d, e, _ = smp[k]
        g2 = trim_head(geoms[k], (d, e, np.full(len(d), 1.0)), dict(Pe, keep_p=0.5, a50_m2=1e30))
        if g2 is None:
            keep[k] = False
            pruned.append(gpd.GeoDataFrame(geometry=[geoms[k]], crs=2193))
            continue
        if g2.length < geoms[k].length - 1e-6:
            pruned.append(gpd.GeoDataFrame(geometry=[substring(geoms[k], 0, geoms[k].length - g2.length)], crs=2193))
            cut = geoms[k].length - g2.length
            dd, ee, aa = smp[k]
            sel = dd >= cut
            smp[k] = (dd[sel] - cut, ee[sel], aa[sel])
            geoms[k] = g2
    S = S.set_geometry(gpd.GeoSeries(geoms, crs=2193))
    smp = [smp[k] for k in np.flatnonzero(keep)]
    S = S[keep].reset_index(drop=True)
    pruned = gpd.GeoDataFrame(pd.concat(pruned, ignore_index=True), geometry="geometry", crs=2193) if pruned else \
        gpd.GeoDataFrame(geometry=[], crs=2193)
    # 4. redrawn on the 1 m channels (mapped lines and detected reaches pull, then the bed)
    raw = S[["geometry"]].copy()
    nodes_xy = snap_nodes(S, prob_files, dem_files, E, ev_all, P) if len(S) else {}
    key = lambda c: (round(c[0], 2), round(c[1], 2))
    S["geometry"] = [refine(g, prob_files, P, dem_files, E, ev_all, (nodes_xy.get(key(g.coords[0]), g.coords[0]),
                                                                    nodes_xy.get(key(g.coords[-1]), g.coords[-1])))
                     for g in S.geometry]
    # 5. channel or flow path, piece by piece along each segment
    rows = []
    for k, r in enumerate(S.itertuples()):
        d, e, _ = smp[k]
        L0 = raw.geometry.iloc[k].length or 1.0
        g = r.geometry
        for a_, b_, has in evidence_runs(d, e >= 0.5, P) or [(0.0, L0, False)]:
            piece = substring(g, a_ / L0 * g.length, b_ / L0 * g.length)
            if piece.length <= 0.5:
                continue
            rows.append(dict(geometry=piece, seg=k, idx=r.idx, idx_ds=r.idx_ds, upa_m2=r.upa_m2, strahler=r.strahler,
                             evidence_share=r.evidence_share, channel=bool(has)))
    N = gpd.GeoDataFrame(rows, geometry="geometry", crs=2193) if rows else gpd.GeoDataFrame(geometry=[], crs=2193)
    # classes from the mapped lines and detected reaches along each piece
    if len(N):
        tmp = N.assign(idx=N.idx, idx_ds=N.idx_ds)
        Rc = R.copy()
        if len(Mp):
            Rm = Mp.rename(columns={"mapped_class": "class"})[["geometry", "class"]].assign(osm_waterway=None)
            Rm.loc[Mp.mapped_source.to_numpy() == "osm", "osm_waterway"] = Mp.waterway[Mp.mapped_source == "osm"].to_numpy()
            Rc = gpd.GeoDataFrame(pd.concat([Rm, Rc], ignore_index=True), geometry="geometry", crs=2193)
        base, src = classes(tmp, Rc, osm_areas, None, dict(P, river_min_m2=np.inf), propagate=False)
        base = [b if ch else "flow path" for b, ch in zip(base, N.channel)]
        src = [s_ if ch else "no evidence (connects)" for s_, ch in zip(src, N.channel)]
        N["class_v2"], N["class_source"] = base, src
        N["channel_source"] = np.where(N.channel, "routed", "flow path")
    # 6. mapped lines the network misses, added (never removed)
    added = []
    if len(Mp) and len(N):
        nt = STRtree(N.geometry.to_numpy())
        for r in Mp.itertuples():
            g = r.geometry
            n = max(int(g.length // 5.0), 1)
            ds = np.linspace(0, g.length, n + 1)
            pts = points([g.interpolate(t).coords[0] for t in ds])
            j = nt.nearest(pts)
            far = np.array([N.geometry.iloc[x].distance(q) for x, q in zip(j, pts)]) > P["evid_dist_m"]
            for a_, b_, f in evidence_runs(ds, far, P):
                if f and b_ - a_ >= P["split_min_m"]:
                    added.append(dict(geometry=substring(g, a_, b_), class_v2=r.mapped_class,
                                      class_source=r.mapped_source, channel_source=r.mapped_source, channel=True,
                                      name=r.name))
    # 7. detected reaches the network misses: added where their evidence reaches the threshold and they can join
    joins, attempts, detected = [], [], []
    if len(R) and len(N):
        nt = STRtree(N.geometry.to_numpy())
        near = np.zeros(len(R), bool)
        for i, g in enumerate(R.geometry):
            idx = nt.query(g, predicate="dwithin", distance=P["evid_dist_m"])
            if len(idx):
                cov = sum(g.intersection(N.geometry.iloc[j].buffer(P["evid_dist_m"])).length for j in idx)
                near[i] = cov >= 0.5 * g.length
        cand = np.flatnonzero(~near & (ev >= thr) & R.intersects(catchment).to_numpy())
        from .catchment_mode import nodes
        from .connectivity import _UF
        if len(cand):
            geoms = [R.geometry.iloc[i] for i in cand]
            ends, xy = nodes(geoms, 0.5)
            uf = _UF(len(xy))
            for a_, b_ in ends:
                uf.union(a_, b_)
            lab = uf.labels()[ends[:, 0]]
            targets = list(N.geometry) + [x["geometry"] for x in added]
            z = sample_z(xy, tiles, cores, bundles)
            for c in np.unique(lab):
                mem = np.flatnonzero(lab == c)
                ln = float(sum(geoms[m].length for m in mem))
                if ln < P["min_piece_m"]:
                    continue
                nn = np.unique(ends[mem].ravel())
                dn = np.array([min(t.distance(q) for t in targets) for q in points(xy[nn])])
                zz = np.round(np.where(np.isfinite(z[nn]), z[nn], np.inf), 1)
                st_ = nn[np.lexsort((zz, dn))[0]]
                line, rec = join_path(xy[st_], z[st_], targets, prob_files, dem_files, roads, cross, P)
                rec["piece_m"] = round(ln, 1)
                attempts.append(rec)
                if line is not None:
                    joins.append(dict(geometry=line, class_v2="flow path", class_source="join", channel_source="join",
                                      channel=False))
                    sub = R.iloc[cand[mem]]
                    for g, cl in zip(sub.geometry, sub.get("class", pd.Series(["drain"] * len(sub))).astype(str)):
                        detected.append(dict(geometry=g, class_v2="stream" if cl == "stream" else "drain",
                                             class_source="detected", channel_source="detected", channel=True))
    parts = [N] + [gpd.GeoDataFrame(x, geometry="geometry", crs=2193) for x in (added, detected, joins) if x]
    N = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=2193)
    # 8. classes consistent downstream (routed pieces, through flow paths)
    if "seg" in N:
        idx_of_seg = {}
        for i_, (sg, a_) in enumerate(zip(N.seg, N.geometry)):
            if pd.notna(sg):
                idx_of_seg.setdefault(int(sg), []).append(i_)
        down = np.full(len(N), -1)
        first = {int(i): k for k, i in enumerate(S.idx)}
        for sg, rows_ in idx_of_seg.items():                            # a segment's pieces are in flow order
            for a_, b_ in zip(rows_[:-1], rows_[1:]):
                down[a_] = b_
            d_seg = first.get(int(S.idx_ds.iloc[sg]))
            if d_seg is not None and d_seg in idx_of_seg:
                down[rows_[-1]] = idx_of_seg[d_seg][0]
        N["upa_m2"] = N.get("upa_m2", pd.Series(np.nan, index=N.index))
        N["class_v2"], N["class_source"] = classes_v3(N, down, P)
    # 9. width and depth: stage B's where a detected reach runs along, else cross-sections
    N["width_m"] = np.nan
    N["depth_m"] = np.nan
    N["width_source"] = None
    if len(R) and "width_m" in R:
        rt = STRtree(R.geometry.to_numpy())
        for i, g in enumerate(N.geometry):
            idx = rt.query(g, predicate="dwithin", distance=3.0)
            if not len(idx):
                continue
            ln = np.array([g.intersection(R.geometry.iloc[j].buffer(3.0)).length for j in idx])
            if ln.sum() >= 0.5 * g.length:
                w_ = R.width_m.iloc[idx].to_numpy(float)
                d_ = R.incision_m.iloc[idx].to_numpy(float) if "incision_m" in R else np.full(len(idx), np.nan)
                ok = np.isfinite(w_)
                if ok.any():
                    N.loc[N.index[i], ["width_m", "depth_m", "width_source"]] = [
                        float(np.average(w_[ok], weights=ln[ok])),
                        float(np.nanmean(d_)) if np.isfinite(d_).any() else np.nan, "stage B"]
    rest = np.flatnonzero(N.width_source.isna().to_numpy())
    if len(rest):
        wd = cross_sections([N.geometry.iloc[i] for i in rest], tiles, cores, bundles, P)
        N.loc[N.index[rest], "width_m"] = [w for w, _ in wd]
        N.loc[N.index[rest], "depth_m"] = [d_ for _, d_ in wd]
        N.loc[N.index[rest], "width_source"] = "cross-sections"
    # upstream area for every piece (largest along it)
    h, w = upa.shape
    ua = []
    for g in N.geometry:
        c = np.array([g.interpolate(t).coords[0] for t in np.linspace(0, g.length, max(int(g.length // 5), 1) + 1)])
        r = ((T5.f - c[:, 1]) // abs(T5.e)).astype(int).clip(0, h - 1)
        q = ((c[:, 0] - T5.c) // T5.a).astype(int).clip(0, w - 1)
        ua.append(float(upa[r, q].max()))
    N["upa_m2"] = np.where(N.upa_m2.notna(), N.upa_m2, ua) if "upa_m2" in N else ua
    X = network_crossings(N, roads, cross, P)
    # coverage of the mapped lines, offsets
    def coverage(M_, net):
        if not len(M_) or not len(net):
            return None
        tt = STRtree(net.geometry.to_numpy())
        tot = cov = 0.0
        for g in M_.geometry:
            n = max(int(g.length // 5.0), 1)
            pts = points([g.interpolate(t).coords[0] for t in np.linspace(0, g.length, n + 1)])
            j = tt.nearest(pts)
            dd = np.array([net.geometry.iloc[x].distance(q) for x, q in zip(j, pts)])
            tot += g.length
            cov += g.length * float((dd <= P["evid_dist_m"]).mean())
        return round(cov / tot, 4) if tot else None
    km = lambda g: round(float(g.length.sum()) / 1000, 3)
    inside = N[N.intersects(catchment)]
    summary = dict(version=3, catchment=str(cat.catchment_id.iloc[0]) if "catchment_id" in cat else cd.name,
                   tiles=len(tiles), params={k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()},
                   evidence=ev_info, outlet=orep,
                   offsets=dict(stage_b=offset_stats(N[N.channel_source == "routed"], R),
                                council=offset_stats(N[N.channel_source == "routed"], CL)),
                   mapped=dict(pieces=int(len(Mp)), km=km(Mp), covered=coverage(Mp, N),
                               osm_km=km(Mp[Mp.mapped_source == "osm"]) if len(Mp) else 0.0,
                               council_km=km(Mp[Mp.mapped_source == "council"]) if len(Mp) else 0.0,
                               added_km=km(gpd.GeoDataFrame(added, geometry="geometry", crs=2193)) if added else 0.0),
                   detected=dict(tried=len(attempts), joined=len(joins),
                                 added_km=km(gpd.GeoDataFrame(detected, geometry="geometry", crs=2193)) if detected else 0.0,
                                 reasons={k: int(v) for k, v in pd.Series([a.get("reason") or "accepted" for a in attempts])
                                          .value_counts().items()} if attempts else {}),
                   pruned_km=km(pruned), network_km=km(N), network_km_in_catchment=km(inside),
                   class_km={k: km(N[N.class_v2 == k]) for k in ("river", "stream", "drain", "flow path")},
                   class_sources={k: int(v) for k, v in N.class_source.value_counts().items()},
                   strahler_km={int(k): km(N[N.strahler == k]) for k in sorted(N.strahler.dropna().unique())}
                   if "strahler" in N else {},
                   crossings=dict(total=int(len(X)), with_candidate=int((~X.needs_structure).sum()) if len(X) else 0,
                                  needs_structure=int(X.needs_structure.sum()) if len(X) else 0,
                                  needs_structure_on_flow_paths=int((X.needs_structure & (X.cls == "flow path")).sum())
                                  if len(X) else 0))
    if trace_points:
        summary["trace"] = trace(trace_points, dict(candidates=cross, stage_b=R, network=N, crossings=X), P)
    if write:
        f = cd / "network_a.gpkg"
        if f.exists():
            f.unlink()
        N.to_file(f, layer="channels", driver="GPKG")
        if len(Mp):
            Mp.to_file(f, layer="mapped", driver="GPKG")
        if len(pruned):
            pruned.to_file(f, layer="pruned", driver="GPKG")
        if len(X):
            X.to_file(f, layer="crossings", driver="GPKG")
        if attempts:
            A_ = pd.DataFrame(attempts)
            gpd.GeoDataFrame(A_, geometry=gpd.points_from_xy(A_.start_x, A_.start_y), crs=2193).to_file(
                f, layer="joins", driver="GPKG")
        o = orep["chosen"]
        gpd.GeoDataFrame([dict(kind="outlet", method=o.get("method"))], geometry=gpd.points_from_xy([o["x"]], [o["y"]]),
                         crs=2193).to_file(f, layer="outlet", driver="GPKG")
        import rasterio
        with rasterio.open(cd / "dem_routed.tif", "w", driver="GTiff", height=shape5[0], width=shape5[1], count=1,
                           dtype="float32", crs="EPSG:2193", transform=T5, nodata=NODATA, compress="deflate") as w_:
            w_.write(z5, 1)
        (cd / "stage_a.json").write_text(json.dumps(summary, indent=1, default=str))
        mid = N.geometry.interpolate(0.5, normalized=True)
        for t in tiles:
            sel = N[mid.within(cores[t].buffer(0.01))]
            tf = Path(runs) / t / "network_a.gpkg"
            if tf.exists():
                tf.unlink()
            if len(sel):
                sel.to_file(tf, layer="channels", driver="GPKG")
    return summary
