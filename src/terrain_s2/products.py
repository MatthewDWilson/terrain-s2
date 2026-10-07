"""Network deliverables (Oct 2026 revision, DESIGN_NOTES §4i).

* ``channels``: the drain/stream network as lines (one per branch), with per-segment features for
  stream/drain classification and a channel-evidence score.
* ``crossings``: culvert candidates with their evidence and a confidence tier: DEM/DSM evidence
  (drain-bed rise at gaps and bumps) and, where road centrelines are supplied, road-drain
  intersections and drains that stop at a road with a drain opposite. Every crossing also keeps a
  DEM-only tier, so the value of the road layer can be measured.
* ``repairs``: continuity joins (gaps without a crossing), for DEM conditioning.

Roads are optional context: they add hypotheses and evidence, never replace the DEM evidence.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

from . import drains

ROAD_NEAR_M = 15.0          # a crossing is "on a road" if within this distance of a centreline
                            # (roadside drains sit ~8-10 m from it in a ~20 m road reserve)
ROADSIDE_M = 20.0           # a drain is a roadside drain if it runs within this distance of a road


def branch_lines(sk, T):
    """Branches as (LineString, cell array) pairs, each extended to touching junction cells."""
    from shapely.geometry import LineString
    k = np.ones((3, 3)); k[1, 1] = 0
    nb = ndi.convolve(sk.astype(np.int16), k, mode="constant")
    junction = sk & (nb >= 3)
    out = []
    for br in drains.branches(sk):
        if len(br) < 2:
            continue
        cells = [tuple(c) for c in br]
        for end, ins in ((cells[0], 0), (cells[-1], None)):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    r, c = end[0] + dr, end[1] + dc
                    if (dr or dc) and 0 <= r < sk.shape[0] and 0 <= c < sk.shape[1] and junction[r, c]:
                        if ins == 0:
                            cells.insert(0, (r, c))
                        else:
                            cells.append((r, c))
                        break
                else:
                    continue
                break
        arr = np.array(cells)
        x, y = drains._to_xy(T, arr[:, 1] + 0.5, arr[:, 0] + 0.5)
        out.append((LineString(np.column_stack([x, y])).simplify(0.5), arr))
    return out


def segment_features(lines, mask, feats, upa, z, T, roads=None):
    """Per-segment features for stream/drain classification and channel evidence."""
    import pandas as pd
    cs = abs(T.a)
    half_w = ndi.distance_transform_edt(mask) * cs          # half-width at the centreline
    road_dist = road_dir = None
    if roads is not None and len(roads):
        road_dist, road_dir = _road_rasters(roads, mask.shape, T)
    rows = []
    for geom, cells in lines:
        r, c = cells[:, 0], cells[:, 1]
        L = geom.length
        p0, p1 = np.array(geom.coords[0]), np.array(geom.coords[-1])
        chord = float(np.hypot(*(p1 - p0)))
        coords = np.array(geom.coords)
        seg = np.diff(coords, axis=0)
        ang = np.arctan2(seg[:, 1], seg[:, 0])
        turn = np.abs(np.angle(np.exp(1j * np.diff(ang)))) if len(ang) > 1 else np.zeros(1)
        main = math.atan2(p1[1] - p0[1], p1[0] - p0[0]) if chord > 0 else 0.0
        row = dict(length_m=L, straightness=chord / L if L > 0 else 1.0,
                   turn_per_100m=float(turn.sum() / max(L, 1) * 100),
                   width_m=float(np.median(2 * half_w[r, c])),
                   incision_m=float(np.nanmedian(np.fmax(feats["tophat_black11"][r, c], feats["tophat_black21"][r, c]))),
                   valley=float(np.nanmedian(feats["valley_s2"][r, c])),
                   log_upa=float(np.log10(1 + np.nanmax(upa[r, c]))),
                   bed_slope=float(abs(z[r[0], c[0]] - z[r[-1], c[-1]]) / max(L, 1)),
                   orient_mod90=float(abs(((math.degrees(main) % 90) + 45) % 90 - 45)))
        if road_dist is not None:
            near = road_dist[r, c] <= ROADSIDE_M
            par = np.abs(np.cos(road_dir[r, c] - main)) >= math.cos(math.radians(20))
            row["roadside_frac"] = float(np.mean(near & par))
        rows.append(row)
    F = pd.DataFrame(rows)
    # channel evidence: clearer incision and longer, connected segments are more certain
    F["channel_evidence"] = 1 / (1 + np.exp(-(4 * (F.incision_m - 0.25) + 0.02 * (F.length_m - 30))))
    return F


def _road_rasters(roads, shape, T):
    """Distance to the nearest road centreline (m) and that road's local direction (rad)."""
    from rasterio.features import rasterize
    cs = abs(T.a)
    burn = rasterize([(g, 1) for g in roads.geometry], out_shape=shape, transform=T, all_touched=True).astype(bool)
    dist, (ir, ic) = ndi.distance_transform_edt(~burn, return_indices=True)
    # local road direction from the rasterised line's structure tensor
    from .channels import skeleton_orientation
    ang = skeleton_orientation(burn, max(2, int(round(6 / cs))))
    return dist * cs, np.nan_to_num(ang[ir, ic])


def classify_streams(F, model=None):
    """Stream/drain probability per segment. With no trained model, a transparent prior is used:
    streams are sinuous, less aligned to the field grid, carry more upstream area."""
    if model is not None:
        cols = model.feature_names_in_
        p = model.predict_proba(F[list(cols)].fillna(0))[:, 1]
    else:
        z = (2.5 * (1 - F.straightness) * 10 + 0.02 * F.turn_per_100m + 1.2 * (F.log_upa - 4)
             + 0.05 * (F.orient_mod90 - 20) - 1.0)
        p = 1 / (1 + np.exp(-z))
    return np.asarray(p)


def crossings(net, z, T, dsm=None, roads=None):
    """Culvert candidates with evidence and tiers. ``net`` from drains.build_drain_network."""
    import pandas as pd
    from shapely.geometry import LineString, Point
    rows = []
    for g in net["gaps"]:
        if g["kind"] == "continuity":
            continue
        rows.append(dict(x=g["x"], y=g["y"], source="gap", dem_h_b=g["rise"], veg_frac=g["veg_frac"],
                         gap_m=g["gap_m"]))
    for b in net["bumps"]:
        rows.append(dict(x=b["x"], y=b["y"], source="bump", dem_h_b=b["h_b"], veg_frac=b.get("veg_frac", np.nan),
                         gap_m=b["length"]))
    C = pd.DataFrame(rows, columns=["x", "y", "source", "dem_h_b", "veg_frac", "gap_m"])
    if roads is not None and len(roads):
        lines = branch_lines(net["skeleton"], T)
        R = roads.union_all()
        extra = []
        for geom, _ in lines:                               # road-drain intersections
            inter = geom.intersection(R)
            for p in getattr(inter, "geoms", [inter]):
                if p.is_empty or p.geom_type != "Point":
                    continue
                h = _bed_bump_at(geom, p, z, T)
                extra.append(dict(x=p.x, y=p.y, source="road_x_drain", dem_h_b=h, veg_frac=np.nan, gap_m=np.nan))
        extra += _road_terminating_pairs(lines, roads, z, T)
        if extra:
            C = pd.concat([C, pd.DataFrame(extra)], ignore_index=True)
    return C


def finalise_crossings(C, use_roads):
    """Merge candidates within 8 m (keeping all sources and the strongest evidence) and tier them."""
    if "road_dist_m" not in C:
        C["road_dist_m"] = np.nan
    C = _merge_points(C, 8.0)
    C["tier_dem_only"] = [_tier(r, use_roads=False) for r in C.itertuples()]
    C["tier"] = [_tier(r, use_roads=use_roads) for r in C.itertuples()]
    return C


def _bed_bump_at(line, p, z, T, half=15.0, crest=6.0):
    """Bed rise at point p along a drain line: max bed within +-crest m above the higher of the
    reference beds beyond (h_b along the drain); NaN if the line is too short either side."""
    s0 = line.project(p)
    ss = np.arange(max(0, s0 - half), min(line.length, s0 + half) + 0.5, 1.0)
    if len(ss) < 5 or s0 - ss[0] < 4 or ss[-1] - s0 < 4:
        return float("nan")
    pts = np.array([line.interpolate(s).coords[0] for s in ss])
    cc, rr = drains._to_cr(T, pts[:, 0], pts[:, 1])
    bed = ndi.minimum_filter(np.nan_to_num(z, nan=np.inf), size=3)[rr.astype(int), cc.astype(int)]
    mid = np.abs(ss - s0) <= crest
    lo, hi = bed[(ss < s0) & ~mid], bed[(ss > s0) & ~mid]
    if len(lo) == 0 or len(hi) == 0:
        return float("nan")
    return float(bed[mid].max() - max(lo.min(), hi.min()))


def _road_terminating_pairs(lines, roads, z, T, near=15.0, across=40.0):
    """A drain end within ``near`` m of a road, pointing at it, with another drain end within
    ``across`` m on the far side: a road culvert hypothesis even if the DEM shows no gap."""
    from shapely.geometry import LineString, Point
    R = roads.union_all()
    ends = []
    for geom, _ in lines:
        cs = np.array(geom.coords)
        for a, b in ((cs[0], cs[min(5, len(cs) - 1)]), (cs[-1], cs[max(-6, -len(cs))])):
            d = a - b
            n = np.hypot(*d)
            if n == 0:
                continue
            if R.distance(Point(a)) <= near:
                ends.append((a, d / n))
    out = []
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            a, da = ends[i]
            b, db = ends[j]
            v = b - a
            L = np.hypot(*v)
            if not (2 < L <= across):
                continue
            u = v / L
            if da @ u >= 0.7 and db @ (-u) >= 0.7 and LineString([a, b]).intersects(R):
                rise, _ = drains.gap_bed_rise(z, T, tuple(a), tuple(b))
                out.append(dict(x=(a[0] + b[0]) / 2, y=(a[1] + b[1]) / 2, source="road_end_pair",
                                dem_h_b=rise, veg_frac=np.nan, gap_m=L))
    return out


def _merge_points(C, dist):
    """Merge candidates within ``dist`` m, keeping all sources and the strongest DEM evidence."""
    import pandas as pd
    from scipy.spatial import cKDTree
    if len(C) < 2:
        C["sources"] = C["source"]
        return C
    t = cKDTree(C[["x", "y"]].values)
    seen, rows = set(), []
    order = np.argsort(-np.nan_to_num(C.dem_h_b.values, nan=-1))
    for i in order:
        if i in seen:
            continue
        grp = [j for j in t.query_ball_point(C.loc[i, ["x", "y"]].values, dist) if j not in seen]
        seen.update(grp)
        r = C.loc[i].to_dict()
        r["sources"] = "+".join(sorted(set(C.loc[grp, "source"])))
        r["road_dist_m"] = float(np.nanmin(C.loc[grp, "road_dist_m"])) if C.loc[grp, "road_dist_m"].notna().any() else np.nan
        for col in ("deck_m",):
            if col in C:
                r[col] = float(np.nanmax(C.loc[grp, col])) if C.loc[grp, col].notna().any() else np.nan
        for col in ("on_stream", "network_link"):
            if col in C:
                r[col] = bool(C.loc[grp, col].fillna(False).astype(bool).any())
        rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True)


def _tier(r, use_roads):
    """Confidence tier from a points score over evidence that separated labelled culverts from
    non-culverts at four sites (DESIGN_NOTES §4j): within 15 m of a road (2; 84% of culverts vs 25%
    of non-culverts), network evidence: a drain gap, a bed bump or a Test A link accepted into the
    network (2; gap 47% vs 2%), two or more independent sources (1; 71% vs 21%), a road-drain source
    (1). High >= 3, medium >= 2; both need a barrier >= 0.3 m and no vegetation flag. Bridges first.
    The DEM-only tier drops the road terms."""
    h = r.dem_h_b if np.isfinite(r.dem_h_b) else 0.0
    veg = (r.veg_frac if np.isfinite(r.veg_frac) else 0.0) >= 0.5
    src = str(r.sources)
    deck = getattr(r, "deck_m", np.nan)
    on_road = use_roads and np.isfinite(r.road_dist_m) and r.road_dist_m <= ROAD_NEAR_M
    road_src = use_roads and any(s in src for s in ("road_x_drain", "road_end_pair"))
    net = any(s in src for s in ("gap", "bump")) or bool(getattr(r, "network_link", False))
    n_src = sum(s in src for s in ("testA", "testB", "gap", "bump", "road_x_drain", "road_end_pair"))
    if (road_src or on_road) and h < 0.3 and np.isfinite(deck) and deck >= 2.0:
        return "bridge"                    # continuous bed under a deck seen in the DSM
    if road_src and h < 0.15:
        return "bridge_or_open"
    score = 2 * net + (n_src >= 2) + (2 * on_road + road_src if use_roads else 0)
    if h >= 0.3 and not veg:
        if score >= 3:
            return "high"
        if score >= 2:
            return "medium"
    return "low"


def merge_test_candidates(C, cands):
    """Add Test A/B candidates (run_aoi output records) to the network crossings."""
    import pandas as pd
    rows = []
    for r in cands:
        src = {"A": "testA", "B": "testB", "A+B": "testA+testB"}[r["test"]]
        rows.append(dict(x=r["crest_x"], y=r["crest_y"], source=src, dem_h_b=r["h_b"],
                         veg_frac=r.get("veg_frac", np.nan), gap_m=r.get("L_b", np.nan)))
    if not rows:
        return C
    return pd.concat([C, pd.DataFrame(rows)], ignore_index=True)


def add_context(C, z, dsm, T, reaches_gdf=None, roads=None):
    """Deck height (max DSM - DEM within 4 m: a bridge deck over a continuous bed), distance to a
    road, and whether the crossing lies on a reach classified as stream."""
    from shapely.geometry import Point
    if dsm is not None and len(C):
        d = np.nan_to_num(dsm - z, nan=0.0)
        dmax = ndi.maximum_filter(d, size=9)
        cc, rr = drains._to_cr(T, C.x.values, C.y.values)
        C["deck_m"] = dmax[rr.astype(int), cc.astype(int)]
    else:
        C["deck_m"] = np.nan
    if roads is not None and len(roads) and len(C):
        R = roads.union_all()
        C["road_dist_m"] = [R.distance(Point(x, y)) for x, y in zip(C.x, C.y)]
    elif "road_dist_m" not in C:
        C["road_dist_m"] = np.nan
    if reaches_gdf is not None and len(reaches_gdf) and len(C):
        S = reaches_gdf[reaches_gdf["class"] == "stream"].union_all()
        C["on_stream"] = [S.distance(Point(x, y)) <= 10 for x, y in zip(C.x, C.y)] if not S.is_empty else False
    else:
        C["on_stream"] = False
    return C
