"""Stage A of channel network v2: catchment mode (design note, 23 Oct 2026).

One sea-draining catchment at a time, from the cleaned tile networks of stage B (``network_b.gpkg``, else stage C's
``network.gpkg``), the tiles' stored probabilities (``channel_prob.tif``) and bundles:

1. **Assemble.** Each tile's reaches clipped to its core (the S2-8 ownership rule: no duplicates from the tile
   buffers), kept where they meet the catchment plus ``overlap_m``. Reaches touched mid-line by another reach's end
   are split there (a junction); ends meeting across a tile edge within ``stitch_m`` are joined.
2. **Outlets.** Nodes within ``outlet_m`` of the coastline, tidal flats or a lake (OSM); nodes at the outer edge of
   the overlap band (provisional: the network leaves the catchment there); nodes at the edge of LiDAR coverage
   inside the catchment (provisional); and, when none of these is on the coast, the lowest dangling end
   (``lowest``).
3. **Orient and order from the outlets upstream.** Shortest paths from every outlet (Dijkstra) over the network,
   walking upstream costing a reach's length plus ``uphill_penalty`` per metre it would have to run uphill beyond
   ``uphill_tol_m``; each reach flows towards its end nearer an outlet. Strahler order and upstream length on the
   resulting tree; a reach still running uphill by more than the tolerance is flagged (an undetected barrier or an
   error).
4. **Connect what is left.** For each piece no outlet reaches, a least-cost path from its lowest node to the
   connected network over the stored U-Net probability (cost 1 + 9 (1 - p); roads ``road_cost`` more unless a
   crossing candidate of ``crossing_tiers`` lies within ``crossing_m``), accepted when it is at most
   ``connect_max_m`` long, rises at most ``connect_rise_m`` above its start (or crosses a road at a crossing
   candidate) and has channel evidence (mean probability ``connect_min_prob``, or under 10 m). The network is then
   oriented again. Pieces still unconnected are dropped, unless at least ``keep_isolated_m`` long (kept, flagged).
5. **Classify in context**: river (a river polygon, the reach's river flag, OSM river, or the U-Net's class),
   stream (OSM stream, the U-Net's class, Strahler order >= ``stream_min_order`` with the chain's stream
   probability >= 0.5), else drain; then the minor drains left as leaves are removed (``cleanup.minor``).
6. **Write back** per catchment (``<catchment>/network_a.gpkg``: channels drawn downstream, outlets, connections,
   dropped; ``stage_a.json``) and per tile (``<runs>/<tile>/network_a.gpkg``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULTS = dict(overlap_m=500.0, touch_m=0.5, stitch_m=5.0, outlet_m=20.0, band_edge_m=5.0, uphill_tol_m=0.5,
                uphill_penalty=50.0, connect_max_m=60.0, connect_rise_m=0.5, connect_min_prob=0.3, connect_free_m=10.0,
                keep_isolated_m=500.0, crossing_m=10.0, crossing_tiers=("high", "medium"), road_cost=20.0,
                stream_min_order=3, min_piece_m=0.5)


# ---------------------------------------------------------------- assembly and topology

def _layer(f, name, bbox=None):
    import geopandas as gpd
    import pyogrio
    f = Path(f)
    if not f.exists() or name not in {n for n, _ in pyogrio.list_layers(f)}:
        return None
    g = gpd.read_file(f, layer=name, bbox=bbox)
    return g.to_crs(2193) if g.crs is not None and g.crs.to_epsg() != 2193 else g


def assemble(tiles, runs, bundles, band, P):
    """Reaches of every tile, clipped to the tile's core and kept where they meet ``band``: a GeoDataFrame with
    ``tile``; also the tiles' cores, crossings, rivers and roads."""
    import geopandas as gpd
    import pandas as pd
    parts, cores, cross, rivers, roads = [], {}, [], [], []
    for t in tiles:
        run, b = Path(runs) / t, Path(bundles) / t
        aoi = gpd.read_file(b / "aoi.geojson").to_crs(2193).union_all()
        cores[t] = aoi
        ch = _layer(run / "network_b.gpkg", "channels")
        src = "b"
        if ch is None:
            ch, src = _layer(run / "network.gpkg", "channels"), "c"
        if ch is not None and len(ch):
            ch = ch[ch.geometry.notna() & ~ch.geometry.is_empty]
            g = ch.geometry.intersection(aoi)
            ch = ch.assign(geometry=g)
            ch = ch[~ch.geometry.is_empty].explode(index_parts=False)
            ch = ch[ch.geometry.geom_type == "LineString"]
            ch = ch[ch.length >= P["min_piece_m"]]
            ch = ch[ch.intersects(band)]
            parts.append(ch.assign(tile=t, stage_in=src))
        for name, acc in (("crossings", cross), ("rivers", rivers)):
            g = _layer(run / "network.gpkg", name)
            if g is not None and len(g):
                acc.append(g[g.intersects(aoi)].assign(tile=t))
        r = _layer(b / "roads.gpkg", _first_layer(b / "roads.gpkg")) if (b / "roads.gpkg").exists() else None
        if r is not None and len(r):
            roads.append(r[["geometry"]])
    cat = lambda xs: gpd.GeoDataFrame(pd.concat(xs, ignore_index=True), geometry="geometry", crs=2193) if xs \
        else gpd.GeoDataFrame(geometry=[], crs=2193)
    reaches = cat(parts).reset_index(drop=True)
    return reaches, cores, cat(cross), cat(rivers), cat(roads)


def _first_layer(f):
    import pyogrio
    return pyogrio.list_layers(f)[0][0]


def split_at_touches(geoms, touch_m):
    """Split lines where another line's end touches their interior (a junction the reaches did not share); returns
    (new geometries, index of the source line for each)."""
    import shapely
    from shapely import STRtree
    from shapely.ops import substring
    tree = STRtree(geoms)
    cuts = {}
    for i, g in enumerate(geoms):
        for e in (shapely.Point(g.coords[0]), shapely.Point(g.coords[-1])):
            for j in tree.query(e, predicate="dwithin", distance=touch_m):
                if j == i:
                    continue
                gj = geoms[j]
                d = gj.project(e)
                if touch_m < d < gj.length - touch_m:
                    cuts.setdefault(int(j), set()).add(round(float(d), 2))
    out, src = [], []
    for i, g in enumerate(geoms):
        ds = sorted(cuts.get(i, ()))
        if not ds:
            out.append(g)
            src.append(i)
            continue
        edges = [0.0] + ds + [g.length]
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a > 1e-6:
                out.append(substring(g, a, b))
                src.append(i)
    return out, np.asarray(src, int)


def nodes(geoms, touch_m):
    """Node id of each line's two ends (ends within ``touch_m`` share a node) and the nodes' coordinates."""
    from scipy.spatial import cKDTree

    from .connectivity import _UF
    pts = np.array([c for g in geoms for c in (g.coords[0][:2], g.coords[-1][:2])], float)
    if not len(pts):
        return np.zeros((0, 2), int), np.zeros((0, 2))
    uf = _UF(len(pts))
    for a, b in cKDTree(pts).query_pairs(touch_m):
        uf.union(a, b)
    _, nid = np.unique(uf.labels(), return_inverse=True)
    xy = np.zeros((nid.max() + 1, 2))
    cnt = np.bincount(nid)
    np.add.at(xy, nid, pts)
    return nid.reshape(-1, 2), xy / cnt[:, None]


def stitch(geoms, tiles_of, cores, P):
    """Short connectors between dangling ends on a tile's core edge and dangling ends of another tile within
    ``stitch_m`` (the same channel, cut at the edge). Returns the connector lines."""
    from scipy.spatial import cKDTree
    from shapely.geometry import LineString, Point
    ends, nid = nodes(geoms, P["touch_m"])
    deg = np.bincount(ends.ravel(), minlength=len(nid))
    cand = []
    for i, g in enumerate(geoms):
        t = tiles_of[i]
        if t is None or t not in cores:                 # connectors and links belong to no tile
            continue
        edge = cores[t].boundary
        for k, c in enumerate((g.coords[0][:2], g.coords[-1][:2])):
            if deg[ends[i, k]] == 1 and edge.distance(Point(c)) <= 1.0:
                cand.append((c, t))
    if len(cand) < 2:
        return []
    xy = np.array([c for c, _ in cand])
    tt = np.array([t for _, t in cand])
    tree = cKDTree(xy)
    used, out = set(), []
    for i, j in sorted(tree.query_pairs(P["stitch_m"]), key=lambda p: np.hypot(*(xy[p[0]] - xy[p[1]]))):
        if tt[i] == tt[j] or i in used or j in used:
            continue
        used |= {i, j}
        if np.hypot(*(xy[i] - xy[j])) > 1e-6:
            out.append(LineString([tuple(xy[i]), tuple(xy[j])]))
    return out


def sample_z(points_xy, tiles, cores, bundles):
    """DEM height at points (from the bundle of the tile whose core holds each, else any covering it)."""
    import rasterio
    from shapely import STRtree, points
    z = np.full(len(points_xy), np.nan)
    owner = np.full(len(points_xy), -1)
    names = list(tiles)
    if len(points_xy):
        tree = STRtree([cores[t].buffer(1.0) for t in names])
        pi, ti = tree.query(points(points_xy), predicate="within")
        first = {}
        for a_, b_ in zip(pi, ti):
            first.setdefault(int(a_), int(b_))
        for a_, b_ in first.items():
            owner[a_] = b_
    for k, t in enumerate(names):
        idx = np.flatnonzero(owner == k)
        if not len(idx):
            continue
        f = Path(bundles) / t / "dem.tif"
        if not f.exists():
            continue
        with rasterio.open(f) as s:
            vals = np.array([v[0] for v in s.sample([tuple(points_xy[i]) for i in idx])], float)
            if s.nodata is not None:
                vals[vals == s.nodata] = np.nan
        z[idx] = vals
    return z


# ---------------------------------------------------------------- outlets and orientation

def outlet_nodes(xy, deg, z, outlets_geom, band, coverage, catchment, P):
    """{node: kind}: coast (OSM coastline, tidal flats, lakes), band (the overlap band's outer edge), coverage (the
    edge of LiDAR coverage inside the catchment), lowest (when nothing is on the coast)."""
    from shapely import STRtree, points
    pts = points(xy)
    out = {}
    if outlets_geom:
        tree = STRtree(np.asarray(outlets_geom, object))
        hit = tree.query(pts, predicate="dwithin", distance=P["outlet_m"])[0]
        for i in np.unique(hit):
            out[int(i)] = "coast"
    edge = band.boundary
    for i in np.flatnonzero(deg == 1):
        if int(i) in out:
            continue
        p = pts[i]
        if edge.distance(p) <= P["band_edge_m"]:
            out[int(i)] = "band"
        elif coverage is not None and catchment.contains(p) and coverage.boundary.distance(p) <= P["band_edge_m"] \
                and not band.boundary.distance(p) <= P["band_edge_m"]:
            out[int(i)] = "coverage"
    if "coast" not in out.values():
        d1 = np.flatnonzero((deg == 1) & np.isfinite(z))
        if len(d1):
            out[int(d1[np.argmin(z[d1])])] = "lowest"
    return out


def orient(ends, lengths, z, outlets, P):
    """Shortest upstream paths from the outlets: (distance per node, predecessor (downstream) node, source outlet
    per node). Walking upstream from a to b costs the length plus ``uphill_penalty`` per metre b lies below a beyond
    ``uphill_tol_m``."""
    import pandas as pd
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    n = int(ends.max()) + 1 if len(ends) else 0
    a, b = ends[:, 0], ends[:, 1]
    za, zb = np.nan_to_num(z[a], nan=0.0), np.nan_to_num(z[b], nan=0.0)
    tol, pen = P["uphill_tol_m"], P["uphill_penalty"]
    c_ab = lengths + pen * np.maximum(0.0, za - zb - tol)      # upstream from a to b: b should be higher
    c_ba = lengths + pen * np.maximum(0.0, zb - za - tol)
    e = pd.DataFrame(dict(u=np.r_[a, b], v=np.r_[b, a], c=np.r_[c_ab, c_ba]))
    e = e[e.u != e.v].groupby(["u", "v"], as_index=False).c.min()
    g = csr_matrix((e.c.to_numpy() + 1e-6, (e.u.to_numpy(), e.v.to_numpy())), shape=(n, n))
    src = np.array(sorted(outlets), int)
    if not len(src):
        return np.full(n, np.inf), np.full(n, -9999), np.full(n, -9999)
    dist, pred, srcs = dijkstra(g, directed=True, indices=src, return_predecessors=True, min_only=True)
    return dist, pred, srcs


def order_and_upstream(ends, lengths, dist, pred):
    """Per line: downstream node, upstream node, Strahler order and upstream length (m, the line included), from the
    shortest-path tree; lines off the tree (braids) take their upstream node's values."""
    n = len(dist)
    pair = {}
    for i, (a, b) in enumerate(ends):
        key = (min(a, b), max(a, b))
        if key not in pair or lengths[i] < lengths[pair[key]]:
            pair[key] = i
    reached = np.isfinite(dist)
    children = [[] for _ in range(n)]
    tree_edge = np.full(n, -1)
    for v in np.flatnonzero(reached):
        p = pred[v]
        if p >= 0:
            children[p].append(v)
            tree_edge[v] = pair.get((min(p, v), max(p, v)), -1)
    order = np.zeros(n, int)
    up = np.zeros(n)
    for v in np.argsort(-np.where(reached, dist, -np.inf)):       # farthest from the outlet first: leaves first
        if not reached[v]:
            continue
        ks = children[v]
        if not ks:
            order[v] = 1
            continue
        o = [order[c] for c in ks]
        m = max(o)
        order[v] = m + 1 if o.count(m) >= 2 else m
        up[v] = sum(up[c] + (lengths[tree_edge[c]] if tree_edge[c] >= 0 else 0.0) for c in ks)
    down = np.full(len(ends), -1)
    upn = np.full(len(ends), -1)
    s_order = np.zeros(len(ends), int)
    s_up = np.zeros(len(ends))
    for i, (a, b) in enumerate(ends):
        if not (reached[a] and reached[b]):
            continue
        d_, u_ = (a, b) if dist[a] <= dist[b] else (b, a)
        down[i], upn[i] = d_, u_
        s_order[i] = order[u_]
        s_up[i] = up[u_] + lengths[i]
    return down, upn, s_order, s_up


# ---------------------------------------------------------------- connecting what is left

_BOUNDS = {}


def _file_bounds(f):
    """A raster's bounds, read once per process (stage A reads small windows many times)."""
    import rasterio
    k = str(f)
    if k not in _BOUNDS:
        with rasterio.open(f) as d:
            _BOUNDS[k] = tuple(d.bounds)
    return _BOUNDS[k]


def _mosaic(files, bounds, res=1.0, nodata=np.nan):
    """A window over the rasters that cover it (first wins), float32, with its transform. Only the files whose
    extent meets the window are opened."""
    import rasterio
    from rasterio.merge import merge
    hit = []
    for f in files:
        if not Path(f).exists():
            continue
        b = _file_bounds(f)
        if not (b[2] <= bounds[0] or b[0] >= bounds[2] or b[3] <= bounds[1] or b[1] >= bounds[3]):
            hit.append(f)
    ds = [rasterio.open(f) for f in hit]
    try:
        if not ds:
            return None, None
        arr, T = merge(ds, bounds=bounds, res=res, nodata=ds[0].nodata if ds[0].nodata is not None else nodata,
                       dtype="float32")
        a = arr[0].astype(np.float32)
        if ds[0].nodata is not None:
            a[a == ds[0].nodata] = np.nan
        return a, T
    finally:
        for d in ds:
            d.close()


def connect_piece(start_xy, z_start, reached_geoms, prob_files, dem_files, roads, crossings, P):
    """A least-cost path from ``start_xy`` to the connected network (``reached_geoms``); returns (line or None,
    record of the attempt)."""
    from rasterio.features import rasterize
    from shapely.geometry import LineString
    from skimage.graph import MCP_Geometric
    m = P["connect_max_m"] + 10.0
    x, y = start_xy
    bounds = (np.floor(x - m), np.floor(y - m), np.ceil(x + m), np.ceil(y + m))
    rec = dict(start_x=float(x), start_y=float(y))
    from shapely.geometry import Point
    sp = Point(x, y)
    tg = [g for g in reached_geoms if g.distance(sp) <= m]
    if not tg:
        return None, dict(rec, reason="no_network_near")
    pr, T = _mosaic(prob_files, bounds)
    if pr is None:
        return None, dict(rec, reason="no_probability")
    p = np.where(np.isfinite(pr) & (pr <= 250), pr / 250.0, 0.0)
    cost = 1.0 + 9.0 * (1.0 - p)
    shape = p.shape
    target = rasterize([(g, 1) for g in tg], out_shape=shape, transform=T, fill=0, all_touched=True) > 0
    road = np.zeros(shape, bool)
    if roads is not None and len(roads):
        rr = [g for g in roads.geometry if g is not None and not g.is_empty]
        if rr:
            road = rasterize([(g, 1) for g in rr], out_shape=shape, transform=T, fill=0, all_touched=True) > 0
    ok_x = np.zeros(shape, bool)
    if crossings is not None and len(crossings) and "tier" in crossings:
        cg = crossings[crossings.tier.astype(str).isin(P["crossing_tiers"])]
        if len(cg):
            ok_x = rasterize([(g.buffer(P["crossing_m"]), 1) for g in cg.geometry], out_shape=shape, transform=T,
                             fill=0) > 0
    cost = np.where(road & ~ok_x, cost + P["road_cost"], cost)
    r0, c0 = int((T.f - y) // abs(T.e)), int((x - T.c) // T.a)
    if not (0 <= r0 < shape[0] and 0 <= c0 < shape[1]):
        return None, dict(rec, reason="outside")
    tr, tc = np.nonzero(target)
    if not len(tr):
        return None, dict(rec, reason="no_network_near")
    mcp = MCP_Geometric(cost)
    cum, _ = mcp.find_costs([(r0, c0)], ends=list(zip(tr, tc)), find_all_ends=False)
    k = int(np.argmin(cum[tr, tc]))
    end = (int(tr[k]), int(tc[k]))
    if not np.isfinite(cum[end]):
        return None, dict(rec, reason="unreachable")
    path = np.array(mcp.traceback(end))
    xs = T.c + (path[:, 1] + 0.5) * T.a
    ys = T.f + (path[:, 0] + 0.5) * T.e
    coords = [(float(x), float(y))] + list(zip(xs.tolist(), ys.tolist()))
    ep = Point(coords[-1])
    near = min(tg, key=lambda g: g.distance(ep))
    q = near.interpolate(near.project(ep))                   # end on the network line itself: a junction
    coords[-1] = (q.x, q.y)
    if len(coords) < 2:
        coords.append((q.x, q.y))
    line = LineString(coords).simplify(0.5)
    zz, Tz = _mosaic(dem_files, bounds)
    rise = np.nan
    if zz is not None:
        rr_ = ((Tz.f - ys) // abs(Tz.e)).astype(int).clip(0, zz.shape[0] - 1)
        cc_ = ((xs - Tz.c) // Tz.a).astype(int).clip(0, zz.shape[1] - 1)
        prof = zz[rr_, cc_]
        if np.isfinite(prof).any() and np.isfinite(z_start):
            rise = float(np.nanmax(prof) - z_start)
    pm = float(p[path[:, 0], path[:, 1]].mean())
    on_road = bool(road[path[:, 0], path[:, 1]].any())
    crossing = bool((road & ok_x)[path[:, 0], path[:, 1]].any())
    rec.update(length_m=round(line.length, 2), rise_m=None if np.isnan(rise) else round(rise, 3),
               prob_mean=round(pm, 3), crosses_road=on_road, at_crossing=crossing)
    if line.length > P["connect_max_m"]:
        return None, dict(rec, reason="too_far")
    if on_road and not crossing:
        return None, dict(rec, reason="barrier")
    if np.isfinite(rise) and rise > P["connect_rise_m"] and not crossing:
        return None, dict(rec, reason="rise")
    if pm < P["connect_min_prob"] and line.length > P["connect_free_m"]:
        return None, dict(rec, reason="weak")
    return line, dict(rec, reason="", accepted=True, is_crossing=crossing)


# ---------------------------------------------------------------- classification

def classify(R, river_polys, P):
    """``class_v2`` (river / stream / drain) and ``class_source`` per reach, from the reach's own evidence and its
    place in the network."""
    from shapely import STRtree
    n = len(R)
    river = R.get("river", False)
    river = np.asarray(river.fillna(False) if hasattr(river, "fillna") else np.zeros(n, bool), bool)
    osm = R["osm_waterway"].astype(object).to_numpy() if "osm_waterway" in R else np.full(n, None)
    mlc = R["ml_class"].astype(object).to_numpy() if "ml_class" in R else np.full(n, None)
    mlp = R["ml_class_p"].to_numpy(float) if "ml_class_p" in R else np.zeros(n)
    ps = R["p_stream"].to_numpy(float) if "p_stream" in R else np.full(n, np.nan)
    order = R["strahler"].to_numpy(int) if "strahler" in R else np.zeros(n, int)
    in_poly = np.zeros(n, bool)
    if river_polys is not None and len(river_polys):
        t = STRtree(river_polys.geometry.to_numpy())
        for i, g in enumerate(R.geometry):
            hits = t.query(g, predicate="intersects")
            if len(hits):
                inside = sum(g.intersection(river_polys.geometry.iloc[h]).length for h in hits)
                in_poly[i] = inside >= 0.5 * g.length
    cls, srcs = [], []
    for i in range(n):
        if in_poly[i] or river[i]:
            c, s = "river", "river polygon"
        elif osm[i] == "river":
            c, s = "river", "osm"
        elif mlc[i] == "river" and np.nan_to_num(mlp[i]) >= 0.5:
            c, s = "river", "u-net class"
        elif osm[i] == "stream":
            c, s = "stream", "osm"
        elif mlc[i] in ("stream", "drain") and np.nan_to_num(mlp[i]) >= 0.5:
            c, s = mlc[i], "u-net class"
        elif order[i] >= P["stream_min_order"] and np.nan_to_num(ps[i], nan=0.0) >= 0.5:
            c, s = "stream", "order and stream probability"
        elif np.nan_to_num(ps[i], nan=0.0) >= 0.5:
            c, s = "stream", "stream probability"
        else:
            c, s = "drain", "default"
        cls.append(c)
        srcs.append(s)
    return cls, srcs


# ---------------------------------------------------------------- the catchment

def _graph(R, cores, bundles, P):
    """Split at touches, stitch tile edges, nodes and heights; returns (lines GeoDataFrame, ends, node xy, z)."""
    import geopandas as gpd
    import pandas as pd
    geoms = list(R.geometry)
    con = stitch(geoms, R.tile.to_numpy(), cores, P)
    if con:
        R = gpd.GeoDataFrame(pd.concat([R, gpd.GeoDataFrame(dict(channel_source=["stitch"] * len(con),
                                                                  tile=[None] * len(con)), geometry=con, crs=2193)], ignore_index=True),
                             geometry="geometry", crs=2193)
    sg, src = split_at_touches(list(R.geometry), P["touch_m"])
    R = R.iloc[src].reset_index(drop=True).set_geometry(gpd.GeoSeries(sg, crs=2193))
    ends, xy = nodes(list(R.geometry), P["touch_m"])
    z = sample_z(xy, list(cores), cores, bundles)
    return R, ends, xy, z


def run_catchment(catchment_dir, runs, bundles, tiles=None, params=None, write=True, log=print):
    """Stage A on one catchment folder (``catchment.gpkg``, from ``scripts/catchment_tiles.py``); returns the
    summary (``stage_a.json``)."""
    import geopandas as gpd
    import pandas as pd
    import shapely

    from . import cleanup
    from . import guidance as G
    P = dict(DEFAULTS, **(params or {}))
    cd = Path(catchment_dir)
    cat = gpd.read_file(cd / "catchment.gpkg", layer="catchment").to_crs(2193)
    catchment = cat.geometry.iloc[0]
    if tiles is None:
        tiles = list(gpd.read_file(cd / "catchment.gpkg", layer="tiles").tile_id)
    tiles = [t for t in tiles if (Path(runs) / t / "network.gpkg").exists()]
    band = catchment.buffer(P["overlap_m"])
    R, cores, cross, rivers, roads = assemble(tiles, runs, bundles, band, P)
    log(f"{len(tiles)} tiles; {len(R)} reach pieces, {R.length.sum() / 1000:.1f} km in the catchment and its band")
    coverage = shapely.union_all(list(cores.values())) if cores else None
    outs = []
    for t in tiles:
        b = Path(bundles) / t
        win = tuple(cores[t].buffer(500).bounds)
        g = G.load(b / "context.gpkg", win)
        a = g["areas"]
        if len(a) and "water_kind" in a:
            outs += [x for x in a[a.water_kind.isin(["lake", "tidal"])].geometry if x is not None]
        outs += [x for x in g["coastline"].geometry if x is not None]
    km_in = float(R.length.sum()) / 1000
    prob_files = [Path(runs) / t / "channel_prob.tif" for t in tiles]
    dem_files = [Path(bundles) / t / "dem.tif" for t in tiles]
    conns, attempts = [], []
    for rnd in range(2):                                    # orient, connect what is left, orient again
        R, ends, xy, z = _graph(R, cores, bundles, P)
        deg = np.bincount(ends.ravel(), minlength=len(xy))
        out_nodes = outlet_nodes(xy, deg, z, outs, band, coverage, catchment, P)
        lengths = R.length.to_numpy(float)
        dist, pred, srcs = orient(ends, lengths, z, out_nodes, P)
        reached_line = np.isfinite(dist[ends[:, 0]]) & np.isfinite(dist[ends[:, 1]])
        log(f"round {rnd + 1}: {len(out_nodes)} outlets ({pd.Series(list(out_nodes.values())).value_counts().to_dict()}); "
            f"{reached_line.mean():.1%} of pieces, {lengths[reached_line].sum() / max(lengths.sum(), 1):.1%} of length "
            "reach an outlet")
        if rnd == 1:
            break
        from .connectivity import _UF
        uf = _UF(len(xy))
        for a, b in ends:
            uf.union(a, b)
        comp_of_node = uf.labels()
        un = ~reached_line
        reached_geoms = list(R.geometry[reached_line])
        from shapely import STRtree, points
        rtree = STRtree(np.asarray(reached_geoms, object)) if reached_geoms else None
        new = []
        for cid in np.unique(comp_of_node[ends[un, 0]]):
            members = np.flatnonzero(un & (comp_of_node[ends[:, 0]] == cid))
            nn = np.unique(ends[members].ravel())
            # the start: the piece's lowest node (to 0.1 m), the one nearest the connected network among equals
            dn = np.full(len(nn), np.inf)
            if rtree is not None:
                pts = points(xy[nn])
                j = rtree.nearest(pts)
                dn = np.array([reached_geoms[k].distance(q) for k, q in zip(j, pts)])
            zz = np.round(np.where(np.isfinite(z[nn]), z[nn], np.inf), 1)
            start = nn[np.lexsort((dn, zz))[0]]
            line, rec = connect_piece(xy[start], z[start], reached_geoms, prob_files, dem_files, roads, cross, P)
            rec.update(component_km=round(float(lengths[members].sum()) / 1000, 3))
            attempts.append(rec)
            if line is not None:
                new.append((line, rec.get("is_crossing", False)))
        if new:
            add = gpd.GeoDataFrame(dict(channel_source=["a_link"] * len(new), is_crossing=[c for _, c in new],
                                        **{"class": ["link"] * len(new)}, tile=[None] * len(new)),
                                   geometry=[g for g, _ in new], crs=2193)
            conns.append(add)
            R = gpd.GeoDataFrame(pd.concat([R, add], ignore_index=True), geometry="geometry", crs=2193)
        log(f"connections: {len(attempts)} pieces tried, {len(new)} joined")
    # unreached: kept when long, else dropped
    from .connectivity import _UF
    uf = _UF(len(xy))
    for a, b in ends:
        uf.union(a, b)
    lab = uf.labels()
    comp_line = lab[ends[:, 0]]
    comp_len = pd.Series(lengths).groupby(comp_line).sum()
    long_iso = ~reached_line & (comp_len.reindex(comp_line).to_numpy() >= P["keep_isolated_m"])
    drop = ~reached_line & ~long_iso
    dropped = [R[drop].assign(reason="unconnected")]
    keep = ~drop
    down, upn, order, up = order_and_upstream(ends, lengths, dist, pred)
    R = R.assign(strahler=order, upstream_km=np.round(up / 1000, 3),
                 dist_outlet_km=np.round(np.where(np.isfinite(dist[np.maximum(down, 0)]), dist[np.maximum(down, 0)],
                                                  np.nan) / 1000, 3),
                 outlet_kind=[out_nodes.get(int(srcs[d]), None) if d >= 0 and srcs[d] >= 0 else None for d in down],
                 isolated=long_iso, uphill=False)
    # draw each reach downstream and flag uphill runs
    geoms = list(R.geometry)
    for i in np.flatnonzero(down >= 0):
        d_xy = xy[down[i]]
        g = geoms[i]
        if np.hypot(*(np.array(g.coords[0][:2]) - d_xy)) < np.hypot(*(np.array(g.coords[-1][:2]) - d_xy)):
            geoms[i] = shapely.reverse(g)
    R = R.set_geometry(gpd.GeoSeries(geoms, crs=2193))
    zu, zd = z[np.maximum(upn, 0)], z[np.maximum(down, 0)]
    R["uphill"] = (down >= 0) & np.isfinite(zu) & np.isfinite(zd) & (zd > zu + P["uphill_tol_m"])
    R = R[keep].reset_index(drop=True)
    # classify, then the minor drains left as leaves go
    R["class_v2"], R["class_source"] = classify(R, rivers, P)
    PB = cleanup._params(params)
    edge = band.boundary.buffer(P["band_edge_m"])
    protected = np.array([g.intersects(edge) for g in R.geometry])
    gone = cleanup.peel(R.geometry.to_numpy(), cleanup.minor(R.assign(**{"class": R.class_v2}), PB) &
                        (R.class_v2.to_numpy() == "drain"), protected, P["touch_m"])
    dropped.append(R[gone].assign(reason="minor_drain"))
    R = R[~gone].reset_index(drop=True)
    D = gpd.GeoDataFrame(pd.concat(dropped, ignore_index=True), geometry="geometry", crs=2193)
    A = gpd.GeoDataFrame(pd.DataFrame(attempts)) if attempts else pd.DataFrame()
    if len(A):
        A = gpd.GeoDataFrame(A, geometry=gpd.points_from_xy(A.start_x, A.start_y), crs=2193)
    O = gpd.GeoDataFrame(dict(kind=list(out_nodes.values())), geometry=gpd.points_from_xy(
        [xy[i][0] for i in out_nodes], [xy[i][1] for i in out_nodes]), crs=2193)
    inside = R[R.intersects(catchment)]
    km = lambda g: round(float(g.length.sum()) / 1000, 3)
    summary = dict(catchment=cat.catchment_id.iloc[0] if "catchment_id" in cat else cd.name, tiles=len(tiles),
                   params={k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()},
                   km_in=round(km_in, 3), km_out=km(R), km_in_catchment=km(inside),
                   reached_share_of_length=round(float(R[~R.isolated].length.sum() / max(R.length.sum(), 1)), 4),
                   isolated_kept=int(R.isolated.sum()), isolated_kept_km=km(R[R.isolated]),
                   outlets={k: int(v) for k, v in O.kind.value_counts().items()},
                   connections=dict(tried=len(attempts), accepted=int(sum(1 for a in attempts if a.get("accepted"))),
                                    crossings=int(sum(1 for a in attempts if a.get("is_crossing"))),
                                    reasons={k: int(v) for k, v in pd.Series([a.get("reason") or "accepted" for a in attempts])
                                             .value_counts().items()} if attempts else {}),
                   dropped={k: km(D[D.reason == k]) for k in D.reason.unique()} if len(D) else {},
                   uphill=dict(reaches=int(R.uphill.sum()), km=km(R[R.uphill])),
                   class_km={k: km(R[R.class_v2 == k]) for k in ("river", "stream", "drain", "link")},
                   class_sources={k: int(v) for k, v in R.class_source.value_counts().items()},
                   strahler_km={int(k): km(R[R.strahler == k]) for k in sorted(R.strahler.unique())})
    if write:
        f = cd / "network_a.gpkg"
        if f.exists():
            f.unlink()
        R.to_file(f, layer="channels", driver="GPKG")
        if len(O):
            O.to_file(f, layer="outlets", driver="GPKG")
        if conns:
            gpd.GeoDataFrame(pd.concat(conns, ignore_index=True), geometry="geometry", crs=2193).to_file(
                f, layer="connections", driver="GPKG")
        if len(A):
            A.to_file(f, layer="attempts", driver="GPKG")
        if len(D):
            D.to_file(f, layer="dropped", driver="GPKG")
        (cd / "stage_a.json").write_text(json.dumps(summary, indent=1, default=str))
        mid = R.geometry.interpolate(0.5, normalized=True)
        for t in tiles:
            sel = R[mid.within(cores[t].buffer(0.01))]
            tf = Path(runs) / t / "network_a.gpkg"
            if tf.exists():
                tf.unlink()
            if len(sel):
                sel.to_file(tf, layer="channels", driver="GPKG")
    return summary
