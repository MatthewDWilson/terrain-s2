"""Connectivity of a run's channel network (integration step 3: measure first, then the clean-up rules).

The rule agreed for the network is that it should be connected. This module measures how far a run's
``channels`` layer is from that, without changing it:

- **Components**: reaches that touch (within ``touch_m``) belong to one component; ``dangling ends`` are reach
  ends no other reach touches.
- **Anchors**: a component is anchored when it can drain somewhere: it reaches the tile edge (within ``edge_m``
  of the AOI boundary, or beyond it: the network continues in the neighbour), meets a river polygon
  (``rivers``, within ``river_m``), has a dangling end at a residual pond (``residual_ponds``, within ``pond_m``:
  a real sink), or has a dangling end at the limit of the floodplain restriction (at least ``fp_limit_m`` from
  the floodplain: the run cut the channel there, it continues upslope). Anything else is **isolated**.
- **Gaps**: for each component, the nearest other component and how they nearly meet, by where the nearest
  points lie: ``end_end`` (a dangling end of each, the small gap that should be joined end to end),
  ``end_line`` (a dangling end of one against the side of the other: a T-junction gap), ``line_line``
  (neither is at an end: lines that nearly touch where one has curved away).
- **Joining**: the isolated length left if every gap up to ``G`` m were joined (components within ``G`` merged,
  then anchored if any part is), for the thresholds in ``JOIN_M``.

``measure(run_dir, aoi)`` returns the tile's summary and the per-component table (with geometry and, for each, the
connector to its nearest component), for ``scripts/network_fragments.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

JOIN_M = (2.0, 5.0, 10.0, 20.0, 50.0)
GAP_BINS = (0.0, 2.0, 5.0, 10.0, 20.0, 50.0, np.inf)


class _UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, i):
        p = self.p
        r = i
        while p[r] != r:
            r = p[r]
        while p[i] != r:
            p[i], i = r, p[i]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)

    def labels(self):
        return np.array([self.find(i) for i in range(len(self.p))])


def _layer(gpkg, name):
    import geopandas as gpd
    import pyogrio
    if not Path(gpkg).exists() or name not in {n for n, _ in pyogrio.list_layers(gpkg)}:
        return None
    return gpd.read_file(gpkg, layer=name)


def _lines(geom):
    if geom.geom_type == "LineString":
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if g.geom_type == "LineString"]


def measure(run_dir, aoi, touch_m: float = 0.5, edge_m: float = 5.0, river_m: float = 2.0, pond_m: float = 5.0,
            fp_limit_m: float = 40.0, end_m: float = 1.0, join_m=JOIN_M):
    """(summary dict, components GeoDataFrame) for one run folder (``network.gpkg``); ``aoi``: the tile's AOI
    (a geometry in the layers' CRS). See the module docstring."""
    import geopandas as gpd
    import shapely
    from shapely import STRtree
    from shapely.ops import nearest_points
    gpkg = Path(run_dir) / "network.gpkg"
    ch = _layer(gpkg, "channels")
    if ch is None or not len(ch):
        return dict(reaches=0), None
    ch = ch[~ch.geometry.is_empty & ch.geometry.notna()].reset_index(drop=True)
    geoms = ch.geometry.to_numpy()
    n = len(geoms)
    # components: reaches that touch
    uf = _UF(n)
    a, b = STRtree(geoms).query(geoms, predicate="dwithin", distance=touch_m)
    for i, j in zip(a, b):
        if i != j:
            uf.union(i, j)
    lab = uf.labels()
    _, comp = np.unique(lab, return_inverse=True)
    ncomp = int(comp.max()) + 1
    # dangling ends: reach ends no other reach touches
    ends, end_reach = [], []
    for i, g in enumerate(geoms):
        for ln in _lines(g)[:1] or []:
            ends += [shapely.Point(ln.coords[0][:2]), shapely.Point(ln.coords[-1][:2])]
            end_reach += [i, i]
    ends = np.array(ends, object)
    end_reach = np.array(end_reach)
    ea, eb = STRtree(geoms).query(ends, predicate="dwithin", distance=touch_m)
    touches = np.zeros(len(ends), int)
    np.add.at(touches, ea[end_reach[ea] != eb], 1)
    dangling = touches == 0
    # components as geometries, with their length, reaches, ML and drain shares
    length = ch.geometry.length.to_numpy()
    src = ch.channel_source.astype(str).to_numpy() if "channel_source" in ch else np.full(n, "physics")
    cls = ch["class"].astype(str).to_numpy() if "class" in ch else np.full(n, "unknown")
    cgeom = [shapely.union_all(geoms[comp == k]) for k in range(ncomp)]
    clen = np.bincount(comp, weights=length, minlength=ncomp)
    cml = np.bincount(comp, weights=length * (src == "ml"), minlength=ncomp)
    cdrain = np.bincount(comp, weights=length * (cls == "drain"), minlength=ncomp)
    creach = np.bincount(comp, minlength=ncomp)
    dend = [[] for _ in range(ncomp)]
    for e, r in zip(ends[dangling], end_reach[dangling]):
        dend[comp[r]].append(e)
    # anchors
    edge = shapely.buffer(shapely.boundary(aoi), edge_m)
    anchor = np.full(ncomp, "", object)
    for k, g in enumerate(cgeom):
        if g.intersects(edge) or not aoi.contains(g):
            anchor[k] = "edge"
    riv = _layer(gpkg, "rivers")
    if riv is not None and len(riv):
        rt = STRtree(riv.geometry.to_numpy())
        for k in np.flatnonzero(anchor == ""):
            if len(rt.query(cgeom[k], predicate="dwithin", distance=river_m)):
                anchor[k] = "river"
    ponds = _layer(gpkg, "residual_ponds")
    if ponds is not None and len(ponds):
        pt = STRtree(ponds.geometry.to_numpy())
        for k in np.flatnonzero(anchor == ""):
            if dend[k] and len(pt.query(np.array(dend[k], object), predicate="dwithin", distance=pond_m)[0]):
                anchor[k] = "pond"
    fp = _layer(gpkg, "floodplain")
    if fp is not None and len(fp):
        fpu = shapely.union_all(fp.geometry.to_numpy())
        for k in np.flatnonzero(anchor == ""):
            if dend[k] and max(fpu.distance(e) for e in dend[k]) >= fp_limit_m:
                anchor[k] = "floodplain_limit"
    isolated = anchor == ""
    # the nearest other component and the kind of gap
    near = np.full(ncomp, -1)
    gap = np.full(ncomp, np.inf)
    kind = np.full(ncomp, "", object)
    connector = [None] * ncomp
    if ncomp > 1:
        ct = STRtree(np.array(cgeom, object))
        qi, qj = ct.query_nearest(np.array(cgeom, object), exclusive=True, all_matches=False)
        for i, j in zip(qi, qj):
            p, q = nearest_points(cgeom[i], cgeom[j])
            d = p.distance(q)
            if d < gap[i]:
                near[i], gap[i] = j, d
                ei = any(p.distance(e) <= end_m for e in dend[i])
                ej = any(q.distance(e) <= end_m for e in dend[j])
                kind[i] = "end_end" if ei and ej else ("end_line" if ei or ej else "line_line")
                connector[i] = shapely.LineString([p, q])
    # joining: merge components within G, anchored if any part is
    total = float(clen.sum())
    join = {}
    if ncomp > 1:
        ct = STRtree(np.array(cgeom, object))
        for G in join_m:
            uj = _UF(ncomp)
            a, b = ct.query(np.array(cgeom, object), predicate="dwithin", distance=G)
            for i, j in zip(a, b):
                if i != j:
                    uj.union(i, j)
            gl = uj.labels()
            anchored_groups = set(gl[~isolated])
            iso = np.array([g not in anchored_groups for g in gl])
            join[f"isolated_share_join_{G:g}m"] = round(float(clen[iso].sum() / total), 4) if total else 0.0
    else:
        for G in join_m:
            join[f"isolated_share_join_{G:g}m"] = round(float(clen[isolated].sum() / total), 4) if total else 0.0
    hist = np.histogram(gap[isolated & np.isfinite(gap)], bins=GAP_BINS)[0]
    summary = dict(reaches=n, components=ncomp, km=round(total / 1000, 3),
                   anchored_edge=int((anchor == "edge").sum()), anchored_river=int((anchor == "river").sum()),
                   anchored_pond=int((anchor == "pond").sum()),
                   anchored_floodplain_limit=int((anchor == "floodplain_limit").sum()),
                   isolated=int(isolated.sum()), isolated_km=round(float(clen[isolated].sum() / 1000), 3),
                   isolated_share=round(float(clen[isolated].sum() / total), 4) if total else 0.0,
                   isolated_ml_share=round(float(cml[isolated].sum() / max(clen[isolated].sum(), 1e-9)), 4),
                   ml_share=round(float(cml.sum() / total), 4) if total else 0.0,
                   isolated_drain_share=round(float(cdrain[isolated].sum() / max(clen[isolated].sum(), 1e-9)), 4),
                   isolated_median_m=round(float(np.median(clen[isolated])), 1) if isolated.any() else None,
                   dangling_ends=int(dangling.sum()),
                   **{(f"gap_{GAP_BINS[i]:g}_{GAP_BINS[i + 1]:g}m" if np.isfinite(GAP_BINS[i + 1])
                       else f"gap_{GAP_BINS[i]:g}plus_m"): int(hist[i]) for i in range(len(hist))},
                   **{f"gap_{k}": int(((kind == k) & isolated).sum()) for k in ("end_end", "end_line", "line_line")},
                   **join)
    comps = gpd.GeoDataFrame(dict(component=np.arange(ncomp), reaches=creach, length_m=np.round(clen, 1),
                                  ml_share=np.round(cml / np.maximum(clen, 1e-9), 3),
                                  drain_share=np.round(cdrain / np.maximum(clen, 1e-9), 3),
                                  dangling_ends=[len(x) for x in dend], anchor=anchor, isolated=isolated,
                                  nearest=near, gap_m=np.round(gap, 2), gap_kind=kind),
                             geometry=cgeom, crs=ch.crs)
    comps["connector_wkt"] = [c.wkt if c is not None else None for c in connector]
    return summary, comps


def connectors(comps):
    """The components' connectors to their nearest component, as a line layer (for QGIS)."""
    import geopandas as gpd
    from shapely import wkt
    c = comps[comps.connector_wkt.notna()]
    return gpd.GeoDataFrame(c[["component", "isolated", "anchor", "gap_m", "gap_kind", "length_m", "ml_share"]].copy(),
                            geometry=[wkt.loads(s) for s in c.connector_wkt], crs=comps.crs)
