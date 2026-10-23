"""Stage B of channel network v2: the clean-up of one tile's network (design note, 23 Oct 2026).

Reads what stage C stored in a run folder (``network.gpkg``: ``channels``, ``links``, ``rivers``, ``floodplain``)
and the tile's OSM outlets (lakes, tidal flats, the coastline: ``guidance.load``), and writes ``network_b.gpkg``
beside it, changing nothing of C's:

1. **Join supported gaps.** A candidate link (``connectivity.candidate_links``) is accepted when it is short enough
   for its kind (``link_max_m``) and either the terrain lets flow along it (it rises at most ``rise_tol_m`` above
   its higher end) and it has channel evidence (U-Net probability or channel mask along it, or it is shorter than
   ``link_free_m``), or, where it crosses a barrier (rises more, or crosses a road), a crossing candidate of a tier
   in ``crossing_tiers`` lies on it (the candidate is then the evidence: an embankment has no channel on it). Links are taken shortest first, at most one per dangling end, and none that
   would join two pieces already joined (no loops). An accepted link across a barrier is a crossing (``is_crossing``).
2. **Drop minor drains.** A drain (class ``drain``, not a river, not mapped by OSM as river, stream or canal) is
   minor when its width (``width_m``, on the LiDAR) is below ``minor_width_m`` or its depth (``incision_m``) below
   ``minor_depth_m``; a council class is never used for size (Matt, 23 Oct). Minor drains are removed from the
   ends of the network inwards, so one that carries a larger channel to the rest stays; links that end up as
   leaves go with them. Reaches at the tile edge are kept: stage A sees the neighbour.
3. **Drop fragments.** A piece left shorter than ``fragment_min_m`` that drains nowhere (no tile edge, river, lake,
   tidal flat or coastline, no floodplain limit; residual ponds no longer count) is removed. Longer isolated pieces
   stay, flagged, for stage A to connect or drop.

Every removal is kept in ``rejected`` with its reason, every candidate link in ``links`` with ``accepted`` and
``reason``; ``stage_b.json`` holds the connectivity before and after, and what each rule removed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULTS = dict(link_max_m=dict(end_end=20.0, end_line=15.0, line_line=10.0), link_free_m=5.0, link_min_prob=0.2,
                link_min_mask=0.3, rise_tol_m=0.3, crossing_tiers=("high", "medium"), minor_width_m=1.5,
                minor_depth_m=0.3, keep_osm=("river", "stream", "canal"), fragment_min_m=100.0, touch_m=0.5,
                edge_m=5.0)


def _params(params):
    P = dict(DEFAULTS, **(params or {}))
    P["link_max_m"] = dict(DEFAULTS["link_max_m"], **((params or {}).get("link_max_m") or {}))
    return P


def outlets(context, window):
    """Lake and tidal-flat polygons and coastline lines from the tile's OSM (``guidance.load``), as geometries."""
    from . import guidance as G
    g = G.load(context, window)
    out = []
    a = g["areas"]
    if len(a) and "water_kind" in a:
        out += list(a[a.water_kind.isin(["lake", "tidal"])].geometry)
    out += list(g["coastline"].geometry)
    return [x for x in out if x is not None and not x.is_empty]


def judge_links(links, P):
    """``accepted`` (candidate) and ``reason`` per link, before the greedy choice: too_long, weak (no channel
    evidence), barrier (rises or crosses a road with no crossing candidate on it); ``is_crossing``."""
    L = links.copy()
    if not len(L):
        for c in ("accepted", "reason", "is_crossing"):
            L[c] = []
        return L
    mx = L.kind.map(P["link_max_m"]).fillna(min(P["link_max_m"].values())).to_numpy(float)
    length = L.length_m.to_numpy(float)
    prob = L.prob_mean.to_numpy(float) if "prob_mean" in L else np.full(len(L), np.nan)
    mask = L.mask_share.to_numpy(float) if "mask_share" in L else np.full(len(L), np.nan)
    rise = np.nan_to_num(L.rise_m.to_numpy(float), nan=0.0) if "rise_m" in L else np.zeros(len(L))
    road = L.crosses_road.fillna(False).astype(bool).to_numpy() if "crosses_road" in L else np.zeros(len(L), bool)
    tier = L.crossing_tier.astype(object).to_numpy() if "crossing_tier" in L else np.full(len(L), None)
    evidence = (length <= P["link_free_m"]) | (np.nan_to_num(prob, nan=-1) >= P["link_min_prob"]) | \
        (np.nan_to_num(mask, nan=-1) >= P["link_min_mask"])
    barrier = (rise > P["rise_tol_m"]) | road
    crossing = np.array([t in P["crossing_tiers"] for t in tier])
    # across a barrier the crossing candidate is the evidence (an embankment has no channel on it); elsewhere the
    # link needs channel evidence of its own
    reason = np.where(length > mx, "too_long", np.where(barrier & ~crossing, "barrier",
                                                         np.where(~barrier & ~evidence, "weak", "")))
    L["accepted"] = reason == ""
    L["reason"] = reason
    L["is_crossing"] = barrier & crossing
    return L


def choose_links(L):
    """The greedy choice among candidate links: shortest first, one per dangling end, no loops. Updates
    ``accepted`` and ``reason`` (``end_used``, ``joined_already``)."""
    from .connectivity import _UF
    L = L.copy()
    if not len(L) or not L.accepted.any():
        return L
    ids = np.unique(np.concatenate([L.from_component.to_numpy(int), L.to_component.to_numpy(int)]))
    pos = {c: i for i, c in enumerate(ids)}
    uf = _UF(len(ids))
    used = set()
    for i in L[L.accepted].sort_values("length_m").index:
        r = L.loc[i]
        a, b = pos[int(r.from_component)], pos[int(r.to_component)]
        start = tuple(np.round(r.geometry.coords[0], 1))
        if r.kind != "line_line" and start in used:
            L.loc[i, ["accepted", "reason"]] = [False, "end_used"]
        elif uf.find(a) == uf.find(b):
            L.loc[i, ["accepted", "reason"]] = [False, "joined_already"]
        else:
            uf.union(a, b)
            if r.kind != "line_line":
                used.add(start)
    return L


def minor(ch, P):
    """Bool per reach: a minor drain (see the module docstring); links (channel_source ``link``) count as minor."""
    cls = ch["class"].astype(str).to_numpy() if "class" in ch else np.full(len(ch), "drain")
    river = ch["river"].fillna(False).astype(bool).to_numpy() if "river" in ch else np.zeros(len(ch), bool)
    osm = ch["osm_waterway"].astype(object).to_numpy() if "osm_waterway" in ch else np.full(len(ch), None)
    w = ch["width_m"].to_numpy(float) if "width_m" in ch else np.full(len(ch), np.inf)
    d = ch["incision_m"].to_numpy(float) if "incision_m" in ch else np.full(len(ch), np.inf)
    small = (np.nan_to_num(w, nan=0.0) < P["minor_width_m"]) | (np.nan_to_num(d, nan=0.0) < P["minor_depth_m"])
    m = (cls == "drain") & ~river & ~np.isin(osm, list(P["keep_osm"])) & small
    if "channel_source" in ch:
        m |= ch.channel_source.astype(str).to_numpy() == "link"
    return m


def peel(geoms, removable, protected, touch_m=0.5):
    """Remove removable reaches from the ends of the network inwards: a removable, unprotected reach with an end no
    live reach touches goes, and its neighbours are looked at again. Returns the bool of removed reaches."""
    import shapely
    from shapely import STRtree
    n = len(geoms)
    tree = STRtree(geoms)
    touch = [[set(), set()] for _ in range(n)]
    ends = []
    for i, g in enumerate(geoms):
        ln = g if g.geom_type == "LineString" else max(g.geoms, key=lambda x: x.length)
        ends.append((shapely.Point(ln.coords[0][:2]), shapely.Point(ln.coords[-1][:2])))
    for i, (a, b) in enumerate(ends):
        for k, e in enumerate((a, b)):
            touch[i][k] = {int(j) for j in tree.query(e, predicate="dwithin", distance=touch_m) if j != i}
    alive = np.ones(n, bool)
    removed = np.zeros(n, bool)
    queue = list(np.flatnonzero(removable & ~protected))
    while queue:
        i = int(queue.pop())
        if not alive[i]:
            continue
        free = any(not any(alive[j] for j in t) for t in touch[i])
        if free:
            alive[i] = False
            removed[i] = True
            for t in touch[i]:
                for j in t:
                    if alive[j] and removable[j] and not protected[j]:
                        queue.append(j)
    return removed


def clean_tile(run_dir, aoi, context=None, window=None, params=None, write=True):
    """Stage B on one run folder; returns the summary (``stage_b.json``). ``aoi``: the tile's AOI (layers' CRS);
    ``context``: the bundle's context.gpkg (for the outlets); ``window``: NZTM bounds for the OSM read."""
    import geopandas as gpd
    import pandas as pd
    import shapely

    from . import connectivity as C
    P = _params(params)
    run = Path(run_dir)
    gpkg = run / "network.gpkg"
    ch = C._layer(gpkg, "channels")
    if ch is None or not len(ch):
        return dict(tile=run.name, reaches=0)
    ch = ch[ch.geometry.notna() & ~ch.geometry.is_empty].reset_index(drop=True)
    crs = ch.crs
    outs = outlets(context, window if window is not None else tuple(aoi.buffer(500).bounds)) if context else []
    kw = dict(outlets=outs, ponds_anchor=False)
    before, _ = C.measure(run, aoi, **kw)
    # 1. links
    links = C._layer(gpkg, "links")
    if links is None:
        links = gpd.GeoDataFrame(geometry=[], crs=crs)
    L = choose_links(judge_links(links, P)) if len(links) else links.assign(accepted=[], reason=[], is_crossing=[])
    acc = L[L.accepted] if len(L) else L
    new = gpd.GeoDataFrame(dict(channel_source="link", link_kind=acc.kind.to_numpy() if len(acc) else [],
                                is_crossing=acc.is_crossing.to_numpy() if len(acc) else [], **{"class": "link"}),
                           geometry=list(acc.geometry), crs=crs)
    work = gpd.GeoDataFrame(pd.concat([ch.assign(link_kind=None, is_crossing=False), new], ignore_index=True),
                            geometry="geometry", crs=crs)
    # 2. minor drains, from the ends inwards; reaches at the tile edge stay for stage A
    edge = shapely.buffer(shapely.boundary(aoi), P["edge_m"])
    geoms = work.geometry.to_numpy()
    protected = np.array([g.intersects(edge) or not aoi.contains(g) for g in geoms])
    gone = peel(geoms, minor(work, P), protected, P["touch_m"])
    rej = [work[gone].assign(reason="minor_drain")]
    work = work[~gone].reset_index(drop=True)
    # 3. fragments: short pieces that drain nowhere
    s1, comps = C.measure(run, aoi, channels=work, **kw)
    frag = np.zeros(len(work), bool)
    if comps is not None and len(comps):
        comp, _, _, _ = C._topology(work.geometry.to_numpy(), P["touch_m"])
        short = comps.isolated.to_numpy() & (comps.length_m.to_numpy(float) < P["fragment_min_m"])
        frag = short[comp]
        work["component"] = comp
        work["isolated"] = comps.isolated.to_numpy()[comp]
    rej.append(work[frag].assign(reason="fragment"))
    work = work[~frag].reset_index(drop=True)
    after, comps2 = C.measure(run, aoi, channels=work, **kw)
    if comps2 is not None and len(comps2):
        comp, _, _, _ = C._topology(work.geometry.to_numpy(), P["touch_m"])
        work["component"] = comp
        work["isolated"] = comps2.isolated.to_numpy()[comp]
        work["anchor"] = comps2.anchor.to_numpy()[comp]
    rejected = gpd.GeoDataFrame(pd.concat(rej, ignore_index=True), geometry="geometry", crs=crs)
    km = lambda g: round(float(g.length.sum()) / 1000, 3)
    summary = dict(tile=run.name, params={k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()},
                   before=before, after=after,
                   links=dict(candidates=int(len(L)), accepted=int(L.accepted.sum()) if len(L) else 0,
                              crossings=int((L.accepted & L.is_crossing).sum()) if len(L) else 0,
                              accepted_km=km(acc) if len(acc) else 0.0,
                              by_reason={k: int(v) for k, v in L.reason.replace("", "accepted").value_counts().items()}
                              if len(L) else {},
                              accepted_by_kind={k: int(v) for k, v in acc.kind.value_counts().items()} if len(acc) else {}),
                   removed=dict(minor_drain_km=km(rejected[rejected.reason == "minor_drain"]),
                                minor_drain_reaches=int((rejected.reason == "minor_drain").sum()),
                                fragment_km=km(rejected[rejected.reason == "fragment"]),
                                fragment_reaches=int((rejected.reason == "fragment").sum())),
                   kept_km=km(work), outlets=len(outs))
    if write:
        f = run / "network_b.gpkg"
        if f.exists():
            f.unlink()
        work.to_file(f, layer="channels", driver="GPKG")
        if len(L):
            L.to_file(f, layer="links", driver="GPKG")
        if len(rejected):
            rejected.to_file(f, layer="rejected", driver="GPKG")
        (run / "stage_b.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def minor_sweep(run_dir, aoi, widths=(1.0, 1.5, 2.0), depths=(0.2, 0.3, 0.4), params=None):
    """Length (km) the minor-drain rule would remove at each (width, depth) pair, links aside: for choosing the
    thresholds. Returns [{width_m, depth_m, removed_km, total_km}]."""
    import shapely

    from . import connectivity as C
    P = _params(params)
    ch = C._layer(Path(run_dir) / "network.gpkg", "channels")
    if ch is None or not len(ch):
        return []
    ch = ch[ch.geometry.notna() & ~ch.geometry.is_empty].reset_index(drop=True)
    geoms = ch.geometry.to_numpy()
    edge = shapely.buffer(shapely.boundary(aoi), P["edge_m"])
    protected = np.array([g.intersects(edge) or not aoi.contains(g) for g in geoms])
    total = float(ch.length.sum()) / 1000
    out = []
    for w in widths:
        for d in depths:
            gone = peel(geoms, minor(ch, dict(P, minor_width_m=w, minor_depth_m=d)), protected, P["touch_m"])
            out.append(dict(width_m=w, depth_m=d, removed_km=round(float(ch[gone].length.sum()) / 1000, 3),
                            total_km=round(total, 3)))
    return out
