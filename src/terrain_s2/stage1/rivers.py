"""River and waterway bed estimation for the ``geofabrics`` backend (the stages plan section 3 left out of the
first Stage 1 templates): GeoFabrics' ``RiverBathymetryGenerator`` and ``WaterwayBedElevationEstimator``,
driven by templates (``templates/rivers.json``, ``templates/waterways.json``) and the profile's ``rivers``
section. Off by default (``TERRAIN_GF_RIVERS``, ``TERRAIN_GF_WATERWAYS``); the benchmark for ML-4 and part of
W7.

**What GeoFabrics 1.1.30 does** (read from its source, ``processor.py`` and ``bathymetry_estimation.py``):

- ``rivers``: one main channel per run. From a river network (REC) it traces upstream from a starting reach
  (``network_id``) through reaches whose upstream area is at least ``area_threshold``. It builds ground and
  vegetation DEMs along that channel (1 m, a corridor of ``river_corridor_width``), aligns the centreline,
  measures widths, bank heights and slopes on cross-sections every ``cross_section_spacing`` m, then estimates
  depths from flow, slope, width and Manning's n (Neal et al. and Rupp and Smart). The trace follows only the
  largest upstream branch at each confluence (the main stem), so a large tributary needs its own run. Flow, n and reach id come
  from the *nearest* network reach to each cross-section.
- Alignment: without ``osm``, the network centreline is aligned to the DEM (lowest ground within
  ``network_alignment_tolerance``). With ``osm: {type, id}``, that one OSM waterway (a way or relation, fetched
  from Overpass) becomes the centreline, clipped to the span between the network's ends projected onto it.
  GeoFabrics does not search OSM for a match: the id is an input.
- ``waterways``: OSM drains, streams and culverts (Overpass), with bed elevations from the DEM so open
  waterways flow downhill and culverts are unblocked. No flows needed.
- The hydrologic DEM then burns each river's bathymetry points inside its polygon (``data_source`` 3,
  "rivers and fans") and the waterways' (4).

**Here:** the network is fetched for the grid plus ``margin_m`` (an ArcGIS layer or item, or a local file;
:func:`fetch_network`), its columns mapped to GeoFabrics' names, reaches clipped to that window, and the main
channels crossing the grid chosen (:func:`select_rivers`): main stems as GeoFabrics traces them, from the
largest reach above the threshold, then from the mouths of large tributaries, at most ``max_rivers``. Each river runs as its own
GeoFabrics call (the runner takes one ``rivers`` block), in ``rivers/<network_id>/`` inside the product folder;
the DEM stage then lists them all in ``data_paths.rivers``.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

GF_COLUMNS = ("id", "to_node", "from_node", "flow", "mannings_n", "area")
RIVERS_TEMPLATE = Path(__file__).parent / "templates" / "rivers.json"
WATERWAYS_TEMPLATE = Path(__file__).parent / "templates" / "waterways.json"


class RiverNetworkError(ValueError):
    """The river network is not configured, or lacks a field the mapping names."""


@dataclass
class River:
    network_id: int
    area_m2: float
    n_reaches: int
    flow_min: float | None
    flow_max: float | None
    osm: dict | None = None

    def record(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class RiverPlan:
    """What :func:`plan` found: the mapped network (GeoDataFrame, EPSG of the profile), the rivers to run,
    and the record that goes into the generator key and the provenance."""
    network: object
    rivers: list = field(default_factory=list)
    source: dict = field(default_factory=dict)

    def record(self) -> dict:
        return {"source": self.source, "rivers": [r.record() for r in self.rivers]}


def config(profile: dict) -> dict:
    return (profile.get("rivers") or {})


def window(bounds, margin_m: float) -> tuple:
    x0, y0, x1, y1 = bounds
    return (x0 - margin_m, y0 - margin_m, x1 + margin_m, y1 + margin_m)


def fetch_network(http, win, profile: dict, *, h_epsg: int = 2193):
    """The river network inside ``win`` (working CRS) with its raw columns, and a record of where it came
    from. ``rivers.network.source``: ``{kind: arcgis, url: <layer URL or ArcGIS item id>}`` or
    ``{kind: file, path: <vector file>}``."""
    net = config(profile).get("network") or {}
    src = net.get("source") or {}
    kind = src.get("kind")
    if kind == "arcgis":
        from ..acquire import arcgis
        g, info = arcgis.query(http, str(src["url"]), win, where=src.get("where", "1=1"),
                               source="rivers-network")
        return g.to_crs(h_epsg) if len(g) else g, {"kind": "arcgis", **{k: info.get(k) for k in ("url", "where", "count")}}
    if kind == "file":
        import geopandas as gpd
        path = Path(str(src["path"])).expanduser()
        from shapely.geometry import box
        g = gpd.read_file(path, bbox=gpd.GeoSeries([box(*win)], crs=h_epsg))   # reprojected to the file's CRS
        g = g.to_crs(h_epsg) if g.crs is not None else g.set_crs(h_epsg)
        from ..key import file_sha256
        return g, {"kind": "file", "path": str(path), "sha256": file_sha256(path), "count": len(g)}
    raise RiverNetworkError("rivers.network.source in the profile must be {kind: arcgis, url: ...} or "
                            "{kind: file, path: ...}")


def map_columns(raw, profile: dict):
    """Rename the network's columns to GeoFabrics' (``id, to_node, from_node, flow, mannings_n, area``).
    ``rivers.network.columns`` maps each to a source field; ``mannings_n`` may instead be a constant
    (``rivers.network.mannings_n``). A missing field raises, listing the fields the layer has."""
    import geopandas as gpd
    net = config(profile).get("network") or {}
    cols = dict(net.get("columns") or {})
    have = [c for c in raw.columns if c != "geometry"]
    lower = {c.lower(): c for c in have}

    def find(name):
        f = cols.get(name)
        if f is None:
            return None
        if f in raw.columns:
            return f
        if str(f).lower() in lower:                      # services differ in case (NZREACH vs nzreach)
            return lower[str(f).lower()]
        raise RiverNetworkError(f"river network field {f!r} (for {name}) not found; the layer has: {have}")

    out = {}
    for name in ("id", "to_node", "from_node", "area", "flow"):
        f = find(name)
        if f is None:
            raise RiverNetworkError(f"rivers.network.columns.{name} is not set in the profile; the layer has: {have}")
        out[name] = raw[f]
    f = find("mannings_n")
    if f is not None:
        out["mannings_n"] = raw[f]
    elif net.get("mannings_n") is not None:
        out["mannings_n"] = float(net["mannings_n"])
    else:
        raise RiverNetworkError("set rivers.network.columns.mannings_n or a constant rivers.network.mannings_n")
    g = gpd.GeoDataFrame(out, geometry=raw.geometry.values, crs=raw.crs)
    for c in ("id", "to_node", "from_node"):
        g[c] = g[c].astype("int64")
    for c in ("area", "flow", "mannings_n"):
        g[c] = g[c].astype("float64")
    return g[list(GF_COLUMNS) + ["geometry"]]


def clip(network, win):
    from shapely.geometry import box
    b = box(*win)
    g = network[network.intersects(b)].copy()
    g["geometry"] = g.geometry.intersection(b)
    return g[~g.geometry.is_empty].reset_index(drop=True)


def upstream(network, start_id: int, area_threshold: float) -> list:
    """The main stem GeoFabrics traces from ``start_id`` (``Channel._get_up_stream_reaches``): repeatedly the
    upstream reach (``to_node`` equal to this reach's ``from_node``) with the largest area, until there is none
    or the largest is below ``area_threshold``. Tributaries are not followed. Returns reach ids, downstream
    first."""
    by_to = {}
    for rid, to, a in zip(network["id"].astype(int), network["to_node"].astype(int), network["area"]):
        by_to.setdefault(to, []).append((float(a), rid))
    frm = dict(zip(network["id"].astype(int), network["from_node"].astype(int)))
    if int(start_id) not in frm:
        return []
    stem = [int(start_id)]
    while True:
        ups = by_to.get(frm[stem[-1]], [])
        if not ups:
            break
        a, rid = max(ups)
        if a < area_threshold or rid in stem:
            break
        stem.append(rid)
    return stem


def select_rivers(network, grid_bounds, area_threshold: float, max_rivers: int = 5) -> list:
    """The channels crossing the grid, as GeoFabrics runs them (one main stem per run). Reaches at or above
    ``area_threshold`` are taken largest first; each one not already on a chosen stem starts a stem (so the
    first is the largest river's lowest reach in the window, later ones the mouths of large tributaries).
    Stems that do not meet the grid box are skipped. At most ``max_rivers``."""
    from shapely.geometry import box
    gbox = box(*grid_bounds)
    big = network[network["area"] >= area_threshold].sort_values("area", ascending=False)
    covered, rivers = set(), []
    for _, o in big.iterrows():
        rid = int(o["id"])
        if rid in covered:
            continue
        stem = upstream(network, rid, area_threshold)
        covered.update(stem)
        reaches = network[network["id"].isin(stem)]
        if not reaches.intersects(gbox).any():
            continue
        fl = reaches["flow"].dropna()
        rivers.append(River(network_id=rid, area_m2=float(o["area"]), n_reaches=len(stem),
                            flow_min=round(float(fl.min()), 4) if len(fl) else None,
                            flow_max=round(float(fl.max()), 4) if len(fl) else None))
        if len(rivers) >= max_rivers:
            break
    return rivers


def plan(http, grid_bounds, profile: dict, log=print) -> RiverPlan:
    """Fetch, map and clip the network around the grid and choose the rivers (no GeoFabrics yet)."""
    from ..key import digest
    c = config(profile)
    h = int(profile["crs"]["horizontal"])
    win = window(grid_bounds, float((c.get("network") or {}).get("margin_m", 1000)))
    raw, src = fetch_network(http, win, profile, h_epsg=h)
    if len(raw) == 0:
        log("rivers: no network reaches in the window")
        return RiverPlan(network=raw, rivers=[], source=dict(src, digest=None))
    net = clip(map_columns(raw, profile), win)
    thr = float(c.get("area_threshold_m2", 1e7))
    rivers = select_rivers(net, grid_bounds, thr, int(c.get("max_rivers", 5)))
    osm = c.get("osm") or {}                     # {network_id: {type, id}} for rivers aligned to OSM
    for r in rivers:
        o = osm.get(r.network_id) or osm.get(str(r.network_id))
        r.osm = dict(o) if o else None
    content = json.loads(net.to_json(drop_id=True))
    src = dict(src, digest=digest(content, 64), area_threshold_m2=thr)
    log(f"rivers: {len(net)} reaches in the window, {len(rivers)} river(s) above {thr / 1e6:g} km2: "
        f"{[r.network_id for r in rivers]}")
    return RiverPlan(network=net, rivers=rivers, source=src)


def _strip(d: dict, keys) -> dict:
    d = copy.deepcopy(d)
    for k in keys:
        d.pop(k, None)
    return d


def river_instructions(base_default: dict, river: River, network_file: Path, profile: dict) -> dict:
    """One ``{"rivers": {...}}`` runner call for ``river``: the rivers template over the product's
    ``default`` block (without its DEM outputs, extents and land, which belong to the main DEM)."""
    from .gf import fill, merge
    c = config(profile)
    t = json.loads(RIVERS_TEMPLATE.read_text())
    stage = fill(t, {"subfolder": f"rivers/{river.network_id}", "network_id": river.network_id,
                     "network_file": str(Path(network_file).resolve()),
                     "area_threshold": float(c.get("area_threshold_m2", 1e7)),
                     "osm": river.osm})
    stage = merge(stage, {"rivers": c.get("parameters") or {}})
    d = copy.deepcopy(base_default)
    d["data_paths"] = _strip(d["data_paths"], ("extents", "land", "raw_dem", "result_dem", "result_geofabric"))
    return {"rivers": merge(d, stage)}


def waterways_instructions(base_default: dict, profile: dict) -> dict:
    """One ``{"waterways": {...}}`` runner call over the product's extents (OSM drains, streams, culverts)."""
    from .gf import fill, merge
    c = profile.get("waterways") or {}
    stage = fill(json.loads(WATERWAYS_TEMPLATE.read_text()), {"subfolder": "waterways"})
    stage = merge(stage, {"waterways": c.get("parameters") or {}})
    d = copy.deepcopy(base_default)
    d["data_paths"] = _strip(d["data_paths"], ("raw_dem", "result_dem", "result_geofabric"))
    d["data_paths"]["extents"] = str((Path(d["data_paths"]["local_cache"]) / base_default["data_paths"]["extents"]).resolve())
    return {"waterways": merge(d, stage)}


def dem_paths(rivers: list, waterways: bool) -> dict:
    """``data_paths`` additions for the DEM stage (relative to the product folder)."""
    out = {}
    if rivers:
        out["rivers"] = [{"extents": f"rivers/{r.network_id}/river_polygon.geojson",
                          "elevations": f"rivers/{r.network_id}/river_bathymetry.geojson"} for r in rivers]
    if waterways:
        out["waterways"] = [{"extents": "waterways/closed_waterways_polygon.geojson",
                             "elevations": "waterways/closed_waterways_elevation.geojson"},
                            {"extents": "waterways/open_waterways_polygon.geojson",
                             "elevations": "waterways/open_waterways_elevation.geojson"}]
    return out
