"""Build one site bundle from a site definition and the source registry (R2 milestone M1).

Bundle (``<out>/<site id>/``):
    dem.tif, dsm.tif          LINZ 1 m clips of the AOI + buffer (EPSG:2193 as published)
    labels.gpkg               harmonised layers: channels, crossings, stopbanks (+ <source>_raw layers)
    context.gpkg              roads, rail, tracks, osm_waterways, osm_culverts
    roads.gpkg                road centrelines for run_network.py --roads (LINZ if fetched, else OSM)
    site.json                 provenance: site, elevation surveys and tiles, every source query
    snapshots/                raw responses + manifest.jsonl (URL, parameters, time, sha256)
"""
from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path

from . import arcgis, linz_elevation, linz_wfs, osm, registry
from .harmonise import harmonise
from .snapshot import SnapshotStore


def build_site(site, sources: dict, out_root, cache_root, http=None, dry_run=False, only=None, log=print, index=None):
    """``index``: the coverage index (registry.load_index), needed when ``site.sources`` is ``auto``."""
    from .http import Http
    out = Path(out_root) / site.id
    out.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(out / "snapshots")
    http = http or Http(store)
    if getattr(http, "store", None) is None:
        http.store = store
    t0 = time.perf_counter()
    prov = dict(site=dict(id=site.id, role=site.role, bounds=site.bounds, buffer_m=site.buffer_m, window=site.window,
                          aoi=site.aoi_path, notes=site.notes),
                built=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), dry_run=dry_run,
                elevation=[], sources=[], errors=[])
    if only in (None, "elevation") and site.elevation:
        for product in site.elevation.get("products", ["dem", "dsm"]):
            try:
                rec = linz_elevation.fetch_product(http, site.elevation["region"], product, site.window,
                                                   out / f"{product}.tif", Path(cache_root) / "linz_stac",
                                                   site.elevation.get("survey"), dry_run=dry_run)
                prov["elevation"].append(rec)
                log(f"[{site.id}] {product}: survey {rec['survey']}, {rec['n_tiles']} tile(s)"
                    + ("" if dry_run else f", {rec['shape'][1]}x{rec['shape'][0]} cells"))
            except Exception as e:  # noqa: BLE001 - recorded, build continues
                prov["errors"].append(dict(step=f"elevation:{product}", error=str(e)))
                log(f"[{site.id}] {product}: FAILED {e}")
    layers = {}
    auto = site.sources == "auto"
    names = site.sources
    if only in (None, "sources") and auto:
        if index is None:
            raise ValueError(f"{site.id}: sources: auto needs the coverage index (scripts/index_sources.py)")
        names = registry.resolve(sources, index, site.window, site.layers, site.exclude)
        prov["sources_auto"] = dict(resolved=names, layers=site.layers, exclude=site.exclude)
        log(f"[{site.id}] sources (auto): {', '.join(names) or 'none'}")
    if only in (None, "sources"):
        for name in names:
            spec = sources.get(name) if isinstance(name, str) else name
            name = name if isinstance(name, str) else spec.get("name", "inline")
            if spec is None:
                prov["errors"].append(dict(step=f"source:{name}", error="not in the source registry"))
                continue
            if spec.get("enabled", True) is False:
                prov["sources"].append(dict(name=name, skipped=spec.get("note", "disabled")))
                continue
            try:
                got = _fetch(http, store, name, spec, site.window, dry_run)
                prov["sources"].append(dict(name=name, **got["meta"]))
                for lyr, g in got.get("layers", {}).items():
                    layers.setdefault(lyr, []).append(g)
                m = got["meta"]
                if dry_run:
                    msg = f"planned; server reports {m['count']} feature(s)" if "count" in m else "planned"
                else:
                    msg = ", ".join(f"{k} {len(v)}" for k, v in got.get("layers", {}).items() if not k.endswith("_raw")) or "0 features"
                if m.get("fallback"):
                    msg += f" (using fallback {m['fallback']}: {m.get('ref') or m.get('url')})"
                log(f"[{site.id}] {name}: {msg}")
                if m.get("missing_fields"):
                    w = dict(step=f"source:{name}", warning=f"fields in sources.yml not in the layer: {', '.join(m['missing_fields'])}")
                    prov.setdefault("warnings", []).append(w)
                    log(f"    WARNING {w['warning']}")
                if not dry_run and not got.get("layers") and not auto:     # auto: coverage is coarse, empty is normal
                    w = dict(step=f"source:{name}", warning="0 features in the window", query_url=m.get("query_url"))
                    prov.setdefault("warnings", []).append(w)
                    if m.get("query_url"):
                        log(f"    check in a browser: {m['query_url']}")
            except Exception as e:  # noqa: BLE001
                prov["errors"].append(dict(step=f"source:{name}", error=str(e)))
                log(f"[{site.id}] {name}: FAILED {e}")
    if not dry_run:
        _write_layers(out, layers, site.window)
        import geopandas as gpd                                 # the AOI (no buffer), for run_network --aoi
        from shapely.geometry import box
        gpd.GeoDataFrame({"site": [site.id]}, geometry=[box(*site.bounds)], crs=2193).to_file(out / "aoi.geojson", driver="GeoJSON")
    prov["seconds"] = round(time.perf_counter() - t0, 1)
    (out / "site.json").write_text(json.dumps(prov, indent=2, default=str))
    return prov


def _fetch(http, store, name, spec, window, dry_run):
    import pandas as pd
    kind = spec["kind"]
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    used = {}
    if kind == "arcgis":                                    # url, else the first fallback that answers
        url, i = arcgis.first_working(http, spec)
        spec = dict(spec, url=url)
        used = dict(fallback=i) if i else {}
        if dry_run:
            miss = arcgis.missing_fields(http.json(url, {"f": "json"}), spec)
            used.update(missing_fields=miss) if miss else None
    if dry_run:
        meta = dict(kind=kind, planned=True, ref=spec.get("url") or spec.get("layer_id") or kind, **used)
        if kind == "arcgis":                                # ask the server how many features would come
            meta["count"], meta["query_url"] = arcgis.count(http, spec["url"], window, spec.get("where", "1=1"))
        return dict(meta=meta)
    if kind == "arcgis":
        g, meta = arcgis.query(http, spec["url"], window, spec.get("where", "1=1"), source=name)
        lyrs = {}
        if len(g):
            lyrs[spec["layer"]] = harmonise(g, spec, name, now)
            lyrs[f"{name}_raw"] = g
            if lyrs[spec["layer"]].attrs.get("missing_fields"):
                used["missing_fields"] = lyrs[spec["layer"]].attrs["missing_fields"]
        return dict(meta=dict(kind=kind, **meta, **used), layers=lyrs)
    if kind == "linz_wfs":
        g, meta = linz_wfs.get_layer(http, spec["layer_id"], window, store=store, source=name)
        lyrs = {spec["layer"]: harmonise(g, spec, name, now)} if len(g) else {}
        return dict(meta=dict(kind=kind, **meta), layers=lyrs)
    if kind == "osm":
        parts, meta = osm.query(http, window, spec.get("endpoint", osm.OVERPASS), source=name)
        lyrs = {}
        for k, target in (("road", "osm_roads"), ("rail", "osm_rail"), ("waterway", "osm_waterways"), ("culvert", "osm_culverts"),
                          ("building", "osm_buildings")):
            if k in parts and len(parts[k]):
                lyrs[target] = parts[k].assign(source=name, retrieved=now)
        return dict(meta=dict(kind=kind, **meta), layers=lyrs)
    if kind == "file":
        import geopandas as gpd
        from shapely.geometry import box
        g = gpd.read_file(spec["path"]).to_crs(2193)
        g = g[g.intersects(box(*window))]
        return dict(meta=dict(kind=kind, path=spec["path"], count=len(g)),
                    layers={spec["layer"]: harmonise(g, spec, name, now)} if len(g) else {})
    raise ValueError(f"unknown source kind {kind!r}")


def _write_layers(out, layers, window=None):
    """Labels keep whole features (a culvert or channel crossing the edge keeps its length); context
    layers and roads are clipped to the window, since sources return whole features (a road or
    railway line of tens of km)."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    label_layers = ("channels", "crossings", "stopbanks", "bridges")

    def clip(g):
        if window is None or not len(g):
            return g
        c = g.clip(box(*window))
        return c[~c.geometry.is_empty].reset_index(drop=True)
    for f in ("labels.gpkg", "context.gpkg", "roads.gpkg"):
        if (out / f).exists():
            (out / f).unlink()
    for lyr, parts in layers.items():
        g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=2193)
        is_label = lyr in label_layers or lyr.endswith("_raw")
        _to_gpkg(g if is_label else clip(g), out / ("labels.gpkg" if is_label else "context.gpkg"), lyr)
    roads = layers.get("roads") or layers.get("osm_roads")
    if roads:
        _to_gpkg(clip(gpd.GeoDataFrame(pd.concat(roads, ignore_index=True), crs=2193)), out / "roads.gpkg", "roads")
    imp = implied_crossings(layers)
    if len(imp):
        _to_gpkg(imp, out / "labels.gpkg", "crossings_implied")


def implied_crossings(layers, dedupe_m=5.0, record_m=15.0):
    """Points where a labelled channel crosses a road or railway: a structure must be there (a culvert,
    or a bridge), recorded or not. Environment Southland's drainage lines cross highways with no culvert
    layer; council registers miss private and some road culverts. ``recorded`` marks a labelled crossing
    within ``record_m``. Roads and rail: LINZ Topo50 if fetched, else OSM."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import MultiPoint, Point

    def gdf(name):
        parts = layers.get(name) or []
        return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=2193) if parts else None
    ch = gdf("channels")
    barriers = [(kind, g) for kind, g in (("road", gdf("roads") if layers.get("roads") else gdf("osm_roads")),
                                          ("rail", gdf("rail") if layers.get("rail") else gdf("osm_rail"))) if g is not None]
    if ch is None or not barriers:
        return gpd.GeoDataFrame(geometry=[], crs=2193)
    rows = []
    for kind, b in barriers:
        idx = b.sindex
        for _, c in ch.iterrows():
            for j in idx.query(c.geometry, predicate="intersects"):
                x = c.geometry.intersection(b.geometry.iloc[j])
                pts = [x] if isinstance(x, Point) else list(x.geoms) if isinstance(x, MultiPoint) else \
                      [g for g in getattr(x, "geoms", []) if isinstance(g, Point)]
                for pt in pts:
                    rows.append(dict(barrier=kind, barrier_source=b.source.iloc[j] if "source" in b else None,
                                     barrier_id=b.source_id.iloc[j] if "source_id" in b else None, channel_source=c.get("source"),
                                     channel_id=c.get("source_id"), channel_class=c.get("class"), geometry=pt))
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs=2193)
    g = gpd.GeoDataFrame(rows, crs=2193)
    keep = []
    for i, pt in enumerate(g.geometry):                           # one point per crossing location
        if all(pt.distance(g.geometry.iloc[k]) > dedupe_m for k in keep):
            keep.append(i)
    g = g.iloc[keep].reset_index(drop=True)
    recs = [r for r in (gdf("crossings"), gdf("bridges")) if r is not None]
    g["recorded"] = g.geometry.apply(lambda p: any(bool((r.distance(p) <= record_m).any()) for r in recs))
    g["type"] = "implied"
    return g


def _to_gpkg(g, path, layer):
    g = g.copy()
    for c in g.columns:
        if c != "geometry" and g[c].dtype == object:
            g[c] = g[c].map(lambda v: v if v is None or isinstance(v, (str, int, float, bool)) else str(v))
    g.to_file(path, layer=layer, driver="GPKG")
