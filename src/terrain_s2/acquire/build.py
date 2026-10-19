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

from . import arcgis, linz_wfs, osm, registry
from .harmonise import harmonise
from .snapshot import SnapshotStore


def build_site(site, sources: dict, out_root, cache_root, http=None, dry_run=False, only=None, log=print, index=None,
               mirror=True):
    """``index``: the coverage index (registry.load_index), needed when ``site.sources`` is ``auto``.
    ``cache_root``: the shared cache (``TERRAIN_DATA_DIR``: STAC documents and item indexes under ``stac/``,
    mirrored COG tiles under ``cog/``, the snapshot store under ``snapshots/``). ``mirror``: keep whole COG
    tiles locally (S2-9)."""
    from .http import Http
    out = Path(out_root) / site.id
    out.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(Path(cache_root) / "snapshots")      # one store for both stages (S2-10)
    http = http or Http(store)
    if getattr(http, "store", None) is None:
        http.store = store
    t0 = time.perf_counter()
    prov = dict(site=dict(id=site.id, role=site.role, bounds=site.bounds, buffer_m=site.buffer_m, window=site.window,
                          aoi=site.aoi_path, notes=site.notes), snapshot_store=str(store.root),
                built=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), dry_run=dry_run,
                elevation=[], sources=[], errors=[])
    if only in (None, "elevation") and site.elevation:
        _elevation(site, out, cache_root, http, dry_run, prov, log, mirror)
    layers = {}
    restricted = {}                                   # S2-11: layers of restricted sources, kept apart
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
                prov["sources"].append(dict(name=name, restricted=bool(spec.get("restricted")), **got["meta"]))
                for lyr, g in got.get("layers", {}).items():
                    if spec.get("restricted"):
                        if not (lyr in LABEL_LAYERS or lyr.endswith("_raw")):
                            raise ValueError(f"restricted source {name} provides '{lyr}': a restricted source may "
                                             "only provide label layers (it must never reach the chain's inputs)")
                        restricted.setdefault(lyr, []).append(g)
                    else:
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
        _write_restricted(out, restricted)
        import geopandas as gpd                                 # the AOI (no buffer), for run_network --aoi
        from shapely.geometry import box
        gpd.GeoDataFrame({"site": [site.id]}, geometry=[box(*site.bounds)], crs=2193).to_file(out / "aoi.geojson", driver="GeoJSON")
    prov["seconds"] = round(time.perf_counter() - t0, 1)
    (out / "site.json").write_text(json.dumps(prov, indent=2, default=str))
    return prov



def refresh_source(bundle, name, sources: dict, cache_root, http=None, log=print):
    """Fetch one source again for a built bundle and merge it in: its layers replace that source's rows in
    labels.gpkg / context.gpkg (rows of other sources kept), roads.gpkg is written from OSM roads only if the
    bundle has none, and site.json loses the source's errors and gains its record (``refreshed``). For a source that
    failed in a batch (Overpass under many parallel workers, wave 3): fix the bundle here, then rerun the tile.
    Implied crossings are not redone (they come from the LINZ roads where those were fetched). Returns the record;
    raises if the fetch fails again (site.json unchanged)."""
    import geopandas as gpd
    import pandas as pd
    import pyogrio
    from shapely.geometry import box

    from .http import Http
    bundle = Path(bundle)
    prov = json.loads((bundle / "site.json").read_text())
    window = prov["site"]["window"]
    store = SnapshotStore(Path(cache_root) / "snapshots")
    http = http or Http(store)
    if getattr(http, "store", None) is None:
        http.store = store
    got = _fetch(http, store, name, sources[name], window, False)
    label_layers = LABEL_LAYERS
    restricted = bool(sources[name].get("restricted"))
    for lyr, g in got.get("layers", {}).items():
        is_label = lyr in label_layers or lyr.endswith("_raw")
        if restricted and not is_label:
            raise ValueError(f"restricted source {name} provides '{lyr}': label layers only (S2-11)")
        f = bundle / (RESTRICTED_FILE if restricted else "labels.gpkg" if is_label else "context.gpkg")
        g = gpd.GeoDataFrame(g, crs=2193)
        if not is_label and len(g):
            g = g.clip(box(*window))
            g = g[~g.geometry.is_empty]
        have = f.exists() and lyr in {n for n, _ in pyogrio.list_layers(f)}
        if have and not lyr.endswith("_raw"):
            old = gpd.read_file(f, layer=lyr)
            if "source" in old:
                old = old[old["source"] != name]
            g = gpd.GeoDataFrame(pd.concat([old.to_crs(2193), g], ignore_index=True), crs=2193)
        if len(g):
            _to_gpkg(g.reset_index(drop=True), f, lyr)
        if lyr == "osm_roads" and not (bundle / "roads.gpkg").exists() and len(g):
            _to_gpkg(g, bundle / "roads.gpkg", "roads")
    prov["errors"] = [e for e in prov.get("errors", []) if e.get("step") != f"source:{name}"]
    prov["sources"] = [x for x in prov.get("sources", []) if x.get("name") != name] + [
        dict(name=name, refreshed=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), **got["meta"])]
    (bundle / "site.json").write_text(json.dumps(prov, indent=2, default=str))
    log(f"[{bundle.name}] {name}: " + (", ".join(f"{k} {len(v)}" for k, v in got.get("layers", {}).items()) or "0 features"))
    return prov

def _fetch(http, store, name, spec, window, dry_run):
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
        ext = spec.get("extract")                       # a local extract (scripts/osm_extract.py) when there is one
        if ext and not Path(ext).is_absolute():
            from .elevation import default_data_dir
            ext = default_data_dir() / ext
        if ext and Path(ext).exists():
            parts, meta = osm.from_extract(ext, window)
        else:
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


def _elevation(site, out, cache_root, http, dry_run, prov, log, mirror=True):
    """DEM and DSM through the shared reader (S2-9): surveys meeting the window, the site's survey first if it
    names one, newest capture next, national fill last; the DSM over the DEM's surveys. Writes
    ``<product>.tif`` and ``<product>_source.tif`` (lidar_source: the survey index per cell)."""
    from . import elevation as E
    products = site.elevation.get("products", ["dem", "dsm"])
    bounds = E.whole_metre_bounds(site.window)
    survey = site.elevation.get("survey")
    try:
        src = E.ElevationSource(http=http, cache_dir=Path(cache_root) / "stac", mirror=mirror,
                                mirror_dir=Path(cache_root) / "cog")
        dem_plan, dsm_plan = src.pair(bounds, priority=[survey] if survey else None)
    except Exception as e:  # noqa: BLE001 - recorded, build continues
        prov["errors"].append(dict(step="elevation:plan", error=str(e)))
        log(f"[{site.id}] elevation: FAILED {e}")
        return
    for product, plan in (("dem", dem_plan), ("dsm", dsm_plan)):
        if product not in products:
            continue
        if plan is None:
            prov["errors"].append(dict(step=f"elevation:{product}", error="no survey covers the window"))
            log(f"[{site.id}] {product}: FAILED no survey covers the window")
            continue
        rec = dict(product=product, bounds=list(bounds), surveys=[s_.record() for s_ in plan.surveys],
                   source_version=plan.source_version(), survey=plan.surveys[0].slug,
                   collection=plan.surveys[0].collection, title=plan.surveys[0].title,
                   licence=plan.surveys[0].licence, capture=list(plan.surveys[0].capture),
                   n_tiles=sum(len(s_.items) for s_ in plan.surveys))
        try:
            if not dry_run:
                read = src.read(plan)
                rec.update(E.write_read(read, out / f"{product}.tif", out / f"{product}_source.tif"))
                rec["used"] = [s_.slug for s_ in read.used]
                rec["tiles"] = [dict(id=i.id, href=i.href, checksum=i.checksum) for s_ in read.used for i in s_.items]
                rec["n_tiles"] = len(rec["tiles"])
                rec["survey"] = read.used[0].slug if read.used else rec["survey"]
            prov["elevation"].append(rec)
            used = rec.get("used") or [s_.slug for s_ in plan.surveys]
            log(f"[{site.id}] {product}: survey {', '.join(used)}, {rec['n_tiles']} tile(s)"
                + ("" if dry_run else f", {rec['shape'][1]}x{rec['shape'][0]} cells, valid {rec['valid_fraction']:.3f}"))
        except Exception as e:  # noqa: BLE001 - recorded, build continues
            prov["errors"].append(dict(step=f"elevation:{product}", error=str(e)))
            log(f"[{site.id}] {product}: FAILED {e}")


LABEL_LAYERS = ("channels", "crossings", "stopbanks", "bridges")
RESTRICTED_FILE = "labels_restricted.gpkg"


def _write_restricted(out, layers):
    """Restricted sources' label layers in their own file (S2-11), which nothing in the chain reads and no export
    copies; training reads it only when asked to (``use_restricted``)."""
    import geopandas as gpd
    import pandas as pd
    f = Path(out) / RESTRICTED_FILE
    if f.exists():
        f.unlink()
    for lyr, parts in layers.items():
        _to_gpkg(gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=2193), f, lyr)


def _write_layers(out, layers, window=None):
    """Labels keep whole features (a culvert or channel crossing the edge keeps its length); context
    layers and roads are clipped to the window, since sources return whole features (a road or
    railway line of tens of km)."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    label_layers = LABEL_LAYERS

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
