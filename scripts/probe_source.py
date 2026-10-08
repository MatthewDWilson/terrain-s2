#!/usr/bin/env python
"""Diagnose one ArcGIS source for one site: layer metadata, feature count in the window, the start
of the first response, and a URL to paste into a browser.

    python scripts/probe_source.py --site sites/canterbury1.yml --source waimakariri_channels
    python scripts/probe_source.py --site sites/canterbury1.yml --source waimakariri_culverts --values
    python scripts/probe_source.py --site sites/canterbury1.yml --source waimakariri_culverts --layers
    python scripts/probe_source.py --site sites/canterbury1.yml --source waimakariri_culverts --find "ASSNBRI = 'SW005887'" --scope folder
    python scripts/probe_source.py --site sites/canterbury1.yml --url <layer URL or Hub item id> --where "CLASSIFICATION3 = 'Culvert'"
    python scripts/probe_source.py --search "Stormwater Pipes Waimakariri"
    python scripts/probe_source.py --source es_stopbanks --values          # no site: the layer's whole extent
    python scripts/probe_source.py --source es_drainage --tile CE09_5000_0505 --values

The window is the site's (--site), a LINZ map-sheet tile's (--tile), a box (--bbox xmin,ymin,xmax,ymax in
EPSG:2193), or, with none of these, the layer's own extent.

--url URL    probe this layer (URL, or ArcGIS Hub/Online item id) instead of the source's own URL
--where W    filter to count with, instead of the source's ``where``
--values     value counts of the layer's text fields for the features in the window (how features are labelled);
             with --where, only the features matching it
--fields     the layer's field names and types
--layers     every layer in the same map service (id, name, geometry type), to find a layer that has moved
--find W     count features matching W in every layer of the service (--scope folder: every service in the
             same folder of the server), e.g. to find which layer holds known asset ids
--search Q   ArcGIS Online item search (open-data portals publish there): id, type, owner, service URL;
             an item id with its layer, e.g. <id>_0, can be used as a source ``url``
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from terrain_s2.acquire import arcgis  # noqa: E402
from terrain_s2.acquire.http import Http  # noqa: E402
from terrain_s2.acquire.site import load_site, load_sources  # noqa: E402

SEARCH = "https://www.arcgis.com/sharing/rest/search"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site")
    ap.add_argument("--tile", help="LINZ map-sheet tile id as the window, e.g. BW24_5000_0703")
    ap.add_argument("--bbox", help="xmin,ymin,xmax,ymax (EPSG:2193) as the window")
    ap.add_argument("--source")
    ap.add_argument("--sources", default=str(ROOT / "sites" / "sources.yml"))
    ap.add_argument("--url", help="layer URL or item id to probe instead of the source's URL")
    ap.add_argument("--where", help="filter to count with instead of the source's where")
    ap.add_argument("--values", nargs="*", metavar="FIELD", help="value counts (all text fields if none named)")
    ap.add_argument("--layers", action="store_true", help="list the layers of the map service")
    ap.add_argument("--fields", action="store_true", help="list the layer's fields")
    ap.add_argument("--find", metavar="WHERE", help="count matches of WHERE in every layer of the service")
    ap.add_argument("--scope", choices=["service", "folder"], default="service", help="where --find looks")
    ap.add_argument("--search", metavar="QUERY", help="ArcGIS Online item search; needs no site or source")
    a = ap.parse_args()
    http = Http()
    if a.search:
        return search(http, a.search)
    if not (a.source or a.url):
        ap.error("one of --source or --url is required (except with --search)")
    site = load_site(a.site, ROOT) if a.site else None
    spec = load_sources(a.sources)[a.source] if a.source else {"kind": "arcgis"}
    if spec["kind"] != "arcgis":
        sys.exit(f"{a.source} is kind {spec['kind']}; this probe is for arcgis sources")
    if a.url:
        spec = dict(spec, url=a.url, fallback_urls=[])
    if a.where:
        spec = dict(spec, where=a.where)
    url, i = arcgis.first_working(http, spec)
    print(f"using {url}" + (f"  (fallback {i}; primary failed)" if i else ""))
    meta = http.json(url, {"f": "json"})
    ext = meta.get("extent") or {}
    sr = ext.get("spatialReference") or {}
    if site is None:
        site = _Window(_window(a, ext, sr))
    print(f"layer: {meta.get('name')} | {meta.get('geometryType')} | maxRecordCount {meta.get('maxRecordCount')} | "
          f"paging {(meta.get('advancedQueryCapabilities') or {}).get('supportsPagination')} | "
          f"formats {meta.get('supportedQueryFormats')}")
    if ext:
        print(f"layer extent: {ext.get('xmin'):.0f}, {ext.get('ymin'):.0f}, {ext.get('xmax'):.0f}, {ext.get('ymax'):.0f} "
              f"(wkid {sr.get('latestWkid') or sr.get('wkid')})")
    # a layer can be published with a fixed filter, so some rows are never served (Before U Dig: culverts?)
    print(f"definition expression: {meta.get('definitionExpression')!r}")
    miss = arcgis.missing_fields(meta, spec)
    print(f"fields used by sources.yml: {', '.join(arcgis.mapped_fields(spec)) or '-'}"
          + (f"  MISSING from the layer: {', '.join(miss)}" if miss else "  (all present)"))
    if a.fields:
        for f in meta.get("fields") or []:
            print(f"  field {f.get('name')} ({(f.get('type') or '').replace('esriFieldType', '')}) {f.get('alias') or ''}")
    print(f"site window (EPSG:2193): {', '.join(f'{v:.0f}' for v in site.window)}")
    for where in dict.fromkeys([spec.get("where", "1=1"), "1=1"]):
        try:
            n, q = arcgis.count(http, url, site.window, where)
            print(f"count where {where!r}: {n}\n    {q}")
        except Exception as e:  # noqa: BLE001
            print(f"count where {where!r}: FAILED {e}")
    if a.layers:
        for lyr in _layers(http, url.rsplit("/", 1)[0]):
            print(f"  layer {lyr.get('id'):>3}: {lyr.get('name')} ({lyr.get('geometryType') or 'group'})")
    if a.find:
        find(http, url, site.window, a.find, a.scope)
    if a.values is not None:
        vw = a.where or "1=1"
        n, vals = arcgis.field_values(http, url, site.window, vw, a.values or None)
        print(f"value counts, all {n} feature(s) in the window (where {vw}):")
        for f, counts in vals.items():
            print(f"  {f}: " + "; ".join(f"{v!r} {c}" for v, c in counts))
        nums = arcgis.numeric_summary(http, url, site.window, vw)
        if nums:
            empty = [f for f, (k, *_) in nums.items() if not k]
            print(f"numeric fields with values (non-null count, min, median, max); {len(empty)} others all null:")
            for f, (k, lo, med, hi) in nums.items():
                if k:
                    print(f"  {f}: {k}, {lo}, {med}, {hi}")
    x0, y0, x1, y1 = site.window
    p = dict(where=spec.get("where", "1=1"), geometry=f"{x0},{y0},{x1},{y1}", geometryType="esriGeometryEnvelope",
             inSR=2193, spatialRel="esriSpatialRelIntersects", outFields="*", outSR=4326, f="geojson", resultRecordCount=2)
    body = http.get(f"{url}/query", p)
    print("first response (geojson, 2 records):", body[:600].decode(errors="replace"))
    print("    ", arcgis.query_url(url, p))


class _Window:
    def __init__(self, window):
        self.window = window


def _window(a, ext, sr):
    """The probe window when no site is given: --tile, --bbox, else the layer extent (in EPSG:2193)."""
    if a.tile:
        from terrain_s2 import grid
        return grid.tile_bounds(a.tile)
    if a.bbox:
        return tuple(float(v) for v in a.bbox.split(","))
    if not ext:
        sys.exit("the layer reports no extent; give --site, --tile or --bbox")
    b = (ext["xmin"], ext["ymin"], ext["xmax"], ext["ymax"])
    wkid = sr.get("latestWkid") or sr.get("wkid")
    if wkid in (2193, None):
        return b
    from rasterio.warp import transform_bounds
    return transform_bounds(f"EPSG:{3857 if wkid == 102100 else wkid}", "EPSG:2193", *b, densify_pts=21)


def _layers(http, service_url):
    d = http.json(service_url, {"f": "json"})
    return [lyr for lyr in (d.get("layers") or []) if not lyr.get("subLayerIds")]


def _services(http, layer_url, scope):
    """Service URLs to search: the layer's own service, or every Map/Feature service in its folder."""
    svc = layer_url.rsplit("/", 1)[0]
    if scope == "service":
        return [svc]
    base, _, rel = svc.partition("/rest/services/")
    folder = rel.rsplit("/", 2)[0] if rel.count("/") >= 2 else ""
    cat = http.json(f"{base}/rest/services/{folder}".rstrip("/"), {"f": "json"})
    return [f"{base}/rest/services/{s['name']}/{s['type']}" for s in cat.get("services") or []
            if s.get("type") in ("MapServer", "FeatureServer")]


def find(http, url, window, where, scope):
    print(f"layers with features where {where!r} (whole layer / site window), scope {scope}:")
    hits, n_lyr, refused = 0, 0, []
    services = _services(http, url, scope)
    for svc in services:
        try:
            layers = _layers(http, svc)
        except Exception as e:  # noqa: BLE001
            print(f"  {svc}: FAILED {e}")
            continue
        for lyr in layers:
            lu = f"{svc}/{lyr['id']}"
            n_lyr += 1
            try:
                d = http.json(f"{lu}/query", {"where": where, "returnCountOnly": "true", "f": "json"})
                if "error" in d:
                    refused.append(f"{lu} {lyr.get('name')}: {(d['error'] or {}).get('message')}")
                    continue                                 # usually: a field in WHERE is absent here
                if not d.get("count"):
                    continue
                n_win, _ = arcgis.count(http, lu, window, where)
                hits += 1
                print(f"  {lu}  {lyr.get('name')}: {d['count']} / {n_win}")
            except Exception as e:  # noqa: BLE001
                print(f"  {lu}  {lyr.get('name')}: FAILED {e}")
    print(f"  searched {len(services)} service(s), {n_lyr} layer(s); {hits} with matches; "
          f"{len(refused)} refused the filter")
    for r in refused[:40]:
        print(f"    refused: {r}")
    if not hits:
        print("  none: no layer that accepted the filter holds a matching feature")


def search(http, q):
    d = http.json(SEARCH, {"q": q, "f": "json", "num": 25, "sortField": "modified", "sortOrder": "desc"})
    print(f"{d.get('total', 0)} item(s) for {q!r} (newest first, max 25):")
    for it in d.get("results") or []:
        print(f"  {it.get('id')}  {it.get('type')}  {it.get('title')!r}  owner {it.get('owner')}\n      {it.get('url') or ''}")


if __name__ == "__main__":
    main()
