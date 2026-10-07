#!/usr/bin/env python
"""Diagnose one ArcGIS source for one site: layer metadata, feature count in the window, the start
of the first response, and a URL to paste into a browser.

    python scripts/probe_source.py --site sites/canterbury1.yml --source waimakariri_channels
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from terrain_s2.acquire import arcgis  # noqa: E402
from terrain_s2.acquire.http import Http  # noqa: E402
from terrain_s2.acquire.site import load_site, load_sources  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--sources", default=str(ROOT / "sites" / "sources.yml"))
    a = ap.parse_args()
    site = load_site(a.site, ROOT)
    spec = load_sources(a.sources)[a.source]
    if spec["kind"] != "arcgis":
        sys.exit(f"{a.source} is kind {spec['kind']}; this probe is for arcgis sources")
    http = Http()
    url = arcgis.resolve_layer(http, spec["url"])
    meta = http.json(url, {"f": "json"})
    if "error" in meta:
        sys.exit(f"layer metadata error: {meta['error']}")
    ext = meta.get("extent", {})
    print(f"layer: {meta.get('name')} | {meta.get('geometryType')} | maxRecordCount {meta.get('maxRecordCount')} | "
          f"paging {(meta.get('advancedQueryCapabilities') or {}).get('supportsPagination')} | formats {meta.get('supportedQueryFormats')}")
    print(f"layer extent: {ext.get('xmin'):.0f}, {ext.get('ymin'):.0f}, {ext.get('xmax'):.0f}, {ext.get('ymax'):.0f} "
          f"(wkid {ext.get('spatialReference', {}).get('latestWkid') or ext.get('spatialReference', {}).get('wkid')})")
    print(f"site window (EPSG:2193): {', '.join(f'{v:.0f}' for v in site.window)}")
    for where in dict.fromkeys([spec.get("where", "1=1"), "1=1"]):
        try:
            n, q = arcgis.count(http, spec["url"], site.window, where)
            print(f"count where {where!r}: {n}\n    {q}")
        except Exception as e:  # noqa: BLE001
            print(f"count where {where!r}: FAILED {e}")
    x0, y0, x1, y1 = site.window
    p = dict(where=spec.get("where", "1=1"), geometry=f"{x0},{y0},{x1},{y1}", geometryType="esriGeometryEnvelope",
             inSR=2193, spatialRel="esriSpatialRelIntersects", outFields="*", outSR=4326, f="geojson", resultRecordCount=2)
    body = http.get(f"{url}/query", p)
    print("first response (geojson, 2 records):", body[:600].decode(errors="replace"))
    print("    ", arcgis.query_url(url, p))


if __name__ == "__main__":
    main()
