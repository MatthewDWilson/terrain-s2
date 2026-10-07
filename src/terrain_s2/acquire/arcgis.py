"""ArcGIS REST feature layers (council asset services, ArcGIS Hub items): envelope query with paging."""
from __future__ import annotations

import json
import re

HUB_ITEM = re.compile(r"^[0-9a-f]{32}(_\d+)?$")


def resolve_layer(http, ref: str) -> str:
    """A layer URL as given, or an ArcGIS Hub/Online item id (``<32 hex>[_<layer>]``) resolved to one."""
    if not HUB_ITEM.match(ref):
        return ref.rstrip("/")
    item, _, layer = ref.partition("_")
    meta = http.json(f"https://www.arcgis.com/sharing/rest/content/items/{item}", {"f": "json"})
    url = meta.get("url")
    if not url:
        raise ValueError(f"ArcGIS item {item} has no service URL")
    return f"{url.rstrip('/')}/{layer or 0}"


class ArcGISError(IOError):
    pass


def _check(d: dict, url: str):
    """ArcGIS reports failures inside an HTTP 200 response: {"error": {"code", "message", "details"}}."""
    if isinstance(d, dict) and "error" in d:
        e = d["error"] or {}
        raise ArcGISError(f"{url}: ArcGIS error {e.get('code')}: {e.get('message')} {e.get('details') or ''}".strip())
    return d


def query_url(url, params):
    """The full GET URL of a query, for pasting into a browser when diagnosing a source."""
    import requests
    return requests.Request("GET", f"{url}/query", params=params).prepare().url


def count(http, ref: str, window_2193, where: str = "1=1"):
    url = resolve_layer(http, ref)
    x0, y0, x1, y1 = window_2193
    p = dict(where=where, geometry=f"{x0},{y0},{x1},{y1}", geometryType="esriGeometryEnvelope", inSR=2193,
             spatialRel="esriSpatialRelIntersects", returnCountOnly="true", f="json")
    d = _check(http.json(f"{url}/query", p), url)
    return int(d.get("count", 0)), query_url(url, p)


def query(http, ref: str, window_2193, where: str = "1=1", out_fields: str = "*", source: str | None = None):
    """All features intersecting the window, as a GeoDataFrame in EPSG:2193.

    GeoJSON is requested in WGS84 (as the format defines) and reprojected here; if the server
    refuses GeoJSON, Esri JSON in EPSG:2193 is requested instead and its geometry converted here."""
    import geopandas as gpd
    import pandas as pd
    url = resolve_layer(http, ref)
    meta = _check(http.json(url, {"f": "json"}), url)
    page = int(meta.get("maxRecordCount") or 1000)
    paging = bool((meta.get("advancedQueryCapabilities") or {}).get("supportsPagination", False))
    x0, y0, x1, y1 = window_2193
    base = dict(where=where, geometry=f"{x0},{y0},{x1},{y1}", geometryType="esriGeometryEnvelope", inSR=2193,
                spatialRel="esriSpatialRelIntersects", outFields=out_fields, outSR=4326, f="geojson")
    fmt = "geojson"
    try:
        probe = dict(base, resultOffset=0, resultRecordCount=1) if paging else dict(base, returnCountOnly="true", f="json")
        _check(json.loads(http.get(f"{url}/query", probe)), url)
    except ArcGISError:
        fmt = "json"                                        # GeoJSON refused: fall back to Esri JSON
        base.update(f="json", outSR=2193)
    frames = []
    if paging:
        off = 0
        while True:
            content = http.get(f"{url}/query", dict(base, resultOffset=off, resultRecordCount=page), source)
            g, more = _read(content, url, fmt)
            if len(g):
                frames.append(g)
            if len(g) == 0 or (not more and len(g) < page):
                break
            off += len(g)
    else:
        ids = _check(http.json(f"{url}/query", dict(base, returnIdsOnly="true", f="json")), url).get("objectIds") or []
        oid = meta.get("objectIdField") or "OBJECTID"
        for k in range(0, len(ids), page):
            chunk = ",".join(str(i) for i in ids[k:k + page])
            g, _ = _read(http.get(f"{url}/query", dict(base, where=f"{oid} IN ({chunk})"), source), url, fmt)
            if len(g):
                frames.append(g)
    info = dict(url=url, where=where, page_size=page, paging=paging, format=fmt,
                query_url=query_url(url, dict(base, resultRecordCount=page) if paging else base))
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=2193), dict(info, count=0)
    crs = 4326 if fmt == "geojson" else 2193
    g = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=crs).to_crs(2193)
    return g, dict(info, count=len(g))


def _read(content: bytes, url: str = "", fmt: str = "geojson"):
    import geopandas as gpd
    d = _check(json.loads(content), url)
    more = bool(d.get("exceededTransferLimit") or (d.get("properties") or {}).get("exceededTransferLimit"))
    feats = d.get("features") or []
    if not feats:
        return gpd.GeoDataFrame(geometry=[], crs=4326 if fmt == "geojson" else 2193), more
    if fmt == "geojson":
        # built from the parsed features, not gpd.read_file: GDAL's GeoJSON/ESRIJSON reader sees
        # exceededTransferLimit and tries to page through the source URL itself (paging is done here)
        return gpd.GeoDataFrame.from_features(feats, crs=4326), more
    rows = [f.get("attributes", {}) for f in feats]
    return gpd.GeoDataFrame(rows, geometry=[_esri_geom(f.get("geometry")) for f in feats], crs=2193), more


def _esri_geom(g):
    """Esri JSON geometry -> shapely (points, polylines, polygons)."""
    from shapely.geometry import LineString, MultiLineString, Point, Polygon
    if not g:
        return None
    if "x" in g:
        return Point(g["x"], g["y"])
    if "paths" in g:
        parts = [LineString(p) for p in g["paths"] if len(p) >= 2]
        return parts[0] if len(parts) == 1 else MultiLineString(parts)
    if "rings" in g:
        rings = [r for r in g["rings"] if len(r) >= 4]
        return Polygon(rings[0], rings[1:]) if rings else None
    return None
