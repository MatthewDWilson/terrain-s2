"""OpenStreetMap via the Overpass API: roads, railways, waterways, culverts (tunnel=culvert), buildings.
Patchy in places: use as context and weak labels, never as the only evidence."""
from __future__ import annotations

OVERPASS = "https://overpass-api.de/api/interpreter"
KEEP = ["highway", "railway", "waterway", "tunnel", "culvert", "bridge", "layer", "name", "surface", "building",
        "service", "access", "intermittent", "usage"]


def query(http, window_2193, endpoint: str = OVERPASS, source: str | None = "osm-overpass"):
    import json

    import geopandas as gpd
    from shapely.geometry import LineString, Polygon

    from .linz_elevation import to_lonlat
    w, s, e, n = to_lonlat(window_2193)
    bb = f"{s},{w},{n},{e}"
    q = (f"[out:json][timeout:120];(way[\"highway\"]({bb});way[\"railway\"]({bb});way[\"waterway\"]({bb});"
         f"way[\"tunnel\"=\"culvert\"]({bb});way[\"building\"]({bb}););out tags geom;")
    d = json.loads(http.get(endpoint, None, source, method="POST", data={"data": q}))
    rows, geoms = [], []
    for el in d.get("elements", []):
        if el.get("type") != "way" or len(el.get("geometry", [])) < 2:
            continue
        t = el.get("tags", {})
        kind = ("culvert" if t.get("tunnel") == "culvert" else "building" if "building" in t else
                "rail" if "railway" in t else "waterway" if "waterway" in t else "road" if "highway" in t else "other")
        xy = [(p["lon"], p["lat"]) for p in el["geometry"]]
        if kind == "building":
            if len(xy) < 4 or xy[0] != xy[-1]:
                continue                                   # unclosed building way: skip
            geoms.append(Polygon(xy))
        else:
            geoms.append(LineString(xy))
        rows.append(dict(osm_id=el["id"], kind=kind, **{k: t.get(k) for k in KEEP}))
    if not rows:
        return {}, dict(count=0, query=q)
    g = gpd.GeoDataFrame(rows, geometry=geoms, crs=4326).to_crs(2193)
    return {k: g[g.kind == k].reset_index(drop=True) for k in ("road", "rail", "waterway", "culvert", "building")
            if (g.kind == k).any()}, \
        dict(count=len(g), query=q, endpoint=endpoint)
