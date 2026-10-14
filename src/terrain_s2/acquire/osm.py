"""OpenStreetMap via the Overpass API: roads, railways, waterways, culverts (tunnel=culvert), buildings.
Patchy in places: use as context and weak labels, never as the only evidence."""
from __future__ import annotations

OVERPASS = "https://overpass-api.de/api/interpreter"
# public mirrors, tried in turn when the main server is busy (504 / 429 under many parallel workers, wave 1)
MIRRORS = ["https://overpass.kumi.systems/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
WAITS_S = [20, 60, 120]                   # between rounds over all servers
KEEP = ["highway", "railway", "waterway", "tunnel", "culvert", "bridge", "layer", "name", "surface", "building",
        "service", "access", "intermittent", "usage"]


def _post(http, endpoints, q, source, sleep=None):
    """The Overpass response from the first server that answers; rounds over all of them with waits between."""
    import time
    sleep = sleep or time.sleep
    errors = []
    for r, wait in enumerate([0] + WAITS_S):
        if wait:
            sleep(wait)
        for ep in endpoints:
            try:
                return http.get(ep, None, source, method="POST", data={"data": q})
            except Exception as e:  # noqa: BLE001 - next server, next round
                errors.append(f"{ep}: {e}")
    raise IOError("Overpass failed on every server: " + " | ".join(errors[-len(endpoints):]))


def query(http, window_2193, endpoint: str = OVERPASS, source: str | None = "osm-overpass", mirrors=None, sleep=None):
    import json

    import geopandas as gpd
    from shapely.geometry import LineString, Polygon

    from .linz_elevation import to_lonlat
    w, s, e, n = to_lonlat(window_2193)
    bb = f"{s},{w},{n},{e}"
    q = (f"[out:json][timeout:120];(way[\"highway\"]({bb});way[\"railway\"]({bb});way[\"waterway\"]({bb});"
         f"way[\"tunnel\"=\"culvert\"]({bb});way[\"building\"]({bb}););out tags geom;")
    eps = [endpoint] + [m for m in (MIRRORS if mirrors is None else mirrors) if m != endpoint]
    d = json.loads(_post(http, eps, q, source, sleep))
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
