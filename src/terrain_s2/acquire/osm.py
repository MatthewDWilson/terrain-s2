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


# ---- a local extract (Geofabrik's .osm.pbf converted once), instead of Overpass (wave 3, 17 Oct 2026: the public
# servers refused parallel workers and were slow even one tile at a time)

EXTRACT_WHERE = dict(
    lines="highway IS NOT NULL OR railway IS NOT NULL OR waterway IS NOT NULL OR other_tags LIKE '%\"tunnel\"=>\"culvert\"%'",
    multipolygons="building IS NOT NULL")


def _hstore(s):
    """GDAL's OSM ``other_tags`` ('"k"=>"v","k2"=>"v2"') as a dict."""
    import re
    if not isinstance(s, str) or not s:
        return {}
    return {k: v.replace('\\"', '"') for k, v in re.findall(r'"((?:[^"\\]|\\.)*)"=>"((?:[^"\\]|\\.)*)"', s)}


def _kind(t):
    return ("culvert" if t.get("tunnel") == "culvert" else "building" if t.get("building") else
            "rail" if t.get("railway") else "waterway" if t.get("waterway") else "road" if t.get("highway") else "other")


def extract_pbf(pbf, out_gpkg, batch_size=200_000, log=print):
    """Convert an OSM extract (.osm.pbf, or .osm) once into a GeoPackage in NZTM with what ``query`` returns: layer
    ``ways`` (roads, rail, waterways, culverts; line geometry) and ``buildings`` (polygons), with ``osm_id``,
    ``kind`` and the ``KEEP`` tags; R-tree indexed, so a tile's window reads in a moment. Streams in batches (NZ:
    a few minutes per layer, little memory). Writes ``<out>.json`` with the source file and its date."""
    import datetime as _dt
    import json
    from pathlib import Path

    import geopandas as gpd
    import pandas as pd
    import pyogrio
    import shapely
    pbf, out = Path(pbf), Path(out_gpkg)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    counts = {}
    for src_layer, dst_layer in (("lines", "ways"), ("multipolygons", "buildings")):
        n = 0
        with pyogrio.raw.open_arrow(str(pbf), layer=src_layer, where=EXTRACT_WHERE[src_layer], batch_size=batch_size,
                                    use_pyarrow=True) as (meta, reader):
            fields = set(meta["fields"])
            for b in reader:
                d = b.to_pandas()
                gcol = next(c for c in d.columns if c not in fields)
                d = d.astype({c: object for c in d.columns if c != gcol}).where(d.notna(), None)   # NaN -> None
                tags = [dict(_hstore(o), **{k: r.get(k) for k in ("highway", "railway", "waterway", "building", "name")
                                            if r.get(k) is not None})
                        for o, r in zip(d.get("other_tags", [None] * len(d)), d.to_dict("records"))]
                rows = pd.DataFrame([dict(osm_id=str(i), kind=_kind(t), **{k: t.get(k) for k in KEEP})
                                     for i, t in zip(d["osm_id"], tags)])
                g = gpd.GeoDataFrame(rows, geometry=shapely.from_wkb(d[gcol].to_numpy()), crs=4326).to_crs(2193)
                if dst_layer == "ways":
                    g = g[g.kind != "other"]
                pyogrio.write_dataframe(g, out, layer=dst_layer, append=n > 0, promote_to_multi=dst_layer == "buildings")
                n += len(g)
                log(f"  {dst_layer}: {n:,}")
        counts[dst_layer] = n
    st = pbf.stat()
    info = dict(source=str(pbf), size_bytes=st.st_size,
                file_date=_dt.datetime.fromtimestamp(st.st_mtime, _dt.timezone.utc).isoformat(timespec="seconds"),
                converted=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), counts=counts)
    Path(str(out) + ".json").write_text(json.dumps(info, indent=1))
    return info


def from_extract(gpkg, window_2193):
    """What ``query`` returns, from a local extract (``extract_pbf``): the ways and buildings meeting the window."""
    import json
    from pathlib import Path

    import pyogrio
    parts = {}
    for layer in ("ways", "buildings"):
        g = pyogrio.read_dataframe(gpkg, layer=layer, bbox=tuple(window_2193))
        if not len(g):
            continue
        for k in (("building",) if layer == "buildings" else ("road", "rail", "waterway", "culvert")):
            s = g[g.kind == k].reset_index(drop=True)
            if len(s):
                parts[k] = s
    j = Path(str(gpkg) + ".json")
    info = json.loads(j.read_text()) if j.exists() else {}
    return parts, dict(count=int(sum(len(v) for v in parts.values())), extract=str(gpkg),
                       extract_date=info.get("file_date"), extract_source=info.get("source"))
