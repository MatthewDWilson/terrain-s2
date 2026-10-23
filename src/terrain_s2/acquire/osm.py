"""OpenStreetMap via the Overpass API or a local extract: roads, railways, waterways, culverts (tunnel=culvert),
buildings; and (channel network v2, Oct 2026) water areas and the coastline: ``water`` polygons with ``water_kind``
``river_area`` (natural=water with water=river / stream / canal / oxbow / riverbank, or waterway=riverbank),
``lake`` (other natural=water) or ``tidal`` (tidal flats, salt marsh, mangrove, mud), and ``coastline`` lines
(natural=coastline). Patchy in places: use as context, guidance and weak labels, never as the only evidence of
a channel's position. Over Overpass, multipolygon relations are not assembled (closed ways only); the extract
(GDAL's OSM driver) assembles them."""
from __future__ import annotations

OVERPASS = "https://overpass-api.de/api/interpreter"
# public mirrors, tried in turn when the main server is busy (504 / 429 under many parallel workers, wave 1)
MIRRORS = ["https://overpass.kumi.systems/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
WAITS_S = [20, 60, 120]                   # between rounds over all servers
KEEP = ["highway", "railway", "waterway", "tunnel", "culvert", "bridge", "layer", "name", "surface", "building",
        "service", "access", "intermittent", "usage", "natural", "water", "wetland"]
RIVER_WATER = {"river", "stream", "canal", "oxbow", "riverbank", "ditch", "drain"}
TIDAL = {"tidalflat", "saltmarsh", "mangrove"}


def water_kind(t: dict):
    """``river_area``, ``lake``, ``tidal`` or None for a polygon's tags."""
    nat, water, ww = t.get("natural"), t.get("water"), t.get("waterway")
    if ww == "riverbank" or (nat == "water" and water in RIVER_WATER):
        return "river_area"
    if nat == "water":
        return "lake"
    if nat == "mud" or (nat == "wetland" and t.get("wetland") in TIDAL):
        return "tidal"
    return None


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
         f"way[\"tunnel\"=\"culvert\"]({bb});way[\"building\"]({bb});way[\"natural\"~\"^(water|wetland|mud|coastline)$\"]({bb}););"
         f"out tags geom;")
    eps = [endpoint] + [m for m in (MIRRORS if mirrors is None else mirrors) if m != endpoint]
    d = json.loads(_post(http, eps, q, source, sleep))
    rows, geoms = [], []
    for el in d.get("elements", []):
        if el.get("type") != "way" or len(el.get("geometry", [])) < 2:
            continue
        t = el.get("tags", {})
        kind = _kind(t)
        xy = [(p["lon"], p["lat"]) for p in el["geometry"]]
        if kind == "water" and t.get("waterway") == "riverbank" and xy[0] != xy[-1]:
            continue
        if kind in ("building", "water"):
            if len(xy) < 4 or xy[0] != xy[-1]:
                continue                                   # unclosed building way: skip
            geoms.append(Polygon(xy))
        else:
            geoms.append(LineString(xy))
        rows.append(dict(osm_id=el["id"], kind=kind, **{k: t.get(k) for k in KEEP}))
    if not rows:
        return {}, dict(count=0, query=q)
    g = gpd.GeoDataFrame(rows, geometry=geoms, crs=4326).to_crs(2193)
    g = _water_kinds(g)
    return {k: g[g.kind == k].reset_index(drop=True) for k in KINDS if (g.kind == k).any()}, \
        dict(count=len(g), query=q, endpoint=endpoint, relations_assembled=False)


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


KINDS = ("road", "rail", "waterway", "culvert", "building", "water", "coastline")


def _kind(t):
    if t.get("tunnel") == "culvert":
        return "culvert"
    if t.get("building"):
        return "building"
    if t.get("natural") == "coastline":
        return "coastline"
    if water_kind(t):
        return "water"
    return ("rail" if t.get("railway") else "waterway" if t.get("waterway") else "road" if t.get("highway")
            else "other")


def _water_kinds(g):
    """``water_kind`` on the water rows (None elsewhere)."""
    if not len(g):
        return g
    g = g.copy()
    g["water_kind"] = [water_kind({k: r.get(k) for k in ("natural", "water", "waterway", "wetland")})
                       if r.get("kind") == "water" else None for r in g.to_dict("records")]
    return g


def _where(layer, fields):
    """The extract's filter for a layer of GDAL's OSM driver, with the tags it holds as fields or in other_tags."""
    def tag(k, values):
        if k in fields:
            return f"{k} IN ({', '.join(repr(v) for v in values)})"
        return "(" + " OR ".join(f"other_tags LIKE '%\"{k}\"=>\"{v}\"%'" for v in values) + ")"
    if layer == "lines":
        return " OR ".join([EXTRACT_WHERE["lines"], tag("natural", ["coastline"])])
    return " OR ".join(["building IS NOT NULL", tag("natural", ["water", "wetland", "mud"]),
                        tag("waterway", ["riverbank"])])


def extract_pbf(pbf, out_gpkg, batch_size=200_000, log=print):
    """Convert an OSM extract (.osm.pbf, or .osm) once into a GeoPackage in NZTM with what ``query`` returns: layer
    ``ways`` (roads, rail, waterways, culverts, coastline; line geometry), ``buildings`` and ``water`` (polygons, with
    ``water_kind``), with ``osm_id``,
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
    for src_layer in ("lines", "multipolygons"):
        fields = set(pyogrio.read_info(str(pbf), layer=src_layer)["fields"])
        with pyogrio.raw.open_arrow(str(pbf), layer=src_layer, where=_where(src_layer, fields), batch_size=batch_size,
                                    use_pyarrow=True) as (meta, reader):
            fields = set(meta["fields"])
            for b in reader:
                d = b.to_pandas()
                gcol = next(c for c in d.columns if c not in fields)
                d = d.astype({c: object for c in d.columns if c != gcol}).where(d.notna(), None)   # NaN -> None
                tags = [dict(_hstore(o), **{k: r.get(k) for k in ("highway", "railway", "waterway", "building", "name",
                                                                  "natural") if r.get(k) is not None})
                        for o, r in zip(d.get("other_tags", [None] * len(d)), d.to_dict("records"))]
                rows = pd.DataFrame([dict(osm_id=str(i if i is not None else r.get("osm_way_id")), kind=_kind(t),
                                          **{k: t.get(k) for k in KEEP})
                                     for i, t, r in zip(d.get("osm_id", [None] * len(d)), tags, d.to_dict("records"))])
                g = gpd.GeoDataFrame(rows, geometry=shapely.from_wkb(d[gcol].to_numpy()), crs=4326).to_crs(2193)
                g = _water_kinds(g)
                parts = ((("ways", g[g.kind.isin(["road", "rail", "waterway", "culvert", "coastline"])]),) if src_layer == "lines"
                         else (("buildings", g[g.kind == "building"]), ("water", g[g.kind == "water"])))
                for dst_layer, part in parts:
                    if not len(part):
                        continue
                    if dst_layer != "water":
                        part = part.drop(columns="water_kind")
                    pyogrio.write_dataframe(part, out, layer=dst_layer, append=counts.get(dst_layer, 0) > 0,
                                            promote_to_multi=dst_layer != "ways")
                    counts[dst_layer] = counts.get(dst_layer, 0) + len(part)
                log("  " + ", ".join(f"{k}: {v:,}" for k, v in counts.items()))
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
    have = {n for n, _ in pyogrio.list_layers(gpkg)}
    for layer in ("ways", "buildings", "water"):
        if layer not in have:
            continue                                   # an extract made before Oct 2026 has no water layer
        g = pyogrio.read_dataframe(gpkg, layer=layer, bbox=tuple(window_2193))
        if not len(g):
            continue
        for k in {"buildings": ("building",), "water": ("water",)}.get(layer, ("road", "rail", "waterway", "culvert",
                                                                               "coastline")):
            s = g[g.kind == k].reset_index(drop=True)
            if len(s):
                parts[k] = s
    j = Path(str(gpkg) + ".json")
    info = json.loads(j.read_text()) if j.exists() else {}
    return parts, dict(count=int(sum(len(v) for v in parts.values())), extract=str(gpkg),
                       extract_date=info.get("file_date"), extract_source=info.get("source"),
                       water_layer="water" in have)
