"""Catchments: the processing units of catchment mode (channel network v2, stage A).

Source: MfE's sea-draining catchments (https://data.mfe.govt.nz/layer/99776-sea-draining-catchments/; CC BY 4.0,
extracted from DOC's Freshwater Ecosystems of New Zealand v1.0 geodatabase), downloaded once as a GeoPackage to
``<TERRAIN_DATA_DIR>/reference/sea_draining_catchments.gpkg`` (``DEFAULT``). Their boundaries are approximate at
LiDAR scale: catchment mode uses them as units with an overlap band, not as hard boundaries.

``load``: the layer in NZTM, with ``catchment_id`` (the layer's own id field, ``catch_id`` in MfE's layer, else the
row number), ``fid`` (the GeoPackage feature id) and ``catchment_name`` (its name field when it has one).
``at(point)``: the catchment holding a point; ``by_id(value, field=...)``: by ``catch_id`` (default) or ``fid``.
``tiles_for(catchment, partition, overlap_m)``: the partition tiles meeting the catchment plus the band, with the
share of each tile inside the catchment itself.
"""
from __future__ import annotations

from pathlib import Path

DEFAULT = Path("reference") / "sea_draining_catchments.gpkg"
ID_FIELDS = ("catchment_id", "catch_id", "catchid", "nzsegment", "nzreach", "objectid", "id")   # matched ignoring case
NAME_FIELDS = ("catchment_name", "Name", "NAME", "name", "CatchName", "CATCH_NAME", "River", "RIVER")


def default_path(data_dir=None) -> Path:
    if data_dir is None:
        from .acquire.elevation import default_data_dir
        data_dir = default_data_dir()
    return Path(data_dir) / DEFAULT


def load(path=None, bbox=None):
    """The catchments (NZTM), with ``catchment_id`` and ``catchment_name``; ``bbox`` (NZTM) reads only those meeting it."""
    import geopandas as gpd
    path = Path(path or default_path())
    if not path.exists():
        raise FileNotFoundError(f"no catchment layer at {path}: download MfE's sea-draining catchments (layer 99776) "
                                "as a GeoPackage there")
    g = gpd.read_file(path, fid_as_index=True)
    g["fid"] = g.index.astype(int)
    if g.crs is not None and g.crs.to_epsg() != 2193:
        g = g.to_crs(2193)
    if bbox is not None:
        from shapely.geometry import box
        g = g[g.intersects(box(*bbox))]
    low = {c.lower(): c for c in g.columns if c != "geometry"}
    idf = next((low[f.lower()] for f in ID_FIELDS if f.lower() in low), None)
    if idf is None:                                   # any field named like a catchment id (catch_id, CatchID_2, ...)
        idf = next((c for lc, c in low.items() if "catch" in lc and "id" in lc), None)
    nmf = next((low[f.lower()] for f in NAME_FIELDS if f.lower() in low), None)
    g = g.copy()
    g["catchment_id"] = g[idf].astype(str) if idf else [str(i) for i in g.index]
    g["catchment_name"] = g[nmf].astype(str) if nmf else None
    g.attrs["id_field"] = idf or "(row number)"
    return g


def by_id(value, path=None, field=None):
    """The catchment row whose ``field`` (default ``catchment_id``, i.e. the layer's id field such as ``catch_id``;
    or ``fid``) equals ``value``, or None; ``id_field`` on the row says which field that was."""
    g = load(path)
    f = field or "catchment_id"
    idf = g.attrs.get("id_field")
    g = g[g[f].astype(str) == str(value)]
    if not len(g):
        return None
    r = g.iloc[0].copy()
    r["id_field"] = "fid" if field == "fid" else idf
    return r


def at(point, path=None):
    """The catchment row holding ``point`` (a shapely Point in NZTM), or None."""
    g = load(path, bbox=(point.x - 1, point.y - 1, point.x + 1, point.y + 1))
    g = g[g.contains(point)]
    return g.iloc[0] if len(g) else None


def tiles_for(catchment, partition, overlap_m: float = 500.0):
    """The partition tiles meeting the catchment geometry buffered by ``overlap_m``, with ``in_catchment`` (share of
    the tile's area inside the catchment itself) and ``overlap_only`` (none of it inside)."""
    import geopandas as gpd
    geom = getattr(catchment, "geometry", catchment)
    band = geom.buffer(overlap_m)
    t = gpd.read_file(partition, layer="tiles", bbox=tuple(band.bounds))
    if t.crs is not None and t.crs.to_epsg() != 2193:
        t = t.to_crs(2193)
    t = t[t.intersects(band)].copy()
    t["in_catchment"] = (t.geometry.intersection(geom).area / t.geometry.area).round(3)
    t["overlap_only"] = t.in_catchment <= 0
    return t.sort_values("tile_id").reset_index(drop=True)
