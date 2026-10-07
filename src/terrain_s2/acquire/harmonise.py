"""Map source attributes to the common label schema (version 1; Matt's review files are version 0).

Per-source mapping in sources.yml:
    layer: channels | crossings | stopbanks | roads | rail | tracks
    const:  {type: culvert}                       # fixed values
    fields: {asset_id: ASSNBRI, length_m: LENGTH_m}  # target: source field
    scale:  {diameter_m: [DIAMETER_mm, 0.001]}    # target: [source field, factor]
    class_map: {field: CLASSIFICATION3, values: {Network Drain: drain, Receiving Waterway: stream}}
Raw attributes are kept in a separate *_raw layer for traceability.
Note: read the schema column ``type`` as ``g["type"]``; ``g.type`` is the GeoDataFrame's geometry type.
"""
from __future__ import annotations

SCHEMA = {
    "channels": ["class", "exists", "source", "source_id", "positional_accuracy", "retrieved"],
    "crossings": ["type", "diameter_m", "width_m", "height_m", "length_m", "cells", "source", "source_id", "retrieved"],
    "stopbanks": ["type", "crest_height_m", "source", "source_id", "retrieved"],
    "roads": ["class", "surface", "lanes", "source", "source_id", "retrieved"],
    "rail": ["status", "source", "source_id", "retrieved"],
    "tracks": ["class", "source", "source_id", "retrieved"],
    "buildings": ["use", "area_m2", "source", "source_id", "retrieved"],
}


def harmonise(g, spec: dict, source: str, retrieved: str):
    import numpy as np
    import pandas as pd
    layer = spec["layer"]
    out = pd.DataFrame(index=g.index)
    for k, v in (spec.get("const") or {}).items():
        out[k] = v
    for k, f in (spec.get("fields") or {}).items():
        out[k] = g[f] if f in g else None
    for k, (f, factor) in (spec.get("scale") or {}).items():
        v = pd.to_numeric(g[f], errors="coerce") if f in g else np.nan
        out[k] = np.where(v > 0, v * factor, np.nan) if f in g else np.nan     # 0 = unknown in council data
    cm = spec.get("class_map")
    if cm:
        target = cm.get("target", "class" if layer != "crossings" else "type")
        out[target] = g[cm["field"]].map(cm["values"]).fillna(cm.get("other", "other")) if cm["field"] in g else None
    if layer == "buildings" and "area_m2" not in out:
        out["area_m2"] = g.geometry.area.values
    out["source"] = source
    out["retrieved"] = retrieved
    if "source_id" not in out:
        idf = spec.get("id_field")
        out["source_id"] = g[idf].astype(str) if idf and idf in g else g.index.astype(str)
    for c in SCHEMA.get(layer, []):
        if c not in out:
            out[c] = None
    import geopandas as gpd
    return gpd.GeoDataFrame(out[SCHEMA.get(layer, list(out.columns))], geometry=g.geometry.values, crs=g.crs)
