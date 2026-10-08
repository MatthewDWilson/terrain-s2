"""LINZ Data Service WFS (Topo50 roads 50329, railways 50319, tracks 50364). Needs an API key in
the LINZ_API_KEY environment variable; the key is never written to snapshots.

WFS 1.0.0 is used on purpose: its bbox and coordinates are always easting, northing. In WFS 2.0,
EPSG:2193 may be read in its official northing, easting order, which silently selects the wrong
area; features returned are also checked against the window.
"""
from __future__ import annotations

import os

BASE = "https://data.linz.govt.nz/services;key={key}/wfs"
LAYERS = {"roads": 50329, "rail": 50319, "tracks": 50364}


def get_layer(http, layer_id: int, window_2193, key: str | None = None, store=None, source=None,
              max_features: int = 100000):
    import geopandas as gpd
    key = key or os.environ.get("LINZ_API_KEY")
    if not key:
        raise RuntimeError("LINZ_API_KEY is not set (create a key at data.linz.govt.nz, account, API keys)")
    x0, y0, x1, y1 = window_2193
    params = dict(service="WFS", version="1.0.0", request="GetFeature", typeName=f"layer-{layer_id}",
                  srsName="EPSG:2193", bbox=f"{x0},{y0},{x1},{y1}", outputFormat="json", maxFeatures=max_features)
    content = http.get(BASE.format(key=key), params)
    if store is not None and source:
        store.put(source, content, BASE.format(key="<redacted>"), params)
    import json
    feats = json.loads(content).get("features") or []
    g = gpd.GeoDataFrame.from_features(feats) if feats else gpd.GeoDataFrame(geometry=[])
    if len(g) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=2193), dict(layer=layer_id, count=0)
    b = g.total_bounds
    g = g.set_crs(4326 if max(abs(b[0]), abs(b[2])) <= 180 else 2193, allow_override=True).to_crs(2193)
    from shapely.geometry import box
    win = box(*window_2193)
    inside = g.intersects(win.buffer(1000))
    if not inside.any():
        # WFS selects by bounding box, so a long line (a 346 km railway) can be returned for a window it
        # never enters. Only features lying entirely elsewhere point to a swapped axis order.
        if box(*g.total_bounds).intersects(win):
            return gpd.GeoDataFrame(geometry=[], crs=2193), dict(layer=layer_id, count=0, bbox_only=len(g))
        raise ValueError(f"LINZ layer {layer_id}: {len(g)} features returned, none near the window (axis order?)")
    if len(g) >= max_features:
        raise ValueError(f"LINZ layer {layer_id}: max_features ({max_features}) reached; split the window")
    return g[inside].reset_index(drop=True), dict(layer=layer_id, count=int(inside.sum()))
