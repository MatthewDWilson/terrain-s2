"""ML-3a: council channel lines for the channel model, with a half-width and a weight per line.

Design: docs/ml3_channel_model.md (section 7, Matt's decisions); configuration: ml/config/channels.yml.

Council lines are read from the bundle's raw layers (``<source>_raw`` in labels.gpkg) and harmonised again with the
current ``sources.yml``, so a field mapping added later (a width, a form) applies to bundles built before it. A
bundle without the raw layer falls back to its harmonised ``channels`` layer.

Half-width per line (``half_width_m``, with ``width_source``), in order:

    council   the council's top width, else its base width, halved, if the width is plausible
    river     the line lies mostly inside a wide-channel polygon (network.gpkg ``rivers``): the polygon is the
              channel, so no half-width (the label rasters burn the polygon)
    physics   the channel map's half-width at the line's cells (distance to the edge of ``channel_mask``, median),
              if enough of the line lies in the mask; meant for snapped lines (council lines are drawn off the bed)
    default   by class (drain, stream, race, other)

Weight per line: the council's ``positive_weight`` times its form weight (Auckland's swales lower).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULTS = dict(
    councils={},
    negative_weight=0.05,
    ignore_ring_m=3.0,
    form_weights={},
    width=dict(plausible_width_m=[0.3, 30.0], physics_min_inside=0.5, physics_half_width_m=[0.5, 4.0],
               river_min_inside=0.5, default_half_width_m=dict(drain=1.0, stream=2.0, race=0.75, other=1.0)),
)
COLUMNS = ["council", "source", "source_id", "class", "form", "width_m", "base_width_m", "weight"]


def load_config(path=None) -> dict:
    """The ML-3 label configuration: DEFAULTS updated from the YAML file (nested dicts merged one level)."""
    import yaml
    c = json.loads(json.dumps(DEFAULTS))
    if path:
        d = yaml.safe_load(Path(path).read_text()) or {}
        for k, v in d.items():
            if isinstance(v, dict) and isinstance(c.get(k), dict):
                c[k].update(v)
            else:
                c[k] = v
    return c


def council_lines(bundle, sources: dict, cfg: dict, crs=None):
    """Council channel lines of a bundle, one row per single line, with council, class, form, widths and weight.

    ``sources``: the parsed ``sources.yml`` entries (``acquire.site.load_sources``)."""
    import geopandas as gpd
    import pandas as pd
    import pyogrio

    from ..acquire.harmonise import harmonise
    lab = Path(bundle) / "labels.gpkg"
    if not lab.exists():
        return gpd.GeoDataFrame(columns=COLUMNS, geometry=[], crs=crs)
    have = {n for n, _ in pyogrio.list_layers(lab)}
    harmonised = None
    parts = []
    for council, c in (cfg.get("councils") or {}).items():
        for s in c.get("sources") or []:
            if f"{s}_raw" in have and s in sources:
                raw = gpd.read_file(lab, layer=f"{s}_raw")
                g = harmonise(raw, sources[s], s, retrieved=None) if len(raw) else None
            elif "channels" in have:
                if harmonised is None:
                    harmonised = gpd.read_file(lab, layer="channels")
                g = harmonised[harmonised["source"] == s] if "source" in harmonised else None
            else:
                g = None
            if g is None or not len(g):
                continue
            g = g.copy()
            g["council"] = council
            w = float(c.get("positive_weight", 1.0))
            form = g["form"] if "form" in g else pd.Series(None, index=g.index)
            fw = cfg.get("form_weights") or {}
            g["weight"] = [w * float(fw.get(f, fw.get("other", 1.0)) if isinstance(f, str) else 1.0) for f in form]
            parts.append(g)
    if not parts:
        return gpd.GeoDataFrame(columns=COLUMNS, geometry=[], crs=crs)
    out = pd.concat([p.to_crs(crs or parts[0].crs) for p in parts], ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=crs or parts[0].crs)
    out = out[out.geometry.notna() & out.geom_type.isin(["LineString", "MultiLineString"])]
    out = out.explode(index_parts=False, ignore_index=True)
    for c in COLUMNS:
        if c not in out:
            out[c] = None
    for c in ("width_m", "base_width_m"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out[COLUMNS + ["geometry"]]


def edge_distance(mask, transform):
    """Distance (m) from each channel cell to the nearest non-channel cell: the half-width at the centreline."""
    from scipy import ndimage as ndi
    return ndi.distance_transform_edt(np.asarray(mask) > 0) * abs(transform.a)


def _cells(line, transform, shape, step):
    """Row, column of the cells along a line, sampled every ``step`` m (cells off the grid dropped)."""
    n = max(int(np.ceil(line.length / step)), 1)
    from ..drains import _to_cr
    pts = [line.interpolate(d) for d in np.linspace(0.0, line.length, n + 1)]
    cf, rf = _to_cr(transform, [p.x for p in pts], [p.y for p in pts])
    c, r = np.floor(cf).astype(int), np.floor(rf).astype(int)
    ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    return r[ok], c[ok], len(pts)


def physics_half_width(line, edt, transform, step=1.0):
    """(median half-width over the line's cells inside the channel map, share of the line inside it)."""
    r, c, n = _cells(line, transform, edt.shape, step)
    v = edt[r, c]
    inside = v > 0
    return (float(np.median(v[inside])) if inside.any() else np.nan), float(inside.sum() / max(n, 1))


def resolve_widths(lines, cfg: dict, edt=None, transform=None, rivers=None):
    """Add ``half_width_m`` and ``width_source`` (council, river, physics, default) to council lines.

    ``edt``: ``edge_distance`` of the channel map on the grid ``transform`` (None: no physics width);
    ``rivers``: wide-channel polygons (GeoSeries or GeoDataFrame in the lines' CRS; None: none)."""
    W = dict(DEFAULTS["width"], **(cfg.get("width") or {}))
    lo, hi = W["plausible_width_m"]
    plo, phi = W["physics_half_width_m"]
    defaults = W["default_half_width_m"]
    out = lines.copy()
    river = None
    if rivers is not None and len(rivers):
        river = (rivers.geometry if hasattr(rivers, "geometry") else rivers).union_all()
    half, source = [], []
    cls_col = out["class"] if "class" in out else [None] * len(out)
    for g, wt, wb, cls in zip(out.geometry, out["width_m"], out["base_width_m"], cls_col):
        w = next((v for v in (wt, wb) if v is not None and np.isfinite(v) and lo <= v <= hi), None)
        if w is not None:
            half.append(w / 2.0); source.append("council")
            continue
        if river is not None and g.length > 0 and g.intersection(river).length / g.length >= W["river_min_inside"]:
            half.append(np.nan); source.append("river")
            continue
        if edt is not None:
            h, inside = physics_half_width(g, edt, transform)
            if inside >= W["physics_min_inside"] and np.isfinite(h):
                half.append(float(np.clip(h, plo, phi))); source.append("physics")
                continue
        cls = cls if isinstance(cls, str) else "drain"
        half.append(float(defaults.get(cls, defaults.get("other", 1.0)))); source.append("default")
    out["half_width_m"] = half
    out["width_source"] = source
    return out
