"""ML-0 step 3: labels for every crossing candidate of a run, from the bundle's labels and the manual reviews.

Sources and their meaning (DESIGN_NOTES section 8: council data reliable, Matt's labels partially reliable,
unlabelled is the target, never a negative):

    kind       from                                          struct   class     weight (config)
    council    council culverts (labels.gpkg crossings)       1        culvert   1.0
    asset      other asset registers (KiwiRail culverts)      1        culvert   1.0
    bridge     LINZ and council bridges (labels.gpkg bridges) 1        bridge    1.0
    manual     Matt's reviews (y / n; unsure is recorded)     1 / 0    culvert   0.6
    implied    labelled channel x road or rail                1        unknown   0.7

Each label is matched to the nearest candidate within its radius, at the crest (the midpoint of a culvert
line; a bridge by distance to its line, since a span can be long). A candidate takes the label of the
highest-priority kind (council, asset, bridge, then manual, then implied); disagreeing labels on one candidate
are recorded as conflicts (a council record wins). Other candidates within a positive label's radius are
``near_positive`` (duplicates of the same structure: neither positive nor unlabelled). Labels inside the core
(the AOI: tile ownership) and the floodplain with no candidate in reach are misses (recall failures for ML-3).

Outputs, beside the run: ``ml_labels.gpkg`` (layers ``candidates``, ``misses``, ``conflicts``) and
``ml_labels.json`` (counts).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PRIORITY = {"council": 0, "asset": 0, "bridge": 0, "manual": 1, "implied": 2}
DEFAULTS = dict(weights=dict(council=1.0, asset=1.0, bridge=1.0, manual=0.6, implied=0.7),
                radius_m=dict(council=15.0, asset=15.0, bridge=15.0, manual=10.0, implied=15.0),
                council_sources=[], asset_sources=["kiwirail_culverts"], manual_root=None, manual=[])


def load_config(path=None) -> dict:
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


def crest(geom):
    """Where a culvert passes under its barrier: the midpoint of a line, or the point itself."""
    if geom.geom_type in ("LineString", "MultiLineString"):
        return geom.interpolate(0.5, normalized=True)
    return geom.centroid


def load_manual(config: dict, crs=2193):
    """Matt's review labels from the files in ``config['manual']`` (in order; a later file wins where two
    labels lie within 5 m). Each entry: file, label (column), optional x and y (columns), round (a name)."""
    import geopandas as gpd
    import pandas as pd
    root = Path(config.get("manual_root") or ".")
    parts = []
    for i, e in enumerate(config.get("manual") or []):
        f = Path(e["file"])
        f = f if f.is_absolute() else root / f
        if not f.exists():
            raise FileNotFoundError(f"manual label file not found: {f}")
        g = gpd.read_file(f)
        g = g.to_crs(crs) if g.crs is not None else g.set_crs(crs)
        if e.get("x") and e.get("y"):
            g = g.set_geometry(gpd.points_from_xy(g[e["x"]], g[e["y"]]), crs=crs)
        lab = g[e["label"]].astype(str).str.strip().str.lower()
        g = g.assign(value=lab.map({"y": 1.0, "n": 0.0, "unsure": np.nan}), raw=lab, file=f.name,
                     round=e.get("round", f.stem), order=i)
        g = g[lab.isin(["y", "n", "unsure"])]
        parts.append(g[["value", "raw", "file", "round", "order", "geometry"]])
    if not parts:
        return gpd.GeoDataFrame(columns=["value", "raw", "file", "round", "order", "geometry"], geometry="geometry", crs=crs)
    m = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=crs).sort_values("order", ascending=False)
    keep, taken = [], []
    for i, p in zip(m.index, m.geometry):                 # newest file first; drop older labels within 5 m
        if all(p.distance(q) > 5.0 for q in taken):
            keep.append(i)
            taken.append(p)
    return m.loc[keep].sort_values("order").reset_index(drop=True)


def _name(kind: str, source) -> str:
    source = str(source)
    return source if source.startswith(kind + ":") else f"{kind}:{source}"


def _labels(bundle: Path, manual, config: dict, crs):
    """All labels as one table: kind, source, source_id, struct, cls, weight, radius, geometry (crest)."""
    import geopandas as gpd
    import pandas as pd
    import pyogrio
    rows = []
    lab = bundle / "labels.gpkg"
    have = {n for n, _ in pyogrio.list_layers(lab)} if lab.exists() else set()
    W, R = config["weights"], config["radius_m"]
    if "crossings" in have:
        c = gpd.read_file(lab, layer="crossings").to_crs(crs)
        for _, r in c.iterrows():
            kind = ("council" if r.source in config["council_sources"] else
                    "asset" if r.source in config["asset_sources"] else None)
            if kind is None:
                continue
            rows.append(dict(kind=kind, source=r.source, source_id=str(r.get("source_id")), struct=1.0, cls="culvert",
                             geometry=crest(r.geometry), match_geom=crest(r.geometry)))
    if "bridges" in have:
        b = gpd.read_file(lab, layer="bridges").to_crs(crs)
        for _, r in b.iterrows():
            rows.append(dict(kind="bridge", source=r.get("source"), source_id=str(r.get("source_id")), struct=1.0,
                             cls="bridge", geometry=crest(r.geometry), match_geom=r.geometry))
    if "crossings_implied" in have:
        im = gpd.read_file(lab, layer="crossings_implied").to_crs(crs)
        for _, r in im.iterrows():
            rows.append(dict(kind="implied", source=f"implied:{r.get('barrier')}", source_id=str(r.get("channel_id")),
                             struct=1.0, cls="unknown", geometry=r.geometry, match_geom=r.geometry))
    if manual is not None and len(manual):
        for _, r in manual.iterrows():
            rows.append(dict(kind="manual", source=f"manual:{r['round']}", source_id=r["file"], struct=r["value"],
                             cls="culvert" if r["value"] == 1 else "", geometry=r.geometry, match_geom=r.geometry))
    if not rows:
        return gpd.GeoDataFrame(columns=["kind", "source", "source_id", "struct", "cls", "weight", "radius", "geometry"],
                                geometry="geometry", crs=crs)
    L = gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry", crs=crs)
    L["weight"] = L.kind.map(W).astype(float)
    L["radius"] = L.kind.map(R).astype(float)
    return L


def label_run(run_dir, bundle_dir, manual=None, config=None, aoi=None, write=True):
    """(candidates, misses, conflicts, report) for one run. ``aoi``: the core (default: the bundle's
    aoi.geojson). ``manual``: from load_manual (labels outside this run are ignored)."""
    import geopandas as gpd
    import pyogrio
    config = config or load_config()
    run_dir, bundle_dir = Path(run_dir), Path(bundle_dir)
    gpkg = run_dir / "network.gpkg"
    C = gpd.read_file(gpkg, layer="crossings")
    crs = C.crs
    core = (gpd.read_file(aoi).to_crs(crs) if aoi else gpd.read_file(bundle_dir / "aoi.geojson").to_crs(crs)).union_all()
    have = {n for n, _ in pyogrio.list_layers(gpkg)}
    fp = gpd.read_file(gpkg, layer="floodplain").to_crs(crs).union_all() if "floodplain" in have else None
    if manual is not None and len(manual):
        manual = manual[manual.intersects(core.buffer(250))]
    L = _labels(bundle_dir, manual, config, crs)
    L["in_core"] = L.geometry.intersects(core)
    L["in_scope"] = L.geometry.intersects(fp) if fp is not None else True

    n = len(C)
    best = [None] * n                                         # (priority, label index)
    near_pos = np.zeros(n, bool)
    cand_labels = [[] for _ in range(n)]
    matched = np.zeros(len(L), bool)
    if n and len(L):
        tree = C.sindex
        for li, r in L.iterrows():
            idx = tree.query(r.match_geom.buffer(r.radius), predicate="intersects")
            if not len(idx):
                continue
            d = C.geometry.iloc[idx].distance(r.match_geom).to_numpy()
            ok = d <= r.radius
            if not ok.any():
                continue
            idx, d = idx[ok], d[ok]
            j = int(idx[np.argmin(d)])
            matched[li] = True
            cand_labels[j].append((li, float(d.min())))
            p = PRIORITY[r.kind]
            if best[j] is None or p < best[j][0]:
                best[j] = (p, li)
            if r.struct == 1:
                for k in idx:
                    if k != j:
                        near_pos[k] = True
    out = C.copy()
    cols = dict(label_kind=[], label_source=[], label_id=[], struct=[], struct_class=[], weight=[], label_dist_m=[],
                n_labels=[], label_sources_all=[])
    conflicts = []
    for j in range(n):
        labs = cand_labels[j]
        if best[j] is None:
            for k, v in (("label_kind", ""), ("label_source", ""), ("label_id", ""), ("struct", np.nan),
                         ("struct_class", ""), ("weight", 0.0), ("label_dist_m", np.nan), ("n_labels", 0),
                         ("label_sources_all", "")):
                cols[k].append(v)
            continue
        r = L.loc[best[j][1]]
        values = {L.loc[li, "struct"] for li, _ in labs if not np.isnan(L.loc[li, "struct"])}
        if len(values) > 1:
            conflicts.append(dict(candidate=j, x=out.x.iloc[j], y=out.y.iloc[j], kept=_name(r.kind, r.source),
                                  labels=";".join(f"{_name(L.loc[li, 'kind'], L.loc[li, 'source'])}={L.loc[li, 'struct']:g}"
                                                  for li, _ in labs), geometry=out.geometry.iloc[j]))
        cols["label_kind"].append(r.kind)
        cols["label_source"].append(r.source)
        cols["label_id"].append(r.source_id)
        cols["struct"].append(r.struct)
        cols["struct_class"].append(r.cls)
        cols["weight"].append(r.weight if not np.isnan(r.struct) else 0.0)
        cols["label_dist_m"].append(dict(labs)[best[j][1]])
        cols["n_labels"].append(len(labs))
        cols["label_sources_all"].append(";".join(sorted({_name(L.loc[li, "kind"], L.loc[li, "source"]) for li, _ in labs})))
    for k, v in cols.items():
        out[k] = v
    out["near_positive"] = near_pos & np.isnan(out.struct.to_numpy(float))
    out["in_core"] = out.geometry.intersects(core)
    out["in_scope"] = out.geometry.intersects(fp) if fp is not None else True
    misses = L[~matched & L.in_core & L.in_scope & (L.struct == 1)].drop(columns="match_geom")
    conflicts = gpd.GeoDataFrame(conflicts, geometry="geometry", crs=crs) if conflicts else \
        gpd.GeoDataFrame(columns=["candidate", "x", "y", "kept", "labels", "geometry"], geometry="geometry", crs=crs)
    core_c = out[out.in_core]
    rep = dict(run=str(run_dir), bundle=str(bundle_dir), candidates=int(n), candidates_in_core=int(len(core_c)),
               positives=int((core_c.struct == 1).sum()), negatives=int((core_c.struct == 0).sum()),
               unsure=int(((core_c.n_labels > 0) & core_c.struct.isna()).sum()),
               near_positive=int(core_c.near_positive.sum()), unlabelled=int((core_c.n_labels == 0).sum()),
               by_kind={k: int(v) for k, v in core_c[core_c.n_labels > 0].label_kind.value_counts().items()},
               labels_in_core={k: int(v) for k, v in L[L.in_core].kind.value_counts().items()},
               misses={k: int(v) for k, v in misses.kind.value_counts().items()}, conflicts=int(len(conflicts)))
    if write:
        f = run_dir / "ml_labels.gpkg"
        if f.exists():
            f.unlink()
        out.to_file(f, layer="candidates", driver="GPKG")
        if len(misses):
            misses.to_file(f, layer="misses", driver="GPKG")
        if len(conflicts):
            conflicts.to_file(f, layer="conflicts", driver="GPKG")
        (run_dir / "ml_labels.json").write_text(json.dumps(rep, indent=1))
    return out, misses, conflicts, rep
