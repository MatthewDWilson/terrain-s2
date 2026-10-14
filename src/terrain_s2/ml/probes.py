"""ML-0 / ML-3: why labelled structures get no candidate ("misses"), measured inside the run.

``probe_run(res, bundle_dir, label_cfg)`` takes every register structure, bridge and implied crossing in the run's
core (the bundle's labels, as ``ml.labels`` reads them), and samples at its crest what the chain saw there: the
candidate features (``features.tabular``), the distance to the nearest candidate, to the channel map and to the
mapped network's centreline, and whether the point is in the floodplain. ``reason`` then says where the chain
lost it, in order:

    found                  a candidate within ``found_m`` (15 m, the label radius)
    position               a candidate within ``position_m`` (30 m): label position, not the chain
    outside_floodplain     the point is outside the floodplain mask (REM above the threshold)
    no_channel             no channel-map cell within ``channel_m`` (10 m): the channel was not detected
    not_in_network         the channel map is there but no network centreline within ``channel_m``: cleaned away
    no_crossing_test       the network passes within ``channel_m`` but no test proposed a crossing

Writes ``ml_probes.parquet`` beside the run (the batch runner does it after the features); ``summarise`` counts
reasons by council and label kind.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

REASONS = ["found", "position", "outside_floodplain", "no_channel", "not_in_network", "no_crossing_test"]
PROBES_VERSION = "1"


def label_points(bundle_dir, label_cfg, crs, core):
    """Register culverts, bridges and implied crossings (crests) in the core, as a GeoDataFrame."""
    import geopandas as gpd
    import pandas as pd

    from .labels import _labels
    L = _labels(Path(bundle_dir), None, label_cfg, crs)
    if not len(L):
        return L
    L = L[L.geometry.intersects(core)].drop(columns=["match_geom"], errors="ignore")
    return gpd.GeoDataFrame(pd.DataFrame(L).reset_index(drop=True), geometry="geometry", crs=crs)


def probe_run(res, bundle_dir, label_cfg=None, found_m=15.0, position_m=30.0, channel_m=10.0):
    import pandas as pd

    from . import features as F
    from .labels import load_config
    label_cfg = label_cfg or load_config()
    st = res.state
    if st is None:
        raise ValueError("probes need the run's arrays: run.network(..., keep=True)")
    T = st["w"].transform
    cs = abs(T.a)
    cands = res.layers.get("crossings")
    crs = cands.crs if cands is not None else st["crs"]
    core = st["aoi"].to_crs(crs).union_all()
    P = label_points(bundle_dir, label_cfg, crs, core)
    if not len(P):
        return pd.DataFrame(columns=["kind", "source", "source_id", "x", "y", "reason"])
    pts = pd.DataFrame({"x": P.geometry.x.to_numpy(), "y": P.geometry.y.to_numpy()})
    bands = F._bands(st)
    tab = F.tabular(pts, bands, T, cs)
    r, c = F._rc(T, pts.x.to_numpy(), pts.y.to_numpy())
    H, W = bands["z"].shape
    r, c = np.clip(r, 0, H - 1), np.clip(c, 0, W - 1)
    d_chan = ndi.distance_transform_edt(~np.asarray(st["net"]["mask"], bool)) * cs
    d_cl = ndi.distance_transform_edt(~np.asarray(st["sk"], bool)) * cs
    fp = st.get("fp")
    from scipy.spatial import cKDTree
    if cands is not None and len(cands):
        dc, _ = cKDTree(np.column_stack([cands.x, cands.y])).query(pts[["x", "y"]].to_numpy())
    else:
        dc = np.full(len(pts), np.inf)
    out = pd.concat([P.drop(columns="geometry").reset_index(drop=True), tab], axis=1)
    out["dist_candidate_m"] = dc
    out["dist_channel_m"] = d_chan[r, c]
    out["dist_centreline_m"] = d_cl[r, c]
    out["in_floodplain"] = np.asarray(fp, bool)[r, c] if fp is not None else True
    reason = np.full(len(out), "no_crossing_test", dtype=object)
    reason[out.dist_centreline_m.to_numpy() > channel_m] = "not_in_network"
    reason[out.dist_channel_m.to_numpy() > channel_m] = "no_channel"
    reason[~out.in_floodplain.to_numpy(bool)] = "outside_floodplain"
    reason[dc <= position_m] = "position"
    reason[dc <= found_m] = "found"
    out["reason"] = reason
    out["probes_version"] = PROBES_VERSION
    return out


def write(res, bundle_dir, out_dir, label_cfg=None):
    t = probe_run(res, bundle_dir, label_cfg)
    t.to_parquet(Path(out_dir) / "ml_probes.parquet", index=False)
    return t


def summarise(tables):
    """Counts of reasons by label kind (and council when the tables carry it), as a DataFrame."""
    import pandas as pd
    t = pd.concat(tables, ignore_index=True)
    keys = [k for k in ("council", "kind") if k in t]
    tab = t.groupby(keys + ["reason"]).size().unstack("reason", fill_value=0)
    return tab.reindex(columns=[r for r in REASONS if r in tab.columns])
