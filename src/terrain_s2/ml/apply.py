"""ML-1 step 9: score a run's crossing candidates with a trained scorer, and add the recorded structures.

``score_run(res, model, bundle_dir)`` (a ``run.network(..., keep=True)`` result) computes the candidates'
features as in training (``features.tabular``, the bundle's context features, for a scorer trained on
channel-model runs the U-Net's evidence from the run's probability in memory, the derived distance to any way),
and adds to the ``crossings`` layer:

- ``p_structure``: the scorer's score (a ranking: unlabelled candidates were weak negatives in training);
- ``p_bridge``: bridge rather than culvert, given a structure;
- ``model_version``: the model's name and the code that trained it;
- ``recorded`` and ``record``: the candidate matches a structure in a register (council, asset) within
  ``match_m`` of its crest. Every recorded structure is in the output, whether or not the model ranks its
  candidate highly, and recorded structures with no candidate are added as rows of their own (``source`` =
  ``record:<source>``, ``tier`` = ``recorded``, ``p_structure`` = 1), as agreed (DESIGN_NOTES section 8).

The rule-based ``tier`` stays, for comparison. Used by ``run.network(..., crossing_model=...)``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


CHANNEL_FEATURES = ("ml_channel", "ml_prob_max", "ml_sharp_max")


def needs_channel_model(model) -> bool:
    """The scorer was trained on channel-model runs (its features include the U-Net's evidence, ML-3c)."""
    return any(f in CHANNEL_FEATURES for f in model.get("features", []))


def model_features(res, bundle_dir, channel=None):
    """The feature table for the run's written candidates, as the dataset builds it. ``channel``: the run's
    channel-model probability and its transform, ``(prob, transform)`` (``ml_prob`` in the run's state), for a
    scorer with the channel features; their order follows ``dataset.tile_rows``: context, then channel, then
    the derived features."""
    from . import dataset, features, train
    tab, _, _ = features.extract(res, None, write=False, with_patches=False)
    tab = features.normalise_source_flags(tab)
    tab = dataset.context_features(Path(bundle_dir), tab)
    if channel is not None:
        prob, T = channel
        tab = dataset.channel_features(None, tab, prob=prob, transform=T, crossings=res.layers.get("crossings"))
    return train.add_derived(tab)


def records(bundle_dir, sources, crs):
    """Register structures (crests) from the bundle's labels: council and asset culverts."""
    import geopandas as gpd
    import pyogrio

    from .labels import crest
    f = Path(bundle_dir) / "labels.gpkg"
    if not f.exists() or "crossings" not in {n for n, _ in pyogrio.list_layers(f)}:
        return gpd.GeoDataFrame(columns=["source", "source_id", "geometry"], geometry="geometry", crs=crs)
    c = gpd.read_file(f, layer="crossings").to_crs(crs)
    c = c[c.source.isin(sources)] if sources else c.iloc[0:0]
    return gpd.GeoDataFrame(c[["source", "source_id"]].reset_index(drop=True),
                            geometry=[crest(g) for g in c.geometry], crs=crs)


def score_run(res, model, bundle_dir, record_sources=(), match_m=15.0):
    """The scored crossings (a GeoDataFrame): see the module docstring. ``model``: train.load_model() or a path."""
    import geopandas as gpd
    import pandas as pd

    from . import train
    M = train.load_model(model) if isinstance(model, (str, Path)) else model
    cands = res.layers["crossings"]
    out = cands.copy()
    channel = None
    if needs_channel_model(M):
        st = res.state or {}
        if st.get("ml_prob") is None:
            raise ValueError(f"the scorer {M.get('meta', {}).get('name', model)} was trained with the channel model's "
                             "evidence (ml_* features): run it with its channel model (a model set, ml/model_sets/)")
        channel = (st["ml_prob"], st["w"].transform)
    if len(cands):
        tab = model_features(res, bundle_dir, channel)
        X = tab.reindex(columns=M["features"]).to_numpy(np.float32)
        out["p_structure"] = M["structure"].predict_proba(X)[:, 1]
        out["p_bridge"] = M["bridge"].predict_proba(X)[:, 1] if M.get("bridge") is not None else np.nan
    else:
        out["p_structure"], out["p_bridge"] = [], []
    meta = M.get("meta", {})
    out["model_version"] = f"{meta.get('name', 'model')}@{meta.get('code', '?')}"
    out["recorded"] = False
    out["record"] = ""
    R = records(bundle_dir, list(record_sources), cands.crs)
    # records outside this run's AOI belong to its neighbours (tile ownership)
    if len(R) and len(cands):
        aoi = res.state["aoi"] if res.state is not None else None
        if aoi is not None:
            R = R[R.intersects(aoi.to_crs(cands.crs).union_all())]
    new = []
    if len(R):
        tree = out.sindex if len(out) else None
        for _, r in R.iterrows():
            j = None
            if tree is not None:
                idx = tree.query(r.geometry.buffer(match_m), predicate="intersects")
                if len(idx):
                    d = out.geometry.iloc[idx].distance(r.geometry).to_numpy()
                    if d.min() <= match_m:
                        j = int(idx[np.argmin(d)])
            name = f"{r.source}:{r.source_id}"
            if j is not None:
                out.iat[j, out.columns.get_loc("recorded")] = True
                prev = out.iat[j, out.columns.get_loc("record")]
                out.iat[j, out.columns.get_loc("record")] = f"{prev};{name}" if prev else name
            else:
                new.append(dict(x=r.geometry.x, y=r.geometry.y, source=f"record:{r.source}", tier="recorded",
                                tier_dem_only="recorded", p_structure=1.0, p_bridge=np.nan, recorded=True, record=name,
                                model_version=out["model_version"].iloc[0] if len(out) else "", geometry=r.geometry))
    if new:
        out = gpd.GeoDataFrame(pd.concat([out, gpd.GeoDataFrame(new, geometry="geometry", crs=cands.crs)],
                                         ignore_index=True), geometry="geometry", crs=cands.crs)
    return out
