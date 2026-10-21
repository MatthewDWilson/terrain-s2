"""ML-0 step 5: one dataset from the tiles of a manifest: features, labels, folds, and a card.

For every tile of the manifest with a finished run (``done.json`` with features):

1. labels are (re)built when missing or older than the run (``labels.label_run``: a rerun replaces the
   candidates, so labels made before it are stale);
2. ``ml_features.parquet`` and the labelled candidates are joined on (x, y) (both are the run's written
   ``crossings`` layer, in the same order; a count or position mismatch stops that tile with a message);
3. tile fields from the manifest are added (council, sheet, split, holdout, region, survey) and each row
   gets ``cand_id`` (``<tile>:<row>``) and its patch (``patch_file``, ``patch_row``: the tile's
   ``ml_patches.npy``, not copied; open with ``np.load(f, mmap_mode="r")``).

4. context features from the bundle's vector layers (``ctx_*``: distance to the nearest LINZ road, its lanes and
   whether it is sealed, to rail, to tracks, to any OSM way with a highway tag (driveways, farm tracks), to the
   nearest building and the number within 50 m; capped at 500 m). They need no rerun of the tiles, and define the
   inventory scope used in training (council registers cover road culverts; KiwiRail, rail).

Writes ``<out>/candidates.parquet``, ``<out>/misses.parquet`` (labels with no candidate in reach: the
recall denominator), ``<out>/tiles.csv`` and ``<out>/card.json`` / ``card.md``: counts per council, fold and
label kind, the feature columns (location and identifiers excluded), and the versions (code, features,
label config, manifest) that make it reproducible.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np

LABEL_COLUMNS = ["label_kind", "label_source", "label_id", "struct", "struct_class", "weight", "label_dist_m",
                 "n_labels", "label_sources_all", "near_positive", "in_core", "in_scope"]
TILE_COLUMNS = ["tile_id", "council", "wave", "sheet", "split", "holdout", "region", "latest_survey"]
NOT_FEATURES = {"x", "y", "cand_id", "patch_file", "patch_row", *LABEL_COLUMNS, *TILE_COLUMNS}


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def tile_rows(tile, run_dir: Path, bundle_dir: Path, manual, label_cfg, relabel="stale", log=print, channel_probs=None):
    """(candidates DataFrame, misses DataFrame, record) for one tile, or (None, None, record) if not ready."""
    import geopandas as gpd
    import pandas as pd

    from . import labels as L
    rec = dict(tile_id=tile.tile_id, run=str(run_dir))
    done = run_dir / "done.json"
    feats = run_dir / "ml_features.parquet"
    if not done.exists() or not feats.exists():
        rec["status"] = "no run with features"
        return None, None, rec
    lab = run_dir / "ml_labels.gpkg"
    stale = (not lab.exists()) or lab.stat().st_mtime < (run_dir / "network.gpkg").stat().st_mtime
    if relabel == "always" or (relabel == "stale" and stale):
        L.label_run(run_dir, bundle_dir, manual, label_cfg)
        rec["relabelled"] = True
    elif stale:
        rec["status"] = "labels older than the run (relabel='never')"
        return None, None, rec
    from .features import normalise_source_flags
    F = normalise_source_flags(pd.read_parquet(feats))           # features v1 and v2 give the same columns
    C = gpd.read_file(lab, layer="candidates")
    if len(F) != len(C) or not (np.allclose(F.x, C.x) and np.allclose(F.y, C.y)):
        rec["status"] = f"features ({len(F)}) and labels ({len(C)}) do not match: rebuild the tile"
        return None, None, rec
    D = F.copy()
    for k in LABEL_COLUMNS:
        D[k] = C[k].to_numpy() if k in C else np.nan
    for k in TILE_COLUMNS:
        D[k] = getattr(tile, k, None)
    D = context_features(bundle_dir, D)
    if channel_probs is not None or (run_dir / "ml_channel_prob.tif").exists():
        D = channel_features(run_dir, D, channel_probs)
    D["cand_id"] = [f"{tile.tile_id}:{i}" for i in range(len(D))]
    D["patch_file"] = str(run_dir / "ml_patches.npy")
    D["patch_row"] = np.arange(len(D))
    M = None
    import pyogrio
    if "misses" in {n for n, _ in pyogrio.list_layers(lab)}:
        m = gpd.read_file(lab, layer="misses")
        M = pd.DataFrame(m.drop(columns="geometry")).assign(x=m.geometry.x, y=m.geometry.y)
        for k in TILE_COLUMNS:
            M[k] = getattr(tile, k, None)
    rec.update(status="ok", candidates=len(D), positives=int((D.struct == 1).sum()), negatives=int((D.struct == 0).sum()),
               misses=0 if M is None else len(M))
    return D, M, rec


CHANNEL_RADIUS_M = 5.0


def channel_features(run_dir: Path, D, channel_probs=None):
    """The channel model's evidence at each candidate (ML-3c), for a scorer trained on runs with it:

        ml_channel      the candidate is on a reach only the U-Net mapped (network.gpkg ``crossings``; NaN for a
                        run without a channel model)
        ml_prob_max     the U-Net's highest probability within 5 m
        ml_sharp_max    the highest sharpness within 5 m (``channel_predict.sharpness_map``: a tight line against
                        a smudge)

    The probability raster is the run's ``ml_channel_prob.tif``, else ``<channel_probs>/<tile>/`` (the out-of-fold
    prediction beside the physics runs: the same held-out model a channel-model batch used)."""
    import geopandas as gpd
    import pyogrio
    import rasterio
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree

    from .channel_predict import sharpness_map
    D = D.copy()
    xy = np.column_stack([D.x.to_numpy(float), D.y.to_numpy(float)])
    net = run_dir / "network.gpkg"
    D["ml_channel"] = np.nan
    if net.exists() and "crossings" in {n for n, _ in pyogrio.list_layers(net)} and len(D):
        c = gpd.read_file(net, layer="crossings")
        if "ml_channel" in c and len(c):
            d, i = cKDTree(np.column_stack([c.geometry.x, c.geometry.y])).query(xy)
            D["ml_channel"] = np.where(d <= 0.5, c.ml_channel.to_numpy(float)[i], np.nan)
    pf = run_dir / "ml_channel_prob.tif"
    if not pf.exists() and channel_probs is not None:
        pf = Path(channel_probs) / run_dir.name / "ml_channel_prob.tif"
    D["ml_prob_max"] = np.nan
    D["ml_sharp_max"] = np.nan
    if pf.exists() and len(D):
        with rasterio.open(pf) as s:
            prob, T = s.read(1), s.transform
        k = 2 * int(round(CHANNEL_RADIUS_M / abs(T.a))) + 1
        pmax = ndi.maximum_filter(np.nan_to_num(prob, nan=0.0), size=k, mode="constant")
        smax = ndi.maximum_filter(sharpness_map(prob, T), size=k, mode="constant")
        r = np.floor((xy[:, 1] - T.f) / T.e).astype(int)
        cc = np.floor((xy[:, 0] - T.c) / T.a).astype(int)
        ok = (r >= 0) & (r < prob.shape[0]) & (cc >= 0) & (cc < prob.shape[1])
        D.loc[ok, "ml_prob_max"] = pmax[r[ok], cc[ok]]
        D.loc[ok, "ml_sharp_max"] = smax[r[ok], cc[ok]]
    return D


CTX_CAP_M = 500.0


def context_features(bundle_dir: Path, D):
    """``ctx_*`` columns from the bundle's context.gpkg and roads.gpkg (see the module docstring)."""
    import geopandas as gpd
    import pyogrio
    from scipy.spatial import cKDTree
    ctx = Path(bundle_dir) / "context.gpkg"
    have = {n for n, _ in pyogrio.list_layers(ctx)} if ctx.exists() else set()
    xy = np.column_stack([D.x.to_numpy(float), D.y.to_numpy(float)])

    def layer(name):
        return gpd.read_file(ctx, layer=name) if name in have else None

    def nearest(g, cap=CTX_CAP_M):
        """distance to the nearest geometry and its row index (-1 beyond the cap)."""
        if g is None or not len(g) or not len(xy):
            return np.full(len(xy), cap), np.full(len(xy), -1)
        pts = gpd.GeoSeries(gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=g.crs)
        j = gpd.sjoin_nearest(gpd.GeoDataFrame(geometry=pts).reset_index(names="i"), g[["geometry"]].reset_index(names="k"),
                              max_distance=cap, distance_col="d").drop_duplicates("i").set_index("i")
        d = j.d.reindex(range(len(xy))).fillna(cap).to_numpy()
        k = j.k.reindex(range(len(xy))).fillna(-1).astype(int).to_numpy()
        return np.minimum(d, cap), k
    out = {}
    roads = layer("roads")
    d, k = nearest(roads)
    out["ctx_dist_road_m"] = d
    if roads is not None and len(roads):
        lanes = pd_num(roads.get("lanes"))
        sealed = roads["surface"].astype(str).str.lower().eq("sealed").to_numpy() if "surface" in roads else None
        out["ctx_road_lanes"] = np.where(k >= 0, lanes[np.maximum(k, 0)], np.nan) if lanes is not None else np.nan
        out["ctx_road_sealed"] = np.where(k >= 0, sealed[np.maximum(k, 0)].astype(float), np.nan) if sealed is not None else np.nan
    else:
        out["ctx_road_lanes"], out["ctx_road_sealed"] = np.nan, np.nan
    out["ctx_dist_rail_m"] = nearest(layer("rail"))[0]
    out["ctx_dist_track_m"] = nearest(layer("tracks"))[0]
    osm = layer("osm_roads")
    out["ctx_dist_osm_way_m"] = nearest(osm)[0]
    b = layer("buildings")
    out["ctx_dist_building_m"] = nearest(b)[0]
    if b is not None and len(b) and len(xy):
        c = b.geometry.centroid
        tree = cKDTree(np.column_stack([c.x, c.y]))
        out["ctx_n_buildings_50m"] = np.array([len(v) for v in tree.query_ball_point(xy, 50.0)], float)
    else:
        out["ctx_n_buildings_50m"] = np.zeros(len(xy))
    import pandas as pd
    return pd.concat([D.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def pd_num(s):
    import pandas as pd
    return None if s is None else pd.to_numeric(s, errors="coerce").to_numpy(float)


def build(manifests, out, runs, bundles, label_cfg, manual=None, relabel="stale", name=None, log=print,
          channel_probs=None) -> dict:
    """Assemble the dataset (see the module docstring); returns the card."""
    import pandas as pd

    from . import code_version
    from .features import FEATURES_VERSION
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=True)
    man = pd.concat([pd.read_csv(m) for m in manifests], ignore_index=True).drop_duplicates("tile_id")
    parts, misses, recs = [], [], []
    for tile in man.itertuples(index=False):
        D, M, rec = tile_rows(tile, Path(runs) / tile.tile_id, Path(bundles) / tile.tile_id, manual, label_cfg, relabel, log,
                              channel_probs)
        recs.append(rec)
        if D is None:
            log(f"{tile.tile_id}: skipped ({rec['status']})")
            continue
        parts.append(D)
        if M is not None:
            misses.append(M)
    if not parts:
        raise ValueError("no tile is ready (finished runs with features)")
    data = pd.concat(parts, ignore_index=True)
    data.to_parquet(out / "candidates.parquet", index=False)
    mis = pd.concat(misses, ignore_index=True) if misses else pd.DataFrame(columns=["kind", *TILE_COLUMNS])
    mis.to_parquet(out / "misses.parquet", index=False)
    pd.DataFrame(recs).to_csv(out / "tiles.csv", index=False)
    feat_cols = [c for c in data.columns if c not in NOT_FEATURES]
    lab = data[data.n_labels > 0]

    def counts(df, by):
        return {str(k): int(v) for k, v in df.groupby(by).size().items()} if len(df) else {}
    card = dict(
        name=name or out.name, built=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        versions=dict(code=code_version(), features=FEATURES_VERSION,
                      label_config=hashlib.sha256(json.dumps(label_cfg, sort_keys=True, default=str).encode()).hexdigest()[:16],
                      manifests={str(m): _sha(m) for m in manifests}),
        tiles=dict(manifest=int(len(man)), included=len(parts), skipped=[r["tile_id"] for r in recs if r.get("status") != "ok"]),
        candidates=int(len(data)), positives=int((data.struct == 1).sum()), negatives=int((data.struct == 0).sum()),
        unsure=int(((data.n_labels > 0) & data.struct.isna()).sum()), near_positive=int(data.near_positive.fillna(False).sum()),
        unlabelled=int((data.n_labels == 0).sum()),
        positives_by_kind=counts(data[data.struct == 1], "label_kind"),
        negatives_by_kind=counts(data[data.struct == 0], "label_kind"),
        by_council={c: dict(tiles=int(g.tile_id.nunique()), candidates=int(len(g)), positives=int((g.struct == 1).sum()),
                            negatives=int((g.struct == 0).sum()), holdout_tiles=int(g[g.holdout == True].tile_id.nunique()))  # noqa: E712
                    for c, g in data.groupby("council")},
        by_split=counts(data, "split"), labelled_by_class=counts(lab[lab.struct == 1], "struct_class"),
        misses=dict(total=int(len(mis)), by_kind=counts(mis, "kind") if len(mis) else {}),
        features=dict(n=len(feat_cols), columns=feat_cols),
        patches=dict(bands=_patch_bands(data), size=64, cell_m=1.0, note="patch_file + patch_row; np.load(mmap_mode='r')"))
    (out / "card.json").write_text(json.dumps(card, indent=1))
    (out / "card.md").write_text(_card_md(card))
    return card


def _patch_bands(data):
    f = Path(data.patch_file.iloc[0]).with_suffix(".json")
    return json.loads(f.read_text())["bands"] if f.exists() else None


def _card_md(c) -> str:
    L = [f"# Dataset {c['name']}", "", f"Built {c['built']}; code {c['versions']['code']}, features "
         f"v{c['versions']['features']}, label config {c['versions']['label_config']}.", "",
         f"Tiles: {c['tiles']['included']} of {c['tiles']['manifest']} in the manifest"
         + (f" (skipped: {', '.join(c['tiles']['skipped'])})" if c['tiles']['skipped'] else "") + ".", "",
         f"Candidates {c['candidates']:,}: positive {c['positives']:,}, negative {c['negatives']:,}, unsure {c['unsure']:,}, "
         f"near a positive {c['near_positive']:,}, unlabelled {c['unlabelled']:,}.", "",
         "| council | tiles | holdout tiles | candidates | positive | negative |", "|---|---|---|---|---|---|"]
    for k, v in c["by_council"].items():
        L.append(f"| {k} | {v['tiles']} | {v['holdout_tiles']} | {v['candidates']:,} | {v['positives']:,} | {v['negatives']:,} |")
    L += ["", "Positives by label kind: " + ", ".join(f"{k} {v}" for k, v in c["positives_by_kind"].items()) + ".",
          "Negatives by label kind: " + (", ".join(f"{k} {v}" for k, v in c["negatives_by_kind"].items()) or "none") + ".",
          "Positive classes: " + ", ".join(f"{k} {v}" for k, v in c["labelled_by_class"].items()) + ".",
          f"Misses (labels with no candidate in reach): {c['misses']['total']} ("
          + ", ".join(f"{k} {v}" for k, v in c["misses"]["by_kind"].items()) + ").", "",
          f"{c['features']['n']} feature columns; patches: {c['patches']['bands']}."]
    return "\n".join(L) + "\n"
