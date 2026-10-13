"""ML-0 step 5: one dataset from the tiles of a manifest: features, labels, folds, and a card.

For every tile of the manifest with a finished run (``done.json`` with features):

1. labels are (re)built when missing or older than the run (``labels.label_run``: a rerun replaces the
   candidates, so labels made before it are stale);
2. ``ml_features.parquet`` and the labelled candidates are joined on (x, y) (both are the run's written
   ``crossings`` layer, in the same order; a count or position mismatch stops that tile with a message);
3. tile fields from the manifest are added (council, sheet, split, holdout, region, survey) and each row
   gets ``cand_id`` (``<tile>:<row>``) and its patch (``patch_file``, ``patch_row``: the tile's
   ``ml_patches.npy``, not copied; open with ``np.load(f, mmap_mode="r")``).

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


def tile_rows(tile, run_dir: Path, bundle_dir: Path, manual, label_cfg, relabel="stale", log=print):
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
    F = pd.read_parquet(feats)
    C = gpd.read_file(lab, layer="candidates")
    if len(F) != len(C) or not (np.allclose(F.x, C.x) and np.allclose(F.y, C.y)):
        rec["status"] = f"features ({len(F)}) and labels ({len(C)}) do not match: rebuild the tile"
        return None, None, rec
    D = F.copy()
    for k in LABEL_COLUMNS:
        D[k] = C[k].to_numpy() if k in C else np.nan
    for k in TILE_COLUMNS:
        D[k] = getattr(tile, k, None)
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


def build(manifests, out, runs, bundles, label_cfg, manual=None, relabel="stale", name=None, log=print) -> dict:
    """Assemble the dataset (see the module docstring); returns the card."""
    import pandas as pd

    from .. import __version__
    from .features import FEATURES_VERSION
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=True)
    man = pd.concat([pd.read_csv(m) for m in manifests], ignore_index=True).drop_duplicates("tile_id")
    parts, misses, recs = [], [], []
    for tile in man.itertuples(index=False):
        D, M, rec = tile_rows(tile, Path(runs) / tile.tile_id, Path(bundles) / tile.tile_id, manual, label_cfg, relabel, log)
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
        versions=dict(code=str(__version__), features=FEATURES_VERSION,
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
