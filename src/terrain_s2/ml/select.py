"""ML-0 step 1: choose training tiles where council data exist, and record their folds.

Tiles come from the partition (LiDAR exists; ``tiles`` layer, one row per LINZ 1:10k tile or quadrant) and
the coverage index (``sources_coverage.geojson``, feature counts per LINZ tile). A wave config names the
councils and, per council, the sources of each label type; a tile is selected for a council when that
council's ``select_by`` sources have at least ``min_features`` in it. Other sources (KiwiRail, LINZ bridges,
OSM) are labels wherever they fall, never a reason to select a tile.

Folds (recorded, applied by the training code):
- ``council``: leave-one-council-out (train on the others, test on this one);
- ``sheet``: the group for blocked cross-validation within a council (tiles of one 1:50k sheet stay together);
- ``holdout``: sheets never trained on (config ``holdout_sheets``: BW24, the sheet of canterbury1 and canterbury2,
  whose manual labels are the long-term test set), and whole councils never trained on (``holdout_councils``: a
  new landscape, the test of a national rollout);
- ``split``: the partition's train / validation / test per sheet, kept for reference. With a few sheets per
  council it is too coarse for evaluation (wave 1: every Waimakariri tile is test or validation, every HBRC
  tile train), hence the folds above.

The manifest (CSV, committed under ``ml/manifests/``) lists, per tile: council, wave, folds, survey and
neighbour fields from the partition, and the per-tile feature counts of every label source in the index.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

LABEL_TYPES = ("crossings", "channels", "stopbanks", "bridges")


def load_config(path) -> dict:
    import yaml
    c = yaml.safe_load(Path(path).read_text())
    for k in ("name", "councils"):
        if k not in c:
            raise ValueError(f"{path}: wave config needs '{k}'")
    c.setdefault("select_by", "crossings")
    c.setdefault("min_features", 1)
    c.setdefault("max_tiles_per_council", None)
    c.setdefault("seed", 0)
    c.setdefault("holdout_sheets", [])
    c.setdefault("holdout_councils", [])
    c.setdefault("exclude_tiles", [])          # tiles of earlier waves (select_tiles.py fills it from exclude_manifests)
    return c


def tile_counts(index: dict, sources) -> dict:
    """{tile_id: {source: count}} from the coverage index (registry.load_index) for these sources."""
    out = {}
    for name in sources:
        if name not in index:
            continue
        for t, n in (index[name][1].get("tiles") or {}).items():
            out.setdefault(t, {})[name] = int(n)
    return out


def select(partition_tiles, index: dict, config: dict):
    """The manifest as a DataFrame. ``partition_tiles``: the partition's ``tiles`` layer (DataFrame or
    GeoDataFrame with tile_id, linz_tile, sheet, split, region, latest_survey, has_dsm, ...)."""
    import pandas as pd
    pt = pd.DataFrame(partition_tiles).drop(columns=["geometry"], errors="ignore")
    key = "linz_tile" if "linz_tile" in pt else "tile_id"
    label_sources = sorted({n for n, (_, p) in index.items() if p.get("enabled", True)
                            and set(p.get("provides", [])) & set(LABEL_TYPES)})
    counts = tile_counts(index, label_sources)
    excluded = set(config.get("exclude_tiles") or [])
    if excluded:
        pt = pt[~pt.tile_id.isin(excluded)]
    rows = []
    for council, by_type in config["councils"].items():
        sel_sources = by_type.get(config["select_by"]) or []
        if not sel_sources:
            raise ValueError(f"council {council}: no '{config['select_by']}' sources in the wave config")
        missing = [s for s in sel_sources if s not in index]
        if missing:
            raise ValueError(f"council {council}: not in the coverage index: {missing}")
        cand = pt[pt[key].map(lambda t: sum(counts.get(t, {}).get(s, 0) for s in sel_sources)) >= config["min_features"]]
        cand = cand.assign(_n=cand[key].map(lambda t: sum(counts.get(t, {}).get(s, 0) for s in sel_sources)))
        cap = config.get("max_tiles_per_council")
        if cap and len(cand) > cap:          # keep the tiles with most council labels, ties by a seeded hash
            cand = cand.assign(_h=cand.tile_id.map(lambda t: _hash(t, config["seed"])))
            cand = cand.sort_values(["_n", "_h"], ascending=[False, True]).head(cap)
        for _, r in cand.iterrows():
            c = counts.get(r[key], {})
            row = dict(tile_id=r.tile_id, linz_tile=r.get(key), council=council, wave=config["name"],
                       split=r.get("split"), sheet=r.get("sheet"), region=r.get("region"),
                       latest_survey=r.get("latest_survey"), has_dsm=r.get("has_dsm"),
                       n_neighbours_same_survey=r.get("n_neighbours_same_survey"), survey_edge=r.get("survey_edge"),
                       holdout=(r.get("sheet") in set(config["holdout_sheets"])
                                or council in set(config.get("holdout_councils") or [])))
            for lt in LABEL_TYPES:
                own = by_type.get(lt) or []
                row[f"council_{lt}"] = sum(c.get(s, 0) for s in own)
            row["label_sources"] = ";".join(f"{s}={n}" for s, n in sorted(c.items()))
            rows.append(row)
    m = pd.DataFrame(rows)
    if len(m):
        dup = m[m.duplicated("tile_id", keep=False)]
        if len(dup):                       # a tile in two councils' areas: keep it once, under the council with more
            m = (m.assign(_n=m[f"council_{config['select_by']}"]).sort_values(["tile_id", "_n"], ascending=[True, False])
                  .drop_duplicates("tile_id").drop(columns="_n"))
        m = m.sort_values(["council", "split", "tile_id"]).reset_index(drop=True)
    return m


def summary(m) -> str:
    if not len(m):
        return "no tiles selected"
    lines = [f"{len(m)} tiles: " + ", ".join(f"{k} {v}" for k, v in m.council.value_counts().items())]
    for council, g in m.groupby("council"):
        lines.append(f"  {council}: {g.sheet.nunique()} sheets ({', '.join(sorted(g.sheet.unique()))}); holdout {int(g.holdout.sum())};"
                     f" partition splits " + ", ".join(f"{k} {v}" for k, v in g.split.value_counts().items())
                     + f"; council crossings {int(g.council_crossings.sum())}, channels {int(g.council_channels.sum())},"
                     f" stopbanks {int(g.council_stopbanks.sum())}")
    return "\n".join(lines)


def _hash(s: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()
