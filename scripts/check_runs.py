#!/usr/bin/env python
"""Completeness and data issues of a manifest's bundles and runs, after a batch (and the channel steps).

    python scripts/check_runs.py --manifest ml/manifests/wave3.csv
    python scripts/check_runs.py --manifest ml/manifests/wave1.csv ml/manifests/wave2.csv ml/manifests/wave3.csv --stage chips

Per tile, from the files only (fast; nothing is recomputed):

    bundle      site.json present; source errors (by source; LINZ_API_KEY missing called out); DEM and DSM present,
                their valid fractions; a DSM covering clearly less than the DEM (DSM gaps: crossings there get no
                DSM - DEM, and the chips' DSM - DEM band is 0)
    run         done.json (and its parameters: saved rasters), failed.json (stale_failed when done.json is newer:
                an earlier attempt), the outputs a finished run has
                (network.gpkg, ml_features.parquet, ml_patches.npy, ml_probes.parquet, the three rasters),
                probes_error, candidates and channel km
    channels    ml_channels.json (snap version), ml_channel_labels (label version), ml_ways.tif, and whether the
                tile has any positives            (--stage channels or chips)
    chips       the tile's chips in <datasets>/<--dataset>/chips   (--stage chips)

Writes <runs>/check_<stage>.csv (one row per tile, an ``issues`` column) and prints the issues counted, then the
tiles with each, so a rerun list can be made from it (``--list ISSUE`` prints just those tile ids).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUN_FILES = ("network.gpkg", "ml_features.parquet", "ml_patches.npy", "ml_probes.parquet",
             "features.tif", "hydro.tif", "conditioned.tif")


def _json(f):
    try:
        return json.loads(Path(f).read_text())
    except (OSError, ValueError):
        return None


def check_tile(t, runs, bundles, stage, chips_dir, dsm_gap):
    b, r = bundles / t, runs / t
    row, issues = dict(tile_id=t), []
    site = _json(b / "site.json")
    if site is None:
        issues.append("no_site_json")
    else:
        errs = site.get("errors") or []
        srcs = sorted({str(e.get("step", "")).replace("source:", "") for e in errs})
        row["source_errors"] = ";".join(srcs)
        if any("LINZ_API_KEY" in str(e.get("error", "")) for e in errs):
            issues.append("linz_key_missing")
        elif srcs:
            issues.append("source_errors")
        ev = {e.get("product"): e for e in site.get("elevation") or []}
        for p in ("dem", "dsm"):
            row[f"{p}_valid"] = ev.get(p, {}).get("valid_fraction")
            row[f"{p}_survey"] = ev.get(p, {}).get("survey")
    for p in ("dem", "dsm"):
        if not (b / f"{p}.tif").exists():
            issues.append(f"no_{p}")
    dv, sv = row.get("dem_valid"), row.get("dsm_valid")
    if dv is not None and dv < 0.5:
        issues.append("dem_mostly_empty")
    if dv is not None and sv is not None and sv < dv - dsm_gap:
        issues.append("dsm_gap")
    done = _json(r / "done.json")
    if (r / "failed.json").exists():
        stale = (r / "done.json").exists() and (r / "done.json").stat().st_mtime > (r / "failed.json").stat().st_mtime
        issues.append("stale_failed" if stale else "failed")      # stale: an earlier attempt, done since
        f = _json(r / "failed.json") or {}
        row["failed_error"] = str(f.get("error", ""))[:200]
    if done is None:
        issues.append("not_done")
    else:
        row.update(crossings=done.get("crossings"), channels_km=done.get("channels_km"),
                   peak_rss_gb=done.get("peak_rss_gb"), rasters=(done.get("params") or {}).get("rasters"))
        if row["rasters"] != "all":
            issues.append("no_saved_rasters")
        if done.get("probes_error"):
            issues.append("probes_error")
        missing = [f for f in RUN_FILES if not (r / f).exists()
                   and not (f in ("features.tif", "hydro.tif", "conditioned.tif") and row["rasters"] != "all")]
        if missing:
            issues.append("missing_outputs")
            row["missing"] = ";".join(missing)
        if not done.get("crossings"):
            issues.append("no_candidates")
    if stage in ("channels", "chips") and done is not None:
        sj = _json(r / "ml_channels.json")
        lj = _json(r / "ml_channel_labels.json")
        row["snap_version"] = sj.get("snap_version") if sj else None
        row["label_version"] = lj.get("label_version") if lj else None
        row["positive_cells"] = lj.get("positive") if lj else None
        if sj is None:
            issues.append("not_snapped")
        if lj is None or not (r / "ml_channel_labels.tif").exists():
            issues.append("no_label_raster")
        elif not lj.get("positive"):
            issues.append("no_positives")
    if stage == "chips":
        if not (r / "ml_ways.tif").exists():
            issues.append("no_ways_raster")
        cj = _json(chips_dir / f"{t}.json") if chips_dir else None
        row["chips"] = len(cj["rows"]) if cj else 0
        if not cj or not (chips_dir / f"{t}.npz").exists():
            issues.append("no_chips")
    row["issues"] = ";".join(issues)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--runs")
    ap.add_argument("--bundles")
    ap.add_argument("--datasets")
    ap.add_argument("--dataset", help="chip dataset name (--stage chips)")
    ap.add_argument("--stage", choices=["batch", "channels", "chips"], default="batch")
    ap.add_argument("--dsm-gap", type=float, default=0.02, help="DSM valid fraction this far below the DEM's")
    ap.add_argument("--list", metavar="ISSUE", help="print only the tile ids with this issue")
    a = ap.parse_args()
    import pandas as pd
    from terrain_s2.acquire.elevation import default_data_dir
    data = default_data_dir()
    runs, bundles = Path(a.runs or data / "runs"), Path(a.bundles or data / "tiles")
    chips_dir = Path(a.datasets or data / "datasets") / a.dataset / "chips" if a.dataset else None
    if a.stage == "chips" and chips_dir is None:
        ap.error("--stage chips needs --dataset")
    m = pd.concat([pd.read_csv(f) for f in a.manifest], ignore_index=True).drop_duplicates("tile_id")
    rows = [check_tile(t, runs, bundles, a.stage, chips_dir, a.dsm_gap) for t in m.tile_id]
    d = pd.DataFrame(rows).merge(m[[c for c in ("tile_id", "council", "wave") if c in m]], on="tile_id", how="left")
    if a.list:
        for t in d[d.issues.str.split(";").apply(lambda v: a.list in v)].tile_id:
            print(t)
        return
    out = runs / f"check_{a.stage}.csv"
    d.to_csv(out, index=False)
    ex = d.assign(issue=d.issues.str.split(";")).explode("issue")
    ex = ex[ex.issue.astype(bool)]
    print(f"{len(d)} tiles; {int((d.issues == '').sum())} without issues")
    if len(ex):
        by = ex.groupby(["issue", "council"]).size().unstack(fill_value=0) if "council" in ex else ex.issue.value_counts()
        print(by.to_string())
        for issue, g in ex.groupby("issue"):
            ids = list(g.tile_id)
            print(f"\n{issue} ({len(ids)}): {' '.join(ids[:12])}{' ...' if len(ids) > 12 else ''}")
        if "source_errors" in d and d.source_errors.fillna("").astype(bool).any():
            se = d.source_errors.fillna("").str.split(";").explode()
            print("\nsource errors by source:", se[se.astype(bool)].value_counts().to_dict())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
