#!/usr/bin/env python
"""Why labelled structures get no candidate (terrain_s2.ml.probes): reasons per council and label kind.

    python scripts/diagnose_misses.py --manifest ml/manifests/wave1.csv                       # tiles with probes
    python scripts/diagnose_misses.py --manifest ml/manifests/wave1.csv --compute --council waimakariri --limit 8

Tiles run since the probes were added have ml_probes.parquet already. --compute reruns the network for tiles without
it (same parameters as their done.json, ~10-16 min per LINZ tile) in a scratch folder, leaving the tile's run as it was. Writes
<runs>/misses_<manifest>.md and .parquet (one row per label, with its reason and the features at its crest).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _table(df):
    try:
        return df.to_markdown()
    except ImportError:                       # tabulate not installed
        return df.to_string()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--runs")
    ap.add_argument("--bundles")
    ap.add_argument("--council", nargs="+")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--compute", action="store_true", help="rerun the network for tiles without probes")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    import pandas as pd

    from terrain_s2.acquire.elevation import default_data_dir
    from terrain_s2.ml import probes
    from terrain_s2.ml.batch import _label_cfg
    from terrain_s2.run import network
    data = default_data_dir()
    runs, bundles = Path(a.runs or data / "runs"), Path(a.bundles or data / "tiles")
    m = pd.read_csv(a.manifest)
    if a.council:
        m = m[m.council.isin(a.council)]
    tables, todo = [], []
    for t in m.itertuples():
        f = runs / t.tile_id / "ml_probes.parquet"
        if f.exists():
            tables.append(pd.read_parquet(f).assign(tile_id=t.tile_id, council=t.council))
        elif (runs / t.tile_id / "done.json").exists():
            todo.append(t)
    print(f"{len(tables)} tiles with probes; {len(todo)} run but without probes (runs: {runs})")
    if not tables and not todo:
        sys.exit(f"none of the manifest's tiles has a run under {runs}: is TERRAIN_DATA_DIR set (or give --runs)?")
    if a.compute:
        todo = todo[:a.limit] if a.limit else todo
        for i, t in enumerate(todo):
            b = bundles / t.tile_id
            params = json.loads((runs / t.tile_id / "done.json").read_text()).get("params", {})
            params = {k: v for k, v in params.items() if k not in ("device", "features")}
            # into a scratch folder: rewriting the tile's own network.gpkg without its features made 4 wave-1 tiles
            # disagree with their ml_features.parquet (a CPU run then a GPU run: a few candidates differ), 16 Oct 2026
            import shutil
            scratch = runs / t.tile_id / "_probe_run"
            res = network(b / "dem.tif", b / "dsm.tif" if (b / "dsm.tif").exists() else None, b / "aoi.geojson",
                          roads=b / "roads.gpkg" if (b / "roads.gpkg").exists() else None, out=scratch,
                          params=params, device=a.device, keep=True)
            pr = probes.write(res, b, runs / t.tile_id, _label_cfg())
            shutil.rmtree(scratch, ignore_errors=True)
            tables.append(pr.assign(tile_id=t.tile_id, council=t.council))
            print(f"[{i + 1}/{len(todo)}] {t.tile_id}: " + ", ".join(f"{k} {v}" for k, v in pr.reason.value_counts().items()))
    if not tables:
        sys.exit("no probes yet: add --compute to make them for tiles run before the probes existed")
    allp = pd.concat(tables, ignore_index=True)
    s = probes.summarise([allp])
    name = Path(a.manifest).stem
    allp.to_parquet(runs / f"misses_{name}.parquet", index=False)
    md = ["# Why labels get no candidate (" + name + ")", "", f"{allp.tile_id.nunique()} tiles, {len(allp)} labels.", "",
          _table(s), ""]
    lost = allp[~allp.reason.isin(["found", "position"])]
    if len(lost):
        md += ["Medians at the crest, labels lost against labels found (upstream area log10 m2, DEM-only barrier "
               "height m, distance to the channel map and centreline m):", ""]
        cols = ["log10_upa_max5", "tophat_white21_max5", "dist_channel_m", "dist_centreline_m", "rem_at"]
        g = allp.assign(lost=~allp.reason.isin(["found", "position"])).groupby(["kind", "lost"])[cols].median().round(2)
        md += [g.to_string(), ""]
    (runs / f"misses_{name}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
