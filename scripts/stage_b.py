#!/usr/bin/env python
"""Stage B of channel network v2 (``cleanup.clean_tile``) over the runs of a manifest: join supported gaps, drop minor
drains and fragments; writes ``network_b.gpkg`` and ``stage_b.json`` in each run folder (stage C's outputs are not
changed).

    python scripts/stage_b.py --manifest ml/manifests/catchment_sh12.csv --runs ~/terrain_data/runs_c
    python scripts/stage_b.py --manifest ... --runs ... --sweep          # also the minor-drain threshold sweep
    python scripts/stage_b.py --manifest ... --runs ... --param minor_width_m=2.0 --param rise_tol_m=0.2

The runs must have been made with stage C's outputs (``c_outputs``: the ``links`` layer; model set ``c_proto``).
Prints, pooled over the tiles: connectivity before and after (components per tile, isolated share of length; residual
ponds no longer count as anchors, OSM lakes, tidal flats and coastline do), links accepted (and how many are
crossings) and why the rest were not, and what the minor-drain and fragment rules removed. ``--sweep`` adds the
length the minor-drain rule would remove at widths 1-2 m and depths 0.2-0.4 m. Writes ``<runs>/stage_b.csv``.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def one(args):
    tile, runs, bundles, params, sweep = args
    import geopandas as gpd
    import pyogrio

    from terrain_s2 import cleanup
    run = Path(runs) / tile
    b = Path(bundles) / tile
    try:
        net = run / "network.gpkg"
        if not net.exists():
            return dict(tile=tile, error="no run")
        crs = pyogrio.read_info(net, layer="channels")["crs"]
        aoi_g = gpd.read_file(b / "aoi.geojson")
        aoi = aoi_g.to_crs(crs).union_all()
        win = tuple(aoi_g.to_crs(2193).union_all().buffer(500).bounds)
        s = cleanup.clean_tile(run, aoi, context=b / "context.gpkg", window=win, params=params)
        if sweep:
            s["sweep"] = cleanup.minor_sweep(run, aoi, params=params)
        return s
    except Exception as e:  # noqa: BLE001 - recorded per tile
        import traceback
        return dict(tile=tile, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-600:])


def _value(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--bundles")
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import pandas as pd

    from terrain_s2.acquire.elevation import default_data_dir
    data = default_data_dir()
    m = pd.concat([pd.read_csv(f) for f in a.manifest], ignore_index=True).drop_duplicates("tile_id")
    params = {k: _value(v) for k, v in (kv.split("=", 1) for kv in a.param)}
    jobs = [(t, a.runs, a.bundles or str(data / "tiles"), params, a.sweep) for t in m.tile_id]
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        res = list(ex.map(one, jobs))
    bad = [r for r in res if r.get("error")]
    ok = [r for r in res if not r.get("error") and r.get("before")]
    print(f"{len(res)} tiles: {len(ok)} cleaned, {len(bad)} failed" + (f" (e.g. {bad[0]['tile']}: {bad[0]['error']})" if bad else ""))
    if not ok:
        return
    rows = []
    for r in ok:
        b, f = r["before"], r["after"]
        rows.append(dict(tile=r["tile"], km_before=b.get("km"), km_after=f.get("km"), components_before=b.get("components"),
                         components_after=f.get("components"), isolated_share_before=b.get("isolated_share"),
                         isolated_share_after=f.get("isolated_share"), isolated_after=f.get("isolated"),
                         links_candidates=r["links"]["candidates"], links_accepted=r["links"]["accepted"],
                         link_crossings=r["links"]["crossings"], minor_drain_km=r["removed"]["minor_drain_km"],
                         fragment_km=r["removed"]["fragment_km"], outlets=r.get("outlets")))
    d = pd.DataFrame(rows)
    d.to_csv(Path(a.runs) / "stage_b.csv", index=False)
    kmb, kma = d.km_before.sum(), d.km_after.sum()
    isob = (d.isolated_share_before * d.km_before).sum() / kmb
    isoa = (d.isolated_share_after * d.km_after).sum() / kma if kma else 0
    print(f"\nchannel length {kmb:.1f} km -> {kma:.1f} km; components per tile {d.components_before.mean():.0f} -> "
          f"{d.components_after.mean():.0f}; isolated share of length {isob:.1%} -> {isoa:.1%} "
          f"({d.isolated_after.sum()} isolated pieces left for stage A)")
    print(f"removed: minor drains {d.minor_drain_km.sum():.1f} km, fragments {d.fragment_km.sum():.1f} km")
    reasons = {}
    kinds = {}
    for r in ok:
        for k, v in r["links"]["by_reason"].items():
            reasons[k] = reasons.get(k, 0) + v
        for k, v in r["links"]["accepted_by_kind"].items():
            kinds[k] = kinds.get(k, 0) + v
    print(f"links: {d.links_candidates.sum()} candidates, {d.links_accepted.sum()} accepted ({d.link_crossings.sum()} "
          f"across a barrier with a crossing candidate); by kind {kinds}; outcomes {reasons}")
    print("\nper tile:\n" + d.drop(columns=["outlets"]).to_string(index=False))
    if a.sweep:
        sw = pd.DataFrame([dict(s, tile=r["tile"]) for r in ok for s in r.get("sweep", [])])
        if len(sw):
            t = sw.groupby(["width_m", "depth_m"]).agg(removed_km=("removed_km", "sum"), total_km=("total_km", "sum"))
            t["share"] = (t.removed_km / t.total_km).round(3)
            print("\nminor-drain sweep (removed if width < w or depth < d; links aside):\n" + t.to_string())
    print(f"\n-> {Path(a.runs) / 'stage_b.csv'}")


if __name__ == "__main__":
    main()
