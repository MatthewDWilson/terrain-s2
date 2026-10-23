#!/usr/bin/env python
"""How connected are the runs' channel networks? (integration step 3: measure before the clean-up rules)

    python scripts/network_fragments.py --manifest ml/manifests/wave1.csv ml/manifests/wave2.csv ml/manifests/wave3.csv
    python scripts/network_fragments.py --manifest ml/manifests/wave3.csv --runs ~/terrain_data/runs --out <csv>
    python scripts/network_fragments.py --manifest ml/manifests/wave3.csv --gaps BY20_10000_0405 CF15_10000_0501

Per tile (``connectivity.measure``): components, how they are anchored (tile edge, river, residual pond, the
floodplain restriction's limit) or isolated, the isolated length and its share mapped only by the U-Net and
classed drain, the nearest-component gaps of the isolated components (distance bins; end-to-end, end-to-line or
line-to-line), and the isolated share left if gaps up to 2-50 m were joined. Prints totals and by council and
writes ``--out`` (default ``<runs>/fragments.csv``). ``--gaps`` writes, for those tiles, ``<runs>/<tile>/
fragments.gpkg`` (layers ``components`` and ``connectors``) for a look in QGIS. Nothing is changed in the runs.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def one(args):
    tile, council, runs, bundles, write = args
    import geopandas as gpd

    from terrain_s2 import connectivity as C
    run = Path(runs) / tile
    try:
        aoi = gpd.read_file(Path(bundles) / tile / "aoi.geojson")
        net = run / "network.gpkg"
        if not net.exists():
            return dict(tile_id=tile, council=council, missing=True)
        import pyogrio
        crs = pyogrio.read_info(net, layer="channels")["crs"] if "channels" in {n for n, _ in pyogrio.list_layers(net)} else None
        g = aoi.to_crs(crs).union_all() if crs else aoi.union_all()
        s, comps = C.measure(run, g)
        if write and comps is not None:
            f = run / "fragments.gpkg"
            comps.drop(columns="connector_wkt").to_file(f, layer="components", driver="GPKG")
            con = C.connectors(comps)
            if len(con):
                con.to_file(f, layer="connectors", driver="GPKG")
        return dict(tile_id=tile, council=council, **s)
    except Exception as e:  # noqa: BLE001 - recorded per tile
        return dict(tile_id=tile, council=council, error=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--runs", help="default <data>/runs_ml3c_f")
    ap.add_argument("--bundles")
    ap.add_argument("--gaps", nargs="*", default=[], help="tiles to write fragments.gpkg for")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args()
    import pandas as pd

    from terrain_s2.acquire.elevation import default_data_dir
    data = default_data_dir()
    runs = Path(a.runs or data / "runs_ml3c_f")
    bundles = Path(a.bundles or data / "tiles")
    m = pd.concat([pd.read_csv(f) for f in a.manifest], ignore_index=True).drop_duplicates("tile_id")
    council = m.council if "council" in m else pd.Series([""] * len(m))
    jobs = [(t, c, str(runs), str(bundles), t in set(a.gaps)) for t, c in zip(m.tile_id, council)]
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        rows = list(ex.map(one, jobs, chunksize=4))
    d = pd.DataFrame(rows)
    out = Path(a.out) if a.out else runs / "fragments.csv"
    d.to_csv(out, index=False)
    bad = d[d.get("error", pd.Series(dtype=str)).notna()] if "error" in d else d.iloc[0:0]
    miss = d[d.get("missing", pd.Series(dtype=bool)).fillna(False).astype(bool)] if "missing" in d else d.iloc[0:0]
    ok = d.drop(bad.index.union(miss.index))
    ok = ok[ok.reaches.fillna(0) > 0]
    print(f"{len(d)} tiles: {len(ok)} measured, {len(miss)} without a run, {len(bad)} failed"
          + (f" (e.g. {bad.error.iloc[0]})" if len(bad) else ""))
    if not len(ok):
        print(f"-> {out}")
        return
    joins = [c for c in ok.columns if c.startswith("isolated_share_join_")]
    gaps = [c for c in ok.columns if c.startswith("gap_") and c.endswith("m")]
    kinds = ["gap_end_end", "gap_end_line", "gap_line_line"]

    def block(g, name):
        km = g.km.sum()
        iso = g.isolated_km.sum()
        mlk = (g.isolated_km * g.isolated_ml_share).sum()
        drk = (g.isolated_km * g.isolated_drain_share).sum()
        joined = {c.replace("isolated_share_join_", "join "): (g[c] * g.km).sum() / km for c in joins}
        return dict(group=name, tiles=len(g), km=round(km, 1), components_per_tile=round(g.components.mean(), 1),
                    isolated_per_tile=round(g.isolated.mean(), 1), isolated_share=round(iso / km, 4),
                    isolated_ml=round(mlk / iso, 3) if iso else 0, ml_share=round((g.ml_share * g.km).sum() / km, 3),
                    isolated_drain=round(drk / iso, 3) if iso else 0,
                    **{k: round(v, 4) for k, v in joined.items()})
    rows = [block(ok, "all")] + [block(g, c) for c, g in ok.groupby("council")]
    print("\nlength isolated (share of all channel length; isolated_ml / _drain: shares of the isolated length "
          "mapped only by the U-Net / classed drain), and what joining gaps up to G m would leave:")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nisolated components by the distance to their nearest component (all tiles):")
    print("  " + ", ".join(f"{c.replace('gap_', '')}: {int(ok[c].sum())}" for c in gaps))
    print("and by the kind of near-meeting: " + ", ".join(f"{k.replace('gap_', '')} {int(ok[k].sum())}" for k in kinds))
    an = ["anchored_edge", "anchored_river", "anchored_pond", "anchored_floodplain_limit", "isolated"]
    print("components by anchor (all tiles): " + ", ".join(f"{c.replace('anchored_', '')} {int(ok[c].sum())}" for c in an))
    if a.gaps:
        print(f"\nfragments.gpkg written for {', '.join(a.gaps)} (in their run folders)")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
