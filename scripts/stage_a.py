#!/usr/bin/env python
"""Stage A of channel network v2 on one catchment's tiles (``routed_network.run_catchment``, version 3, 29 Oct:
mapped lines always represented, detected channels added by evidence, routed paths without evidence only as flow
paths, one outlet), or the first version (``--v1``: ``catchment_mode``).

    python scripts/stage_a.py --catchment sh12 --runs ~/terrain_data/runs_c --trace 1642255.6,6075854.7
    python scripts/stage_a.py --catchment sh12 --runs ~/terrain_data/runs_c --param a50_m2=50000 --param res_m=10

``--catchment``: a folder name under ``<TERRAIN_DATA_DIR>/catchments`` (``catchment_tiles.py`` writes it), or a path.
The runs should have stage B's ``network_b.gpkg`` (else stage C's network is used) and stage C's
``channel_prob.tif``. Version 2 writes ``<catchment>/network_a.gpkg`` (channels: routed segments drawn downstream
with Strahler order, upstream area, evidence and p_channel, class_v2 and its source; joined drains and their joins;
crossings with their candidate's p_structure or needs_structure; the outlet; pruned heads; join attempts),
``dem_routed.tif`` (the conditioned coarse DEM), ``stage_a.json`` and ``<runs>/<tile>/network_a.gpkg``. ``--trace X,Y``
(repeatable) reports what each stage has near a point (a known culvert, say).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _value(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catchment", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--bundles")
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--trace", action="append", default=[], metavar="X,Y")
    ap.add_argument("--v1", action="store_true", help="the first version (catchment_mode)")
    a = ap.parse_args()
    from terrain_s2 import catchment_mode as A
    from terrain_s2.acquire.elevation import default_data_dir
    data = default_data_dir()
    cd = Path(a.catchment)
    if not (cd / "catchment.gpkg").exists():
        cd = data / "catchments" / a.catchment
    params = {k: _value(v) for k, v in (kv.split("=", 1) for kv in a.param)}
    if a.v1:
        s = A.run_catchment(cd, a.runs, a.bundles or data / "tiles", params=params)
        keys = ("km_in", "km_out", "km_in_catchment", "reached_share_of_length", "isolated_kept", "isolated_kept_km",
                "outlets", "connections", "dropped", "uphill", "class_km", "class_sources", "strahler_km")
    else:
        from terrain_s2 import routed_network as RN
        pts = [tuple(float(v) for v in t.split(",")) for t in a.trace]
        s = RN.run_catchment(cd, a.runs, a.bundles or data / "tiles", params=params, trace_points=pts)
        keys = ("evidence", "outlet", "offsets", "mapped", "detected", "pruned_km", "network_km", "network_km_in_catchment",
                "class_km", "class_sources", "strahler_km", "crossings", "trace")
    print("\n" + "\n".join(f"{k}: {s.get(k)}" for k in keys))
    print(f"\n-> {cd / 'network_a.gpkg'}\n-> {cd / 'stage_a.json'}")


if __name__ == "__main__":
    main()
