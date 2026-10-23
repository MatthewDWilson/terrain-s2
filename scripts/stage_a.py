#!/usr/bin/env python
"""Stage A of channel network v2 (``catchment_mode.run_catchment``): catchment mode on one catchment's tiles.

    python scripts/stage_a.py --catchment sh12 --runs ~/terrain_data/runs_c
    python scripts/stage_a.py --catchment sh12 --runs ~/terrain_data/runs_c --param connect_max_m=80

``--catchment``: a folder name under ``<TERRAIN_DATA_DIR>/catchments`` (``catchment_tiles.py`` writes it), or a path.
The runs should have stage B's ``network_b.gpkg`` (else stage C's network is used) and stage C's
``channel_prob.tif``. Writes ``<catchment>/network_a.gpkg`` (channels drawn downstream, with Strahler order,
upstream length, distance to the outlet, the outlet's kind, class_v2 and its source, uphill and isolated flags;
outlets; connections; attempts; dropped), ``<catchment>/stage_a.json``, and ``<runs>/<tile>/network_a.gpkg``.
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
    a = ap.parse_args()
    from terrain_s2 import catchment_mode as A
    from terrain_s2.acquire.elevation import default_data_dir
    data = default_data_dir()
    cd = Path(a.catchment)
    if not (cd / "catchment.gpkg").exists():
        cd = data / "catchments" / a.catchment
    params = {k: _value(v) for k, v in (kv.split("=", 1) for kv in a.param)}
    s = A.run_catchment(cd, a.runs, a.bundles or data / "tiles", params=params)
    keys = ("km_in", "km_out", "km_in_catchment", "reached_share_of_length", "isolated_kept", "isolated_kept_km",
            "outlets", "connections", "dropped", "uphill", "class_km", "class_sources", "strahler_km")
    print("\n" + "\n".join(f"{k}: {s.get(k)}" for k in keys))
    print(f"\n-> {cd / 'network_a.gpkg'}\n-> {cd / 'stage_a.json'}")


if __name__ == "__main__":
    main()
