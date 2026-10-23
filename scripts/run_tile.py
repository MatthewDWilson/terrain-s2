#!/usr/bin/env python
"""Run partition tiles by Stage 2's own route (``run.tile``): the bundle (built if missing), then the network
chain with a model set.

    python scripts/run_tile.py --tiles BY20_10000_0405 --model-set ml3c_v1
    python scripts/run_tile.py --tiles BY20_10000_0405 --model-set ml3c_v1 --device cpu --out <data>/scratch/cpu

Outputs go to ``--out/<tile>`` (default ``<TERRAIN_DATA_DIR>/network/<tile>``). Per tile it prints the device,
the total time, the U-Net's time, the candidates and recorded structures, and peak memory; ``--device cpu``
measures the chain without a GPU (as an EDDIE worker may have to run it). ``--param key=value`` sets any
``run.network`` option (values parsed as JSON where they can be, e.g. ``--param rasters=\\"hydro\\"``).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _value(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--model-set", default="ml3c_v1", help="a set in ml/model_sets/, a path, or 'none'")
    ap.add_argument("--out", help="parent folder of the tiles' outputs (default <data>/network)")
    ap.add_argument("--device", default="auto", help="auto, cpu or gpu")
    ap.add_argument("--bundles")
    ap.add_argument("--partition")
    ap.add_argument("--no-build", action="store_true", help="use existing bundles only")
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    from terrain_s2.run import tile
    params = dict(device=a.device)
    for kv in a.param:
        k, _, v = kv.partition("=")
        params[k] = _value(v)
    ms = None if str(a.model_set).lower() == "none" else a.model_set
    for t in a.tiles:
        out = Path(a.out) / t if a.out else None
        r = tile(t, model_set=ms, out=out, bundles=a.bundles, partition=a.partition, build=not a.no_build, params=params)
        cm = r.info.get("channel_model") or {}
        sc = r.info.get("scored") or {}
        print(json.dumps(dict(tile=t, device=r.info.get("device"), seconds=r.info.get("seconds"),
                              unet_device=cm.get("device"), unet_s=cm.get("seconds"),
                              candidates=sum((r.info.get("crossings") or {}).values()), recorded=sc.get("recorded"),
                              added_records=sc.get("added_records"), peak_rss_gb=r.perf.get("peak_rss_gb"),
                              model_set=(r.info.get("model_set") or {}).get("name"), out=str(r.out))))


if __name__ == "__main__":
    main()
