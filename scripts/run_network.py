#!/usr/bin/env python
"""Drain/stream network and crossing products for an AOI (DESIGN_NOTES §4i).

    python scripts/run_network.py --dem dem.tif [--dsm dsm.tif] --aoi aoi.geojson \
        [--roads roads.gpkg] [--stream-model stream_model.joblib] --out out/site

Writes ``network.gpkg`` with layers:
  channels   reaches (chained through junctions, smoothed, generalised): class (stream/drain),
             p_stream (after smoothing along the network), p_stream_raw, class_confidence,
             through_culvert, and the reach features
  crossings  culvert candidates: x, y, sources (gap / bump / testA / testB / road_x_drain /
             road_end_pair), dem_h_b, deck_m (DSM deck over the crossing), on_stream
             (drain-bed rise, m), veg_frac, road_dist_m, tier (with roads if given), tier_dem_only
  repairs    continuity joins (gaps whose bed rises < 0.15 m): DEM-conditioning repairs
Roads are optional; without them the road-based sources are absent and tier = tier_dem_only.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from terrain_s2.run import network  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", nargs="+", required=True)
    ap.add_argument("--dsm", nargs="*", default=None)
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--aoi-crs")
    ap.add_argument("--roads", help="optional road centrelines (any vector format)")
    ap.add_argument("--stream-model", help="optional trained stream/drain model (joblib)")
    ap.add_argument("--buffer", type=float, default=200.0)
    ap.add_argument("--max-hand", type=float, default=10.0, help="floodplain: height above nearest drainage (m); <= 0 disables")
    ap.add_argument("--no-clean", action="store_true", help="keep isolated pieces and short dangling reaches")
    ap.add_argument("--assume-crs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", action="store_true", help="cProfile the run: <out>/profile.prof and profile_top.txt")
    ap.add_argument("--height-model", choices=["rem", "hand"], default="rem",
                    help="height above drainage for the floodplain and rasters: relative elevation model (IDW of channel "
                         "elevations, continuous) or HAND (nearest drainage along flow); default rem")
    ap.add_argument("--no-conditioned", action="store_true",
                    help="skip the conditioned hydrology pass (conditioned.tif, residual_ponds)")
    ap.add_argument("--burn-m", type=float, default=0.25, help="channel burn below the local bed, conditioned pass (m)")
    ap.add_argument("--context", help="context.gpkg with rivers_rec2 to seed upstream area at the window edge "
                                      "(default: context.gpkg beside the DEM, if present)")
    ap.add_argument("--rasters", choices=["all", "hydro", "none"], default="all",
                    help="write features.tif (DEM feature stack) and hydro.tif (HAND, upstream area, ...), cropped "
                         "to the AOI, with rasters.json describing the bands (all: both; hydro: hydro.tif only)")
    a = ap.parse_args()
    run = network(a.dem, a.aoi, a.out, dsm=a.dsm, roads=a.roads, stream_model=a.stream_model, buffer=a.buffer,
                  max_hand=a.max_hand, clean=not a.no_clean, aoi_crs=a.aoi_crs, assume_crs=a.assume_crs,
                  device=a.device, height_model=a.height_model, conditioned=not a.no_conditioned, burn_m=a.burn_m,
                  context=a.context, rasters=a.rasters, profile=a.profile)
    print(json.dumps(run.info))


if __name__ == "__main__":
    main()
