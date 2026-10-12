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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from terrain_s2.run import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
