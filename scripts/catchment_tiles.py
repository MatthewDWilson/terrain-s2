#!/usr/bin/env python
"""The partition tiles of a sea-draining catchment, as a manifest for run_tiles.py (channel network v2).

    python scripts/catchment_tiles.py --site sh12 --out ml/manifests/catchment_sh12.csv
    python scripts/catchment_tiles.py --xy 1640000 6080000 --overlap 500

The catchment is the one holding the site's AOI centre (``--site``: ``sites/<site>.yml``) or a point (``--xy``,
NZTM), from ``catchments.load`` (MfE sea-draining catchments; ``--catchments`` for another path). Lists every
partition tile meeting the catchment plus an overlap band (``--overlap`` m), with the share of each inside the
catchment, and writes the manifest (``tile_id``, ``council`` = ``catchment_<slug>``, ``wave`` = ``catchment``, the
partition's fields, ``in_catchment``, ``overlap_only``) and the catchment itself to
``<TERRAIN_DATA_DIR>/catchments/<slug>/catchment.gpkg`` (layers ``catchment`` and ``tiles``). Prints the catchment, its
area, the number of tiles and how much of the catchment the LiDAR tiles cover (the partition holds LiDAR tiles only:
the rest is outside coverage, where the network will end at a provisional outlet).
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--site", help="a site in sites/ (its AOI's centre)")
    g.add_argument("--xy", nargs=2, type=float, metavar=("X", "Y"), help="a point in NZTM")
    ap.add_argument("--catchments", help="the catchment layer (default <data>/reference/sea_draining_catchments.gpkg)")
    ap.add_argument("--partition", help="default <data>/partition/nz_tiles.gpkg")
    ap.add_argument("--overlap", type=float, default=500.0, help="band around the catchment (m)")
    ap.add_argument("--slug", help="name for the outputs (default the site, else the catchment id)")
    ap.add_argument("--out", help="manifest CSV (default ml/manifests/catchment_<slug>.csv)")
    a = ap.parse_args()
    import geopandas as gpd
    import yaml
    from shapely.geometry import Point

    from terrain_s2 import catchments, tiles
    from terrain_s2.acquire.elevation import default_data_dir
    data = default_data_dir()
    if a.site:
        site = yaml.safe_load((ROOT / "sites" / f"{a.site}.yml").read_text())
        aoi = gpd.read_file(ROOT / site["aoi"]).to_crs(2193)
        pt = aoi.union_all().centroid
    else:
        pt = Point(*a.xy)
    c = catchments.at(pt, a.catchments)
    if c is None:
        sys.exit(f"no catchment holds {pt.x:.0f}, {pt.y:.0f}")
    slug = a.slug or a.site or re.sub(r"[^A-Za-z0-9_-]+", "_", str(c.catchment_id))
    part = a.partition or tiles.default_partition(data)
    t = catchments.tiles_for(c, part, a.overlap)
    area_km2 = c.geometry.area / 1e6
    covered = t.geometry.intersection(c.geometry).area.sum() / c.geometry.area if len(t) else 0.0
    keep = [k for k in ("linz_tile", "split", "region", "latest_survey", "has_dsm", "sheet") if k in t.columns]
    m = t[["tile_id"] + keep + ["in_catchment", "overlap_only"]].copy()
    m.insert(1, "council", f"catchment_{slug}")
    m.insert(2, "wave", "catchment")
    m["catchment_id"] = c.catchment_id
    m["catchment_name"] = c.catchment_name
    out = Path(a.out or ROOT / "ml" / "manifests" / f"catchment_{slug}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out, index=False)
    cd = data / "catchments" / slug
    cd.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame([dict(catchment_id=c.catchment_id, catchment_name=c.catchment_name, area_km2=round(area_km2, 2),
                           lidar_share=round(float(covered), 4), overlap_m=a.overlap)],
                     geometry=[c.geometry], crs=2193).to_file(cd / "catchment.gpkg", layer="catchment", driver="GPKG")
    if len(t):
        t.to_file(cd / "catchment.gpkg", layer="tiles", driver="GPKG")
    print(f"catchment {c.catchment_id} ({c.catchment_name}): {area_km2:,.1f} km2; {len(t)} tiles "
          f"({int((~t.overlap_only).sum())} with area inside, {int(t.overlap_only.sum())} in the {a.overlap:g} m band only); "
          f"LiDAR tiles cover {covered:.1%} of it")
    print(f"-> {out}\n-> {cd / 'catchment.gpkg'}")


if __name__ == "__main__":
    main()
