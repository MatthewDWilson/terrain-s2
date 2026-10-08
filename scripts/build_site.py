#!/usr/bin/env python
"""Assemble site bundles automatically (R2 milestone M1): LiDAR DEM/DSM from the LINZ nz-elevation
bucket, council inventories (ArcGIS REST), roads/rail/tracks (LINZ Topo50 WFS, OpenStreetMap).

    python scripts/build_site.py --site sites/canterbury1.yml --out D:/eddie_data/sites --cache D:/eddie_data/cache
    python scripts/build_site.py --site sites/*.yml --dry-run        # plan only: surveys, tiles, queries

LINZ Topo50 layers need an API key: set LINZ_API_KEY (data.linz.govt.nz, My account, API keys).
Keep --out and --cache outside the repository and OneDrive. See DESIGN_NOTES §7 and the R2/R3 design.
"""
import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from terrain_s2.acquire.build import build_site  # noqa: E402
from terrain_s2.acquire import registry  # noqa: E402
from terrain_s2.acquire.site import load_site, load_sources  # noqa: E402


def tile_site(tile_id, partition, buffer_m):
    """A site for one partition tile: its bounds, its LINZ survey and region, its split as role."""
    import geopandas as gpd
    from terrain_s2 import grid
    from terrain_s2.acquire.site import Site
    if not partition:
        sys.exit("--tile needs --partition (for the tile's region and survey)")
    t = gpd.read_file(partition, layer="tiles", where=f"tile_id = '{tile_id}'")
    if not len(t):
        sys.exit(f"{tile_id} is not in {partition}")
    r = t.iloc[0]
    return Site(id=tile_id, bounds=grid.tile_bounds(tile_id), buffer_m=buffer_m, role=r.split, sources="auto",
                elevation=dict(region=r.region, products=["dem", "dsm"] if r.has_dsm else ["dem"], survey=r.latest_survey),
                notes=f"partition tile; LINZ tile {r.linz_tile}; neighbours with the same survey: {r.n_neighbours_same_survey}/8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", nargs="+", default=[], help="site definition(s); globs allowed")
    ap.add_argument("--tile", nargs="+", default=[], help="LINZ tile id(s) from the partition, e.g. BW24_10000_0402")
    ap.add_argument("--partition", help="partition GeoPackage (region, survey, split for --tile)")
    ap.add_argument("--buffer", type=float, default=200.0, help="buffer for --tile sites (m)")
    ap.add_argument("--sources", default=str(ROOT / "sites" / "sources.yml"))
    ap.add_argument("--index", help="coverage index (default: sources_coverage.geojson beside sources.yml)")
    ap.add_argument("--out", default="data/sites")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--only", choices=["elevation", "sources"])
    ap.add_argument("--dry-run", action="store_true", help="discover surveys/tiles and list queries; download nothing")
    a = ap.parse_args()
    sources = load_sources(a.sources)
    idx_path = Path(a.index) if a.index else registry.default_index_path(a.sources)
    index = registry.load_index(idx_path) if idx_path.exists() else None
    registry_file = Path(a.sources).resolve()
    paths = [p for s in a.site for p in (glob.glob(s) or [s]) if Path(p).resolve() != registry_file]   # sites/*.yml
    if not paths and not a.tile:
        ap.error("give --site and/or --tile")
    sites = [load_site(p, ROOT) for p in paths] + [tile_site(t, a.partition, a.buffer) for t in a.tile]
    bad = 0
    for site in sites:
        prov = build_site(site, sources, a.out, a.cache, dry_run=a.dry_run, only=a.only, index=index)
        print(f"[{site.id}] done in {prov['seconds']} s; errors: {len(prov['errors'])}; -> {Path(a.out) / site.id}")
        for e in prov["errors"]:
            print(f"    ERROR {e['step']}: {e['error']}")
        for w in prov.get("warnings", []):
            print(f"    WARNING {w['step']}: {w['warning']}")
        bad += bool(prov["errors"])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
