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
from terrain_s2.acquire.site import load_site, load_sources  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", nargs="+", required=True, help="site definition(s); globs allowed")
    ap.add_argument("--sources", default=str(ROOT / "sites" / "sources.yml"))
    ap.add_argument("--out", default="data/sites")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--only", choices=["elevation", "sources"])
    ap.add_argument("--dry-run", action="store_true", help="discover surveys/tiles and list queries; download nothing")
    a = ap.parse_args()
    sources = load_sources(a.sources)
    paths = [p for s in a.site for p in (glob.glob(s) or [s])]
    bad = 0
    for p in paths:
        site = load_site(p, ROOT)
        prov = build_site(site, sources, a.out, a.cache, dry_run=a.dry_run, only=a.only)
        print(f"[{site.id}] done in {prov['seconds']} s; errors: {len(prov['errors'])}; -> {Path(a.out) / site.id}")
        for e in prov["errors"]:
            print(f"    ERROR {e['step']}: {e['error']}")
        for w in prov.get("warnings", []):
            print(f"    WARNING {w['step']}: {w['warning']}")
        bad += bool(prov["errors"])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
