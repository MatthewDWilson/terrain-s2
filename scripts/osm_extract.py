#!/usr/bin/env python
"""Convert an OSM extract (Geofabrik's new-zealand-latest.osm.pbf) once into the GeoPackage bundles read OSM from.

    python scripts/osm_extract.py --pbf ~/terrain_data/osm/new-zealand-latest.osm.pbf

Writes <TERRAIN_DATA_DIR>/osm/new-zealand-osm.gpkg (the `extract` of the `osm` entry in sites/sources.yml; --out to
change) with layers `ways` (roads, rail, waterways, culverts) and `buildings`, in NZTM, plus a .json with the source
file's date. From then on bundle builds and refresh_source.py read OSM from it instead of Overpass. Run it again with
a newer download to update (a few minutes per layer).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    from terrain_s2.acquire import osm
    from terrain_s2.acquire.elevation import default_data_dir
    from terrain_s2.acquire.site import load_sources
    ext = load_sources(ROOT / "sites" / "sources.yml")["osm"].get("extract", "osm/new-zealand-osm.gpkg")
    out = Path(a.out) if a.out else (Path(ext) if Path(ext).is_absolute() else default_data_dir() / ext)
    print(f"{a.pbf} -> {out}")
    info = osm.extract_pbf(Path(a.pbf).expanduser(), out)
    print(f"done: {info['counts']}; source dated {info['file_date']}")


if __name__ == "__main__":
    main()
