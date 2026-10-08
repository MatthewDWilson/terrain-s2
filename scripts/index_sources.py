#!/usr/bin/env python
"""Build the source coverage index (sites/sources_coverage.geojson) from sources.yml.

    python scripts/index_sources.py --partition D:/eddie_data/partition/nz_tiles.gpkg
    python scripts/index_sources.py --only es_drainage es_stopbanks      # re-index some, keep the rest

ArcGIS layers: the extent from the service, a feature count per 1:50k map sheet in that extent, then
per 1:10k tile inside the sheets with features (limited to the partition's sheets and tiles with
--partition, i.e. where LiDAR exists); the coverage is the union of tiles with features. Sources
whose URL is not resolved yet (``url: TODO``) are recorded as unresolved, not queried. LINZ WFS layers are national (nz); OSM is global; ``coverage:`` in
sources.yml overrides. Disabled sources are indexed too (marked), so enabling one needs no re-index.
Re-run when sources are added or their data change; the file is small and diffs cleanly in git.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from terrain_s2.acquire import registry  # noqa: E402
from terrain_s2.acquire.http import Http  # noqa: E402
from terrain_s2.acquire.site import load_sources  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=str(ROOT / "sites" / "sources.yml"))
    ap.add_argument("--out", help="default: sources_coverage.geojson beside sources.yml")
    ap.add_argument("--partition", help="partition GeoPackage (layer sheets): count only sheets with LiDAR")
    ap.add_argument("--only", nargs="+", help="re-index these sources, keep the others from the existing index")
    a = ap.parse_args()
    sources = load_sources(a.sources)
    out = Path(a.out) if a.out else registry.default_index_path(a.sources)
    sheets = tiles = None
    if a.partition:
        import geopandas as gpd
        sheets = set(gpd.read_file(a.partition, layer="sheets").sheet)
        tiles = set(gpd.read_file(a.partition, layer="linz_tiles", ignore_geometry=True).tile_id)
        print(f"partition: {len(sheets)} sheets, {len(tiles)} LINZ tiles with LiDAR")
    old = {}
    if out.exists():
        import json
        old = {f["properties"]["name"]: f for f in json.loads(out.read_text())["features"]}
    http = Http()
    feats = []
    for name, spec in sources.items():
        if a.only and name not in a.only:
            if name in old:
                feats.append(old[name])
            continue
        try:
            f = registry.coverage_record(http, name, spec, sheets, tiles=tiles)
        except Exception as e:  # noqa: BLE001 - keep the previous record, say so
            print(f"  {name}: FAILED {e}" + ("; keeping the previous record" if name in old else ""))
            if name in old:
                feats.append(old[name])
            continue
        p = f["properties"]
        print(f"  {name}: {p['coverage']}" + (f", {p.get('n_features')} feature(s)" if "n_features" in p else ""))
        feats.append(f)
    registry.write_index(feats, out)
    print(f"{len(feats)} sources -> {out}")


if __name__ == "__main__":
    main()
