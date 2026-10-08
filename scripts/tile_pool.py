#!/usr/bin/env python
"""Join the LiDAR partition with the source coverage index: the pool of tiles with labels.

    python scripts/tile_pool.py --partition D:/eddie_data/partition/nz_tiles.gpkg

Adds layer ``pool`` to the partition GeoPackage (or --out): the processing tiles where at least one
label source (crossings, channels, stopbanks) has features in the tile (counted per LINZ 1:10k tile
by index_sources.py; a source indexed only by sheet counts at sheet level), with the sources per
label type and the feature counts (for a quadrant, those of its LINZ tile: the label may lie in a
neighbouring quadrant). Context sources (roads, buildings, OSM) are
national and not listed. Re-run whenever the index or the partition changes.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from terrain_s2.acquire import registry  # noqa: E402

LABEL_TYPES = ("crossings", "channels", "stopbanks")


def pool(tiles, index, label_types=LABEL_TYPES):
    """tiles: GeoDataFrame (tile_id, sheet, split, ..., EPSG:2193); index: registry.load_index()."""
    import geopandas as gpd
    t4326 = tiles.to_crs(4326)
    cols = {lt: [[] for _ in range(len(tiles))] for lt in label_types}
    nfeat = [0] * len(tiles)
    for name, (geom, p) in index.items():
        if not p.get("enabled", True):
            continue
        kinds = [lt for lt in p.get("provides", []) if lt in label_types]
        if not kinds:
            continue
        per_tile, sheets = p.get("tiles"), p.get("sheets")
        key = None
        if per_tile:                                       # counted per LINZ tile; a quadrant takes its tile's
            key = tiles.linz_tile if "linz_tile" in tiles else tiles.tile_id
            hit, counts = key.isin(per_tile).to_numpy(), per_tile
        elif sheets:
            key, hit, counts = tiles.sheet, tiles.sheet.isin(sheets).to_numpy(), sheets
        else:
            hit, counts = t4326.intersects(geom).to_numpy(), {}
        for i in hit.nonzero()[0]:
            for k in kinds:
                cols[k][i].append(name)
            nfeat[i] += int(counts.get(key.iloc[i], 0)) if key is not None else 0
    out = tiles.copy()
    for k in label_types:
        out[k] = [",".join(sorted(v)) for v in cols[k]]
    out["n_label_types"] = sum((out[k] != "").astype(int) for k in label_types)
    out["label_features"] = nfeat
    return gpd.GeoDataFrame(out[out.n_label_types > 0], crs=tiles.crs)


def main():
    import geopandas as gpd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--index", default=str(ROOT / "sites" / "sources_coverage.geojson"))
    ap.add_argument("--out", help="GeoPackage for the pool layer (default: the partition file)")
    a = ap.parse_args()
    tiles = gpd.read_file(a.partition, layer="tiles")
    p = pool(tiles, registry.load_index(a.index))
    p.to_file(a.out or a.partition, layer="pool", driver="GPKG")
    print(f"{len(p)} of {len(tiles)} tiles have label sources -> {a.out or a.partition} (layer pool)")
    for k in LABEL_TYPES:
        print(f"  {k}: {(p[k] != '').sum()} tiles; " + ", ".join(f"{s} {n}" for s, n in
              p[k][p[k] != ""].str.split(",").explode().value_counts().items()))
    print("  by split: " + ", ".join(f"{s} {n}" for s, n in p.split.value_counts().items()))


if __name__ == "__main__":
    main()
