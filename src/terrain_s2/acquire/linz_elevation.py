"""LiDAR DEM/DSM from the LINZ nz-elevation bucket (public S3, Cloud-Optimised GeoTIFFs, STAC).

Collections are found from the catalogue (path ``<region>/<survey>/<product>_1m/2193/collection.json``);
the survey covering the AOI with the latest capture is chosen unless one is named. A collection
lists its tiles by name only (e.g. ``BS24_10000_0502``), so an index of tile footprints is built
once per collection from the item records and cached; only the window around the AOI is then read
from the COGs (windowed HTTP range reads, no full-tile download). Native CRS is EPSG:2193.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

ROOT = "https://nz-elevation.s3-ap-southeast-2.amazonaws.com/"
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
                GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", AWS_NO_SIGN_REQUEST="YES")


def to_lonlat(bounds_2193):
    from rasterio.warp import transform_bounds
    return transform_bounds("EPSG:2193", "EPSG:4326", *bounds_2193, densify_pts=21)


def _intersects(a, b):
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def find_collections(http, region: str, product: str, root: str = ROOT):
    """Collection URLs for a region and product (dem / dsm) at 1 m."""
    cat = http.json(urljoin(root, "catalog.json"), source="linz-stac-catalog")
    out = []
    for l in cat.get("links", []):
        if l.get("rel") != "child":
            continue
        h = l["href"].lstrip("./")
        parts = h.split("/")
        if len(parts) >= 4 and parts[0] == region and parts[2] == f"{product}_1m":
            out.append(urljoin(root, h))
    return out


def choose_collection(http, region, product, aoi_lonlat, survey=None, root=ROOT):
    """(url, collection, considered): the named survey, else the latest-captured covering one."""
    considered = []
    for url in find_collections(http, region, product, root):
        c = http.json(url, source=f"linz-stac-{region}-{product}")
        bbox = c["extent"]["spatial"]["bbox"][0]
        end = c["extent"]["temporal"]["interval"][0][1] or ""
        slug = c.get("linz:slug") or url.split("/")[-4]
        considered.append(dict(url=url, slug=slug, end=end, intersects=_intersects(bbox, aoi_lonlat)))
        if survey and slug == survey:
            return url, c, considered
    if survey:
        raise ValueError(f"survey {survey!r} not found for {region} {product}; considered {[c['slug'] for c in considered]}")
    hits = [c for c in considered if c["intersects"]]
    if not hits:
        raise ValueError(f"no {region} {product}_1m collection intersects the AOI; considered {[c['slug'] for c in considered]}")
    best = max(hits, key=lambda c: c["end"])
    return best["url"], http.json(best["url"]), considered


def tile_index(http, coll_url, coll, cache_dir, workers: int = 16):
    """Footprints of all tiles in a collection: [{id, bbox, href, checksum}], cached per collection."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = (coll.get("linz:slug", "c") + "_" + coll.get("linz:geospatial_category", "x") + "_" +
           str(coll.get("updated", ""))[:10]).replace(":", "-")
    f = cache_dir / f"tiles_{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    links = [l for l in coll.get("links", []) if l.get("rel") == "item"]

    def one(l):
        url = urljoin(coll_url, l["href"])
        it = http.json(url)
        asset = next((a for a in it.get("assets", {}).values()
                      if "geotiff" in str(a.get("type", "")) or str(a.get("href", "")).lower().endswith((".tif", ".tiff"))), None)
        return dict(id=it.get("id"), bbox=it.get("bbox"), href=urljoin(url, asset["href"]) if asset else None,
                    checksum=(asset or {}).get("file:checksum"), item=url)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        idx = [r for r in ex.map(one, links) if r["bbox"] and r["href"]]
    f.write_text(json.dumps(idx))
    return idx


def select_tiles(index, aoi_lonlat):
    return [t for t in index if _intersects(t["bbox"], aoi_lonlat)]


def fetch_window(tiles, window_2193, out_path, assume_crs=None):
    """Read the window from the tiles' COGs (HTTP range reads) and write a GeoTIFF clip."""
    import numpy as np
    import rasterio

    from .. import dataio
    paths = [("/vsicurl/" + t["href"]) if str(t["href"]).startswith(("http://", "https://")) else
             str(t["href"]).replace("file://", "") for t in tiles]
    with rasterio.Env(**GDAL_ENV):
        w = dataio.read_window(paths, window_2193, 0.0, assume_crs=assume_crs)
    z = np.where(np.isfinite(w.z), w.z, -9999.0).astype(np.float32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", driver="GTiff", height=z.shape[0], width=z.shape[1], count=1, dtype="float32",
                       crs=w.crs, transform=w.transform, nodata=-9999.0, compress="deflate", predictor=3,
                       tiled=True) as d:
        d.write(z, 1)
        d.update_tags(source_tiles=",".join(t["id"] for t in tiles))
    return dict(path=str(out_path), tiles=[dict(id=t["id"], href=t["href"], checksum=t["checksum"]) for t in tiles],
                shape=list(z.shape), crs=w.crs.to_string(), uncovered_m=w.uncovered_m)


def fetch_product(http, region, product, window_2193, out_path, cache_dir, survey=None, root=ROOT, dry_run=False):
    aoi_ll = to_lonlat(window_2193)
    url, coll, considered = choose_collection(http, region, product, aoi_ll, survey, root)
    idx = tile_index(http, url, coll, cache_dir)
    tiles = select_tiles(idx, aoi_ll)
    if not tiles:
        raise ValueError(f"{coll.get('linz:slug')}: no tiles intersect the window")
    rec = dict(product=product, collection=url, survey=coll.get("linz:slug"), title=coll.get("title"),
               licence=coll.get("license"), capture=coll["extent"]["temporal"]["interval"][0],
               considered=considered, n_tiles=len(tiles))
    if dry_run:
        rec["tiles"] = [t["id"] for t in tiles]
        return rec
    rec.update(fetch_window(tiles, window_2193, out_path))
    return rec
