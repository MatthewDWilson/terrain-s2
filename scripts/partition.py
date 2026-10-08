#!/usr/bin/env python
"""Partition NZ into processing tiles where LiDAR elevation exists (LINZ map-sheet grid).

    python scripts/partition.py --out D:/eddie_data/partition/nz_tiles.gpkg --cache D:/eddie_data/cache
    python scripts/partition.py --unit quadrant --out D:/eddie_data/partition/nz_quadrants.gpkg --cache D:/eddie_data/cache
    python scripts/partition.py --check D:/eddie_data/cache/linz_stac      # grid vs cached tile footprints

Reads the nz-elevation STAC catalogue: every ``<region>/<survey>/dem_1m/2193/collection.json`` (and
``dsm_1m``) lists its tiles by name, and the names are LINZ map-sheet ids, so tile footprints come
from the grid (``terrain_s2.grid``) without fetching ~10^5 item records. One request per collection,
cached under ``<cache>/linz_collections``; ``--refresh`` re-reads them.

Writes a GeoPackage:
  linz_tiles   one row per LINZ elevation tile (1:10k): surveys covering it, latest survey and its
               capture end, region, DSM present
  tiles        the processing units with their LINZ tile's attributes and a split. --unit tile (default,
               production): the LINZ 1:10k tile itself (4.8 x 7.2 km, one COG), read with a buffer from
               its neighbours. --unit quadrant (testing): its four quarters, 2.4 x 3.6 km, named by the
               tile and a cardinal suffix (BW24_10000_0402_NW). n_neighbours_same_survey (0-8) and
               survey_edge say whether the buffer comes from the same survey.
  sheets       1:50k sheets with tile counts per split
``split`` (train / validation / test) is assigned per 1:50k sheet (24 x 36 km) from a hash of its id,
so neighbouring tiles share a split and spatial leakage between them is avoided. Sheets holding a
site from sites/*.yml with role test or validation take that role (canterbury1 and canterbury2 make
BW24 a test sheet), so no training tile shares a sheet with a held-out site.
"""
import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from terrain_s2 import grid  # noqa: E402

STAC = "https://nz-elevation.s3-ap-southeast-2.amazonaws.com/"
# LINZ also publishes a national mosaic (new-zealand/new-zealand/dem_1m), tiled by 1:50k sheet and
# dated by its latest input. It must not pick the survey (it would win every tile and cover whole
# sheets, sea included): tiles record whether it covers them, which helps fill buffers at survey edges.
NATIONAL = {"new-zealand"}
COLL = re.compile(r"^\.?/?([^/]+)/([^/]+)/(dem|dsm)_1m/2193/collection\.json$")


def split_of(sheet: str, fractions=(0.8, 0.1, 0.1)) -> str:
    u = int(hashlib.sha1(sheet.encode()).hexdigest()[:8], 16) / 16 ** 8
    return "train" if u < fractions[0] else "validation" if u < fractions[0] + fractions[1] else "test"


def collections(http, cache: Path, refresh=False, root=STAC, log=print):
    """[{region, survey, product, end, tiles: [ids], url, updated}] for every 1 m DEM/DSM collection."""
    cat = http.json(urljoin(root, "catalog.json"))
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for link in cat.get("links", []):
        m = COLL.match(link.get("href", "")) if link.get("rel") == "child" else None
        if not m:
            continue
        region, survey, product = m.groups()
        url = urljoin(root, link["href"].lstrip("./"))
        f = cache / f"{region}__{survey}__{product}.json"
        if f.exists() and not refresh:
            rec = json.loads(f.read_text())
        else:
            c = http.json(url)
            items = [Path(it["href"]).stem for it in c.get("links", []) if it.get("rel") == "item"]
            rec = dict(region=region, survey=c.get("linz:slug") or survey, product=product, url=url,
                       updated=c.get("updated"), end=(c["extent"]["temporal"]["interval"][0][1] or "")[:10],
                       start=(c["extent"]["temporal"]["interval"][0][0] or "")[:10], title=c.get("title"),
                       licence=c.get("license"), tiles=items)
            f.write_text(json.dumps(rec))
        out.append(rec)
    log(f"{len(out)} collections ({sum(r['product'] == 'dem' for r in out)} DEM, "
        f"{sum(r['product'] == 'dsm' for r in out)} DSM)")
    return out


def linz_tiles(colls, log=print):
    """{tile id: {surveys, latest_survey, latest_end, region, has_dsm}} over all DEM collections."""
    dem, dsm, bad, national = defaultdict(list), set(), set(), set()
    for c in colls:
        if c["region"] in NATIONAL:
            national.update((c["product"], t) for t in c["tiles"])
            continue
        for t in c["tiles"]:
            try:
                grid.parse(t)
            except ValueError:
                bad.add(t)
                continue
            if c["product"] == "dem":
                dem[t].append(c)
            else:
                dsm.add((c["survey"], t))
    if bad:
        log(f"{len(bad)} tile names are not map-sheet ids and were skipped, e.g. {sorted(bad)[:3]}")
    nat = {grid.parent(t, 50000) for p, t in national if p == "dem" and _is_id(t)}
    out = {}
    for t, cs in dem.items():
        latest = max(cs, key=lambda c: c["end"])
        out[t] = dict(surveys=",".join(sorted(c["survey"] for c in cs)), n_surveys=len(cs), latest_survey=latest["survey"],
                      latest_end=latest["end"], region=latest["region"], has_dsm=(latest["survey"], t) in dsm,
                      in_national=grid.parent(t, 50000) in nat)
    if national:
        log(f"national mosaic: {len(nat)} 1:50k sheets (recorded per tile as in_national, not used to choose surveys)")
    return out


def _is_id(t):
    try:
        grid.parse(t)
        return True
    except ValueError:
        return False


def site_sheets(site_files, root=ROOT):
    """{sheet: role} for sheets holding a held-out site (test beats validation)."""
    from terrain_s2.acquire.site import load_site
    out = {}
    for f in site_files:
        try:
            st = load_site(f, root)
        except ValueError:
            continue                                         # sources.yml and other non-site files
        if st.role in ("test", "validation"):
            for sh in grid.NZSheetGrid(50000).tiles(st.window):
                out[sh] = "test" if "test" in (st.role, out.get(sh)) else st.role
    return out


def processing_tiles(lt: dict, unit: str, fractions, forced=None):
    """Processing units over the LINZ tiles (``unit``: tile or quadrant), each with its LINZ tile's
    attributes. A LINZ tile finer than 1:10k (none in the catalogue at 8 Oct 2026) maps to its 1:10k parent."""
    forced = forced or {}
    out = {}
    for t, a in lt.items():
        t10 = t if grid.parse(t)[3] == 10000 else grid.parent(t, 10000)
        for i in (grid.quadrants(t10) if unit == "quadrant" else [t10]):
            prev = out.get(i)
            if prev is None or a["latest_end"] > prev["latest_end"]:
                sheet = grid.parse(t10)[0]
                out[i] = dict(a, linz_tile=t, sheet=sheet, split=forced.get(sheet) or split_of(sheet, fractions))
    return out


def add_neighbours(pt: dict, lt: dict, unit: str):
    """Neighbour counts for the buffer: the 8 surrounding units that have LiDAR at all, and that share
    the unit's latest survey (a buffer read across a survey edge mixes captures)."""
    tid = grid.quadrant_id if unit == "quadrant" else grid.NZSheetGrid(10000).tile_id
    for i, a in pt.items():
        x0, y0, x1, y1 = grid.tile_bounds(i)
        w, h = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        nb = [tid(cx + dx * w, cy + dy * h) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        a["n_neighbours_lidar"] = sum(n in pt for n in nb)
        a["n_neighbours_same_survey"] = sum(n in pt and pt[n]["latest_survey"] == a["latest_survey"] for n in nb)
        a["survey_edge"] = a["n_neighbours_same_survey"] < 8
    return pt


def to_gdf(rows: dict, id_col="tile_id"):
    import geopandas as gpd
    from shapely.geometry import box
    ids = sorted(rows)
    return gpd.GeoDataFrame([dict({id_col: i}, **rows[i]) for i in ids],
                            geometry=[box(*grid.tile_bounds(i)) for i in ids], crs=2193)


def check(cache_dir, log=print):
    """Compare grid footprints with the STAC item bboxes cached by build_site.py (tiles_*.json)."""
    from rasterio.warp import transform_bounds
    n, bad = 0, []
    for f in sorted(Path(cache_dir).glob("tiles_*.json")):
        for t in json.loads(f.read_text()):
            try:
                g = grid.tile_bounds(t["id"])
            except ValueError:
                bad.append((t["id"], "not a map-sheet id"))
                continue
            # a lon/lat bbox of an NZTM square, taken back to NZTM, is larger than the square (grid
            # convergence, up to ~4 deg): so test that the grid tile lies inside it and shares its centre
            s = transform_bounds("EPSG:4326", "EPSG:2193", *t["bbox"], densify_pts=21)
            n += 1
            inside = g[0] >= s[0] - 1 and g[1] >= s[1] - 1 and g[2] <= s[2] + 1 and g[3] <= s[3] + 1
            dc = max(abs((g[0] + g[2]) - (s[0] + s[2])), abs((g[1] + g[3]) - (s[1] + s[3]))) / 2
            if not inside or dc > 0.05 * (g[2] - g[0]):
                bad.append((t["id"], f"grid {[round(v) for v in g]} not centred in stac {[round(v) for v in s]}"))
    log(f"checked {n} cached tile footprints against the grid; {len(bad)} disagree")
    for b in bad[:20]:
        log(f"  {b[0]}: {b[1]}")
    return n, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="GeoPackage to write")
    ap.add_argument("--cache", default="data/cache", help="cache root (collections under <cache>/linz_collections)")
    ap.add_argument("--unit", default="tile", choices=["tile", "quadrant"],
                    help="processing unit: the LINZ 1:10k tile (production) or its quadrants (testing)")
    ap.add_argument("--split", default="0.8,0.1,0.1", help="train,validation,test fractions (by 1:50k sheet)")
    ap.add_argument("--refresh", action="store_true", help="re-read the collections")
    ap.add_argument("--sites", default=str(ROOT / "sites" / "*.yml"),
                    help="site files whose test/validation roles fix their sheets' split")
    ap.add_argument("--check", metavar="DIR", help="check the grid against cached tile footprints (tiles_*.json) and exit")
    a = ap.parse_args()
    if a.check:
        _, bad = check(a.check)
        sys.exit(1 if bad else 0)
    if not a.out:
        ap.error("--out is required (or --check)")
    from terrain_s2.acquire.http import Http
    fr = tuple(float(v) for v in a.split.split(","))
    colls = collections(Http(), Path(a.cache) / "linz_collections", a.refresh)
    lt = linz_tiles(colls)
    import glob
    forced = site_sheets(sorted(glob.glob(a.sites)))
    if forced:
        print("held-out sheets from site files: " + ", ".join(f"{k} {v}" for k, v in sorted(forced.items())))
    pt = add_neighbours(processing_tiles(lt, a.unit, fr, forced), lt, a.unit)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    to_gdf(lt).to_file(out, layer="linz_tiles", driver="GPKG")
    tiles = to_gdf(pt)
    tiles.to_file(out, layer="tiles", driver="GPKG")
    sheets = tiles.groupby("sheet").agg(n_tiles=("tile_id", "size"), split=("split", "first"),
                                        region=("region", "first")).reset_index()
    to_gdf({r.sheet: dict(n_tiles=int(r.n_tiles), split=r.split, region=r.region) for r in sheets.itertuples()},
           "sheet").to_file(out, layer="sheets", driver="GPKG")
    meta = dict(built=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), unit=a.unit, split=fr,
                collections=[{k: c[k] for k in ("region", "survey", "product", "updated", "end")} for c in colls])
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    km2 = tiles.area.sum() / 1e6
    print(f"{len(lt)} LINZ tiles; {len(tiles)} processing units ({a.unit}, {km2:,.0f} km2); "
          f"split {tiles.split.value_counts().to_dict()}; -> {out}")
    print(tiles.groupby("region").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
