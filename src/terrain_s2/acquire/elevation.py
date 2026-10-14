"""One elevation reader for both stages (integration plan S2-9; DESIGN_NOTES section 9 item 8).

``ElevationSource`` reads LINZ nz-elevation (a static STAC of 1 m COGs) for any area:

- every ``<region>/<survey>/<product>_1m/<epsg>`` collection meeting the area is a candidate, whatever the
  region (the region of a site or tile no longer picks the survey);
- order: the profile's ``priority`` surveys first (a newer survey is not always better), then newest capture,
  then most recently updated; ``exclude`` surveys never; the national mosaic last, as fill only;
- cells are filled survey by survey in that order, and ``lidar_source`` records which survey (its index)
  gave each cell;
- the DSM follows the DEM: ``pair()`` plans the DSM over the DEM's surveys in the DEM's order;
- ``plan.source_version()`` lists collections, ``updated`` and every item's ``file:checksum`` (the generator
  key, S2-13);
- with ``mirror`` (default on), whole COG tiles are kept under ``<cache_dir>/../cog`` (``TERRAIN_DATA_DIR/cog``
  with the default layout), keyed by item id and checked against ``file:checksum`` (sha2-256 multihash), so
  repeated reads of the same tiles (training runs, both stages) download once. ``file://`` items are read in
  place.

It implements the interface ``terrain_s2.stage1.source`` documents (Stage 1's raster backend uses it through
``get_source``), and replaces ``linz_elevation.fetch_product`` for Stage 2 bundles (``acquire/build.py``).
``linz_elevation`` stays: its STAC helpers are used here and the name remains importable. Stage 2 does not
import Stage 1: the record types are defined here with the same fields as Stage 1's stopgap.

    from terrain_s2.acquire.elevation import ElevationSource
    src = ElevationSource()                                  # NZ defaults, cache under TERRAIN_DATA_DIR
    dem_plan, dsm_plan = src.pair((1568600, 5197000, 1573800, 5204600))
    dem = src.read(dem_plan); dsm = src.read(dsm_plan)       # z, transform, crs, lidar_source, used
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin

import numpy as np

from . import linz_elevation as le

DEFAULT_PROFILE = {
    "name": "nz",
    "crs": {"horizontal": 2193, "vertical": 7839},
    "elevation": {
        "stac_root": None,                      # linz_elevation.ROOT, read when a source is made
        "products": {"dem": "dem_1m", "dsm": "dsm_1m"},
        "national": {"dem": "new-zealand/new-zealand/dem_1m/2193/collection.json"},
        "priority": [],
        "exclude": [],
        "nodata": -9999,
    },
}


class NoElevationData(LookupError):
    """No survey covers the requested area."""


@dataclass
class Item:
    id: str
    href: str
    checksum: str | None
    bbox: tuple


@dataclass
class Survey:
    index: int                      # 1 = highest priority
    slug: str
    collection: str
    title: str | None
    capture: tuple                  # (start, end) ISO strings
    updated: str | None
    national: bool
    items: list = field(default_factory=list)
    region: str | None = None
    licence: str | None = None

    def record(self, with_items: bool = True) -> dict:
        d = {"index": self.index, "survey": self.slug, "collection": self.collection, "title": self.title,
             "capture": list(self.capture), "updated": self.updated, "national": self.national,
             "region": self.region, "licence": self.licence}
        if with_items:
            d["items"] = [{"id": i.id, "href": i.href, "checksum": i.checksum} for i in self.items]
        return d


@dataclass
class SourcePlan:
    product: str
    bounds: tuple
    crs: str
    surveys: list

    def mapping(self) -> dict:
        return {s.slug: s.index for s in self.surveys}

    def source_version(self) -> dict:
        return {"product": self.product,
                "surveys": [{"collection": s.collection, "updated": s.updated,
                             "items": sorted([i.id, i.checksum] for i in s.items)} for s in self.surveys]}


@dataclass
class ElevationRead:
    z: np.ndarray
    transform: tuple
    crs: str
    lidar_source: np.ndarray
    plan: SourcePlan
    used: list

    @property
    def valid_fraction(self) -> float:
        return float(np.isfinite(self.z).mean()) if self.z.size else 0.0


def crs_string(profile: dict) -> str:
    c = profile["crs"]
    return f"EPSG:{c['horizontal']}+{c['vertical']}" if c.get("vertical") else f"EPSG:{c['horizontal']}"


def whole_metre_bounds(bounds) -> tuple:
    """Bounds expanded outward to whole metres (the 1 m grid LINZ tiles lie on)."""
    x0, y0, x1, y1 = bounds
    return (float(np.floor(x0)), float(np.floor(y0)), float(np.ceil(x1)), float(np.ceil(y1)))


def default_data_dir() -> Path:
    from ..settings import from_env
    return Path(from_env().data_dir)


class ElevationSource:
    """See the module docstring. ``cache_dir``: STAC documents and item indexes (default
    ``TERRAIN_DATA_DIR/stac``); ``mirror``: keep whole COG tiles under ``mirror_dir`` (default
    ``<cache_dir>/../cog``); ``ttl_s``: how long catalogue and collection documents are trusted."""

    def __init__(self, profile: dict | None = None, http=None, cache_dir=None, stac_root=None, mirror: bool = True,
                 mirror_dir=None, ttl_s: float = 86400.0, workers: int = 16):
        self.profile = profile or DEFAULT_PROFILE
        ev = self.profile["elevation"]
        self.root = stac_root or ev.get("stac_root") or le.ROOT
        if not self.root.endswith("/"):
            self.root += "/"
        if http is None:
            from .http import Http
            http = Http()
        self.http = http
        self.cache_dir = Path(cache_dir) if cache_dir else default_data_dir() / "stac"
        self.mirror = mirror
        self.mirror_dir = Path(mirror_dir) if mirror_dir else self.cache_dir.parent / "cog"
        self.ttl_s, self.workers = ttl_s, workers
        self.epsg = int(self.profile["crs"]["horizontal"])
        self.crs = crs_string(self.profile)

    # --- STAC, with a short-lived local cache (item indexes are keyed by each collection's ``updated``) ---
    def _json(self, url: str):
        f = self.cache_dir / "json" / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")
        if f.exists() and time.time() - f.stat().st_mtime < self.ttl_s:
            return json.loads(f.read_text())
        d = self.http.json(url)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(d))
        return d

    def collections(self, product: str = "dem"):
        """[(url, region, survey, national)] for the product at 1 m in the profile's CRS."""
        ev = self.profile["elevation"]
        folder = ev.get("products", {}).get(product, f"{product}_1m")
        national = (ev.get("national") or {}).get(product)
        cat = self._json(urljoin(self.root, "catalog.json"))
        out = []
        for link in cat.get("links", []):
            if link.get("rel") != "child":
                continue
            h = link["href"]
            h = h[2:] if h.startswith("./") else h
            parts = h.split("/")
            if len(parts) >= 5 and parts[2] == folder and parts[3] == str(self.epsg):
                out.append((urljoin(self.root, h), parts[0], parts[1], bool(national) and h == national))
        return out

    def plan(self, bounds, product: str = "dem", surveys=None) -> SourcePlan:
        """Surveys covering ``bounds`` (whole metres, in the profile's horizontal CRS) in priority order.
        ``surveys``: survey slugs to use, in this order (e.g. the DEM's, for the DSM); the national mosaic
        is still appended as fill."""
        ll = _lonlat_bounds(bounds, self.epsg)
        ev = self.profile["elevation"]
        exclude = set(ev.get("exclude") or [])
        priority = list(surveys) if surveys is not None else list(ev.get("priority") or [])
        cols = [c for c in self.collections(product) if c[2] not in exclude]
        if surveys is not None:
            cols = [c for c in cols if c[2] in priority or c[3]]
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            metas = list(ex.map(lambda c: self._json(c[0]), cols))
        cands = []
        for (url, region, slug, national), c in zip(cols, metas):
            bbox = c["extent"]["spatial"]["bbox"][0]
            if not le._intersects(bbox, ll):
                continue
            interval = c["extent"]["temporal"]["interval"][0]
            cands.append(dict(url=url, coll=c, region=region, slug=c.get("linz:slug") or slug, folder=slug,
                              national=national, start=interval[0] or "", end=interval[1] or "",
                              updated=c.get("updated") or ""))

        def order(c):
            if c["national"]:
                return (2, 0, "", "", c["slug"])
            for key in (c["slug"], c["folder"]):
                if key in priority:
                    return (0, priority.index(key), "", "", c["slug"])
            return (1, 0, _neg(c["end"]), _neg(c["updated"]), c["slug"])   # newest capture, then updated
        cands.sort(key=order)
        out = []
        for c in cands:
            idx = le.tile_index(self.http, c["url"], c["coll"], self.cache_dir / "tiles")
            tiles = le.select_tiles(idx, ll)
            if not tiles:
                continue
            out.append(Survey(index=len(out) + 1, slug=c["slug"], collection=c["url"], title=c["coll"].get("title"),
                              capture=(c["start"], c["end"]), updated=c["updated"] or None, national=c["national"],
                              items=[Item(t["id"], t["href"], t.get("checksum"), tuple(t["bbox"])) for t in tiles],
                              region=c["region"], licence=c["coll"].get("license")))
        if not out:
            raise NoElevationData(f"no {product} survey covers {tuple(round(b) for b in bounds)} "
                                  f"(EPSG:{self.epsg}); {len(cols)} collections checked")
        return SourcePlan(product=product, bounds=tuple(bounds), crs=self.crs, surveys=out)

    def pair(self, bounds, priority=None):
        """(DEM plan, DSM plan or None): the DSM over the DEM's surveys, in the DEM's order. ``priority``:
        survey slugs first for the DEM (e.g. a site's or tile's survey)."""
        if priority:
            prof = json.loads(json.dumps(self.profile))
            prof["elevation"]["priority"] = list(priority) + [s for s in prof["elevation"].get("priority") or []
                                                              if s not in priority]
            dem = ElevationSource(prof, self.http, self.cache_dir, self.root, self.mirror, self.mirror_dir,
                                  self.ttl_s, self.workers).plan(bounds, "dem")
        else:
            dem = self.plan(bounds, "dem")
        try:
            dsm = self.plan(bounds, "dsm", surveys=[s.slug for s in dem.surveys if not s.national])
        except NoElevationData:
            dsm = None
        return dem, dsm

    # --- reading ---
    def local(self, item: Item, survey: Survey) -> str:
        """A readable path for the item: the mirrored copy for http(s) items (downloaded once, checked
        against file:checksum), else the item's own path."""
        href = str(item.href)
        if not self.mirror or not href.startswith(("http://", "https://")):
            return href
        rel = survey.collection[len(self.root):] if survey.collection.startswith(self.root) else survey.slug
        d = self.mirror_dir / Path(rel).parent
        f = d / (Path(href.split("?")[0]).name or f"{item.id}.tif")
        side = f.with_suffix(f.suffix + ".json")
        if f.exists() and side.exists():
            meta = json.loads(side.read_text())
            if meta.get("checksum") == item.checksum and meta.get("href") == href:
                return str(f)
        content = self.http.get(href, ext="tif")
        ok = _check(content, item.checksum)
        if ok is False:
            raise IOError(f"{href}: content does not match file:checksum {item.checksum}")
        d.mkdir(parents=True, exist_ok=True)
        # Parallel workers often need the same neighbouring tile: each writes its own temporary file, and if
        # the rename fails because another worker has (or is reading) the finished tile, that tile is used.
        # Seen on Windows in wave 1 (WinError 5 and 32 on a shared '.part' name, 14 Oct 2026).
        import os
        import uuid
        tmp = f.with_name(f"{f.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part")
        tmp.write_bytes(content)
        meta = json.dumps({"href": href, "checksum": item.checksum, "verified": ok, "bytes": len(content),
                           "item": item.id, "survey": survey.slug})
        for attempt in range(5):
            try:
                os.replace(tmp, f)
                break
            except PermissionError:
                if f.exists() and f.stat().st_size == len(content):
                    tmp.unlink(missing_ok=True)
                    break
                time.sleep(0.5 * (attempt + 1))
        else:
            tmp.unlink(missing_ok=True)
            raise IOError(f"could not place {f} (in use by another process)")
        stmp = side.with_name(f"{side.name}.{os.getpid()}.part")
        stmp.write_text(meta)
        try:
            os.replace(stmp, side)
        except PermissionError:
            stmp.unlink(missing_ok=True)
        return str(f)

    def read(self, plan: SourcePlan) -> ElevationRead:
        x0, y0, x1, y1 = plan.bounds
        if any(abs(v - round(v)) > 1e-6 for v in plan.bounds):
            raise ValueError(f"bounds must lie on whole metres for a 1 m read: {plan.bounds}")
        H, W = int(round(y1 - y0)), int(round(x1 - x0))
        z = np.full((H, W), np.nan, np.float32)
        src = np.full((H, W), -1, np.int16)
        used = []
        nodata = self.profile["elevation"].get("nodata")
        import rasterio
        with rasterio.Env(**le.GDAL_ENV):
            for s in plan.surveys:
                if not np.isnan(z).any():
                    break
                put_any = False
                for it in s.items:
                    data, off = _read_item(self.local(it, s), plan.bounds, nodata)
                    if data is None:
                        continue
                    r0, c0 = off
                    h, w = data.shape
                    tz = z[r0:r0 + h, c0:c0 + w]
                    put = np.isnan(tz) & np.isfinite(data)
                    if put.any():
                        tz[put] = data[put]
                        src[r0:r0 + h, c0:c0 + w][put] = s.index
                        put_any = True
                if put_any:
                    used.append(s)
        return ElevationRead(z=z, transform=(1.0, 0.0, float(x0), 0.0, -1.0, float(y1)), crs=plan.crs,
                             lidar_source=src, plan=plan, used=used)


def write_read(read: ElevationRead, path, source_path=None, nodata=-9999.0):
    """Write a read as a GeoTIFF (float32, nodata -9999, deflate) and its lidar_source (int16) beside it."""
    import rasterio
    from affine import Affine
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    z = np.where(np.isfinite(read.z), read.z, nodata).astype(np.float32)
    T = Affine(*read.transform)
    common = dict(driver="GTiff", height=z.shape[0], width=z.shape[1], count=1, crs=read.crs, transform=T,
                  compress="deflate", tiled=True)
    with rasterio.open(path, "w", dtype="float32", nodata=nodata, predictor=3, **common) as d:
        d.write(z, 1)
        d.update_tags(source_tiles=",".join(i.id for s in read.used for i in s.items),
                      surveys=json.dumps({s.index: s.slug for s in read.plan.surveys}))
    if source_path:
        with rasterio.open(source_path, "w", dtype="int16", nodata=-1, **common) as d:
            d.write(read.lidar_source, 1)
            d.update_tags(surveys=json.dumps({s.index: s.slug for s in read.plan.surveys}))
    return dict(path=str(path), source_path=str(source_path) if source_path else None, shape=list(z.shape),
                crs=read.crs, valid_fraction=round(read.valid_fraction, 6))


def _check(content: bytes, checksum: str | None):
    """True / False against a sha2-256 multihash (``1220<hex>``); None if there is nothing to check."""
    if not checksum or not str(checksum).startswith("1220") or len(checksum) != 68:
        return None
    return hashlib.sha256(content).hexdigest() == checksum[4:].lower()


def _lonlat_bounds(bounds, epsg: int):
    from pyproj import Transformer
    return tuple(Transformer.from_crs(epsg, 4326, always_xy=True).transform_bounds(*bounds, densify_pts=21))


def _neg(s: str) -> str:
    """Sort key for descending ISO strings."""
    return "".join(chr(0x10FFFF - ord(ch)) for ch in (s or ""))


def _read_item(href: str, bounds, nodata=None):
    """The part of one 1 m COG inside ``bounds``: (array, (row, col) offset in the target), or (None, None).
    Tiles not on the whole-metre grid are resampled bilinearly (with a warning)."""
    import rasterio
    from rasterio.windows import from_bounds
    path = (("/vsicurl/" + href) if str(href).startswith(("http://", "https://"))
            else str(href).replace("file://", ""))
    x0, y0, x1, y1 = bounds
    with rasterio.open(path) as s:
        b = s.bounds
        ix0, iy0, ix1, iy1 = max(x0, b.left), max(y0, b.bottom), min(x1, b.right), min(y1, b.top)
        if ix1 <= ix0 or iy1 <= iy0:
            return None, None
        nd = s.nodata if s.nodata is not None else nodata
        aligned = (abs(s.res[0] - 1) < 1e-9 and abs(s.res[1] - 1) < 1e-9
                   and abs((s.transform.c - x0) - round(s.transform.c - x0)) < 1e-6
                   and abs((s.transform.f - y1) - round(s.transform.f - y1)) < 1e-6)
        if aligned:
            ix0, iy0, ix1, iy1 = (np.floor(ix0), np.floor(iy0), np.ceil(ix1), np.ceil(iy1))
            win = from_bounds(ix0, iy0, ix1, iy1, transform=s.transform).round_offsets().round_lengths()
            data = s.read(1, window=win, masked=True).astype(np.float32).filled(np.nan)
            wt = s.window_transform(win)
            c0, r0 = int(round(wt.c - x0)), int(round(y1 - wt.f))
        else:
            from affine import Affine
            from rasterio.warp import Resampling, reproject
            warnings.warn(f"{href}: not on the whole-metre 1 m grid; resampled (bilinear)")
            c0, r0 = int(np.floor(ix0 - x0)), int(np.floor(y1 - iy1))
            w, h = int(np.ceil(ix1 - x0)) - c0, int(np.ceil(y1 - iy0)) - r0
            data = np.full((h, w), np.nan, np.float32)
            reproject(rasterio.band(s, 1), data, dst_transform=Affine(1, 0, x0 + c0, 0, -1, y1 - r0),
                      dst_crs=s.crs, dst_nodata=np.nan, src_nodata=nd, resampling=Resampling.bilinear)
    if nd is not None and np.isfinite(nd):
        data[data == nd] = np.nan
    H, W = int(round(y1 - y0)), int(round(x1 - x0))
    data = data[:max(0, H - r0), :max(0, W - c0)]
    if r0 < 0 or c0 < 0:
        data = data[max(0, -r0):, max(0, -c0):]
        r0, c0 = max(r0, 0), max(c0, 0)
    return data, (r0, c0)
