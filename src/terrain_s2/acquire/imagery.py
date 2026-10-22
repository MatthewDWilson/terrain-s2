"""LINZ aerial imagery (nz-imagery: a static STAC of RGB / RGBNIR COGs, organised by region and survey).

First use: ``scripts/review_imagery.py``, the cheap check of imagery against the channel reviews (whether
imagery separates the model's real drains from its false detections before it is tried as U-Net input).

- Collections are ``<region>/<survey>/<product>/<epsg>/collection.json`` with ``product`` in ``products``
  (default ``rgb`` and ``rgbnir``); ``regions`` narrows the catalogue to the LINZ regions given (the regions of
  a tile's elevation surveys: the two buckets share region slugs), which keeps the number of collection
  documents fetched small.
- Satellite and scanned (historical) imagery are left out (``linz:geospatial_category`` or title), and so is
  any survey coarser than ``max_gsd`` (read from the slug, e.g. ``..._0.3m``).
- Items are chosen by their map-sheet id (``grid.tile_bounds``) without fetching every item document: only
  the items meeting the area are fetched (ids that do not parse are fetched to read their bbox).
- Order: ``prefer="newest"`` (latest capture first) or ``"date"`` (capture midpoint closest to ``target``,
  e.g. the LiDAR survey's). ``read`` fills the window survey by survey in that order.
- ``read`` reads each item onto a regular grid at ``res`` m (average resampling: COG overviews serve coarse
  reads), so only the COG blocks the window needs are fetched over HTTP; the items must share the grid's CRS
  (EPSG:2193); bands R, G, B (and NIR where a survey has it) as float32 0-255, NaN
  where no survey covers a cell; ``survey`` gives the index of the survey each cell came from.

Licences differ by survey (``licence`` and ``providers`` on each survey record, from its collection).
"""
from __future__ import annotations

import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import numpy as np

from . import linz_elevation as le

ROOT = "https://nz-imagery.s3-ap-southeast-2.amazonaws.com/"
PRODUCTS = ("rgb", "rgbnir")
EXCLUDE_CATEGORIES = ("satellite", "scanned", "historic")
GSD_RE = re.compile(r"_(\d+(?:\.\d+)?)m(?:_|$)")


@dataclass
class ImagerySurvey:
    index: int                      # 1 = first in the order
    slug: str
    region: str
    product: str                    # rgb / rgbnir
    collection: str
    title: str | None
    capture: tuple                  # (start, end) ISO strings
    gsd: float | None
    category: str | None
    licence: str | None
    providers: list = field(default_factory=list)
    items: list = field(default_factory=list)       # [{id, href, checksum, bounds}]

    def record(self) -> dict:
        return dict(index=self.index, survey=self.slug, region=self.region, product=self.product,
                    collection=self.collection, title=self.title, capture=list(self.capture), gsd=self.gsd,
                    category=self.category, licence=self.licence, providers=self.providers,
                    items=[i["id"] for i in self.items])


@dataclass
class ImageryRead:
    data: np.ndarray                # (bands, H, W) float32, NaN where not covered
    bands: tuple                    # ("R", "G", "B") or ("R", "G", "B", "NIR")
    survey: np.ndarray              # (H, W) int16: index of the survey used, -1 none
    transform: object               # affine.Affine
    surveys: list                   # the ImagerySurvey records used

    @property
    def valid(self):
        return np.isfinite(self.data[0])


def gsd_of(slug: str):
    m = GSD_RE.search(slug or "")
    return float(m.group(1)) if m else None


def _date(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


def midpoint(capture):
    """The midpoint of a (start, end) capture interval, or None."""
    a, b = (_date(capture[0]), _date(capture[1])) if capture else (None, None)
    a, b = a or b, b or a
    return a + (b - a) / 2 if a else None


def _intersects(a, b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


class ImagerySource:
    """See the module docstring. ``cache_dir``: STAC documents (default ``TERRAIN_DATA_DIR/stac``, shared with
    the elevation reader); ``ttl_s``: how long catalogue and collection documents are trusted."""

    def __init__(self, http=None, cache_dir=None, root: str = ROOT, products=PRODUCTS, epsg: int = 2193,
                 max_gsd: float | None = 0.5, exclude_categories=EXCLUDE_CATEGORIES, ttl_s: float = 7 * 86400.0,
                 workers: int = 16):
        if http is None:
            from .http import Http
            http = Http()
        self.http = http
        if cache_dir is None:
            from .elevation import default_data_dir
            cache_dir = default_data_dir() / "stac"
        self.cache_dir = Path(cache_dir)
        self.root = root if root.endswith("/") else root + "/"
        self.products, self.epsg, self.max_gsd = tuple(products), int(epsg), max_gsd
        self.exclude = tuple(exclude_categories or ())
        self.ttl_s, self.workers = ttl_s, workers

    def _json(self, url: str, ttl: float | None = None):
        from . import cachefile
        f = self.cache_dir / "json" / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")
        ttl = self.ttl_s if ttl is None else ttl
        if f.exists() and time.time() - f.stat().st_mtime < ttl:
            d = cachefile.read(f)
            if d is not None:
                return d
        d = self.http.json(url)
        cachefile.write(f, d)
        return d

    def collections(self, regions=None):
        """[(url, region, slug, product)] in the catalogue, for the products, in the profile's CRS."""
        cat = self._json(urljoin(self.root, "catalog.json"))
        regions = set(regions) if regions else None
        out = []
        for link in cat.get("links", []):
            if link.get("rel") != "child":
                continue
            h = link["href"][2:] if link["href"].startswith("./") else link["href"]
            parts = h.split("/")
            if (len(parts) >= 5 and parts[2] in self.products and parts[3] == str(self.epsg)
                    and (regions is None or parts[0] in regions)):
                out.append((urljoin(self.root, h), parts[0], parts[1], parts[2]))
        return out

    def _excluded(self, coll: dict, slug: str):
        text = f"{coll.get('linz:geospatial_category', '')} {coll.get('title', '')}".lower()
        if any(e in text for e in self.exclude):
            return True
        g = gsd_of(coll.get("linz:slug") or slug)
        return self.max_gsd is not None and g is not None and g > self.max_gsd

    def _items(self, coll_url: str, coll: dict, bounds):
        from .. import grid
        keep, unknown = [], []
        for l in coll.get("links", []):
            if l.get("rel") != "item":
                continue
            stem = Path(l["href"].split("?")[0]).stem
            try:
                b = grid.tile_bounds(stem)
            except (ValueError, KeyError, TypeError, IndexError):
                unknown.append(l)
                continue
            if _intersects(b, bounds):
                keep.append((l, b))
        ll = le.to_lonlat(bounds)
        unknown = [(l, None) for l in unknown]

        def one(lb):
            l, b = lb
            url = urljoin(coll_url, l["href"])
            try:
                it = self._json(url, ttl=365 * 86400.0)
            except (IOError, ValueError):                   # an item that cannot be read leaves a hole
                return None
            if b is None:
                bb = it.get("bbox")
                if not bb or not le._intersects(bb, ll):
                    return None
                from rasterio.warp import transform_bounds
                b = transform_bounds("EPSG:4326", f"EPSG:{self.epsg}", *bb[:4], densify_pts=21)
            asset = next((a for a in it.get("assets", {}).values()
                          if "geotiff" in str(a.get("type", "")) or str(a.get("href", "")).lower().endswith((".tif", ".tiff"))),
                         None)
            if asset is None:
                return None
            return dict(id=it.get("id") or Path(url).stem, href=urljoin(url, asset["href"]),
                        checksum=asset.get("file:checksum"), bounds=tuple(float(v) for v in b))
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return [r for r in ex.map(one, keep + unknown) if r is not None]

    def surveys(self, bounds, regions=None, prefer: str = "newest", target=None):
        """Surveys covering ``bounds`` (EPSG:2193), in order: ``prefer="newest"`` or ``"date"`` (capture
        midpoint closest to ``target``, a datetime or ISO string; newest when ``target`` is None)."""
        bounds = tuple(float(v) for v in bounds)
        ll = le.to_lonlat(bounds)
        cols = self.collections(regions)
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            metas = list(ex.map(lambda c: self._json(c[0]), cols))
        cands = []
        for (url, region, slug, product), c in zip(cols, metas):
            if self._excluded(c, slug):
                continue
            bbox = (c.get("extent", {}).get("spatial", {}).get("bbox") or [[]])[0]
            if len(bbox) >= 4 and not le._intersects(bbox, ll):
                continue
            interval = ((c.get("extent", {}).get("temporal", {}).get("interval")) or [[None, None]])[0]
            cands.append((url, region, slug, product, c, (interval[0] or "", interval[1] or "")))
        tgt = _date(target) if isinstance(target, str) else target

        def order(c):
            m = midpoint(c[5])
            if m is None:
                return (1, 0.0, c[2])
            if prefer == "date" and tgt is not None:
                return (0, abs((m - tgt).total_seconds()), c[2])
            end = _date(c[5][1]) or m
            return (0, -(end - datetime(1900, 1, 1)).total_seconds(), c[2])          # newest capture first
        cands.sort(key=order)
        out = []
        for url, region, slug, product, c, capture in cands:
            items = self._items(url, c, bounds)
            if not items:
                continue
            out.append(ImagerySurvey(index=len(out) + 1, slug=c.get("linz:slug") or slug, region=region,
                                     product=product, collection=url, title=c.get("title"), capture=capture,
                                     gsd=gsd_of(c.get("linz:slug") or slug),
                                     category=c.get("linz:geospatial_category"), licence=c.get("license"),
                                     providers=[p.get("name") for p in c.get("providers", []) if p.get("name")],
                                     items=items))
        return out

    def read(self, surveys, bounds, res: float = 0.5, _open=None) -> ImageryRead:
        """Mosaic of ``surveys`` (in their order) on a grid of ``res`` m over ``bounds`` (snapped outwards to
        whole cells). ``_open``: a dict kept between calls, holding open datasets (pass the same dict for many
        small reads of one tile)."""
        import rasterio
        from affine import Affine
        from rasterio.enums import ColorInterp, Resampling
        from rasterio.windows import from_bounds
        x0, y0, x1, y1 = bounds
        x0, y0 = np.floor(x0 / res) * res, np.floor(y0 / res) * res
        x1, y1 = np.ceil(x1 / res) * res, np.ceil(y1 / res) * res
        W, H = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
        T = Affine(res, 0.0, float(x0), 0.0, -res, float(y1))
        nir = any(s.product == "rgbnir" for s in surveys)
        names = ("R", "G", "B", "NIR") if nir else ("R", "G", "B")
        data = np.full((len(names), H, W), np.nan, np.float32)
        src_ix = np.full((H, W), -1, np.int16)
        used = []
        opened = _open if _open is not None else {}
        box = (float(x0), float(y0), float(x1), float(y1))
        with rasterio.Env(**le.GDAL_ENV):
            for s in surveys:
                if not np.isnan(data[0]).any():
                    break
                put_any = False
                for it in s.items:
                    if not _intersects(it["bounds"], box):
                        continue
                    href = str(it["href"])
                    path = ("/vsicurl/" + href) if href.startswith(("http://", "https://")) else href.replace("file://", "")
                    ds = opened.get(path)
                    if ds is None:
                        ds = opened[path] = rasterio.open(path)
                    ci = list(ds.colorinterp)
                    colour = [i + 1 for i, c in enumerate(ci) if c != ColorInterp.alpha]
                    want = colour[:len(names)] if s.product == "rgbnir" else colour[:3]
                    if len(want) < 3:
                        continue                                 # a single-band (panchromatic) source
                    # the output cells lying wholly inside the item (sheet edges fall on whole metres, so
                    # neighbouring items cover the rest); read at the output resolution, overviews allowed
                    b = ds.bounds
                    c0 = max(0, int(np.ceil((b.left - x0) / res - 1e-6)))
                    c1 = min(W, int(np.floor((b.right - x0) / res + 1e-6)))
                    r0 = max(0, int(np.ceil((y1 - b.top) / res - 1e-6)))
                    r1 = min(H, int(np.floor((y1 - b.bottom) / res + 1e-6)))
                    if c1 <= c0 or r1 <= r0:
                        continue
                    win = from_bounds(x0 + c0 * res, y1 - r1 * res, x0 + c1 * res, y1 - r0 * res, ds.transform)
                    shape = (r1 - r0, c1 - c0)
                    vals = ds.read(want, window=win, out_shape=(len(want),) + shape,
                                   resampling=Resampling.average).astype(np.float32)
                    mask = ds.read_masks(want[0], window=win, out_shape=shape, resampling=Resampling.nearest) > 0
                    sub = data[:, r0:r1, c0:c1]
                    free = np.isnan(sub[0]) & mask
                    if not free.any():
                        continue
                    for k in range(len(want)):
                        sub[k][free] = vals[k][free]
                    src_ix[r0:r1, c0:c1][free] = s.index
                    put_any = True
                if put_any:
                    used.append(s)
        if _open is None:
            for ds in opened.values():
                ds.close()
        return ImageryRead(data=data, bands=names, survey=src_ix, transform=T, surveys=used)


def close(opened: dict):
    """Close the datasets an ``_open`` dict holds."""
    for ds in opened.values():
        try:
            ds.close()
        except Exception:  # noqa: BLE001
            pass
    opened.clear()
