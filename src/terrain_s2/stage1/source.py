"""Elevation source for the ``raster`` backend: the interface Stage 1 needs from the shared reader (S2-9).

S2-9 (owned by the Stage 2 agent) generalises ``acquire/linz_elevation.py`` into ``acquire/elevation.py``: one
reader for both stages, with overlapping surveys newest first, national-mosaic fill, DEM and DSM from the same
survey, a per-cell survey index and STAC checksums. Until it lands, :class:`InterimLinzSource` provides the
same behaviour for the DEM from the existing ``linz_elevation`` functions, so the raster backend can be built
and tested now. When ``terrain_s2.acquire.elevation`` defines ``ElevationSource`` with this interface,
:func:`get_source` uses it instead and this stopgap can be deleted.

Interface (what ``acquire.elevation.ElevationSource`` should provide)::

    src = ElevationSource(profile: dict, http=None, cache_dir=None, stac_root=None)
    plan = src.plan(bounds, product="dem")     # bounds in the profile's horizontal CRS; cheap (STAC only)
    plan.surveys                               # [Survey], in priority order; index 1 = highest
    plan.source_version()                      # JSON for the generator key: collections, updated, item ids
                                               # and file:checksum
    read = src.read(plan)                      # ElevationRead on the 1 m grid of plan.bounds
    read.z, read.transform, read.crs           # float32 NaN = no data; affine 6-tuple; "EPSG:2193+7839"
    read.lidar_source                          # int16, Survey.index per cell, -1 = none
    read.used                                  # surveys that contributed cells

Bounds passed to ``plan`` must lie on whole metres (Stage 1 snaps them).
"""
from __future__ import annotations

import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin

import numpy as np


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

    def record(self, with_items: bool = True) -> dict:
        d = {"index": self.index, "survey": self.slug, "collection": self.collection, "title": self.title,
             "capture": list(self.capture), "updated": self.updated, "national": self.national}
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


def get_source(profile: dict, http=None, cache_dir=None, stac_root=None):
    """The shared reader (S2-9) when present, else the interim one."""
    try:
        from ..acquire import elevation as shared
    except ImportError:
        shared = None
    if shared is not None and hasattr(shared, "ElevationSource"):
        return shared.ElevationSource(profile, http=http, cache_dir=cache_dir, stac_root=stac_root)
    return InterimLinzSource(profile, http=http, cache_dir=cache_dir, stac_root=stac_root)


class InterimLinzSource:
    """LINZ nz-elevation STAC: every ``<region>/<survey>/<product>_1m/<epsg>`` collection meeting the area,
    priority surveys first, then newest capture first, the national mosaic last (fill only)."""

    def __init__(self, profile: dict, http=None, cache_dir=None, stac_root=None, ttl_s: float = 86400.0,
                 workers: int = 16):
        from ..acquire import linz_elevation
        self.le = linz_elevation
        self.profile = profile
        ev = profile["elevation"]
        self.root = stac_root or ev["stac_root"]
        if not self.root.endswith("/"):
            self.root += "/"
        if http is None:
            from ..acquire.http import Http
            http = Http()
        self.http = http
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / "terrain_data" / "stac"
        self.ttl_s, self.workers = ttl_s, workers
        self.epsg = int(profile["crs"]["horizontal"])
        from .profile import crs_string
        self.crs = crs_string(profile)

    # --- STAC with a short-lived local cache (catalogue and collections change; item indexes are keyed by
    # the collection's ``updated`` in linz_elevation.tile_index) ---
    def _json(self, url: str):
        import hashlib
        f = self.cache_dir / "json" / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")
        if f.exists() and time.time() - f.stat().st_mtime < self.ttl_s:
            return json.loads(f.read_text())
        d = self.http.json(url)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(d))
        return d

    def collections(self, product: str = "dem"):
        """[(url, region, survey, national)] for the product at 1 m in the profile's CRS."""
        folder = self.profile["elevation"].get("products", {}).get(product, f"{product}_1m")
        national = self.profile["elevation"].get("national", {}).get(product)
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

    def plan(self, bounds, product: str = "dem") -> SourcePlan:
        from .aoi import lonlat_bounds
        ll = lonlat_bounds(bounds, f"EPSG:{self.epsg}")
        ev = self.profile["elevation"]
        exclude, priority = set(ev.get("exclude") or []), list(ev.get("priority") or [])
        cols = [c for c in self.collections(product) if c[2] not in exclude]
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            metas = list(ex.map(lambda c: self._json(c[0]), cols))
        cands = []
        for (url, region, slug, national), c in zip(cols, metas):
            bbox = c["extent"]["spatial"]["bbox"][0]
            if not self.le._intersects(bbox, ll):
                continue
            interval = c["extent"]["temporal"]["interval"][0]
            cands.append(dict(url=url, coll=c, slug=c.get("linz:slug") or slug, national=national,
                              start=interval[0] or "", end=interval[1] or "", updated=c.get("updated") or ""))

        def order(c):
            if c["national"]:
                return (2, 0, "", "", c["slug"])
            if c["slug"] in priority:
                return (0, priority.index(c["slug"]), "", "", c["slug"])
            # newest capture first, then most recently updated, then name
            return (1, 0, _neg(c["end"]), _neg(c["updated"]), c["slug"])
        cands.sort(key=order)
        surveys = []
        for c in cands:
            idx = self.le.tile_index(self.http, c["url"], c["coll"], self.cache_dir / "tiles")
            tiles = self.le.select_tiles(idx, ll)
            if not tiles:
                continue
            surveys.append(Survey(index=len(surveys) + 1, slug=c["slug"], collection=c["url"],
                                  title=c["coll"].get("title"), capture=(c["start"], c["end"]),
                                  updated=c["updated"] or None, national=c["national"],
                                  items=[Item(t["id"], t["href"], t.get("checksum"), tuple(t["bbox"])) for t in tiles]))
        if not surveys:
            raise NoElevationData(f"no {product} survey covers {tuple(round(b) for b in bounds)} "
                                  f"(EPSG:{self.epsg}); {len(cols)} collections checked")
        return SourcePlan(product=product, bounds=tuple(bounds), crs=self.crs, surveys=surveys)

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
        with rasterio.Env(**self.le.GDAL_ENV):
            for s in plan.surveys:
                if not np.isnan(z).any():
                    break
                put_any = False
                for it in s.items:
                    data, (r0, c0) = _read_item(it.href, plan.bounds, nodata)
                    if data is None:
                        continue
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


def _neg(s: str) -> str:
    """Sort key for descending ISO strings."""
    return "".join(chr(0x10FFFF - ord(ch)) for ch in (s or ""))


def _read_item(href: str, bounds, nodata=None):
    """The part of one 1 m COG inside ``bounds``: (array, (row, col) offset in the target), or (None, None).
    Tiles not on the whole-metre grid are resampled bilinearly (with a warning)."""
    import rasterio
    from rasterio.windows import from_bounds
    path = ("/vsicurl/" + href) if str(href).startswith(("http://", "https://")) else str(href).replace("file://", "")
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
