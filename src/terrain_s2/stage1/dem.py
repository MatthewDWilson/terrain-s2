"""Stage 1 entry point: an AOI and a parameter set in, one DEM product (section 4.3 contract) out.

    from terrain_s2.settings import from_env
    from terrain_s2.stage1.dem import make
    r = make("POLYGON((...))", from_env(TERRAIN_RESOLUTION="8"))     # WKT in EPSG:4326 by default
    r.paths["netcdf"], r.key, r.reused

Products live in ``<TERRAIN_PRODUCT_DIR>/<generator_key>/<grid_id>/``. The generator key (6.5) identifies how a
product was made; ``grid_id`` (a hash of the snapped grid bounds and CRS) separates AOIs made the same way
from the same inputs. An identical request finds the finished product (``product.json``) and reuses it. The
database registry and spatial reuse are in :mod:`terrain_s2.store`.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST = "product.json"
LOCK = ".lock"


@dataclass
class Stage1Result:
    key: str
    info: dict
    product: str
    backend: str
    resolution: int
    crs: str
    product_dir: str
    paths: dict
    aoi_wkt: str
    grid_bounds: list
    extent_wkt: str | None
    gap_fraction: float
    reused: bool = False
    created: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_manifest(cls, path) -> "Stage1Result":
        d = json.loads(Path(path).read_text())
        d["reused"] = True
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def grid_id(bounds, crs) -> str:
    from ..key import digest
    return digest({"grid": list(bounds), "crs": crs}, 12)


def make(aoi, settings=None, *, aoi_crs=None, http=None, store=None, source=None, overrides: dict | None = None,
         log=print, gf_runner=None, lock_timeout_s: float = 6 * 3600) -> Stage1Result:
    """Make (or reuse) one Stage 1 product. See the module docstring.

    ``source``: an elevation source for the raster backend (default: the shared reader, see
    :mod:`.source`). ``overrides``: a dict merged into the GeoFabrics instructions (part of the key).
    ``gf_runner``: replaces the GeoFabrics call (tests).
    """
    from ..key import generator_key, stage1_components
    from ..settings import from_env, validate
    from . import aoi as A
    from .profile import crs_string, load

    s = settings or from_env()
    validate(s)
    profile = load(s.profile)
    crs = crs_string(profile)
    hcrs = f"EPSG:{profile['crs']['horizontal']}"
    geom = A.to_working(aoi, hcrs, aoi_crs)
    res = int(s.resolution)
    bounds = A.grid_bounds(geom, s.buffer_m, res)
    data_dir = Path(s.data_dir)
    if store is None:
        from ..acquire.snapshot import SnapshotStore
        store = SnapshotStore(data_dir / "snapshots")
    if http is None and (source is None or s.backend == "geofabrics"):
        from ..acquire.http import Http
        http = Http(store=store)

    backend, plan, src, datasets, mapping = s.backend, None, None, None, None
    fallback_note = None
    for attempt in range(2):
        if backend == "raster":
            from .source import NoElevationData, get_source
            src = source or get_source(profile, http=http, cache_dir=data_dir / "stac", stac_root=s.stac_root)
            try:
                plan = src.plan(bounds, "dem")
                break
            except NoElevationData as e:
                if not s.allow_fallback or attempt or s.product == "geofabric":
                    raise
                fallback_note, backend = f"raster: {e}; fell back to geofabrics", "geofabrics"
                if http is None:
                    from ..acquire.http import Http
                    http = Http(store=store)
        else:
            from .catalogue import NoPointClouds, discover, mapping as mk_mapping
            try:
                datasets, _ = discover(http, bounds, hcrs, profile)
                mapping = mk_mapping(datasets)
                break
            except NoPointClouds as e:
                if not s.allow_fallback or attempt or s.product == "geofabric":
                    raise
                fallback_note, backend = f"geofabrics: {e}; fell back to raster", "raster"
    if fallback_note:
        log(fallback_note)
        s = s.replace(backend=backend)

    if backend == "raster":
        from . import raster
        source_version, configuration = plan.source_version(), raster.configuration(s)
    else:
        from . import gf
        source_version = {"datasets": [d.record() for d in datasets], "mapping": mapping}
        configuration = gf.configuration(s, overrides)
    key, info = generator_key(stage1_components(
        product=s.product, profile=profile["name"], backend=backend, resolution=res, land_source=s.land_source,
        buffer_m=s.buffer_m, source_version=source_version, configuration=configuration))
    name = f"{s.product}_{res}m"
    gid = grid_id(bounds, crs)
    out_dir = s.products / key / gid
    manifest = out_dir / MANIFEST
    if manifest.exists():
        log(f"reusing {out_dir}")
        return Stage1Result.from_manifest(manifest)

    out_dir.mkdir(parents=True, exist_ok=True)
    with _Lock(out_dir / LOCK, lock_timeout_s, log):
        if manifest.exists():          # built by another worker while we waited
            return Stage1Result.from_manifest(manifest)
        t0 = time.perf_counter()
        if backend == "raster":
            r = raster.build(plan, src, settings=s, profile=profile, aoi_geom=geom, bounds=bounds, out_dir=out_dir,
                             name=name, key=key, info=info, http=http, store=store, log=log)
        else:
            land_file, land_rec = _gf_land(s, profile, geom, hcrs, out_dir, name, http, store, datasets, data_dir, log)
            r = gf.build(settings=s, profile=profile, aoi_geom=geom, bounds=bounds, out_dir=out_dir, name=name,
                         key=key, info=info, datasets=datasets, mapping=mapping, land_file=land_file,
                         land_rec=land_rec, cache_dir=data_dir / "geofabrics", overrides=overrides, log=log,
                         runner=gf_runner)
        result = Stage1Result(key=key, info=info, product=s.product, backend=backend, resolution=res, crs=crs,
                              product_dir=str(out_dir), paths=r["paths"], aoi_wkt=geom.wkt, grid_bounds=list(bounds),
                              extent_wkt=r["extent"].wkt if r["extent"] is not None else None,
                              gap_fraction=r["gap_fraction"],
                              created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                              extra={"seconds": round(time.perf_counter() - t0, 2),
                                     **({"fallback": fallback_note} if fallback_note else {})})
        tmp = manifest.with_name(MANIFEST + ".part")
        tmp.write_text(json.dumps(result.to_dict(), indent=1, default=str))
        tmp.replace(manifest)
    return result


def _gf_land(s, profile, geom, hcrs, out_dir, name, http, store, datasets, data_dir, log):
    from . import land
    from .catalogue import fetch_tile_indexes
    out = out_dir / f"{name}_land.geojson"
    coverage = None
    if s.land_source == "coverage":
        try:
            idx = fetch_tile_indexes(http, datasets, data_dir / "geofabrics" / "downloads")
            coverage = land.tile_index_coverage(idx, hcrs)
        except Exception as e:  # noqa: BLE001 - no tile index: GeoFabrics treats the AOI as land
            log(f"coverage land: tile indexes unavailable ({e}); the AOI is used as land")
            return None, {"land_source": "coverage", "note": f"tile indexes unavailable: {e}"}
    return land.build(s.land_source, geom, hcrs, out, profile=profile, coverage=coverage, http=http, store=store,
                      linz_key=s.linz_api_key, land_file=s.land_file)


class _Lock:
    """One builder per product folder: an exclusive lock file. Others wait for it to go (then reuse the
    product); a lock older than ``timeout_s`` is taken over."""

    def __init__(self, path, timeout_s, log):
        self.path, self.timeout_s, self.log, self.waited, self.acquired = Path(path), timeout_s, log, False, False

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {_dt.datetime.now(_dt.timezone.utc).isoformat()}".encode())
                os.close(fd)
                self.acquired = True
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > self.timeout_s:
                    self.log(f"taking over stale lock {self.path} ({age:.0f} s old)")
                    self.path.unlink(missing_ok=True)
                    continue
                if not self.waited:
                    self.log(f"waiting for another build of {self.path.parent}")
                self.waited = True
                if time.time() - t0 > self.timeout_s:
                    raise TimeoutError(f"{self.path} held for over {self.timeout_s} s")
                time.sleep(2.0)
                if (self.path.parent / MANIFEST).exists():
                    return self

    def __exit__(self, *exc):
        if self.acquired:
            self.path.unlink(missing_ok=True)
        return False
