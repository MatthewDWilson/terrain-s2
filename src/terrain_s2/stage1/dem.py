"""Stage 1 entry point: an AOI and a parameter set in, one DEM product (section 4.3 contract) out.

    from terrain_s2.settings import from_env
    from terrain_s2.stage1.dem import ensure, make
    r = make("POLYGON((...))", from_env(TERRAIN_RESOLUTION="8"))     # WKT in EPSG:4326 by default
    r.paths["netcdf"], r.key, r.reused
    row = ensure("POLYGON((...))", settings)                          # make + the product registry

Products live in ``<TERRAIN_PRODUCT_DIR>/<generator_key>/<grid_id>/``. The generator key (6.5) identifies how a
product was made and from which inputs; ``grid_id`` (a hash of the snapped grid bounds and CRS) separates AOIs
made the same way from the same inputs. An identical request finds the finished product (``product.json``) and
reuses it. :func:`ensure` adds the database registry and spatial reuse (:mod:`terrain_s2.store`).

Steps: :func:`prepare` resolves the AOI, the grid and the inputs (STAC or otCatalog queries only) and forms the
key; :func:`build` reads or runs GeoFabrics and writes the product.
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


@dataclass
class Prepared:
    settings: object
    profile: dict
    crs: str
    hcrs: str
    geom: object
    bounds: tuple
    backend: str
    key: str
    info: dict
    name: str
    out_dir: Path
    plan: object = None
    src: object = None
    datasets: list | None = None
    mapping: dict | None = None
    http: object = None
    store: object = None
    overrides: dict | None = None
    fallback_note: str | None = None


def grid_id(bounds, crs) -> str:
    from ..key import digest
    return digest({"grid": list(bounds), "crs": crs}, 12)


def prepare(aoi, settings=None, *, aoi_crs=None, http=None, store=None, source=None, overrides: dict | None = None,
            log=print) -> Prepared:
    """Resolve the AOI, the snapped grid and the inputs, and form the generator key. No raster is read.

    ``source``: an elevation source for the raster backend (default: the shared reader, :mod:`.source`).
    ``overrides``: merged into the GeoFabrics instructions (part of the key). ``store``: the snapshot store.
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

    def _http():
        nonlocal http
        if http is None:
            from ..acquire.http import Http
            http = Http(store=store)
        return http

    backend, plan, src, datasets, mapping, note = s.backend, None, None, None, None, None
    for attempt in range(2):
        if backend == "raster":
            from .source import NoElevationData, get_source
            src = source or get_source(profile, http=_http(), cache_dir=data_dir / "stac", stac_root=s.stac_root)
            try:
                plan = src.plan(bounds, "dem")
                break
            except NoElevationData as e:
                if not s.allow_fallback or attempt or s.product == "geofabric":
                    raise
                note, backend = f"raster: {e}; fell back to geofabrics", "geofabrics"
        else:
            from .catalogue import NoPointClouds, discover, mapping as mk_mapping
            try:
                datasets, _ = discover(_http(), bounds, hcrs, profile)
                mapping = mk_mapping(datasets)
                break
            except NoPointClouds as e:
                if not s.allow_fallback or attempt or s.product == "geofabric":
                    raise
                note, backend = f"geofabrics: {e}; fell back to raster", "raster"
    if note:
        log(note)
        s = s.replace(backend=backend)

    if backend == "raster":
        from . import raster
        source_version, configuration = plan.source_version(), raster.configuration(s)
    else:
        from . import gf
        source_version = {"datasets": [d.record() for d in datasets], "mapping": mapping}
        configuration = gf.configuration(s, overrides)
    from ..key import digest, file_sha256
    configuration["profile"] = digest(profile, 64)           # CRS, sources, overrides, exclusions
    if s.land_source == "file":
        configuration["land_file_sha256"] = file_sha256(s.land_file)
    key, info = generator_key(stage1_components(
        product=s.product, profile=profile["name"], backend=backend, resolution=res, land_source=s.land_source,
        buffer_m=s.buffer_m, source_version=source_version, configuration=configuration))
    return Prepared(settings=s, profile=profile, crs=crs, hcrs=hcrs, geom=geom, bounds=bounds, backend=backend,
                    key=key, info=info, name=f"{s.product}_{res}m", out_dir=s.products / key / grid_id(bounds, crs),
                    plan=plan, src=src, datasets=datasets, mapping=mapping, http=http, store=store,
                    overrides=overrides, fallback_note=note)


def build(prep: Prepared, *, log=print, gf_runner=None, lock_timeout_s: float = 6 * 3600) -> Stage1Result:
    """Write the product (or reuse the finished one in its folder). ``gf_runner`` replaces the GeoFabrics call
    (tests)."""
    s, out_dir = prep.settings, prep.out_dir
    manifest = out_dir / MANIFEST
    if manifest.exists():
        log(f"reusing {out_dir}")
        return Stage1Result.from_manifest(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    with _Lock(out_dir / LOCK, lock_timeout_s, log):
        if manifest.exists():          # built by another worker while we waited
            return Stage1Result.from_manifest(manifest)
        t0 = time.perf_counter()
        kw = dict(settings=s, profile=prep.profile, aoi_geom=prep.geom, bounds=prep.bounds, out_dir=out_dir,
                  name=prep.name, key=prep.key, info=prep.info, log=log)
        if prep.backend == "raster":
            from . import raster
            r = raster.build(prep.plan, prep.src, http=prep.http, store=prep.store, **kw)
        else:
            from . import gf
            land_file, land_rec = _gf_land(prep, log)
            r = gf.build(datasets=prep.datasets, mapping=prep.mapping, land_file=land_file, land_rec=land_rec,
                         cache_dir=Path(s.data_dir) / "geofabrics", overrides=prep.overrides, runner=gf_runner, **kw)
        result = Stage1Result(key=prep.key, info=prep.info, product=s.product, backend=prep.backend,
                              resolution=int(s.resolution), crs=prep.crs, product_dir=str(out_dir), paths=r["paths"],
                              aoi_wkt=prep.geom.wkt, grid_bounds=list(prep.bounds),
                              extent_wkt=r["extent"].wkt if r["extent"] is not None else None,
                              gap_fraction=r["gap_fraction"],
                              created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                              extra={"seconds": round(time.perf_counter() - t0, 2),
                                     **({"fallback": prep.fallback_note} if prep.fallback_note else {})})
        tmp = manifest.with_name(MANIFEST + ".part")
        tmp.write_text(json.dumps(result.to_dict(), indent=1, default=str))
        tmp.replace(manifest)
    return result


def make(aoi, settings=None, *, aoi_crs=None, http=None, store=None, source=None, overrides: dict | None = None,
         log=print, gf_runner=None, lock_timeout_s: float = 6 * 3600) -> Stage1Result:
    """Make (or reuse) one Stage 1 product: :func:`prepare` then :func:`build`."""
    prep = prepare(aoi, settings, aoi_crs=aoi_crs, http=http, store=store, source=source, overrides=overrides, log=log)
    return build(prep, log=log, gf_runner=gf_runner, lock_timeout_s=lock_timeout_s)


def ensure(aoi, settings=None, *, registry=None, aoi_crs=None, http=None, store=None, source=None,
           overrides: dict | None = None, log=print, gf_runner=None):
    """Make (or find) a product and register it: the ``terrain.dem`` process. Returns the registry row
    (:class:`terrain_s2.store.Product`). With ``spatial_reuse``, a stored product with the same configuration
    whose grid contains this one, and whose inputs here are unchanged, is clipped instead of building."""
    from ..settings import from_env
    from ..store import Store
    s = settings or from_env()
    prep = prepare(aoi, s, aoi_crs=aoi_crs, http=http, store=store, source=source, overrides=overrides, log=log)
    reg = registry or Store(settings=prep.settings, srid=int(prep.profile["crs"]["horizontal"]))
    hit = reg.lookup(prep.key, prep.info, prep.geom, prep.bounds, spatial_reuse=prep.settings.spatial_reuse)
    if hit is not None:
        log(f"registry hit: product {hit.id} ({hit.generator_key})")
        return hit
    return reg.register(build(prep, log=log, gf_runner=gf_runner))


def _gf_land(prep: Prepared, log):
    from . import land
    from .catalogue import fetch_tile_indexes
    s = prep.settings
    out = prep.out_dir / f"{prep.name}_land.geojson"
    coverage = None
    if s.land_source == "coverage":
        try:
            idx = fetch_tile_indexes(prep.http, prep.datasets, Path(s.data_dir) / "geofabrics" / "downloads")
            coverage = land.tile_index_coverage(idx, prep.hcrs)
        except Exception as e:  # noqa: BLE001 - no tile index: GeoFabrics treats the AOI as land
            log(f"coverage land: tile indexes unavailable ({e}); the AOI is used as land")
            return None, {"land_source": "coverage", "note": f"tile indexes unavailable: {e}"}
    from shapely.geometry import box
    return land.build(s.land_source, box(*prep.bounds), prep.hcrs, out, profile=prep.profile, coverage=coverage,
                      http=prep.http, store=prep.store, linz_key=s.linz_api_key, land_file=s.land_file)


class _Lock:
    """One builder per product folder: an OS lock on ``.lock`` (``fcntl.flock`` on POSIX, ``msvcrt.locking`` on
    Windows). The kernel releases it when the holder exits or is killed, so a crashed build never blocks later
    requests; waiters poll, and return early if the product appears. The lock file is left in place (deleting a
    locked file races with waiters). Works across containers sharing a local volume; not on NFS without locking.
    """

    def __init__(self, path, timeout_s, log, poll_s: float = 2.0):
        self.path, self.timeout_s, self.log, self.poll_s = Path(path), timeout_s, log, poll_s
        self.waited = self.acquired = False
        self._f = None

    def _try(self) -> bool:
        try:
            import fcntl
            fcntl.flock(self._f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except ImportError:
            import msvcrt
            try:
                self._f.seek(0)
                msvcrt.locking(self._f.fileno(), msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                return False
        except OSError:
            return False

    def __enter__(self):
        t0 = time.time()
        self._f = open(self.path, "a+")
        while True:
            if self._try():
                self.acquired = True
                self._f.seek(0)
                self._f.truncate()
                self._f.write(f"{os.getpid()} {_dt.datetime.now(_dt.timezone.utc).isoformat()}")
                self._f.flush()
                return self
            if not self.waited:
                self.log(f"waiting for another build of {self.path.parent}")
            self.waited = True
            if (self.path.parent / MANIFEST).exists():
                return self
            if time.time() - t0 > self.timeout_s:
                self._f.close()
                raise TimeoutError(f"{self.path} held for over {self.timeout_s} s")
            time.sleep(self.poll_s)

    def __exit__(self, *exc):
        try:
            if self.acquired:
                try:
                    import fcntl
                    fcntl.flock(self._f.fileno(), fcntl.LOCK_UN)
                except ImportError:
                    import msvcrt
                    self._f.seek(0)
                    msvcrt.locking(self._f.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self._f.close()
        return False
