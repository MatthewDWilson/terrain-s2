"""``TERRAIN_*`` settings (integration plan 6.1), read from the environment.

The same names configure the library, the CLI, the ``eddie_terrain`` module and the compatibility shim used
by FReDT and Smart Ideas. Only the standard library is imported here.

=============================  ===========================================  ==================================
Variable                       Meaning                                      Default
=============================  ===========================================  ==================================
``TERRAIN_BACKEND``            ``raster`` or ``geofabrics``                 ``raster`` (until W7 decides)
``TERRAIN_RESOLUTION``         integer metres: 1, 2, 4 or 8                 ``8``
``TERRAIN_PRODUCT``            ``dem`` or ``geofabric`` (adds ``zo``)       ``dem``
``TERRAIN_LAND_SOURCE``        ``coverage``, ``topo50``,                    ``coverage``
                               ``topo50_mangrove``, ``file``
``TERRAIN_LIDAR_CLASSES``      comma-separated LAS classes                  ``2``
``TERRAIN_BUFFER_M``           AOI buffer, metres (200 for Stage 2 input)   ``10``
``TERRAIN_SPATIAL_REUSE``      reuse a stored product containing the AOI    ``false`` (FReDT: ``true``)
``TERRAIN_PROFILE``            ``nz``                                       ``nz``
``TERRAIN_DATA_DIR``           cache root: COG and vector snapshots, LAZ    ``~/terrain_data``
``TERRAIN_PRODUCT_DIR``        products, ``<dir>/<generator_key>/``         ``$TERRAIN_DATA_DIR/products``
``TERRAIN_STAC_ROOT``          root of the elevation STAC catalogue         the profile's (LINZ nz-elevation)
``TERRAIN_DB_URL``             SQLAlchemy URL of the product registry       EDDIE's Postgres settings, else
                                                                            SQLite in ``TERRAIN_DATA_DIR``
``TERRAIN_GF_MEMORY_LIMIT``    GeoFabrics dask memory limit per worker      ``4GiB``
``TERRAIN_GF_CORES``           GeoFabrics dask workers                      ``2``
``TERRAIN_GF_DOWNLOAD_LIMIT``  GeoFabrics download limit, GB                ``100``
``TERRAIN_ALLOW_FALLBACK``     fall back to ``raster`` when no point        ``false``
                               clouds are found (and the reverse)
``LINZ_API_KEY``               LINZ Data Service key (Topo50 land sources)  none
``LAND_FILE``                  polygon file for ``land_source=file``        none
=============================  ===========================================  ==================================
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

BACKENDS = ("raster", "geofabrics")
PRODUCTS = ("dem", "geofabric")
LAND_SOURCES = ("coverage", "topo50", "topo50_mangrove", "file")
RESOLUTIONS = (1, 2, 4, 8)


def _bool(s: str | None, default: bool) -> bool:
    if s in (None, ""):
        return default
    v = s.strip().lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise ValueError(f"not a boolean: {s!r}")


def _env(name, default=None, env=None):
    v = (env if env is not None else os.environ).get(name)
    return default if v in (None, "") else v


@dataclass(frozen=True)
class Settings:
    backend: str = "raster"
    resolution: int = 8
    product: str = "dem"
    land_source: str = "coverage"
    lidar_classes: tuple = (2,)
    buffer_m: float = 10.0
    spatial_reuse: bool = False
    profile: str = "nz"
    data_dir: Path = field(default_factory=lambda: Path.home() / "terrain_data")
    product_dir: Path | None = None
    stac_root: str | None = None
    db_url: str | None = None
    gf_memory_limit: str = "4GiB"
    gf_cores: int = 2
    gf_download_limit_gb: float = 100.0
    allow_fallback: bool = False
    linz_api_key: str | None = None
    land_file: str | None = None

    @property
    def products(self) -> Path:
        return Path(self.product_dir) if self.product_dir else Path(self.data_dir) / "products"

    def replace(self, **kw) -> "Settings":
        known = {f.name for f in fields(self)}
        bad = set(kw) - known
        if bad:
            raise TypeError(f"unknown settings: {sorted(bad)}")
        return replace(self, **kw)

    def public(self) -> dict:
        """Settings without secrets (for logs and provenance)."""
        return {f.name: (str(getattr(self, f.name)) if isinstance(getattr(self, f.name), Path) else getattr(self, f.name))
                for f in fields(self) if f.name not in ("linz_api_key", "db_url")}


def from_env(env: dict | None = None, **overrides) -> Settings:
    """Settings from ``TERRAIN_*`` variables (``env`` defaults to ``os.environ``), then ``overrides``."""
    e = (lambda n, d=None: _env(n, d, env))
    classes = e("TERRAIN_LIDAR_CLASSES", "2")
    s = Settings(
        backend=e("TERRAIN_BACKEND", "raster"),
        resolution=int(e("TERRAIN_RESOLUTION", "8")),
        product=e("TERRAIN_PRODUCT", "dem"),
        land_source=e("TERRAIN_LAND_SOURCE", "coverage"),
        lidar_classes=tuple(int(c) for c in str(classes).replace(" ", "").split(",") if c),
        buffer_m=float(e("TERRAIN_BUFFER_M", "10")),
        spatial_reuse=_bool(e("TERRAIN_SPATIAL_REUSE"), False),
        profile=e("TERRAIN_PROFILE", "nz"),
        data_dir=Path(e("TERRAIN_DATA_DIR", str(Path.home() / "terrain_data"))).expanduser(),
        product_dir=(Path(e("TERRAIN_PRODUCT_DIR")).expanduser() if e("TERRAIN_PRODUCT_DIR") else None),
        stac_root=e("TERRAIN_STAC_ROOT"),
        db_url=e("TERRAIN_DB_URL"),
        gf_memory_limit=e("TERRAIN_GF_MEMORY_LIMIT", "4GiB"),
        gf_cores=int(e("TERRAIN_GF_CORES", "2")),
        gf_download_limit_gb=float(e("TERRAIN_GF_DOWNLOAD_LIMIT", "100")),
        allow_fallback=_bool(e("TERRAIN_ALLOW_FALLBACK"), False),
        linz_api_key=e("LINZ_API_KEY"),
        land_file=e("LAND_FILE"),
    )
    s = s.replace(**overrides) if overrides else s
    validate(s)
    return s


def validate(s: Settings) -> None:
    from .io.contract import check_resolution
    if s.backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {s.backend!r}")
    if s.product not in PRODUCTS:
        raise ValueError(f"product must be one of {PRODUCTS}, got {s.product!r}")
    if s.product == "geofabric" and s.backend != "geofabrics":
        raise ValueError("product 'geofabric' (with roughness zo) needs backend 'geofabrics'")
    if s.land_source not in LAND_SOURCES:
        raise ValueError(f"land_source must be one of {LAND_SOURCES}, got {s.land_source!r}")
    if s.land_source == "file" and not s.land_file:
        raise ValueError("land_source 'file' needs LAND_FILE")
    r = check_resolution(s.resolution)
    if r not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS} m, got {r}")
    if s.buffer_m < 0:
        raise ValueError("buffer_m must be >= 0")
