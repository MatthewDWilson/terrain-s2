"""Product registry (plan 6.6): table ``terrain_product``, lookup, reuse and clipping to an AOI.

Base install only (SQLAlchemy 2, shapely, the contract), so FReDT and Smart Ideas can use it through
:mod:`terrain_s2.client`. PostgreSQL/PostGIS in deployments; SQLite works for tests and single-user runs
(geometries are then WKT text, and spatial tests are done with shapely, as they are on PostGIS too: the
candidates are first narrowed by key in SQL).

Columns: ``id``, ``product``, ``generator_key`` (indexed), ``config_key`` (indexed; the key without the source
version, see :mod:`terrain_s2.key`), ``generator_info`` (JSON/JSONB), ``resolution``, ``aoi`` and ``extent``
(geometry, working CRS), ``grid`` (the product's snapped grid box), ``paths`` (JSON), ``parent_id`` (set when
clipped from a larger product), ``created_at``.

``config_key`` and ``grid`` are additions to the plan's table: they let a larger product serve a smaller AOI
when only the inputs under the larger one differ. Legacy ``user_dem`` and ``hydro_dem`` rows are never read.

Lookup (``Store.lookup``):

1. exact ``generator_key`` with ``aoi`` equal to the request within 0.1 m;
2. with ``spatial_reuse``: an earlier clip (child) with the same ``config_key`` and an equal AOI, or a product
   with the same ``config_key`` whose grid contains the request's grid and whose inputs under the request are
   unchanged; that product is clipped to the request's grid and the clip stored as a child row.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import TypeDecorator, UserDefinedType

TABLE = "terrain_product"
metadata = sa.MetaData()


class _PGGeometry(UserDefinedType):
    cache_ok = True

    def __init__(self, srid: int):
        self.srid = srid

    def get_col_spec(self, **kw):
        return f"geometry(Geometry,{self.srid})"


class Geometry(TypeDecorator):
    """PostGIS ``geometry(Geometry, srid)`` on PostgreSQL (written as EWKT, read as hex EWKB), WKT text elsewhere.
    Python values are shapely geometries."""
    impl = sa.Text
    cache_ok = True

    def __init__(self, srid: int = 2193):
        super().__init__()
        self.srid = srid

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(_PGGeometry(self.srid))
        return dialect.type_descriptor(sa.Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        wkt = value if isinstance(value, str) else value.wkt
        return f"SRID={self.srid};{wkt}" if dialect.name == "postgresql" else wkt

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        import shapely
        if isinstance(value, (bytes, memoryview)):
            return shapely.from_wkb(bytes(value))
        s = str(value)
        if s[:1] in "0123456789ABCDEFabcdef" and all(c in "0123456789ABCDEFabcdef" for c in s[:16]):
            return shapely.from_wkb(bytes.fromhex(s))
        if s.upper().startswith("SRID="):
            s = s.split(";", 1)[1]
        return shapely.from_wkt(s)


def table(srid: int = 2193, md: sa.MetaData | None = None) -> sa.Table:
    md = md if md is not None else metadata
    if TABLE in md.tables:
        return md.tables[TABLE]
    json_t = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    t = sa.Table(
        TABLE, md,
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("product", sa.Text, nullable=False),
        sa.Column("generator_key", sa.Text, nullable=False, index=True),
        sa.Column("config_key", sa.Text, nullable=False, index=True),
        sa.Column("generator_info", json_t, nullable=False),
        sa.Column("resolution", sa.Integer, nullable=False),
        sa.Column("aoi", Geometry(srid), nullable=False),
        sa.Column("extent", Geometry(srid)),
        sa.Column("grid", Geometry(srid)),
        sa.Column("paths", json_t, nullable=False),
        sa.Column("parent_id", sa.Integer, sa.ForeignKey(f"{TABLE}.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    sa.Index(f"ix_{TABLE}_aoi_gist", t.c.aoi, postgresql_using="gist").ddl_if(dialect="postgresql")
    return t


@dataclass
class Product:
    id: int
    product: str
    generator_key: str
    config_key: str
    generator_info: dict
    resolution: int
    aoi: object
    extent: object
    grid: object
    paths: dict
    parent_id: int | None
    created_at: object

    @property
    def netcdf(self) -> str:
        return self.paths["netcdf"]

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        for k in ("aoi", "extent", "grid"):
            d[k] = d[k].wkt if d[k] is not None else None
        d["created_at"] = str(d["created_at"])
        return d


def default_url(settings=None) -> str:
    """``TERRAIN_DB_URL``, else EDDIE's ``POSTGRES_*`` settings, else SQLite in ``TERRAIN_DATA_DIR``."""
    import os
    if settings is not None and settings.db_url:
        return settings.db_url
    if os.environ.get("TERRAIN_DB_URL"):
        return os.environ["TERRAIN_DB_URL"]
    e = os.environ
    if e.get("POSTGRES_DB") and e.get("POSTGRES_USER"):
        host = e.get("POSTGRES_HOST", "localhost")
        port = e.get("POSTGRES_PORT", "5432")
        return f"postgresql://{e['POSTGRES_USER']}:{e.get('POSTGRES_PASSWORD', '')}@{host}:{port}/{e['POSTGRES_DB']}"
    data = Path(settings.data_dir) if settings is not None else Path(e.get("TERRAIN_DATA_DIR", Path.home() / "terrain_data"))
    data.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{data / 'terrain_products.sqlite'}"


class Store:
    def __init__(self, engine_or_url=None, srid: int = 2193, settings=None):
        if engine_or_url is None:
            engine_or_url = default_url(settings)
        self.engine = sa.create_engine(engine_or_url) if isinstance(engine_or_url, str) else engine_or_url
        if hasattr(engine_or_url, "engine") and not isinstance(engine_or_url, sa.engine.Engine):
            self.engine = engine_or_url.engine           # a Connection
        self.srid = srid
        self.md = sa.MetaData()
        self.t = table(srid, self.md)
        self._ensured = False

    def ensure(self):
        """Create the table (and PostGIS, where allowed) at first use."""
        if self._ensured:
            return self
        with self.engine.begin() as c:
            if c.dialect.name == "postgresql":
                try:
                    c.execute(sa.text("CREATE EXTENSION IF NOT EXISTS postgis"))
                except Exception:  # noqa: BLE001 - no rights: the extension must already exist
                    pass
            self.md.create_all(c, checkfirst=True)
        self._ensured = True
        return self

    # --- rows -------------------------------------------------------------------------------------------
    def _row(self, r) -> Product:
        m = r._mapping
        info = m["generator_info"]
        paths = m["paths"]
        return Product(id=m["id"], product=m["product"], generator_key=m["generator_key"], config_key=m["config_key"],
                       generator_info=json.loads(info) if isinstance(info, str) else info, resolution=m["resolution"],
                       aoi=m["aoi"], extent=m["extent"], grid=m["grid"],
                       paths=json.loads(paths) if isinstance(paths, str) else paths, parent_id=m["parent_id"],
                       created_at=m["created_at"])

    def get(self, pid: int) -> Product | None:
        self.ensure()
        with self.engine.connect() as c:
            r = c.execute(sa.select(self.t).where(self.t.c.id == pid)).first()
        return self._row(r) if r else None

    def register(self, result, parent_id: int | None = None) -> Product:
        """Store a :class:`terrain_s2.stage1.dem.Stage1Result` (or a dict with the same fields); returns the
        row. An identical key and AOI already registered returns the existing row."""
        import shapely
        from shapely.geometry import box
        d = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        aoi = shapely.from_wkt(d["aoi_wkt"])
        existing = self._exact(d["key"], aoi)
        if existing is not None:
            return existing
        vals = dict(product=d["product"], generator_key=d["key"], config_key=d["info"]["config_key"],
                    generator_info=d["info"], resolution=int(d["resolution"]), aoi=aoi,
                    extent=shapely.from_wkt(d["extent_wkt"]) if d.get("extent_wkt") else None,
                    grid=box(*d["grid_bounds"]), paths=d["paths"], parent_id=parent_id)
        self.ensure()
        with self.engine.begin() as c:
            pid = c.execute(sa.insert(self.t).values(**vals).returning(self.t.c.id)).scalar_one()
        return self.get(pid)

    def _candidates(self, **eq) -> list[Product]:
        self.ensure()
        q = sa.select(self.t)
        for k, v in eq.items():
            q = q.where(self.t.c[k] == v)
        with self.engine.connect() as c:
            rows = [self._row(r) for r in c.execute(q.order_by(self.t.c.created_at.desc(), self.t.c.id.desc()))]
        return [r for r in rows if all(Path(p).exists() for k, p in r.paths.items() if k in ("netcdf",))]

    def _exact(self, key: str, aoi, tol: float = 0.1) -> Product | None:
        for r in self._candidates(generator_key=key):
            if _equal(r.aoi, aoi, tol):
                return r
        return None

    def lookup(self, key: str, info: dict, aoi, grid_bounds, *, spatial_reuse: bool = False, out_root=None,
               tol: float = 0.1) -> Product | None:
        """The stored product for this request, or None (see the module docstring). ``info`` is the request's
        generator info (its ``config_key`` and ``source_version`` are used for reuse)."""
        from shapely.geometry import box
        hit = self._exact(key, aoi, tol)
        if hit is not None or not spatial_reuse:
            return hit
        want = box(*grid_bounds)
        cands = self._candidates(config_key=info["config_key"])
        for r in cands:                                   # an earlier clip for the same AOI
            if r.parent_id is not None and _equal(r.aoi, aoi, tol) and sources_compatible(info, r.generator_info):
                return r
        for r in cands:
            if r.grid is not None and r.grid.buffer(1e-6).contains(want) and sources_compatible(info, r.generator_info):
                return self.clip(r, aoi, grid_bounds, out_root=out_root)
        return None

    def find_for_request(self, request: dict, geometry, *, contains: bool = True) -> list[Product]:
        """Products whose ``generator_info.request`` matches every item of ``request`` and whose AOI contains
        (or, with ``contains=False``, intersects) ``geometry``; newest first. Used by the compatibility shim,
        which knows the settings but not the inputs."""
        out = []
        for r in self._candidates():
            req = r.generator_info.get("request", {})
            if any(req.get(k) != v for k, v in request.items()):
                continue
            g = r.aoi.buffer(0.1)
            if (g.contains(geometry) if contains else g.intersects(geometry)):
                out.append(r)
        return out

    def clip(self, parent: Product, aoi, grid_bounds, *, out_root=None) -> Product:
        """Cut ``parent`` to ``grid_bounds`` (aligned to its grid), write it in the contract beside the parent
        and register it as a child row."""
        import datetime as _dt

        from .io import contract as C
        from .key import digest
        p = C.read_dem(parent.netcdf)
        a, _, c, _, e, f = p.transform
        x0, y0, x1, y1 = grid_bounds
        c0, r0 = (x0 - c) / a, (f - y1) / -e
        c1, r1 = (x1 - c) / a, (f - y0) / -e
        if any(abs(v - round(v)) > 1e-6 for v in (c0, r0, c1, r1)):
            raise ValueError(f"grid {grid_bounds} is not aligned to the parent's grid {p.transform}")
        c0, r0, c1, r1 = (int(round(v)) for v in (c0, r0, c1, r1))
        sl = (slice(r0, r1), slice(c0, c1))
        cut = lambda v: None if v is None else v[sl]  # noqa: E731
        prov = dict(p.provenance, clipped_from={"id": parent.id, "generator_key": parent.generator_key,
                                                "netcdf": parent.netcdf}, grid_bounds=list(grid_bounds))
        child = C.DemProduct(z=p.z[sl], transform=(a, 0.0, float(x0), 0.0, e, float(y1)), crs=p.crs,
                             resolution=p.resolution, product=p.product, generator=p.generator,
                             data_source=cut(p.data_source), lidar_source=cut(p.lidar_source),
                             lidar_mapping={k: v for k, v in p.lidar_mapping.items() if k != "no LiDAR"},
                             zo=cut(p.zo), modification_source=cut(p.modification_source), provenance=prov,
                             attrs={k: v for k, v in p.attrs.items() if k in ("geofabrics_instructions", "history")})
        gid = digest({"grid": list(grid_bounds), "crs": p.crs}, 12)
        root = Path(out_root) if out_root else Path(parent.netcdf).parent.parent
        out_dir = root / gid
        name = Path(parent.netcdf).stem
        paths = C.write_dem(out_dir / f"{name}.nc", child)
        geom = C.valid_footprint(child.z, child.transform)
        paths["extents"] = str(C.write_extents(out_dir / f"{name}_extents.geojson", geom, p.crs))
        rec = dict(key=parent.generator_key, info=parent.generator_info, product=parent.product,
                   resolution=parent.resolution, aoi_wkt=aoi.wkt, extent_wkt=geom.wkt if geom is not None else None,
                   grid_bounds=list(grid_bounds), paths=paths,
                   created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
        (out_dir / "product.json").write_text(json.dumps(dict(rec, parent_id=parent.id), indent=1, default=str))
        return self.register(rec, parent_id=parent.id)


def _equal(a, b, tol: float) -> bool:
    return a is not None and b is not None and a.hausdorff_distance(b) <= tol


def sources_compatible(request_info: dict, stored_info: dict) -> bool:
    """True when the inputs the request would use are the ones the stored product used: every requested
    (collection, item, checksum) is in the stored product, in the same survey priority order (raster); every
    requested point-cloud dataset is in the stored mapping, in the same order (GeoFabrics)."""
    a, b = request_info.get("source_version") or {}, stored_info.get("source_version") or {}
    if "surveys" in a:
        sb = {s["collection"]: (i, {tuple(x) for x in s["items"]}) for i, s in enumerate(b.get("surveys", []))}
        order = []
        for s in a["surveys"]:
            if s["collection"] not in sb:
                return False
            i, items = sb[s["collection"]]
            if not {tuple(x) for x in s["items"]} <= items:
                return False
            order.append(i)
        return order == sorted(order)
    if "mapping" in a:
        ma, mb = a.get("mapping") or {}, b.get("mapping") or {}
        if not set(ma) <= set(mb):
            return False
        return sorted(ma, key=ma.get) == sorted(ma, key=mb.get)
    return a == b
