"""Light client for consumers (FReDT, Smart Ideas): find, open and clip stored terrain products.

Base install only. Consumers import ``terrain_s2.client`` (and :mod:`.compat`), never ``eddie_terrain`` (v5
plan section 5 item 11). Products are made on the terrain worker (``eddie_terrain.tasks.ensure_dem``, queue
``terrain``); this client reads the registry and the files on the shared ``stored_data`` volume.
"""
from __future__ import annotations


class TerrainClient:
    def __init__(self, conn_or_url=None, settings=None):
        from ..settings import from_env
        from ..store import Store
        self.settings = settings or from_env()
        self.store = Store(conn_or_url, settings=self.settings)

    def request(self) -> dict:
        """The request part of the generator info these settings produce (for matching stored products)."""
        s = self.settings
        from ..stage1.profile import load
        return {"product": s.product, "profile": load(s.profile)["name"], "resolution": int(s.resolution),
                "land_source": s.land_source, "buffer_m": s.buffer_m}

    def find(self, geometry, *, backend: str | None = None, contains: bool = True):
        """Newest stored product made with these settings whose AOI contains ``geometry`` (working CRS), or None.
        ``backend`` defaults to the settings' backend; pass ``"any"`` to accept either."""
        req = self.request()
        b = backend or self.settings.backend
        if b != "any":
            req["backend"] = b
        rows = self.store.find_for_request(req, geometry, contains=contains)
        return rows[0] if rows else None

    def get(self, product_id: int):
        return self.store.get(product_id)

    def open(self, product_id: int):
        """The product's netCDF as an xarray Dataset."""
        from ..io.contract import open_dem
        return open_dem(self.get(product_id).netcdf)

    def clip(self, product, geometry):
        """A child product cut to ``geometry``'s bounding box on the product's grid (registered; reused if it
        exists)."""
        from ..stage1.aoi import snap
        g = product.grid.bounds                       # snapped to the resolution, so snapping aligns to it
        b = snap(geometry.bounds, product.resolution)
        bounds = (max(b[0], g[0]), max(b[1], g[1]), min(b[2], g[2]), min(b[3], g[3]))
        for c in self.store._candidates(config_key=product.config_key):
            if c.parent_id == product.id and c.grid is not None and c.grid.bounds == bounds:
                return c
        return self.store.clip(product, geometry, bounds)
