"""``terrain_product`` (integration plan 6.6), created at first use in EDDIE's database.

The table definition lives in the library (:mod:`terrain_s2.store`), so FReDT and Smart Ideas read the same
registry through ``terrain_s2.client`` without importing this module. Legacy ``user_dem`` and ``hydro_dem`` rows
are never read.
"""
from terrain_s2.store import TABLE, Store, table

__all__ = ["TABLE", "Store", "table", "get_store"]


def get_store(srid: int = 2193) -> Store:
    """The registry on EDDIE's database engine (``eddie.digitaltwin.setup_environment.get_database``)."""
    from eddie.digitaltwin import setup_environment
    return Store(setup_environment.get_database(), srid=srid).ensure()
