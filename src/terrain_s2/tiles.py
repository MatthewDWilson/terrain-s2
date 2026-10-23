"""Tiles: a partition tile's bundle, for the batch and for the production tile route (``run.tile``).

A tile's bundle (``<bundles>/<tile>/``) is what Stage 2's own acquisition builds for it (``acquire.build``): the
1 m DEM and DSM through the shared elevation reader, roads, context (OSM ways, buildings, REC2 rivers), labels
(council and asset records) and ``site.json`` (provenance). ``ensure_bundle`` builds it when it is missing or
unusable (``bundle_ok``: a DEM and no elevation error) and returns its folder. Tile ownership and merging (S2-8)
will join this module.

Defaults (``TERRAIN_DATA_DIR`` = ``data``): bundles ``data/tiles``, the partition ``data/partition/nz_tiles.gpkg``
(``scripts/partition.py``), sources ``sites/sources.yml`` with its coverage index beside it.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def bundle_ok(bundle) -> bool:
    """A usable bundle: site.json without elevation errors, and a DEM (a build that failed on elevation
    still writes site.json: BJ39_10000_0303 in wave 1 was skipped as built and then had no DEM)."""
    bundle = Path(bundle)
    f = bundle / "site.json"
    if not f.exists() or not (bundle / "dem.tif").exists():
        return False
    try:
        errs = json.loads(f.read_text()).get("errors", [])
    except ValueError:
        return False
    return not any(str(e.get("step", "")).startswith("elevation") for e in errs)


def default_partition(data_dir=None) -> Path:
    if data_dir is None:
        from .acquire.elevation import default_data_dir
        data_dir = default_data_dir()
    return Path(data_dir) / "partition" / "nz_tiles.gpkg"


def ensure_bundle(tile_id: str, bundles=None, partition=None, buffer_m: float = 200.0, sources=None, index=None,
                  cache=None, mirror: bool = True, build: bool = True, log=print):
    """``(bundle folder, build errors)`` for ``tile_id``: the existing bundle if usable, else (``build``) a new one.
    Raises when there is no usable bundle afterwards (no DEM: an elevation step failed)."""
    from .acquire.elevation import default_data_dir
    data = default_data_dir()
    bundles = Path(bundles or data / "tiles")
    bundle = bundles / tile_id
    errors = []
    if not bundle_ok(bundle):
        if not build:
            raise FileNotFoundError(f"no usable bundle for {tile_id} at {bundle} (and build=False)")
        from .acquire import registry
        from .acquire.build import build_site
        from .acquire.site import load_sources, tile_site
        sources_path = sources or str(ROOT / "sites" / "sources.yml")
        idx_path = Path(index) if index else registry.default_index_path(sources_path)
        site = tile_site(tile_id, partition or default_partition(data), buffer_m)
        prov = build_site(site, load_sources(sources_path), str(bundles), cache or str(data),
                          index=registry.load_index(idx_path), mirror=mirror, log=log)
        errors = list(prov.get("errors", []))
        if any(str(e.get("step", "")).startswith("elevation") for e in errors):
            raise RuntimeError(f"elevation failed: {errors}")
    return bundle, errors


def bundle_inputs(bundle) -> dict:
    """The run's inputs from a bundle: dem, dsm (or None), aoi, roads (or None)."""
    b = Path(bundle)
    return dict(dem=b / "dem.tif", dsm=b / "dsm.tif" if (b / "dsm.tif").exists() else None, aoi=b / "aoi.geojson",
                roads=b / "roads.gpkg" if (b / "roads.gpkg").exists() else None)
