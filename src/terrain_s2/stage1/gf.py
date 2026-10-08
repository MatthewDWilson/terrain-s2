"""``geofabrics`` backend (plan 6.3): an unmodified, pinned GeoFabrics driven only through
``geofabrics.runner.from_instructions_dict``, with instruction templates held as data (``templates/``).

1. Snap the buffered AOI (done by the caller); the snapped box goes in as ``data_paths.extents``.
2. Discover point clouds with ``otCatalog`` and rank them newest first into ``dataset_mapping.lidar``
   (:mod:`.catalogue`), done by the caller so the generator key can be formed first.
3. CRS: per-dataset overrides come only from the profile; otherwise GeoFabrics uses the LAZ headers.
4. Land: one file from :mod:`.land` as ``data_paths.land``.
5. Fill the template, save it beside the product as ``instructions.json``, run GeoFabrics.
6. Extents: the valid-data footprint of ``z`` (decimated by 8 above 25 M cells) as ``<id>_extents.geojson``.
7. Finish: CF attributes, the compound CRS and ``terrain_provenance`` in the netCDF, the COG copy at 1 m,
   the provenance JSON.

GeoFabrics (GPL-3.0) is imported only inside :func:`run_geofabrics`, so nothing else needs it installed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import time
from pathlib import Path

from ..io import contract

HERE = Path(__file__).parent / "templates"
TEMPLATES = {"dem": "dem.json", "geofabric": "dem_roughness.json"}
PIN = "1.1.30"
_PLACEHOLDER = re.compile(r"^\{([a-z_]+)\}$")


class _Drop:
    pass


DROP = _Drop()


def template_path(product: str) -> Path:
    return HERE / TEMPLATES[product]


def template_sha256(product: str) -> str:
    return hashlib.sha256(template_path(product).read_bytes()).hexdigest()


def load_template(product: str) -> dict:
    return json.loads(template_path(product).read_text())


def fill(obj, values: dict):
    """Fill ``"{name}"`` placeholders. A string that is exactly one placeholder takes the value's type; a value
    of None drops its key (and dicts left empty by that are dropped too). Keys starting with ``_`` are
    comments and are removed. Unknown placeholders raise."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k).startswith("_"):
                continue
            fv = fill(v, values)
            if fv is DROP or (isinstance(fv, dict) and not fv and isinstance(v, (dict, str)) and v != {}):
                continue
            out[k] = fv
        return out
    if isinstance(obj, list):
        return [x for x in (fill(v, values) for v in obj) if x is not DROP]
    if isinstance(obj, str):
        m = _PLACEHOLDER.match(obj)
        if m:
            name = m.group(1)
            if name not in values:
                raise KeyError(f"template placeholder {{{name}}} has no value")
            v = values[name]
            return DROP if v is None else copy.deepcopy(v)
        if "{" in obj:
            try:
                return obj.format(**values)
            except KeyError as e:
                raise KeyError(f"template placeholder {e} has no value") from None
        return obj
    return obj


def merge(a: dict, b: dict) -> dict:
    """Recursive merge, ``b`` wins (for per-request ``overrides``)."""
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def configuration(settings, overrides: dict | None = None) -> dict:
    from ..key import digest
    return {"lidar_classes": list(settings.lidar_classes), "template": TEMPLATES[settings.product],
            "template_sha256": template_sha256(settings.product),
            "overrides": digest(overrides or {}, 64) if overrides else None}


def datasets_block(datasets, profile: dict) -> dict | None:
    """``datasets.lidar.open_topography``: {name: {} or {"crs": {...}}} (CRS overrides from the profile only)."""
    if not datasets:
        return None
    over = (profile.get("geofabrics") or {}).get("crs_overrides") or {}
    return {d.name: ({"crs": dict(over[d.name])} if d.name in over else {}) for d in datasets}


def instructions(*, settings, profile: dict, out_dir, name: str, datasets, mapping: dict | None, land_file,
                 cache_dir, overrides: dict | None = None) -> dict:
    res = int(settings.resolution)
    values = {
        "h_crs": int(profile["crs"]["horizontal"]), "v_crs": int(profile["crs"]["vertical"]),
        "resolution": res, "chunk_size": max(100, int(800 // res)),
        "memory_limit": settings.gf_memory_limit, "cores": int(settings.gf_cores),
        "download_limit": settings.gf_download_limit_gb,
        # GeoFabrics needs the result folder inside local_cache, so local_cache is the product folder and the
        # shared download cache (LAZ tiles, tile indexes) is given as an absolute path.
        "product_dir": str(Path(out_dir).resolve()), "downloads": str((Path(cache_dir) / "downloads").resolve()),
        "id": name, "land_file": str(Path(land_file).resolve()) if land_file else None,
        "datasets": datasets_block(datasets, profile), "mapping": dict(mapping) if mapping else None,
        "lidar_classes": list(settings.lidar_classes),
    }
    ins = fill(load_template(settings.product), values)
    return merge(ins, overrides) if overrides else ins


def result_path(ins: dict, product: str) -> Path:
    dp = ins["default"]["data_paths"]
    key = "result_geofabric" if product == "geofabric" else "result_dem"
    p = Path(dp[key])
    return p if p.is_absolute() else Path(dp["local_cache"]) / dp["subfolder"] / p


def run_geofabrics(ins: dict) -> None:
    """Call the pinned GeoFabrics runner. Root logging, which GeoFabrics reconfigures, is restored after."""
    import geofabrics
    from geofabrics import runner
    v = getattr(geofabrics, "__version__", None)
    if v is None:
        from importlib.metadata import version
        v = version("geofabrics")
    if v != PIN:
        raise RuntimeError(f"geofabrics {PIN} is pinned, found {v}")
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    try:
        runner.from_instructions_dict(ins)
    finally:
        for h in list(root.handlers):
            if h not in handlers:
                root.removeHandler(h)
                try:
                    h.close()
                except Exception:  # noqa: BLE001
                    pass
        for h in handlers:
            if h not in root.handlers:
                root.addHandler(h)
        root.setLevel(level)


def build(*, settings, profile: dict, aoi_geom, bounds, out_dir, name: str, key: str, info: dict, datasets,
          mapping, land_file, land_rec: dict, cache_dir, overrides: dict | None = None, log=print,
          runner=None) -> dict:
    """Run GeoFabrics for one product and bring its output up to the contract."""
    import geopandas as gpd
    from shapely.geometry import box

    from ..key import provenance
    from .profile import crs_string
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crs = crs_string(profile)
    h = f"EPSG:{profile['crs']['horizontal']}"
    gpd.GeoDataFrame(geometry=[box(*bounds)], crs=h).to_file(out_dir / f"{name}.geojson", driver="GeoJSON")
    ins = instructions(settings=settings, profile=profile, out_dir=out_dir, name=name, datasets=datasets,
                       mapping=mapping, land_file=land_file, cache_dir=cache_dir, overrides=overrides)
    (out_dir / "instructions.json").write_text(json.dumps(ins, indent=1))
    # GeoFabrics skips a stage whose output file exists; outputs left by an interrupted build (no manifest, or
    # the caller would have reused the product) must not be taken as finished.
    dp = ins["default"]["data_paths"]
    for k in ("raw_dem", "result_dem", "result_geofabric"):
        if k in dp:
            f = Path(dp[k]) if Path(dp[k]).is_absolute() else Path(dp["local_cache"]) / dp["subfolder"] / dp[k]
            f.unlink(missing_ok=True)
    (runner or run_geofabrics)(ins)
    t_gf = time.perf_counter() - t0
    nc = result_path(ins, settings.product)
    if not nc.exists():
        raise RuntimeError(f"GeoFabrics finished without writing {nc}")
    if nc.parent.resolve() != out_dir.resolve():
        raise RuntimeError(f"GeoFabrics wrote {nc} outside the product folder {out_dir}")
    p = contract.read_dem(nc)
    res = contract.resolution_from_description(_desc(nc))
    if res != int(settings.resolution):
        raise RuntimeError(f"{nc}: description resolution {res} != requested {settings.resolution}")
    geom = contract.valid_footprint(p.z, p.transform)
    paths = {"netcdf": str(nc), "instructions": str(out_dir / "instructions.json"),
             "extents": str(contract.write_extents(out_dir / f"{name}_extents.geojson", geom, crs))}
    if land_file:
        paths["land"] = str(land_file)
    import numpy as np
    valid = np.isfinite(p.z)
    gap = float(1.0 - valid.mean()) if valid.size else 1.0
    counts = {}
    if p.data_source is not None:
        u, c = np.unique(p.data_source[np.isfinite(p.data_source)].astype(int), return_counts=True)
        counts = {int(a): int(b) for a, b in zip(u, c)}
    prov = provenance(key, info, inputs=[d.record() for d in datasets or []],
                      parameters={"settings": settings.public(), "grid_bounds": list(bounds),
                                  "dataset_mapping": mapping or {}},
                      timings={"geofabrics_s": round(t_gf, 2), "total_s": round(time.perf_counter() - t0, 2)},
                      extra={"backend": "geofabrics", "gap_fraction": round(gap, 6), "land": land_rec,
                             "data_source_counts": counts, "instructions_file": "instructions.json"})
    contract.finish_netcdf(nc, crs, prov)
    if int(settings.resolution) == 1:
        paths["cog"] = str(contract.write_cog(contract.cog_path(nc), p.z, p.transform, crs))
    paths["provenance"] = str(contract.write_provenance(contract.provenance_path(nc), prov))
    log(f"geofabrics: {p.z.shape[1]} x {p.z.shape[0]} cells at {settings.resolution} m, "
        f"{len(datasets or [])} dataset(s), gaps {gap:.2%}")
    return {"paths": paths, "extent": geom, "gap_fraction": gap, "crs": crs, "provenance": prov}


def _desc(nc) -> str:
    with contract.open_dem(nc, decode=False) as ds:
        return str(ds.attrs.get("description", ""))
