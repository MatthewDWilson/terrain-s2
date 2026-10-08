"""``terrain`` command line (W6).

    terrain dem --aoi examples/aoi_whirinaki_sh12.geojson --resolution 8 [--backend geofabrics] [--register]
    terrain compare A.nc B.nc [--out report.md]        # regression harness
    terrain network ...                                # Stage 2 (delegates to run.py, S2-1, or the script)
    terrain condition ...                              # Stage 2 conditioning (S2-6, when present)

Settings default to the ``TERRAIN_*`` environment (see :mod:`terrain_s2.settings`); options override them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _dem(a) -> int:
    from .settings import from_env
    over = {k: v for k, v in dict(backend=a.backend, resolution=a.resolution, product=a.product,
                                  land_source=a.land_source, land_file=a.land_file, buffer_m=a.buffer,
                                  profile=a.profile, stac_root=a.stac_root, db_url=a.db).items() if v is not None}
    if a.data_dir:
        over["data_dir"] = Path(a.data_dir)
    if a.product_dir:
        over["product_dir"] = Path(a.product_dir)
    if a.spatial_reuse:
        over["spatial_reuse"] = True
    if a.allow_fallback:
        over["allow_fallback"] = True
    if a.lidar_classes:
        over["lidar_classes"] = tuple(int(c) for c in a.lidar_classes.split(","))
    s = from_env(**over)
    aoi = a.aoi
    log = (lambda *x: print(*x, file=sys.stderr)) if not a.quiet else (lambda *x: None)
    if a.register:
        from .stage1.dem import ensure
        row = ensure(aoi, s, aoi_crs=a.aoi_crs, log=log)
        out = {"product_id": row.id, "generator_key": row.generator_key, "paths": row.paths,
               "resolution": row.resolution, "parent_id": row.parent_id}
    else:
        from .stage1.dem import make
        r = make(aoi, s, aoi_crs=a.aoi_crs, log=log)
        out = {"generator_key": r.key, "reused": r.reused, "paths": r.paths, "resolution": r.resolution,
               "backend": r.backend, "gap_fraction": r.gap_fraction, "grid_bounds": r.grid_bounds}
    text = json.dumps(out, indent=1, default=str)
    if a.out_json:
        Path(a.out_json).write_text(text)
    print(text)
    return 0


def _compare(a) -> int:
    from .stage1.compare import compare, markdown
    rep = compare(_resolve_nc(a.a), _resolve_nc(a.b))
    md = markdown(rep)
    if a.out:
        out = Path(a.out)
        out.write_text(md)
        out.with_suffix(".json").write_text(json.dumps(rep, indent=1, default=str))
    print(md)
    return 1 if "error" in rep else 0


def _resolve_nc(p) -> str:
    """A product netCDF from a path to it, to its folder, or to its product.json."""
    p = Path(p)
    if p.is_dir():
        found = [p / "product.json"] if (p / "product.json").exists() else sorted(p.rglob("product.json"))
        if len(found) != 1:
            raise SystemExit(f"{p}: expected one product.json beneath it, found {len(found)}")
        p = found[0]
    if p.name == "product.json":
        return json.loads(p.read_text())["paths"]["netcdf"]
    return str(p)


def _network(argv) -> int:
    """Stage 2. Uses ``terrain_s2.run`` (S2-1) when it provides ``main``; otherwise the existing script. A
    Stage 1 product (its folder, product.json or .nc) given as ``--dem`` is read through its 1 m COG copy
    until S2-2 reads netCDF directly."""
    argv = list(argv)
    if "--dem" in argv:
        i = argv.index("--dem") + 1
        while i < len(argv) and not argv[i].startswith("--"):
            argv[i] = _stage1_input(argv[i])
            i += 1
    try:
        from . import run as s2run
        if hasattr(s2run, "main"):
            return int(s2run.main(argv) or 0)
    except ImportError:
        pass
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_network.py"
    if not script.exists():
        print("terrain network needs terrain_s2.run (S2-1) or a source checkout with scripts/run_network.py",
              file=sys.stderr)
        return 2
    import runpy
    sys.argv = [str(script)] + argv
    runpy.run_path(str(script), run_name="__main__")
    return 0


def _stage1_input(p: str) -> str:
    q = Path(p)
    if q.is_dir() or q.name == "product.json" or q.suffix == ".nc":
        nc = Path(_resolve_nc(q))
        cog = nc.with_suffix(".tif")
        try:
            import importlib
            dataio = importlib.import_module("terrain_s2.dataio")
            if getattr(dataio, "READS_NETCDF", False):
                return str(nc)
        except ImportError:
            pass
        if not cog.exists():
            raise SystemExit(f"{nc}: no 1 m COG copy beside it; Stage 2 needs a 1 m Stage 1 product")
        return str(cog)
    return p


def _condition(argv) -> int:
    try:
        from . import condition
    except ImportError:
        condition = None
    if condition is None or not hasattr(condition, "main"):
        print("terrain condition: Stage 2 conditioning (S2-6, terrain_s2.condition) is not available yet",
              file=sys.stderr)
        return 2
    return int(condition.main(list(argv)) or 0)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="terrain", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dem", help="Stage 1 DEM for an AOI")
    d.add_argument("--aoi", required=True, help="vector file, WKT or GeoJSON")
    d.add_argument("--aoi-crs", help="CRS of a WKT AOI (default EPSG:4326) or of a file without one")
    d.add_argument("--resolution", type=int)
    d.add_argument("--backend", choices=["raster", "geofabrics"])
    d.add_argument("--product", choices=["dem", "geofabric"])
    d.add_argument("--land-source", choices=["coverage", "topo50", "topo50_mangrove", "file"])
    d.add_argument("--land-file")
    d.add_argument("--lidar-classes", help="comma-separated, e.g. 2,9")
    d.add_argument("--buffer", type=float, help="AOI buffer in metres (200 when feeding Stage 2)")
    d.add_argument("--profile")
    d.add_argument("--data-dir")
    d.add_argument("--product-dir", help="where products are written (a separate folder per harness run)")
    d.add_argument("--stac-root")
    d.add_argument("--db", help="registry URL (with --register)")
    d.add_argument("--spatial-reuse", action="store_true")
    d.add_argument("--allow-fallback", action="store_true")
    d.add_argument("--register", action="store_true", help="use the product registry (ensure_dem)")
    d.add_argument("--out-json")
    d.add_argument("--quiet", action="store_true")
    c = sub.add_parser("compare", help="regression harness: B minus A")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out", help="report .md (a .json beside it)")
    sub.add_parser("network", help="Stage 2 network products (arguments passed through)", add_help=False)
    sub.add_parser("condition", help="Stage 2 conditioning (arguments passed through)", add_help=False)
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("network", "condition"):
        return (_network if argv[0] == "network" else _condition)(argv[1:])
    a = parser().parse_args(argv)
    return {"dem": _dem, "compare": _compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
