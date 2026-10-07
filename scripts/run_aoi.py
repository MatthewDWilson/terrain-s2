#!/usr/bin/env python
"""Phase 2a runbook (design §11): DEM (+DSM) window for an AOI -> features, candidates, quick-look.

Example (test site):
    python scripts/run_aoi.py --dem data/AW27.tif --dsm data/AW27_dsm.tif \
        --aoi data/aoi.geojson --buffer 200 --out out/sh12 --device auto

Several --dem tiles may be given (neighbours), and are mosaicked over the window.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from terrain_s2 import __version__, dataio, pipeline, quicklook  # noqa: E402
from terrain_s2.backend import get_backend  # noqa: E402
from terrain_s2.config import Params  # noqa: E402


def _plain(v):
    """numpy scalars -> Python, so GeoPackage columns get clean types."""
    if isinstance(v, np.generic):
        return v.item()
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", nargs="+", required=True, help="bare-earth DEM tile(s)")
    ap.add_argument("--dsm", nargs="*", default=None, help="DSM tile(s) on the same grid (optional)")
    ap.add_argument("--aoi", help="AOI vector file (any CRS); default: whole first DEM tile")
    ap.add_argument("--bounds", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                    help="AOI bounds in the DEM CRS (alternative to --aoi)")
    ap.add_argument("--buffer", type=float, default=200.0, help="context buffer around the AOI (m)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--hash-inputs", action="store_true", help="sha256 the input files (slow for 1:50k tiles)")
    ap.add_argument("--no-raster", action="store_true", help="skip writing features.tif")
    ap.add_argument("--assume-crs", help="relabel inputs with this CRS (no reprojection), e.g. EPSG:2193+7839 "
                                         "for LINZ tiles mislabelled EPSG:9528")
    ap.add_argument("--aoi-crs", help="relabel the AOI with this CRS (no reprojection), e.g. EPSG:2193")
    ap.add_argument("--reproject-to", help="warp inputs to this projected CRS (resamples; prefer native data)")
    ap.add_argument("--resolution", type=float, default=1.0, help="cell size (m) when reprojecting")
    ap.add_argument("--resampling", default="bilinear", help="resampling when reprojecting")
    ap.add_argument("--no-gates", action="store_true", help="keep kind/vegetation/approach as attributes, not exclusions")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import geopandas as gpd
    import rasterio

    dem_crs = dataio.processing_crs(a.dem[0], a.assume_crs, a.reproject_to)
    with rasterio.open(a.dem[0]) as s:
        tile_bounds = tuple(s.bounds)
        if a.reproject_to:                            # tile bounds into the processing CRS
            from rasterio.warp import transform_bounds
            src_h = dataio.horizontal_crs(dataio.effective_crs(a.dem[0], s.crs, tile_bounds, a.assume_crs,
                                                               allow_geographic=True, res=s.res))
            tile_bounds = transform_bounds(dataio._rio_crs(src_h), dataio._rio_crs(dem_crs), *tile_bounds,
                                           densify_pts=21)
    aoi_geom = None
    if a.aoi:
        aoi = dataio.read_aoi(a.aoi, dem_crs, a.aoi_crs)
        aoi_geom = aoi.union_all()
        bounds = tuple(aoi_geom.bounds)
    elif a.bounds:
        bounds = tuple(a.bounds)
    else:
        bounds = tile_bounds

    t0 = time.perf_counter()
    rp = dict(assume_crs=a.assume_crs, reproject_to=a.reproject_to, resolution=a.resolution, resampling=a.resampling)
    win = dataio.read_window(a.dem, bounds, a.buffer, **rp)
    dsm = None
    if a.dsm:
        dw = dataio.read_window(a.dsm, bounds, a.buffer, **rp)
        if dw.z.shape != win.z.shape or not np.allclose(tuple(dw.transform)[:6], tuple(win.transform)[:6]):
            raise SystemExit("DSM grid does not match DEM grid")
        dsm = dw.z
    t_read = time.perf_counter() - t0
    print(f"read {win.z.shape[1]}x{win.z.shape[0]} cells in {t_read:.1f}s; "
          f"valid {np.isfinite(win.z).mean():.1%}; uncovered {win.uncovered_m:.0f} m")

    params = Params()
    if a.no_gates:
        params.candidates.hard_gates = False
    be = get_backend(a.device)
    sv = "|".join(f"{Path(p).name}" for p in a.dem)
    res = pipeline.run(win.z, win.transform, params, be, dsm=dsm, core=bounds, source_version=sv)
    if aoi_geom is not None:
        from shapely.geometry import Point
        keep = lambda r: aoi_geom.contains(Point(r["crest_x"], r["crest_y"]))  # noqa: E731
        res.breaches = [r for r in res.breaches if keep(r)]
        res.candidates = [r for r in res.candidates if keep(r)]
        res.test_a = [r for r in res.test_a if keep(r)]
    res.timings["read"] = t_read
    print("timings (s):", {k: round(v, 2) for k, v in res.timings.items()}, "device:", res.device)
    print(f"pits {res.breach.n_pits}, unresolved {res.breach.n_unresolved}, "
          f"stored breaches {len(res.breach.seed)}, in AOI {len(res.breaches)}")
    print(f"channel ends {len(res.network.ends.rc)}, bridging paths {len(res.network.paths)}, "
          f"Test A records in AOI {len(res.test_a)}")
    n = {k: sum(r['test'] == k for r in res.candidates) for k in ('A', 'B', 'A+B')}
    print(f"candidates {len(res.candidates)}: A {n['A']}, B {n['B']}, A+B {n['A+B']}")

    # vectors
    def gdf(recs, geom="geometry"):
        if not recs:
            return None
        keys = sorted({k for r in recs for k in r} - {"geometry", "path_geometry"})
        rows = [{k: _plain(r.get(k)) for k in keys} for r in recs]
        return gpd.GeoDataFrame(rows, geometry=[r[geom] for r in recs], crs=dataio.horizontal_crs(win.crs))

    gpkg = out / "candidates.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    obstructions = [r for r in res.breaches if r.get("kind") == "channel_obstruction"
                    and r["max_cut"] >= params.candidates.min_barrier_height_m]
    print(f"channel obstructions (false barriers inside channels, for DEM conditioning): {len(obstructions)}")
    for layer, recs, geom in (("candidates", res.candidates, "geometry"),
                              ("channel_obstructions", obstructions, "geometry"),
                              ("test_a", res.test_a, "geometry"),
                              ("breaches", res.breaches, "geometry"),
                              ("breach_paths", res.breaches, "path_geometry")):
        g = gdf(recs, geom)
        if g is not None:
            g.to_file(gpkg, layer=layer, driver="GPKG")

    # rasters
    breach_depth = np.where(np.isfinite(win.z), win.z - res.breach.z_breached, np.nan).astype(np.float32)
    if not a.no_raster:
        bands = dict(res.feats)
        bands["breach_depth"] = breach_depth
        bands["depression_depth"] = np.where(np.isfinite(win.z), res.dep.depth, np.nan)
        bands["log10_upstream_area_m2"] = np.log10(np.maximum(res.upa, 1.0))
        bands["channel_mask"] = res.network.mask.astype(np.float32)
        bands["channel_centreline"] = res.network.skeleton.astype(np.float32)
        if dsm is not None:
            bands["dsm_minus_dem"] = dsm - win.z
        dataio.write_stack(out / "features.tif", bands, win.transform, win.crs)

    quicklook.render(out / "quicklook.png", win.z, win.transform, res.breaches, res.candidates,
                     breach_depth, res.upa, aoi_bounds=bounds, title=", ".join(Path(p).name for p in a.dem),
                     channels=res.network.mask, test_a=res.test_a)

    # provenance (T5)
    import numba, pyflwdir, scipy
    prov = {
        "tool": "terrain_s2", "version": __version__, "device": res.device,
        "inputs": {"dem": [dataio.file_provenance(p, a.hash_inputs) for p in a.dem],
                   "dsm": [dataio.file_provenance(p, a.hash_inputs) for p in (a.dsm or [])]},
        "aoi_bounds": bounds, "buffer_m": a.buffer, "window_bounds": win.bounds, "assume_crs": a.assume_crs,
        "reprojection": ({"to": a.reproject_to, "resolution": a.resolution, "resampling": a.resampling}
                         if a.reproject_to else None),
        "uncovered_m": win.uncovered_m, "grid": {"shape": list(win.z.shape), "transform": list(win.transform)[:6],
                                                 "crs": win.crs.to_wkt()},
        "params": params.to_dict(), "timings_s": res.timings,
        "counts": {"pits": res.breach.n_pits, "unresolved": res.breach.n_unresolved,
                   "breaches_in_aoi": len(res.breaches), "channel_ends": int(len(res.network.ends.rc)),
                   "bridging_paths": len(res.network.paths), "test_a_records_in_aoi": len(res.test_a),
                   "candidates": len(res.candidates),
                   "candidates_by_test": {k: sum(r["test"] == k for r in res.candidates) for k in ("A", "B", "A+B")}},
        "env": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "numba": numba.__version__, "pyflwdir": pyflwdir.__version__, "platform": platform.platform()},
    }
    (out / "run.json").write_text(json.dumps(prov, indent=2, default=str))
    print(f"wrote {out}/candidates.gpkg, features.tif, quicklook.png, run.json")


if __name__ == "__main__":
    main()
