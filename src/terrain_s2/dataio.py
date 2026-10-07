"""Raster and vector I/O. Reads only the window needed (LINZ 1:50k tiles are ~860 M cells)."""
from __future__ import annotations

import hashlib
import math
import os
import warnings
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.transform import Affine

# rasterio 1.5.1 composes transforms with `*`, a pending deprecation in affine 3 (to be fixed
# upstream). Silence only that message, only when raised from rasterio (e.g. in windows.py).
warnings.filterwarnings("ignore", message="Use `@` matmul", category=PendingDeprecationWarning,
                        module="rasterio")


@dataclass
class Window:
    z: np.ndarray            # float32, NaN = nodata
    transform: Affine
    crs: object
    bounds: tuple            # (xmin, ymin, xmax, ymax) actually read
    uncovered_m: float       # how far the requested window extends beyond the inputs (0 = none)
    sources: list


def horizontal_crs(crs):
    """Horizontal part of a (possibly compound or bound) CRS, as a pyproj CRS."""
    import pyproj
    c = pyproj.CRS.from_user_input(crs.to_wkt() if hasattr(crs, "to_wkt") else crs)
    if c.is_bound:
        c = c.source_crs
    if c.is_compound:
        c = c.sub_crs_list[0]
    return c


def _looks_geographic(bounds, res=None):
    """Bounds within degree ranges (and, if given, pixel sizes far below a metre)."""
    x0, y0, x1, y1 = bounds
    in_deg = max(abs(x0), abs(x1)) <= 360 and max(abs(y0), abs(y1)) <= 90
    return in_deg and (res is None or max(abs(res[0]), abs(res[1])) < 0.01)


def vertical_crs(crs):
    """Vertical part of a compound CRS (pyproj), or None."""
    import pyproj
    c = pyproj.CRS.from_user_input(crs.to_wkt() if hasattr(crs, "to_wkt") else crs)
    if c.is_compound and len(c.sub_crs_list) > 1:
        return c.sub_crs_list[1]
    return None


def effective_crs(name, crs, bounds, assume_crs=None, allow_geographic=False, res=None):
    """Validate a raster's CRS and return the CRS to use (rasterio CRS).

    Compound CRSs (e.g. NZTM 2000 + NZVD2016, ``EPSG:2193+7839``) are accepted: only the horizontal
    part must be projected in metres. ``assume_crs`` relabels a file whose metadata is wrong (no
    reprojection). A geographic label on coordinates that are clearly not degrees (some LINZ Data
    Service exports are labelled ``EPSG:9528`` = NZGD2000 + NZVD2016, geographic, while holding
    NZTM metres) is refused with a hint, rather than silently relabelled.
    """
    from rasterio.crs import CRS
    if assume_crs:
        new = CRS.from_user_input(assume_crs)
        if (crs is not None and not horizontal_crs(crs).is_projected and horizontal_crs(new).is_projected
                and _looks_geographic(bounds, res)):
            raise ValueError(
                f"{name}: this raster really is geographic ({crs.to_string()[:40]}; bounds "
                f"{bounds[0]:.4f}, {bounds[1]:.4f}; pixel size {res[0] if res else '?'}), so relabelling it as "
                f"{assume_crs} would be wrong. Download it in its native CRS (LINZ elevation: EPSG:2193), "
                f"or pass --reproject-to EPSG:2193 (resamples).")
        crs = new
    if crs is None:
        raise ValueError(f"{name}: no CRS; pass assume_crs (e.g. 'EPSG:2193+7839' for NZTM + NZVD2016)")
    h = horizontal_crs(crs)
    if not h.is_projected and allow_geographic and _looks_geographic(bounds):
        return crs
    if not h.is_projected:
        x0, y0, x1, y1 = bounds
        if max(abs(x0), abs(x1)) > 360 or max(abs(y0), abs(y1)) > 90:
            raise ValueError(
                f"{name}: CRS is labelled {crs.to_string()[:60]} (horizontal {h.name}, geographic, degrees), "
                f"but its coordinates ({x0:.0f}, {y0:.0f}) are not degrees: the CRS metadata is wrong. If "
                "the data are NZTM 2000 + NZVD2016, pass assume_crs='EPSG:2193+7839' (--assume-crs on the "
                "command line); this only relabels, it does not reproject.")
        raise ValueError(
            f"{name}: a projected horizontal CRS in metres is required (got {h.name}, geographic). LDS exports "
            "are reprojected into the CRS chosen at download: download LINZ elevation in its native EPSG:2193 "
            "(recommended; heights stay NZVD2016), or pass --reproject-to EPSG:2193 (resamples).")
    units = (h.axis_info[0].unit_name or "").lower() if h.axis_info else ""
    if units not in ("metre", "meter", "m", "metres", "meters"):
        raise ValueError(f"{name}: horizontal CRS units are {units!r}, metres required")
    return crs


def read_aoi(path, target_crs, aoi_crs=None):
    """Read an AOI file and return it in ``target_crs`` (a horizontal, projected CRS).

    ``aoi_crs`` relabels the AOI (no reprojection) when its CRS metadata is wrong. An AOI whose
    label is geographic but whose coordinates cannot be degrees is refused with a hint: a GeoJSON
    without a ``crs`` member is read as WGS84 (RFC 7946), and QGIS projects using a raster
    mislabelled EPSG:9528 save NZTM coordinates under that geographic label.
    """
    import geopandas as gpd
    aoi = gpd.read_file(path)
    if aoi_crs:
        aoi = aoi.set_crs(aoi_crs, allow_override=True)
    elif aoi.crs is None:
        raise ValueError(f"{path}: the AOI has no CRS; pass --aoi-crs (e.g. EPSG:2193)")
    b = aoi.total_bounds
    if not horizontal_crs(aoi.crs).is_projected and (max(abs(b[0]), abs(b[2])) > 360 or max(abs(b[1]), abs(b[3])) > 90):
        raise ValueError(
            f"{path}: the AOI is labelled {aoi.crs.to_string()[:40]} (geographic, degrees) but its coordinates "
            f"({b[0]:.0f}, {b[1]:.0f}) are not degrees. A GeoJSON without a 'crs' member is read as WGS84, and "
            "QGIS projects using a raster mislabelled EPSG:9528 save NZTM coordinates that way. Pass "
            "--aoi-crs EPSG:2193 (relabel only), or re-save the AOI in EPSG:2193.")
    out = aoi.to_crs(target_crs)
    if not np.all(np.isfinite(out.total_bounds)):
        raise ValueError(f"{path}: AOI bounds are not finite after transforming to {target_crs}")
    return out


def snap_bounds(bounds, transform):
    """Expand bounds outward to the source pixel grid."""
    xmin, ymin, xmax, ymax = bounds
    a, e, c, f = transform.a, transform.e, transform.c, transform.f
    x0 = c + math.floor((xmin - c) / a) * a
    x1 = c + math.ceil((xmax - c) / a) * a
    y1 = f + math.floor((ymax - f) / e) * e   # e < 0
    y0 = f + math.ceil((ymin - f) / e) * e
    return (x0, y0, x1, y1)


MAX_WINDOW_CELLS = 2_000_000_000          # ~8 GB as float32; anything larger is almost surely a CRS mix-up


def processing_crs(path, assume_crs=None, reproject_to=None):
    """Horizontal CRS (pyproj) that AOIs and outputs use for this input."""
    if reproject_to:
        return horizontal_crs(_rio_crs(reproject_to))
    with rasterio.open(path) as s:
        return horizontal_crs(effective_crs(path, s.crs, tuple(s.bounds), assume_crs, res=s.res))


def _rio_crs(x):
    from rasterio.crs import CRS
    return x if isinstance(x, CRS) else CRS.from_user_input(x.to_wkt() if hasattr(x, "to_wkt") else x)


def read_window(paths, bounds, buffer_m: float = 0.0, assume_crs=None,
                max_cells: int = MAX_WINDOW_CELLS, reproject_to=None, resolution: float = 1.0,
                resampling: str = "bilinear") -> Window:
    """Mosaic ``paths`` (one or more tiles) over ``bounds`` + ``buffer_m``, on the source grid.

    Passing neighbouring tiles is how edge effects are avoided at tile boundaries. Tiles may
    differ in vertical CRS labelling (e.g. EPSG:2193 and EPSG:2193+7839) but must share the
    horizontal CRS and resolution. ``bounds`` are in that horizontal CRS.
    """
    paths = [str(p) for p in paths]
    if reproject_to:
        return _read_window_reprojected(paths, bounds, buffer_m, assume_crs, max_cells, reproject_to,
                                        resolution, resampling)
    srcs = [rasterio.open(p) for p in paths]
    try:
        s0 = srcs[0]
        crs = effective_crs(paths[0], s0.crs, tuple(s0.bounds), assume_crs, res=s0.res)
        h0 = horizontal_crs(crs)
        for s in srcs[1:]:
            cs_ = effective_crs(s.name, s.crs, tuple(s.bounds), assume_crs, res=s.res)
            if not horizontal_crs(cs_).equals(h0) or not np.allclose(s.res, s0.res):
                raise ValueError(f"{s.name}: horizontal CRS or resolution differs from {paths[0]}")
        xmin, ymin, xmax, ymax = bounds
        if not np.all(np.isfinite(bounds)):
            raise ValueError(f"requested bounds are not finite: {bounds} (check the AOI's CRS)")
        want = snap_bounds((xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m),
                           s0.transform)
        ux0 = min(s.bounds.left for s in srcs); uy0 = min(s.bounds.bottom for s in srcs)
        ux1 = max(s.bounds.right for s in srcs); uy1 = max(s.bounds.top for s in srcs)
        inputs = f"inputs cover ({ux0:.0f}, {uy0:.0f}, {ux1:.0f}, {uy1:.0f})"
        if want[2] <= ux0 or want[0] >= ux1 or want[3] <= uy0 or want[1] >= uy1:
            raise ValueError(f"requested window ({want[0]:.0f}, {want[1]:.0f}, {want[2]:.0f}, {want[3]:.0f}) "
                             f"does not overlap the inputs; {inputs}. Check the AOI's CRS.")
        cells = ((want[2] - want[0]) / s0.transform.a) * ((want[3] - want[1]) / -s0.transform.e)
        if cells > max_cells:
            raise ValueError(f"requested window ({want[0]:.0f}, {want[1]:.0f}, {want[2]:.0f}, {want[3]:.0f}) is "
                             f"{cells:.2e} cells (limit {max_cells:.1e}); {inputs}. Check the AOI's CRS.")
        uncovered = max(ux0 - want[0], uy0 - want[1], want[2] - ux1, want[3] - uy1, 0.0)
        z, tr = _mosaic(srcs, want)
        if uncovered > 0:
            warnings.warn(f"Requested window extends {uncovered:.0f} m beyond the input tiles; "
                          "results near that edge may show edge effects. Supply neighbouring tiles.")
        return Window(z, tr, crs, want, float(uncovered), paths)
    finally:
        for s in srcs:
            s.close()


def _read_window_reprojected(paths, bounds, buffer_m, assume_crs, max_cells, reproject_to, resolution,
                             resampling):
    """Warp tiles onto a ``resolution`` grid in ``reproject_to`` covering bounds + buffer.

    Only the horizontal CRS is warped (heights are not vertically transformed); the output label
    combines the target's horizontal CRS with the source's vertical CRS, e.g. NZGD2000 + NZVD2016
    becomes NZTM 2000 + NZVD2016. Resampling smooths micro-relief: prefer data in a native
    projected CRS where available.
    """
    import pyproj
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_bounds
    tgt = _rio_crs(reproject_to)
    th = horizontal_crs(tgt)
    if not th.is_projected:
        raise ValueError(f"reproject_to must be a projected CRS in metres (got {th.name})")
    srcs = [rasterio.open(p) for p in paths]
    try:
        scrs = [effective_crs(s.name, s.crs, tuple(s.bounds), assume_crs, allow_geographic=True, res=s.res)
                for s in srcs]
        v = vertical_crs(scrs[0])
        for s, c in zip(srcs[1:], scrs[1:]):
            vc = vertical_crs(c)
            if (v is None) != (vc is None) or (v is not None and not v.equals(vc)):
                raise ValueError(f"{s.name}: vertical CRS differs from {paths[0]}")
        out_crs = tgt
        if vertical_crs(tgt) is None and v is not None:       # keep the source's vertical label
            out_crs = _rio_crs(pyproj.CRS(pyproj.crs.CompoundCRS(f"{th.name} + {v.name}", [th, v]).to_wkt()))
        xmin, ymin, xmax, ymax = bounds
        if not np.all(np.isfinite(bounds)):
            raise ValueError(f"requested bounds are not finite: {bounds} (check the AOI's CRS)")
        r = float(resolution)
        want = (math.floor((xmin - buffer_m) / r) * r, math.floor((ymin - buffer_m) / r) * r,
                math.ceil((xmax + buffer_m) / r) * r, math.ceil((ymax + buffer_m) / r) * r)
        sb = [transform_bounds(_rio_crs(horizontal_crs(c)), _rio_crs(th), *s.bounds, densify_pts=21)
              for s, c in zip(srcs, scrs)]
        ux0 = min(b[0] for b in sb); uy0 = min(b[1] for b in sb)
        ux1 = max(b[2] for b in sb); uy1 = max(b[3] for b in sb)
        inputs = f"inputs cover ({ux0:.0f}, {uy0:.0f}, {ux1:.0f}, {uy1:.0f}) in {th.name}"
        if want[2] <= ux0 or want[0] >= ux1 or want[3] <= uy0 or want[1] >= uy1:
            raise ValueError(f"requested window {tuple(round(w) for w in want)} does not overlap the inputs; "
                             f"{inputs}. Check the AOI's CRS.")
        W, H = int(round((want[2] - want[0]) / r)), int(round((want[3] - want[1]) / r))
        if W * H > max_cells:
            raise ValueError(f"requested window is {W * H:.2e} cells (limit {max_cells:.1e}); {inputs}.")
        tr = Affine(r, 0.0, want[0], 0.0, -r, want[3])
        z = np.full((H, W), np.nan, np.float32)
        for s, c in zip(srcs, scrs):
            with WarpedVRT(s, src_crs=_rio_crs(horizontal_crs(c)), crs=_rio_crs(th), transform=tr, width=W,
                           height=H, resampling=Resampling[resampling], src_nodata=s.nodata,
                           nodata=np.nan, dtype="float32") as vrt:
                data = vrt.read(1)
            put = np.isnan(z) & np.isfinite(data)
            z[put] = data[put]
        uncovered = max(ux0 - want[0], uy0 - want[1], want[2] - ux1, want[3] - uy1, 0.0)
        if uncovered > 0:
            warnings.warn(f"Requested window extends {uncovered:.0f} m beyond the input tiles; "
                          "results near that edge may show edge effects. Supply neighbouring tiles.")
        warnings.warn(f"Inputs were reprojected to {th.name} at {r} m ({resampling}); resampling smooths "
                      "micro-relief. Prefer data in a native projected CRS.")
        return Window(z, tr, out_crs, want, float(uncovered), paths)
    finally:
        for s in srcs:
            s.close()


def _mosaic(srcs, want):
    """Read each tile's intersection with ``want`` onto one grid (no resampling, first tile wins).

    Tiles must share the pixel grid of the first tile; a tile that is not aligned to it is refused
    rather than resampled. Their CRS labels may differ in the vertical part only (checked by the
    caller), which rasterio.merge would reject.
    """
    from rasterio.windows import from_bounds
    s0 = srcs[0]
    a, e = s0.transform.a, s0.transform.e
    tr = Affine(a, 0.0, want[0], 0.0, e, want[3])
    W = int(round((want[2] - want[0]) / a))
    H = int(round((want[3] - want[1]) / -e))
    z = np.full((H, W), np.nan, np.float32)
    for s in srcs:
        fx = (s.transform.c - want[0]) / a
        fy = (want[3] - s.transform.f) / -e
        if abs(fx - round(fx)) > 1e-6 or abs(fy - round(fy)) > 1e-6:
            raise ValueError(f"{s.name}: not aligned to the pixel grid of {s0.name}; resample it first")
        x0, y0 = max(want[0], s.bounds.left), max(want[1], s.bounds.bottom)
        x1, y1 = min(want[2], s.bounds.right), min(want[3], s.bounds.top)
        if x1 <= x0 or y1 <= y0:
            continue
        win = from_bounds(x0, y0, x1, y1, transform=s.transform).round_offsets().round_lengths()
        data = s.read(1, window=win, masked=True).astype(np.float32).filled(np.nan)
        if s.nodata is not None and np.isfinite(s.nodata):
            data[data == s.nodata] = np.nan
        c0 = int(round((s.transform.c + win.col_off * a - want[0]) / a))
        r0 = int(round((want[3] - (s.transform.f + win.row_off * e)) / -e))
        h, w = data.shape
        h, w = min(h, H - r0), min(w, W - c0)
        tgt = z[r0:r0 + h, c0:c0 + w]
        put = np.isnan(tgt) & np.isfinite(data[:h, :w])
        tgt[put] = data[:h, :w][put]
    return z, tr


def file_provenance(path, with_hash: bool = False) -> dict:
    st = os.stat(path)
    d = {"path": str(path), "size": st.st_size, "mtime": st.st_mtime}
    with rasterio.open(path) as s:
        d.update(crs=s.crs.to_string() if s.crs else None, res=list(s.res), nodata=s.nodata,
                 tags=s.tags())
    if with_hash:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        d["sha256"] = h.hexdigest()
    return d


def write_stack(path, bands: dict, transform, crs):
    names = list(bands)
    first = bands[names[0]]
    with rasterio.open(path, "w", driver="GTiff", height=first.shape[0], width=first.shape[1],
                       count=len(names), dtype="float32", crs=crs, transform=transform,
                       nodata=np.nan, compress="deflate", predictor=3, tiled=True,
                       blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER") as dst:
        for i, n in enumerate(names, 1):
            dst.write(bands[n].astype(np.float32), i)
            dst.set_band_description(i, n)
