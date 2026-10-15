"""ML-3a step 3: move council channel lines onto the channel bed in the DEM.

Council lines are drawn up to tens of metres off the bed (docs/ml3_channel_model.md, section 2). Each line is moved
to the least-cost path through a corridor of +-``corridor_m`` around it, where the cost per metre falls with the bed
depth (the black top-hats: depth of incised features narrower than 11 m, and half that of 21 m, which also picks up
the broad hollows of paddocks) and rises with the distance from the council line:

    cost = base + exp(-depth / depth_scale_m) + lateral * (distance / corridor_m) ** 2

Long lines are snapped in pieces of ``piece_m`` (each piece starting where the last one ended), so memory and time
stay small whatever the line's length. The path's share of cells on a bed (11 m top-hat >= ``bed_depth_m``; on
BW24_10000_0402, 0.10 m is the 90th percentile off the drains) decides whether the line counts as snapped; lines that are not keep their council geometry and are flagged (the label
rasters down-weight them).

The check (``bed_check``) is independent of the cost: at points along a line, across it (+-``check_half_m``), the
DEM's lowest point should lie within ``check_tolerance_m`` of the line, or, where the cross-section has real banks
(relief >= ``check_relief_m``), the line should lie in its lowest quarter (a wide, flat bed has no single lowest
point: Environment Southland's scheme drains, 17 Oct).
"""
from __future__ import annotations

import numpy as np

DEFAULTS = dict(corridor_m=15.0, piece_m=200.0, base=0.02, depth_scale_m=0.15, lateral=0.3, bed_depth_m=0.10,
                min_bed_frac=0.6, simplify_m=0.75, check_step_m=5.0, check_half_m=6.0, check_tolerance_m=2.0,
                check_relief_m=0.2, check_low_share=0.25)
DEPTH_BANDS = ("tophat_black11", "tophat_black21")


def _band(bands, name, shape=None):
    v = bands.get(name)
    if v is None:
        return np.zeros(shape, np.float32)
    v = np.asarray(v, dtype=np.float32)
    return np.where(np.isfinite(v), np.maximum(v, 0.0), 0.0)


def bed_depth(bands: dict):
    """(depth for the cost, depth for the bed test), m: max(11 m top-hat, half the 21 m top-hat), and the 11 m
    top-hat alone (the 21 m one is high in paddock hollows: median 0.04 m, 90th percentile 0.18 m off the drains)."""
    b11 = _band(bands, "tophat_black11", np.shape(bands["tophat_black21"]) if "tophat_black21" in bands else None)
    b21 = _band(bands, "tophat_black21", b11.shape)
    return np.maximum(b11, 0.5 * b21), b11


def _rc(transform, x, y):
    return (np.floor((np.asarray(y) - transform.f) / transform.e).astype(int),
            np.floor((np.asarray(x) - transform.c) / transform.a).astype(int))


def _xy(transform, r, c):
    return transform.c + (np.asarray(c) + 0.5) * transform.a, transform.f + (np.asarray(r) + 0.5) * transform.e


def _pieces(line, piece_m):
    from shapely.ops import substring
    n = max(1, int(np.ceil(line.length / piece_m)))
    cuts = np.linspace(0.0, line.length, n + 1)
    return [substring(line, a, b) for a, b in zip(cuts[:-1], cuts[1:])]


def _snap_piece(piece, depth, transform, P, start_rc=None):
    """Least-cost path for one piece; returns (rows, cols) on the full grid, or None when the piece is off the grid."""
    from rasterio.features import rasterize
    from scipy import ndimage as ndi
    from skimage.graph import MCP_Geometric
    cs = abs(transform.a)
    R = P["corridor_m"]
    x0, y0, x1, y1 = piece.buffer(R + 2 * cs).bounds
    r0, c0 = _rc(transform, x0, y1)
    r1, c1 = _rc(transform, x1, y0)
    r0, c0 = max(int(r0), 0), max(int(c0), 0)
    r1, c1 = min(int(r1) + 1, depth.shape[0]), min(int(c1) + 1, depth.shape[1])
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform
    wt = window_transform(Window(c0, r0, c1 - c0, r1 - r0), transform)
    on = rasterize([(piece, 1)], out_shape=(r1 - r0, c1 - c0), transform=wt, all_touched=True, dtype=np.uint8) > 0
    if not on.any():
        return None
    dist = ndi.distance_transform_edt(~on) * cs
    corridor = dist <= R
    d = depth[r0:r1, c0:c1]
    cost = P["base"] + np.exp(-d / P["depth_scale_m"]) + P["lateral"] * (dist / R) ** 2
    cost = np.where(corridor, cost, np.inf)

    def disc(pt):
        rr, cc = _rc(wt, *pt)
        yy, xx = np.ogrid[:cost.shape[0], :cost.shape[1]]
        m = ((yy - rr) ** 2 + (xx - cc) ** 2) * cs * cs <= R * R
        return [tuple(p) for p in np.argwhere(m & corridor)]

    starts = [(start_rc[0] - r0, start_rc[1] - c0)] if start_rc is not None else disc(piece.coords[0])
    starts = [s for s in starts if 0 <= s[0] < cost.shape[0] and 0 <= s[1] < cost.shape[1] and corridor[s]]
    ends = disc(piece.coords[-1])
    if not starts or not ends:
        return None
    mcp = MCP_Geometric(cost, fully_connected=True)
    cum, _ = mcp.find_costs(starts, ends, find_all_ends=False)
    e = min(ends, key=lambda p: cum[p])
    if not np.isfinite(cum[e]):
        return None
    path = np.array(mcp.traceback(e))
    return path[:, 0] + r0, path[:, 1] + c0


def snap_line(line, depth, transform, params=None, bed=None):
    """(snapped LineString or None, info: bed_frac, offset_med_m) for one council line. ``depth`` drives the cost,
    ``bed`` (default ``depth``) the bed test."""
    from shapely.geometry import LineString
    P = dict(DEFAULTS, **(params or {}))
    rows, cols, start = [], [], None
    for piece in _pieces(line, P["piece_m"]):
        got = _snap_piece(piece, depth, transform, P, start)
        if got is None:
            start = None
            continue
        r, c = got
        if rows and start is not None:
            r, c = r[1:], c[1:]                       # the joint is the previous piece's last cell
        rows.append(r); cols.append(c)
        start = (int(got[0][-1]), int(got[1][-1]))
    if not rows:
        return None, dict(bed_frac=np.nan, offset_med_m=np.nan)
    r, c = np.concatenate(rows), np.concatenate(cols)
    if len(r) < 2:
        return None, dict(bed_frac=np.nan, offset_med_m=np.nan)
    on_bed = (depth if bed is None else bed)[r, c] >= P["bed_depth_m"]
    x, y = _xy(transform, r, c)
    g = LineString(np.column_stack([x, y]))
    from shapely import points
    off = line.distance(points(np.column_stack([x, y])))
    return g.simplify(P["simplify_m"]), dict(bed_frac=float(on_bed.mean()), offset_med_m=float(np.median(off)))


def snap_lines(lines, depth, transform, params=None, bed=None):
    """Snap every line; adds ``snapped`` (bool), ``bed_frac``, ``offset_med_m`` and keeps ``geometry_council``.
    A line counts as snapped when its path lies on a bed for at least ``min_bed_frac`` of its length; otherwise it
    keeps the council geometry."""
    P = dict(DEFAULTS, **(params or {}))
    out = lines.copy()
    out["geometry_council"] = out.geometry
    geo, snapped, frac, off = [], [], [], []
    for g in out.geometry:
        s, info = snap_line(g, depth, transform, P, bed)
        ok = s is not None and np.isfinite(info["bed_frac"]) and info["bed_frac"] >= P["min_bed_frac"]
        geo.append(s if ok else g); snapped.append(bool(ok))
        frac.append(info["bed_frac"]); off.append(info["offset_med_m"])
    out["snapped"], out["bed_frac"], out["offset_med_m"] = snapped, frac, off
    out = out.set_geometry(geo, crs=lines.crs)
    return out


def share_on(line, grid, transform, step=1.0):
    """Share of a line's sample points (every ``step`` m) on True cells of ``grid``; NaN off the grid."""
    n = max(int(np.ceil(line.length / step)), 1)
    pts = [line.interpolate(d) for d in np.linspace(0.0, line.length, n + 1)]
    r, c = _rc(transform, [p.x for p in pts], [p.y for p in pts])
    ok = (r >= 0) & (r < grid.shape[0]) & (c >= 0) & (c < grid.shape[1])
    return float(np.mean(grid[r[ok], c[ok]])) if ok.any() else np.nan


def bed_check(line, dem, transform, params=None):
    """Share of a line's check points on the bed: across the line (+-check_half_m), the DEM's lowest point lies within
    check_tolerance_m of it, or the cross-section has relief >= check_relief_m and the line lies in its lowest
    check_low_share of that relief. NaN when no point is on the grid. Independent of the snapping cost."""
    from scipy.ndimage import map_coordinates
    P = dict(DEFAULTS, **(params or {}))
    cs = abs(transform.a)
    n = int(line.length // P["check_step_m"])
    if n < 1:
        return np.nan
    offs = np.arange(-P["check_half_m"], P["check_half_m"] + cs / 2, cs / 2)
    hits = []
    for s in (np.arange(n) + 0.5) * P["check_step_m"]:
        p, q = line.interpolate(max(s - 1.0, 0)), line.interpolate(min(s + 1.0, line.length))
        t = np.array([q.x - p.x, q.y - p.y])
        if not np.linalg.norm(t):
            continue
        t /= np.linalg.norm(t)
        m = line.interpolate(s)
        xs, ys = m.x - offs * t[1], m.y + offs * t[0]
        c = (xs - transform.c) / transform.a - 0.5
        r = (ys - transform.f) / transform.e - 0.5
        z = map_coordinates(dem, [r, c], order=1, mode="constant", cval=np.nan)
        if not np.isfinite(z).all():
            continue
        lo, hi = float(z.min()), float(z.max())
        near = abs(offs[int(np.argmin(z))]) <= P["check_tolerance_m"]
        z0 = float(z[int(np.argmin(np.abs(offs)))])
        low = (hi - lo) >= P["check_relief_m"] and (z0 - lo) <= P["check_low_share"] * (hi - lo)
        hits.append(near or low)
    return float(np.mean(hits)) if hits else np.nan
