"""Crossing candidates (design §4.2) from least-cost breach paths.

Each stored breach path runs pit -> outlet. It is extended upstream along the
main upstream flow path and downstream along D8 flow (both on the breached
DEM), giving a profile across the barrier. From that profile:

* barrier span: contiguous path cells around the deepest cut with cut > threshold
* L_b: span length; crest: highest DEM cell in the span
* h_b = crest - max(z_up, z_dn), beds = minimum DEM within ``bed_window`` either side
* raised: white top-hat (embankment scale) at the crest >= threshold
* elongation: L/W of the raised region around the crest (second moments)
* channel_up / channel_dn: incision at probes beyond the barrier ends

Test B (dammed depression) and a first form of Test A (interrupted channel,
using channel evidence at both ends rather than a vectorised network, which
comes with §4.1) are then evaluated. Everything is in metres (T1).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .config import CandidateParams
from .hydro import BreachResult, Depressions, NODATA, pond_stats


@dataclass
class Context:
    z: np.ndarray                 # original DEM (NaN = nodata)
    transform: object
    breach: BreachResult
    dep: Depressions
    upa: np.ndarray               # upstream area on breached DEM, m²
    idxs_ds: np.ndarray           # pyflwdir downstream index (flat)
    idxs_us_main: np.ndarray      # pyflwdir main upstream index (flat, -1 none)
    feats: dict                   # feature_stack output
    dsm: np.ndarray | None = None
    core: tuple | None = None     # (xmin, ymin, xmax, ymax) to keep; None = all
    source_version: str = ""
    net: object | None = None     # channels.Network, for Test A and the Test B kind

    def __post_init__(self):
        self.valid = np.isfinite(self.z)
        self.z_valid_filled = np.where(self.valid, self.z, NODATA).astype(np.float64)
        self._near = {}

    def channel_near(self, radius_m):
        """Channel map dilated by radius_m (cached)."""
        if radius_m not in self._near:
            cs = abs(self.transform.a)
            dist = ndi.distance_transform_edt(~self.net.mask) * cs
            self._near[radius_m] = dist <= radius_m
        return self._near[radius_m]


def _xy(transform, flat_idx, ncols):
    r, c = np.divmod(np.asarray(flat_idx), ncols)
    return transform.c + (c + 0.5) * transform.a, transform.f + (r + 0.5) * transform.e


def _cumdist(cells, ncols, cellsize):
    r, c = np.divmod(cells, ncols)
    step = np.hypot(np.diff(r), np.diff(c)) * cellsize
    return np.concatenate([[0.0], np.cumsum(step)])


def _walk(start, nxt, max_m, ncols, cellsize):
    out, d, i = [], 0.0, start
    while d < max_m:
        j = int(nxt[i])
        if j < 0 or j == i:
            break
        ri, ci = divmod(i, ncols)
        rj, cj = divmod(j, ncols)
        d += math.hypot(ri - rj, ci - cj) * cellsize
        out.append(j)
        i = j
    return out


def _elongation(mask_win, rc_center):
    """L/W ratio and major-axis angle (rad) of the component containing rc_center."""
    lab, _ = ndi.label(mask_win, structure=np.ones((3, 3), bool))
    l0 = lab[rc_center]
    if l0 == 0:
        ys, xs = np.nonzero(lab)
        if ys.size == 0:
            return 0.0, np.nan, 0
        k = np.argmin((ys - rc_center[0]) ** 2 + (xs - rc_center[1]) ** 2)
        l0 = lab[ys[k], xs[k]]
    ys, xs = np.nonzero(lab == l0)
    if ys.size < 5:
        return 1.0, np.nan, ys.size
    cov = np.cov(np.vstack([xs, -ys]).astype(float))
    w, v = np.linalg.eigh(cov)
    ratio = math.sqrt(max(w[1], 1e-9) / max(w[0], 1e-9))
    angle = math.atan2(v[1, 1], v[0, 1])
    return ratio, angle, ys.size


def candidates(ctx: Context, p: CandidateParams | None = None):
    """Return (all_breaches, candidates) as lists of dict records with geometry."""
    from shapely.geometry import LineString, Point

    p = p or CandidateParams()
    z = ctx.z
    nr, ncols = z.shape
    cs = abs(ctx.transform.a)
    zflat = z.ravel()
    relief = ctx.feats.get(f"relief_med{p.channel_relief_scale_m:g}")
    black = ctx.feats.get("tophat_black21")
    white = ctx.feats[f"tophat_white{p.raised_tophat_scale_m:g}"]
    white_filled = np.nan_to_num(white)
    incision = np.fmax(-relief if relief is not None else -np.inf, black if black is not None else -np.inf)
    inc_flat = incision.ravel()
    white_flat = white.ravel()
    ext_m = p.bed_window_m + p.channel_probe_m + 5.0
    win = int(round(p.elongation_window_m / cs))

    records = []
    br = ctx.breach
    for k in range(len(br.seed)):
        cells, znew = br.path(k)
        if cells.size < 2:
            continue
        cut = zflat[cells] - znew
        up = _walk(int(cells[0]), ctx.idxs_us_main, ext_m, ncols, cs)[::-1]
        dn = _walk(int(cells[-1]), ctx.idxs_ds, ext_m, ncols, cs)
        prof = np.concatenate([np.asarray(up, np.int64), cells, np.asarray(dn, np.int64)])
        d = _cumdist(prof, ncols, cs)
        off = len(up)
        # barrier span around the deepest cut
        imax = int(np.argmax(cut))
        if cut[imax] <= p.cut_threshold_m:
            continue
        s = imax
        while s > 0 and cut[s - 1] > p.cut_threshold_m:
            s -= 1
        e = imax
        while e < cut.size - 1 and cut[e + 1] > p.cut_threshold_m:
            e += 1
        s_p, e_p = s + off, e + off
        span = prof[s_p:e_p + 1]
        zs = zflat[span]
        L_b = d[e_p] - d[s_p] + cs
        ic = int(np.nanargmax(zs))
        crest = int(span[ic])
        z_crest = float(zs[ic])
        zp = zflat[prof]
        up_sel = (d >= d[s_p] - p.bed_window_m) & (d < d[s_p])
        dn_sel = (d > d[e_p]) & (d <= d[e_p] + p.bed_window_m)
        z_up = float(np.nanmin(zp[up_sel])) if up_sel.any() else float(zflat[span[0]])
        z_dn = float(np.nanmin(zp[dn_sel])) if dn_sel.any() else float(zflat[span[-1]])
        h_b = z_crest - max(z_up, z_dn)
        crest_width = float(np.sum(zs >= z_crest - 0.15)) * cs
        # channel evidence at probes beyond the barrier
        pu = (d >= d[s_p] - p.channel_probe_m - 5.0) & (d <= d[s_p] - p.channel_probe_m)
        pd = (d >= d[e_p] + p.channel_probe_m) & (d <= d[e_p] + p.channel_probe_m + 5.0)
        inc_up = float(np.nanmax(inc_flat[prof[pu]])) if pu.any() else np.nan
        inc_dn = float(np.nanmax(inc_flat[prof[pd]])) if pd.any() else np.nan
        ch_up = bool(inc_up >= p.channel_min_incision_m)
        ch_dn = bool(inc_dn >= p.channel_min_incision_m)
        # raised elongated feature around the crest
        raised_h = float(np.nanmax(white_flat[span]))
        raised = raised_h >= p.raised_min_height_m
        rc, cc = divmod(crest, ncols)
        r0, r1 = max(rc - win, 0), min(rc + win + 1, nr)
        c0, c1 = max(cc - win, 0), min(cc + win + 1, ncols)
        wwin = white[r0:r1, c0:c1]
        yy, xx = np.ogrid[r0 - rc:r1 - rc, c0 - cc:c1 - cc]
        disk = (yy**2 + xx**2) <= win**2
        thr = max(p.raised_min_height_m, 0.3 * raised_h)
        elong_mom, _, n_raised = _elongation((wwin > thr) & disk, (rc - r0, cc - c0))
        # path direction across the barrier; elongation by axis continuity, and crossing angle
        xs_, ys_ = _xy(ctx.transform, [span[0], span[-1]], ncols)
        pdir = math.atan2(ys_[1] - ys_[0], xs_[1] - xs_[0])
        zprof_ = zflat[prof]
        w50 = _half_height_width(zprof_, d, s_p + ic, max(z_up, z_dn), h_b, cs)
        if h_b >= p.min_barrier_height_m and L_b < p.max_barrier_length_m and raised:
            elong, angle, run = _axis_elongation(white_filled, (rc, cc), pdir, w50, raised_h, p, cs)
        else:                                          # fails anyway: skip the (slower) crest following
            elong, angle, run = float("nan"), float("nan"), float("nan")
        cross_angle = _angle_between_axes(pdir, angle)
        elongated = bool(np.isfinite(elong) and elong >= p.min_elongation)
        # pond retained by this barrier (not the whole fill-based depression,
        # which would give nested bumps inside a big pond the big pond's size)
        seed = int(cells[0])
        level = min(z_crest, float(zflat[seed] + ctx.dep.depth.ravel()[seed]))
        dep_area, dep_vol, dep_depth, trunc = pond_stats(
            ctx.z_valid_filled, ctx.valid, seed, level, cs, p.pond_max_cells)
        upa_dn = float(ctx.upa.ravel()[span[-1]])
        # tests
        geom_ok = (h_b >= p.min_barrier_height_m) and (L_b < p.max_barrier_length_m) and raised and elongated
        test_b = (geom_ok and float(cut.max()) >= p.min_barrier_height_m
                  and dep_depth >= p.min_depression_depth_m and dep_area >= p.min_depression_area_m2)
        both_ends = geom_ok and ch_up and ch_dn            # local evidence only; the real Test A is test_a()
        kind, veg = _kind_and_veg(ctx, p, prof, d, s_p, e_p, span, angle)
        rec = dict(
            test="B", kind=kind, veg_frac=veg,
            max_cut=float(cut.max()), h_b=h_b, L_b=L_b, z_crest=z_crest, z_up=z_up, z_dn=z_dn,
            crest_width=crest_width, width_half_height=w50, raised_h=raised_h, elongation=elong, raised_run=run,
            elongation_moments=elong_mom, cross_angle=cross_angle, inc_up=inc_up, inc_dn=inc_dn, channel_up=ch_up, channel_dn=ch_dn,
            dep_depth=dep_depth, dep_area=dep_area, dep_volume=dep_vol, pond_truncated=bool(trunc),
            upstream_area=upa_dn,
            breach_length=float(br.length_m[k]), breach_cost=float(br.cost[k]),
            channels_both_ends=bool(both_ends), test_a=False, test_b=bool(test_b),
            vegetation_affected=bool(np.isfinite(veg) and veg >= p.veg_frac),
            # a bank (stopbank / spoil bank beside a channel) or vegetation artefact is kept as a
            # record but is not a crossing candidate (labels, SH12 Oct 2026)
            is_candidate=bool(test_b and (not p.hard_gates or (kind not in ("bank", "channel_obstruction")
                              and not (np.isfinite(veg) and veg >= p.veg_frac)))),
        )
        if ctx.dsm is not None:
            diff = ctx.dsm.ravel()[span] - zs
            rec["dsm_minus_dem_max"] = float(np.nanmax(diff))
            rec["dsm_minus_dem_mean"] = float(np.nanmean(diff))
        # geometry: cutline from bed_window upstream to bed_window downstream
        g_sel = (d >= d[s_p] - p.bed_window_m) & (d <= d[e_p] + p.bed_window_m)
        gx, gy = _xy(ctx.transform, prof[g_sel], ncols)
        cx, cy = _xy(ctx.transform, [crest], ncols)
        rec["crest_x"], rec["crest_y"] = float(cx[0]), float(cy[0])
        rec["geometry"] = LineString(np.column_stack([gx, gy])) if gx.size >= 2 else Point(cx[0], cy[0])
        px, py = _xy(ctx.transform, cells, ncols)
        rec["path_geometry"] = LineString(np.column_stack([px, py]))
        records.append(rec)

    if ctx.core is not None:
        x0, y0, x1, y1 = ctx.core
        records = [r for r in records if x0 <= r["crest_x"] <= x1 and y0 <= r["crest_y"] <= y1]

    cands = _merge([r for r in records if r["is_candidate"]], p.merge_distance_m)
    for r in records + cands:
        key = f"{round(r['crest_x'])}_{round(r['crest_y'])}_{ctx.source_version}"
        r["id"] = hashlib.sha1(key.encode()).hexdigest()[:12]
    return records, cands


def _merge(recs, dist_m):
    """Keep one record per cluster of crests within dist_m (largest depression volume, then cut)."""
    if len(recs) < 2:
        return list(recs)
    xy = np.array([[r["crest_x"], r["crest_y"]] for r in recs])
    order = sorted(range(len(recs)), key=lambda i: (-recs[i]["dep_volume"], -recs[i]["max_cut"]))
    tree = cKDTree(xy)
    taken = np.zeros(len(recs), bool)
    out = []
    for i in order:
        if taken[i]:
            continue
        nb = tree.query_ball_point(xy[i], dist_m)
        taken[nb] = True
        r = dict(recs[i])
        r["n_merged"] = len(nb)
        out.append(r)
    return out


def _half_height_width(zprof, d, ic, base, h_b, cs):
    """Width of the barrier at half height, along the profile, around crest index ic."""
    thr = base + 0.5 * h_b
    s = e = ic
    while s > 0 and zprof[s - 1] > thr:
        s -= 1
    while e < len(zprof) - 1 and zprof[e + 1] > thr:
        e += 1
    return float(d[e] - d[s] + cs)


def _axis_elongation(white, rc, path_angle, width, raised_h, p, cs):
    """Elongation by crest following (cf. design §5.2): from the crest, step along the raised
    feature's axis, re-centring on the highest top-hat value across it, until the feature ends or
    the crest turns more than 60 deg from its starting direction. Elongation = crest run / L_b.
    ``width`` is the barrier width at half height (crest plus upper batters), which barely grows
    with fill height, so tall embankments across narrow valleys are not penalised. The reach is
    1.5 * min_elongation * width + 20 m each side.
    Returns (elongation, axis_angle_rad, run_m)."""
    thr = max(p.raised_min_height_m, 0.3 * raised_h)
    nr, nc = white.shape
    step, snap = 2.0, 3.0
    # local axis window scales with the barrier: a driveway joining a road within a few metres
    # must not be measured as the road (AOI2), while large embankments use up to axis_local_m
    local_m = min(p.axis_local_m, max(4.0, 1.5 * width))
    reach = 1.5 * p.min_elongation * width + 20.0
    offs = np.arange(-snap, snap + cs, cs)

    def val(x, y):                                     # x, y in metres relative to the crest
        r = int(round(rc[0] - y / cs)); c = int(round(rc[1] + x / cs))
        if 0 <= r < nr and 0 <= c < nc:
            return white[r, c]
        return -1.0

    def trace(ax0, sign):
        ux, uy = sign * math.cos(ax0), sign * math.sin(ax0)
        x = y = 0.0
        run = 0.0
        a0 = (ux, uy)
        pts = []                                       # crest points within axis_local_m
        while run < reach:
            nx_, ny_ = x + step * ux, y + step * uy
            px, py = -uy, ux                           # across the axis
            vals = [val(nx_ + o * px, ny_ + o * py) for o in offs]
            # re-centre on the crest, with a small penalty for moving sideways so the trace does
            # not wander across a flat crest on noise; real batters drop far more than this
            k = int(np.argmax([v - p.crest_snap_penalty * abs(o) for v, o in zip(vals, offs)]))
            if vals[k] < thr:
                break
            nx_, ny_ = nx_ + offs[k] * px, ny_ + offs[k] * py
            dx, dy = nx_ - x, ny_ - y
            dn = math.hypot(dx, dy)
            run += dn
            vx, vy = 0.7 * ux + 0.3 * dx / dn, 0.7 * uy + 0.3 * dy / dn
            vn = math.hypot(vx, vy)
            ux, uy = vx / vn, vy / vn
            if ux * a0[0] + uy * a0[1] < 0.5:           # turned more than 60 deg
                break
            x, y = nx_, ny_
            if run <= local_m:
                pts.append((x, y))
        return run, pts

    best = None
    for off in sorted(np.radians(np.arange(-45.0, 45.1, 15.0)), key=abs):   # ties -> nearest perpendicular
        ax = path_angle + np.pi / 2 + off
        r1, p1 = trace(ax, 1.0)
        r2, p2 = trace(ax, -1.0)
        run = r1 + r2
        if best is None or run > best[0] * 1.02:
            # the axis is the principal direction of the traced crest near the crossing (within
            # axis_local_m each side), so a driveway is not measured as the road it joins
            P = np.array([(0.0, 0.0)] + p1 + p2)
            if len(P) >= 3:
                w_, v_ = np.linalg.eigh(np.cov(P.T))
                axis = math.atan2(v_[1, 1], v_[0, 1])
            else:
                axis = ax
            best = (run, axis)
    run, ax = best
    return run / max(width, cs), ax, run


# --------------------------------------------------------------------------- shared helpers
def _veg_fraction(ctx, p, span):
    if ctx.dsm is None or len(span) == 0:
        return float("nan")
    diff = ctx.dsm.ravel()[span] - ctx.z.ravel()[span]
    return float(np.mean(diff > p.veg_dsm_m))


def _channel_axis(sk, rc, radius):
    """Orientation (rad) of skeleton cells within radius of rc, or nan."""
    r0, c0 = rc
    nr, nc = sk.shape
    rs, cs = slice(max(r0 - radius, 0), min(r0 + radius + 1, nr)), slice(max(c0 - radius, 0), min(c0 + radius + 1, nc))
    ys, xs = np.nonzero(sk[rs, cs])
    if ys.size < 4:
        return float("nan")
    w, v = np.linalg.eigh(np.cov(np.vstack([xs, -ys]).astype(float)))
    return math.atan2(v[1, 1], v[0, 1])


def _angle_between_axes(a, b):
    """Acute angle (deg) between two undirected axes."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(np.degrees(abs(math.remainder(a - b, math.pi))))


def _near_channel(mask, flat_cells, ncols, radius):
    if len(flat_cells) == 0:
        return False
    r, c = np.divmod(np.asarray(flat_cells), ncols)
    nr = mask.shape[0]
    for rr, cc in zip(r, c):
        if mask[max(rr - radius, 0):min(rr + radius + 1, nr), max(cc - radius, 0):cc + radius + 1].any():
            return True
    return False


def _kind_and_veg(ctx, p, prof, d, s_p, e_p, span, raised_angle):
    """Test B kind from the channel network: channel_obstruction, crossing, bank or overflow.

    ``channel_obstruction``: the whole cutline lies within ``obstruction_radius_m`` of the channel
    map, i.e. a false barrier inside a channel (typically vegetation over a river or drain). It is
    a DEM-conditioning repair (§6.1), not a structure.
    """
    veg = _veg_fraction(ctx, p, span)
    if ctx.net is None:
        return "unknown", veg
    ncols = ctx.z.shape[1]
    cs = abs(ctx.transform.a)
    near = ctx.channel_near(p.obstruction_radius_m)
    cut = prof[(d >= d[s_p] - p.bed_window_m) & (d <= d[e_p] + p.bed_window_m)]
    if len(cut) and near.ravel()[cut].mean() >= p.obstruction_frac:
        return "channel_obstruction", veg
    rad = max(1, int(round(3.0 / cs)))
    pu = prof[(d >= d[s_p] - p.channel_probe_m - 5.0) & (d <= d[s_p] - p.channel_probe_m)]
    pd = prof[(d >= d[e_p] + p.channel_probe_m) & (d <= d[e_p] + p.channel_probe_m + 5.0)]
    up = _near_channel(ctx.net.mask, pu, ncols, rad)
    dn = _near_channel(ctx.net.mask, pd, ncols, rad)
    if up and dn:
        return "crossing", veg
    if dn and len(pd):
        r, c = divmod(int(pd[len(pd) // 2]), ncols)
        ch = _channel_axis(ctx.net.skeleton, (r, c), int(round(10.0 / cs)))
        if _angle_between_axes(ch, raised_angle) <= p.bank_parallel_deg:
            return "bank", veg
    return "overflow", veg


# --------------------------------------------------------------------------- Test A
def test_a(ctx: Context, p: CandidateParams | None = None):
    """Evaluate Test A (§4.2) on every gap-bridging path of ctx.net. Returns records."""
    from shapely.geometry import LineString

    p = p or CandidateParams()
    if ctx.net is None:
        return []
    z = ctx.z
    zf = z.ravel()
    nr, ncols = z.shape
    cs = abs(ctx.transform.a)
    white = ctx.feats[f"tophat_white{p.raised_tophat_scale_m:g}"]
    wf = np.nan_to_num(white).ravel()
    win = int(round(p.elongation_window_m / cs))
    out = []
    for pi, path in enumerate(ctx.net.paths):
        if path.size < 3:
            continue
        end_k = ctx.net.path_end[pi]
        zs = zf[path]
        d = _cumdist(path, ncols, cs)
        up = d <= p.bed_window_m
        dn = d >= d[-1] - p.bed_window_m
        z_up, z_dn = float(np.nanmin(zs[up])), float(np.nanmin(zs[dn]))
        ic = int(np.nanargmax(zs))
        z_crest = float(zs[ic])
        base = max(z_up, z_dn)
        h_b = z_crest - base
        if h_b < p.min_barrier_height_m:
            continue
        thr = base + max(0.1, p.barrier_frac * h_b)
        s = e = ic
        while s > 0 and zs[s - 1] > thr:
            s -= 1
        while e < len(zs) - 1 and zs[e + 1] > thr:
            e += 1
        span = path[s:e + 1]
        L_b = float(d[e] - d[s] + cs)
        raised_h = float(wf[span].max())
        crest = int(path[ic])
        rc, cc = divmod(crest, ncols)
        r0, r1 = max(rc - win, 0), min(rc + win + 1, nr)
        c0, c1 = max(cc - win, 0), min(cc + win + 1, ncols)
        yy, xx = np.ogrid[r0 - rc:r1 - rc, c0 - cc:c1 - cc]
        disk = (yy**2 + xx**2) <= win**2
        rthr = max(p.raised_min_height_m, 0.3 * raised_h)
        elong_mom, _, _ = _elongation((np.nan_to_num(white[r0:r1, c0:c1]) > rthr) & disk, (rc - r0, cc - c0))
        xs_, ys_ = _xy(ctx.transform, [path[s], path[e]], ncols)
        pdir = math.atan2(ys_[1] - ys_[0], xs_[1] - xs_[0])
        w50 = _half_height_width(zs, d, ic, base, h_b, cs)
        elong, angle, run = _axis_elongation(np.nan_to_num(white), (rc, cc), pdir, w50, raised_h, p, cs)
        cross = _angle_between_axes(pdir, angle)
        crest_width = float(np.sum(zf[span] >= z_crest - 0.15)) * cs
        upa = ctx.upa.ravel()
        # kind: at a crossing at least one channel approaches the barrier transversely; if both
        # run along it, the path only hops a bank between parallel channels (design §5.3 stopbank)
        rad = int(round(10.0 / cs))
        approach = ctx.net.skeleton & ~ctx.net.toe if ctx.net.toe is not None else ctx.net.skeleton

        def ch_angle(cell):
            rc_ = divmod(int(cell), ncols)
            a = _channel_axis(approach, rc_, rad)      # the approaching channel, toe ditches excluded
            if not np.isfinite(a):
                a = _channel_axis(ctx.net.skeleton, rc_, rad)
            return _angle_between_axes(a, angle)

        # start of the path: the channel end itself - its approach over up to 15 m
        ax_, ay_ = ctx.net.ends.approach[end_k]
        ang_up = _angle_between_axes(math.atan2(ay_, ax_), angle)
        approach_len = float(ctx.net.ends.approach_len[end_k])
        ang_dn = ch_angle(path[-1])                    # target side: the channel reached
        rc_t = divmod(int(path[-1]), ncols)
        ax_t = _channel_axis(approach, rc_t, rad)
        if not np.isfinite(ax_t):
            ax_t = _channel_axis(ctx.net.skeleton, rc_t, rad)
        ang_cont = _angle_between_axes(math.atan2(ay_, ax_), ax_t)   # approach vs channel beyond
        # A crossing needs the channel end itself to approach the barrier transversely over
        # >= min_approach_m. (Allowing a transverse *target* instead admitted short stubs; all
        # five such SH12 candidates were labelled "no culvert", Oct 2026.) Or the channel
        # *continues* through: both sides >= continuation_min_deg off the barrier axis and within
        # continuation_max_dev_deg of each other (oblique stream crossings; a bank hop between
        # channels running along the barrier still fails).
        start_transverse = approach_len >= p.min_approach_m and ang_up >= p.transverse_deg
        continuation = (approach_len >= p.min_approach_m and ang_up >= p.continuation_min_deg
                        and np.isfinite(ang_dn) and ang_dn >= p.continuation_min_deg
                        and np.isfinite(ang_cont) and ang_cont <= p.continuation_max_dev_deg)
        kind = "crossing" if (start_transverse or continuation) else "bank"
        rec = dict(
            test="A", kind=kind, channel_angle_up=ang_up, channel_angle_dn=ang_dn, approach_length=approach_len,
            continuation_angle=ang_cont, continuation=bool(continuation),
            veg_frac=_veg_fraction(ctx, p, span),
            max_cut=h_b, h_b=h_b, L_b=L_b, z_crest=z_crest, z_up=z_up, z_dn=z_dn,
            crest_width=crest_width, width_half_height=w50, raised_h=raised_h, elongation=elong, raised_run=run,
            elongation_moments=elong_mom, cross_angle=cross, upstream_area=float(max(upa[path[0]], upa[path[-1]])), path_length=float(d[-1]),
            dep_depth=0.0, dep_area=0.0, dep_volume=0.0,
        )
        min_L = p.min_barrier_length_m if p.hard_gates else 0.0
        rec["test_a"] = bool(raised_h >= p.raised_min_height_m and elong >= p.min_elongation
                             and min_L <= L_b < p.max_barrier_length_m
                             and (np.isfinite(cross) and cross >= p.min_cross_angle_deg))
        rec["test_b"] = False
        veg = rec["veg_frac"]
        rec["vegetation_affected"] = bool(np.isfinite(veg) and veg >= p.veg_frac)
        rec["is_candidate"] = rec["test_a"] and (not p.hard_gates or (not rec["vegetation_affected"]
                                                                       and kind == "crossing"))
        if ctx.dsm is not None:
            diff = ctx.dsm.ravel()[span] - zf[span]
            rec["dsm_minus_dem_max"] = float(np.nanmax(diff))
            rec["dsm_minus_dem_mean"] = float(np.nanmean(diff))
        cx, cy = _xy(ctx.transform, [crest], ncols)
        rec["crest_x"], rec["crest_y"] = float(cx[0]), float(cy[0])
        px, py = _xy(ctx.transform, path, ncols)
        rec["geometry"] = LineString(np.column_stack([px, py]))
        rec["path_geometry"] = rec["geometry"]
        out.append(rec)
    if ctx.core is not None:
        x0, y0, x1, y1 = ctx.core
        out = [r for r in out if x0 <= r["crest_x"] <= x1 and y0 <= r["crest_y"] <= y1]
    return out


def combine(b_cands, a_recs, merge_distance_m):
    """Merge Test A and Test B candidates; a location found by both keeps both flags."""
    a_cands = _merge([r for r in a_recs if r["is_candidate"]], merge_distance_m)
    out = [dict(r) for r in b_cands]
    tree = cKDTree(np.array([[r["crest_x"], r["crest_y"]] for r in b_cands])) if b_cands else None
    for r in a_cands:
        if tree is not None:
            dist, k = tree.query([r["crest_x"], r["crest_y"]])
            if dist <= merge_distance_m:
                out[k]["test_a"] = True
                out[k]["test"] = "A+B"
                continue
        out.append(dict(r))
    return out
