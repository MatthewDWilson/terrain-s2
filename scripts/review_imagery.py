#!/usr/bin/env python
"""Cheap check of LINZ aerial imagery against the channel reviews, before imagery is tried as U-Net input.

    python scripts/review_imagery.py                          # every <data>/labels/manual/review_*.gpkg
    python scripts/review_imagery.py --prefer newest --out <data>/labels/manual/review_imagery_newest.csv

The question: does the imagery separate the model's real drains from its false detections where the LiDAR
measures (sharpness, building cover: ``review_features.py``) cannot, above all the sharp rural false lines on
BH41 (road edges, fences, tracks?). Nothing is trained.

Per tile, the imagery surveys over it (``acquire/imagery.py``: LINZ nz-imagery, aerial RGB or RGBNIR, gsd at
most --max-gsd), ordered by capture closest to the tile's LiDAR survey (``--prefer lidar``, the default: what a
U-Net would be given) or newest first (``--prefer newest``: closest to the basemap the reviews were answered
on, so partly circular). Per reviewed reach, points every --step m along it and, at each, the imagery across
it: the core (--core m either side of the line) against both banks (--ring m out on each side), after one
lateral shift per reach (within +-``--shift`` m, chosen by the brightness line contrast) for the
misregistration between imagery and LiDAR. Indices: brightness, excess green ((2G - R - B) / (R + G + B)) and
NDVI where the survey has NIR, each standardised per tile and survey (median, MAD over the reaches' windows)
so surveys of different colour compare. Measures per index:

    _core, _side    levels on the line and on the banks (standardised)
    _line           core against both banks the same way (darker or greener than both); 0 where the banks
                    disagree (a step: a road edge or field boundary), median over the points
    _absline        the size of _line whichever way
    _edge           bank against bank (the step's size)
    _twoside        share of points where the core differs from both banks the same way, by at least
                    --min-contrast (standardised units)
    _consistent     the same, agreeing in sign with the reach's median _line

Then, on the model-only reaches with a y / n answer (on a model component, as ``review_features.py``): each
measure's AUC (probability that a real drain scores higher; < 0.5 means lower), with a bootstrap 95 % interval,
overall, per review, on the reaches the current filter keeps (--filter: sharpness and building cover, from
``review_features.csv``), and per tile for the strongest; then filter + one imagery threshold, the threshold
and its direction chosen leaving each tile out in turn (thresholds tuned on the same 6 tiles flatter; the
in-sample figure is printed beside it). Writes --out (CSV, one row per reviewed reach, geometry as WKT);
``--save-windows`` writes each reach's imagery window (GeoTIFF, uint8) for a look in QGIS.
"""
import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("review_features", ROOT / "scripts" / "review_features.py")
RF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RF)

MEASURES = ("core", "side", "line", "absline", "edge", "twoside", "consistent")
H = 0.5                                                # spacing of the samples across the reach (m)


def lidar_info(bundle: Path):
    """(DEM capture (start, end) or None, the regions of the tile's elevation surveys or None)."""
    try:
        site = json.loads((bundle / "site.json").read_text())
    except (OSError, ValueError):
        return None, None
    cap, regions = None, set()
    for rec in site.get("elevation") or []:
        if rec.get("product") == "dem" and rec.get("capture"):
            cap = tuple(rec["capture"])
        for s in rec.get("surveys") or []:
            if s.get("region") and not s.get("national"):
                regions.add(s["region"])
    return cap, (sorted(regions) or None)


def cross_points(geom, step, offsets):
    """Sample coordinates across a line: (X, Y) of shape (points, offsets), along normals every ``step`` m."""
    import numpy as np
    if geom.geom_type == "MultiLineString":
        from shapely.ops import linemerge
        geom = linemerge(geom)
        if geom.geom_type == "MultiLineString":
            geom = max(geom.geoms, key=lambda g: g.length)
    L = geom.length
    d = np.arange(step / 2, L, step) if L > step else np.array([L / 2])
    p = np.array([geom.interpolate(x).coords[0][:2] for x in d])
    a = np.array([geom.interpolate(max(0.0, x - 1.0)).coords[0][:2] for x in d])
    b = np.array([geom.interpolate(min(L, x + 1.0)).coords[0][:2] for x in d])
    t = b - a
    n = np.hypot(t[:, 0], t[:, 1])
    n[n == 0] = 1.0
    t /= n[:, None]
    nx, ny = -t[:, 1], t[:, 0]                          # left of the direction of travel
    return p[:, 0, None] + nx[:, None] * offsets[None, :], p[:, 1, None] + ny[:, None] * offsets[None, :]


def sample(read, X, Y):
    """Bilinear samples of each band at (X, Y): {band: array}, NaN off the imagery; and the survey index
    at the centre points (nearest)."""
    import numpy as np
    from scipy import ndimage as ndi
    T = read.transform
    col = (X - T.c) / T.a - 0.5
    row = (Y - T.f) / T.e - 0.5
    out = {}
    for k, name in enumerate(read.bands):
        out[name] = ndi.map_coordinates(read.data[k], [row, col], order=1, mode="constant", cval=np.nan)
    sv = ndi.map_coordinates(read.survey.astype(np.float32), [row, col], order=0, mode="constant", cval=-1)
    return out, sv


def indices(b):
    """Brightness, excess green and (with NIR) NDVI from sampled or pixel bands (0-255)."""
    import numpy as np
    R, G, B = b["R"], b["G"], b["B"]
    s = R + G + B
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {"bright": s / 765.0, "exg": np.where(s > 0, (2 * G - R - B) / s, np.nan)}
        if "NIR" in b and np.isfinite(b["NIR"]).any():
            N = b["NIR"]
            out["ndvi"] = np.where(N + R > 0, (N - R) / (N + R), np.nan)
    return out


def line_contrast(c, l, r):
    import numpy as np
    dl, dr = c - l, c - r
    same = (np.sign(dl) == np.sign(dr)) & (dl != 0)
    v = np.where(same, np.sign(dl) * np.minimum(np.abs(dl), np.abs(dr)), 0.0)
    return np.where(np.isfinite(dl) & np.isfinite(dr), v, np.nan)


def profile_parts(V, offsets, shift, core, ring):
    """Per point: the core, left and right bank means of V (points x offsets) at a lateral shift."""
    import numpy as np
    o = offsets - shift
    cm = np.abs(o) <= core + 1e-9
    lm = (o >= ring[0] - 1e-9) & (o <= ring[1] + 1e-9)
    rm = (o <= -ring[0] + 1e-9) & (o >= -ring[1] - 1e-9)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(V[:, cm], 1), np.nanmean(V[:, lm], 1), np.nanmean(V[:, rm], 1)


def reach_measures(ix, offsets, shift_m, core, ring, stats, survey_ix, min_points, min_contrast=0.5):
    """The measures of one reach from its sampled indices ``ix`` ({name: points x offsets}, raw values)."""
    import warnings

    import numpy as np
    rec = {}
    z = {}
    for k, V in ix.items():
        med, mad = stats.get((survey_ix, k), (np.nan, np.nan))
        z[k] = (V - med) / mad if np.isfinite(med) and mad > 0 else V * np.nan
    shifts = np.arange(-shift_m, shift_m + H / 2, H)
    best, bs = -1.0, 0.0
    for s in shifts:                                    # one shift per reach, by the brightness line contrast
        c, l, r = profile_parts(z["bright"], offsets, s, core, ring)
        ln = line_contrast(c, l, r)
        if np.isfinite(ln).sum() >= min_points:
            v = abs(float(np.nanmedian(ln)))
            if v > best + 1e-9 or (abs(v - best) <= 1e-9 and abs(s) < abs(bs)):
                best, bs = v, float(s)
    rec["img_shift_m"] = bs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k, V in z.items():
            c, l, r = profile_parts(V, offsets, bs, core, ring)
            ln = line_contrast(c, l, r)
            ok = np.isfinite(ln)
            if ok.sum() < min_points:
                for m in MEASURES:
                    rec[f"{k}_{m}"] = np.nan
                continue
            med = float(np.nanmedian(ln))
            rec.update({f"{k}_core": float(np.nanmedian(c)), f"{k}_side": float(np.nanmedian((l + r) / 2)),
                        f"{k}_line": med, f"{k}_absline": float(np.nanmedian(np.abs(ln))),
                        f"{k}_edge": float(np.nanmedian(np.abs(l - r))),
                        f"{k}_twoside": float((np.abs(ln[ok]) >= min_contrast).mean()),
                        f"{k}_consistent": float(((np.sign(ln[ok]) == np.sign(med))
                                                  & (np.abs(ln[ok]) >= min_contrast)).mean())})
    return rec


def tile_imagery(tile, rows, bundles, src, a, save_dir=None):
    """Imagery measures for one tile's reviewed reaches: (records, survey summary)."""
    import numpy as np
    from terrain_s2 import grid
    from terrain_s2.acquire import imagery as IM
    cap, regions = lidar_info(Path(bundles) / tile)
    target = IM.midpoint(cap) if cap else None
    prefer = "date" if a.prefer == "lidar" and target is not None else "newest"
    summary = dict(tile=tile, lidar_capture=f"{cap[0][:10]}..{cap[1][:10]}" if cap else None, regions=regions,
                   prefer=prefer)
    try:
        surveys = src.surveys(grid.tile_bounds(tile), regions=regions, prefer=prefer, target=target)
    except Exception as e:  # noqa: BLE001 - recorded, the tile is left without imagery
        summary["error"] = str(e)[:200]
        surveys = []
    summary["surveys"] = [f"{s.slug} ({s.product}, {s.capture[0][:7]}..{s.capture[1][:7]})" for s in surveys]
    if not surveys:
        return [dict(r, img_survey=None) for r in rows.drop(columns="geometry").to_dict("records")], summary
    r1 = a.ring[1] + a.shift
    offsets = np.arange(-r1, r1 + H / 2, H)
    g = rows.to_crs(2193)
    opened, samples, pix = {}, [], {}
    for rec, geom in zip(g.drop(columns="geometry").to_dict("records"), g.geometry):
        x0, y0, x1, y1 = geom.bounds
        m = r1 + 2.0
        read = src.read(surveys, (x0 - m, y0 - m, x1 + m, y1 + m), res=a.res, _open=opened)
        X, Y = cross_points(geom, a.step, offsets)
        b, sv = sample(read, X, Y)
        centre = sv[:, len(offsets) // 2]
        centre = centre[centre >= 0]
        six = int(np.bincount(centre.astype(int)).argmax()) if len(centre) else -1
        samples.append((rec, geom, b, six, len(X)))
        v = read.valid
        for si in np.unique(read.survey[v]):            # pixels of the window for the standardisation
            sel = np.flatnonzero((read.survey == si).ravel() & v.ravel())
            sel = sel[np.random.default_rng(len(samples)).permutation(len(sel))[:4000]]
            pix.setdefault(int(si), []).append(read.data.reshape(len(read.bands), -1)[:, sel])
        if save_dir is not None and v.any():
            _save_window(save_dir / f"{tile}_{rec.get('review_id', len(samples))}_{rec.get('answer', '')}.tif", read)
    IM.close(opened)
    stats = {}
    for si, parts in pix.items():
        arr = np.concatenate(parts, 1)
        bands = {n: arr[k] for k, n in enumerate(read.bands)}
        for k, v in indices(bands).items():
            v = v[np.isfinite(v)]
            if len(v):
                med = float(np.median(v))
                stats[(si, k)] = (med, float(np.median(np.abs(v - med)) * 1.4826) or 1.0)
    by_ix = {s.index: s for s in surveys}
    out = []
    for rec, geom, b, six, npts in samples:
        s = by_ix.get(six)
        rec = dict(rec, img_survey=s.slug if s else None, img_product=s.product if s else None,
                   img_gsd=s.gsd if s else None, img_capture=f"{s.capture[0][:10]}..{s.capture[1][:10]}" if s else None,
                   img_points=npts, wkt=geom.wkt)
        if s is not None and target is not None and IM.midpoint(s.capture) is not None:
            rec["img_gap_years"] = round(abs((IM.midpoint(s.capture) - target).days) / 365.25, 1)
        ok = np.isfinite(b["R"][:, len(offsets) // 2])
        rec["img_valid"] = float(ok.mean()) if npts else 0.0
        if s is not None:
            rec.update(reach_measures(indices(b), offsets, a.shift, a.core, a.ring, stats, six,
                                      max(3, int(0.5 * npts)), getattr(a, "min_contrast", 0.5)))
        out.append(rec)
    summary["used"] = sorted({r["img_survey"] for r in out if r.get("img_survey")})
    return out, summary


def _save_window(path, read):
    import numpy as np
    import rasterio
    path.parent.mkdir(parents=True, exist_ok=True)
    d = np.nan_to_num(read.data[:3], nan=0).clip(0, 255).astype("uint8")
    with rasterio.open(path, "w", driver="GTiff", height=d.shape[1], width=d.shape[2], count=3, dtype="uint8",
                       crs="EPSG:2193", transform=read.transform, compress="deflate", photometric="RGB") as f:
        f.write(d)


def auc_ci(y, n, higher=True, boot=1000, seed=0):
    import numpy as np
    y, n = np.asarray(y, float), np.asarray(n, float)
    y, n = y[np.isfinite(y)], n[np.isfinite(n)]
    a = RF.auc(y, n, higher)
    if not len(y) or not len(n) or boot <= 0:
        return a, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = [RF.auc(rng.choice(y, len(y)), rng.choice(n, len(n)), higher) for _ in range(boot)]
    return a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def best_threshold(y, n):
    """(score, threshold, direction) maximising keeps-y + drops-n - 1 on y / n values; direction "high" keeps
    values >= threshold, "low" keeps <= threshold. A reach without the measure is kept."""
    import numpy as np
    y, n = np.asarray(y, float), np.asarray(n, float)
    vals = np.unique(np.round(np.concatenate([y[np.isfinite(y)], n[np.isfinite(n)]]), 4))
    top = (-1.0, None, None)
    for v in vals:
        for d in ("high", "low"):
            ky = np.mean(_keep(y, v, d)) if len(y) else 0.0
            dn = 1 - np.mean(_keep(n, v, d)) if len(n) else 0.0
            if ky + dn - 1 > top[0] + 1e-12:
                top = (ky + dn - 1, float(v), d)
    return top


def _keep(x, v, d):
    import numpy as np
    x = np.asarray(x, float)
    with np.errstate(invalid="ignore"):
        k = x >= v if d == "high" else x <= v
    return k | ~np.isfinite(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviews", help="glob of review GeoPackages (default <data>/labels/manual/review_*.gpkg)")
    ap.add_argument("--bundles")
    ap.add_argument("--features", help="review_features.py's CSV (default <data>/labels/manual/review_features.csv)")
    ap.add_argument("--prefer", choices=["lidar", "newest"], default="lidar",
                    help="survey order: capture closest to the tile's LiDAR (default) or newest first")
    ap.add_argument("--res", type=float, default=0.5, help="imagery grid (m)")
    ap.add_argument("--max-gsd", type=float, default=0.5, help="leave out surveys coarser than this (m)")
    ap.add_argument("--step", type=float, default=2.0, help="points along the reach every this (m)")
    ap.add_argument("--core", type=float, default=1.0, help="core half-width (m)")
    ap.add_argument("--ring", type=float, nargs=2, default=(5.0, 8.0), help="banks from / to (m) either side")
    ap.add_argument("--shift", type=float, default=2.0, help="lateral shift searched for registration (m)")
    ap.add_argument("--min-contrast", type=float, default=0.5,
                    help="line contrast (standardised) a point needs to count in _twoside and _consistent")
    ap.add_argument("--filter", type=float, nargs=2, default=(0.15, 0.05), metavar=("MIN_SHARPNESS", "MAX_URBAN"),
                    help="the current ML-3c filter, for the reaches it keeps")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples for the AUC intervals (0: none)")
    ap.add_argument("--top", type=int, default=6, help="measures shown per tile and in the rules")
    ap.add_argument("--save-windows", metavar="DIR", help="write each reach's imagery window here (GeoTIFF)")
    ap.add_argument("--out")
    a = ap.parse_args()
    import numpy as np
    import pandas as pd
    from terrain_s2.acquire.elevation import default_data_dir
    from terrain_s2.acquire.imagery import ImagerySource
    data = default_data_dir()
    bundles = Path(a.bundles or data / "tiles")
    manual = data / "labels" / "manual"
    files = sorted(glob.glob(a.reviews or str(manual / "review_*.gpkg")))
    if not files:
        ap.error("no review files")
    rv = RF.load_reviews(files)
    print(f"{len(files)} review files, {len(rv)} confident answers on {rv.tile.nunique()} tiles")
    src = ImagerySource(max_gsd=a.max_gsd)
    save = Path(a.save_windows) if a.save_windows else None
    rows, summ = [], []
    for t, g in rv.groupby("tile"):
        r, s = tile_imagery(t, g, bundles, src, a, save)
        rows += r
        summ.append(s)
        print(f"{t}: LiDAR {s.get('lidar_capture')}; imagery used {', '.join(s.get('used') or []) or 'none'}"
              + (f"  ERROR {s['error']}" if s.get("error") else ""))
    d = pd.DataFrame(rows)
    fpath = Path(a.features) if a.features else manual / "review_features.csv"
    key = [c for c in ("review", "review_id") if c in d]
    if fpath.exists() and "review_id" in key:
        f = pd.read_csv(fpath)
        cols = [c for c in ("sharpness", "urban_reach", "prob_on_reach", "component") if c in f]
        if "review_id" in f and "review" in f:
            d = d.merge(f[key + cols], on=key, how="left")
    else:
        print(f"note: {fpath} not found (or no review_id): run review_features.py first for the filter's subset")
    out = Path(a.out) if a.out else manual / ("review_imagery.csv" if a.prefer == "lidar" else f"review_imagery_{a.prefer}.csv")
    d.to_csv(out, index=False)

    # --- the surveys
    print("\nimagery per tile (survey of most reaches; gap = capture midpoints, imagery against LiDAR):")
    cols = [c for c in ("review", "tile", "img_survey", "img_product", "img_gsd", "img_capture", "img_gap_years") if c in d]
    if "img_survey" in d and d.img_survey.notna().any():
        t = d[d.img_survey.notna()].groupby(cols[:4], dropna=False).agg(
            reaches=("answer", "size"), gap_years=("img_gap_years", "median") if "img_gap_years" in d else ("answer", "size"),
            shift_m=("img_shift_m", lambda v: float(np.median(np.abs(v)))), valid=("img_valid", "median"))
        print(t.round(2).to_string())
    else:
        print("no imagery read for any tile")
        print(f"\n-> {out}")
        return
    mo = d[(d.kind == "model_only") & d.answer.isin(["y", "n"])]
    if "component" in mo:
        mo = mo[mo.component.notna()]
    ms = [c for c in d.columns if c.split("_", 1)[-1] in MEASURES and c.split("_", 1)[0] in ("bright", "exg", "ndvi")]
    ms = [m for m in ms if mo[m].notna().any()]
    have_filter = {"sharpness", "urban_reach"} <= set(mo.columns)
    if have_filter:
        keep_f = (mo.sharpness.fillna(-1) >= a.filter[0]) & (mo.urban_reach.fillna(1) <= a.filter[1])
        kept = mo[keep_f]
    y, n = mo[mo.answer == "y"], mo[mo.answer == "n"]
    print(f"\nmodel-only reaches: {len(y)} y, {len(n)} n"
          + (f"; the filter (sharpness >= {a.filter[0]}, building cover <= {a.filter[1]}) keeps "
             f"{int((kept.answer == 'y').sum())} y, {int((kept.answer == 'n').sum())} n" if have_filter else ""))

    # --- AUC per measure: overall, per review, on the filter's survivors
    def aucs(t):
        yy, nn = t[t.answer == "y"], t[t.answer == "n"]
        return {m: auc_ci(yy[m], nn[m], True, a.boot) for m in ms}
    groups = [("all", mo)] + [(r, g) for r, g in mo.groupby("review")] + ([("kept", kept)] if have_filter else [])
    res = {name: aucs(g) for name, g in groups}
    print("\nAUC (P(real > false); < 0.5: real lower) [95 % bootstrap]; columns: " +
          ", ".join(f"{name} {int((g.answer == 'y').sum())}y/{int((g.answer == 'n').sum())}n" for name, g in groups))
    order = sorted(ms, key=lambda m: -abs(res["kept" if have_filter else "all"][m][0] - 0.5)
                   if np.isfinite(res["kept" if have_filter else "all"][m][0]) else 0)
    w = max(len(m) for m in ms)
    for m in order:
        cells = []
        for name, _ in groups:
            v, lo, hi = res[name][m]
            cells.append(f"{v:.2f} [{lo:.2f}-{hi:.2f}]" if np.isfinite(v) and np.isfinite(lo) else
                         (f"{v:.2f}" if np.isfinite(v) else "  -  "))
        print(f"{m:{w}s}  " + "   ".join(f"{c:17s}" for c in cells))
    if have_filter:
        print("for comparison (LiDAR, review_features.py): sharpness and building cover on the same reaches:")
        for m, hi_ in (("sharpness", True), ("urban_reach", False)):
            v = RF.auc(y[m], n[m], hi_)
            print(f"  {m}: AUC {v:.2f} ({'higher' if hi_ else 'lower'} for real)")

    # --- per tile, the strongest measures
    top = order[:a.top]
    print("\nper tile (AUC; tiles with both answers): " + ", ".join(top))
    for t, g in mo.groupby("tile"):
        yy, nn = g[g.answer == "y"], g[g.answer == "n"]
        if not len(yy) or not len(nn):
            print(f"  {t} ({len(yy)}y/{len(nn)}n): one answer only")
            continue
        print(f"  {t} ({len(yy)}y/{len(nn)}n): " + "  ".join(f"{RF.auc(yy[m], nn[m]):.2f}" for m in top))

    # --- filter + one imagery threshold, leaving each tile out
    if have_filter:
        base_y, base_n = keep_f[mo.answer == "y"].mean(), 1 - keep_f[mo.answer == "n"].mean()
        print(f"\nfilter + one imagery threshold (applied to the reaches the filter keeps): keeps y, drops n over all "
              f"{len(y)} y / {len(n)} n model-only reaches\n  filter alone: keeps {base_y:.0%}, drops {base_n:.0%}")
        print(f"  {'measure':{w}s}  in-sample (rule)                  leave-one-tile-out")
        for m in top:
            sc, v, dirn = best_threshold(kept[kept.answer == "y"][m], kept[kept.answer == "n"][m])
            k_in = keep_f.to_numpy() & _keep(mo[m], v, dirn) if v is not None else keep_f.to_numpy()
            ky_in, dn_in = k_in[(mo.answer == "y").to_numpy()].mean(), 1 - k_in[(mo.answer == "n").to_numpy()].mean()
            k_lo = keep_f.to_numpy().copy()
            for t in mo.tile.unique():
                tr = kept[kept.tile != t]
                _, v2, d2 = best_threshold(tr[tr.answer == "y"][m], tr[tr.answer == "n"][m])
                sel = (mo.tile == t).to_numpy()
                if v2 is not None:
                    k_lo[sel] &= _keep(mo[m].to_numpy()[sel], v2, d2)
            ky_lo, dn_lo = k_lo[(mo.answer == "y").to_numpy()].mean(), 1 - k_lo[(mo.answer == "n").to_numpy()].mean()
            rule = f"{'>=' if dirn == 'high' else '<='} {v:.3g}" if v is not None else "-"
            print(f"  {m:{w}s}  keeps {ky_in:.0%}, drops {dn_in:.0%} ({rule:9s})     keeps {ky_lo:.0%}, drops {dn_lo:.0%}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
