#!/usr/bin/env python
"""Make labelling materials from a run: candidate points (GeoJSON) and image galleries.

    python scripts/review_candidates.py --run out/sh12 --dem data/AW27.tif --dsm data/AW27_dsm.tif \
        [--labels previous_review.geojson ...] --out out/sh12/review

Writes:
  candidates_for_review.geojson   one point per candidate at its crest, numbered as in the
                                  quick-look, with blank `what_is_it` / `culvert_y_n_unsure` fields.
                                  Labels from earlier reviews within 10 m are carried across.
  gallery_test_A.png, gallery_test_B.png, gallery_channel_obstructions.png
                                  80 m chips: hillshade, 0.25 m contours, vegetation > 2 m (green),
                                  path or cutline (red), crest (white).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from terrain_s2 import dataio  # noqa: E402
from terrain_s2.quicklook import hillshade  # noqa: E402

FIELDS = ["num", "test", "kind", "id", "crest_x", "crest_y", "h_b", "L_b", "width_half_height",
          "crest_width", "raised_h", "raised_run", "elongation", "cross_angle", "approach_length",
          "veg_frac", "dsm_minus_dem_max", "upstream_area", "dep_area", "prev_label", "prev_culvert"]


def gallery(G, z, dsm, T, fname, title, half=40):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(G) == 0:
        return
    cols = 6
    rows = max(1, math.ceil(len(G) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(24, 4.2 * rows), constrained_layout=True, squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    for ax, (_, rw) in zip(axs.flat, G.iterrows()):
        ax.axis("on")
        c, r = ~T @ (rw.crest_x, rw.crest_y)
        r, c = int(r), int(c)
        rs, cs = slice(max(r - half, 0), r + half), slice(max(c - half, 0), c + half)
        x0 = T.c + cs.start * T.a
        y1 = T.f + rs.start * T.e
        zz = z[rs, cs]
        ext = (x0, x0 + zz.shape[1] * T.a, y1 + zz.shape[0] * T.e, y1)
        ax.imshow(hillshade(zz, abs(T.a), vexag=3), cmap="gray", extent=ext)
        if dsm is not None:
            veg = dsm[rs, cs] - zz
            ax.imshow(np.where(veg > 2, 1, np.nan), cmap="Greens", vmin=0, vmax=1.5, alpha=0.35, extent=ext)
        X = np.linspace(ext[0] + .5, ext[1] - .5, zz.shape[1])
        Y = np.linspace(ext[3] - .5, ext[2] + .5, zz.shape[0])
        if np.isfinite(zz).any():
            ax.contour(X, Y, zz, levels=np.arange(np.floor(np.nanmin(zz)), np.nanmax(zz), 0.25),
                       colors="y", linewidths=0.3)
        xs, ys = rw.geometry.xy if rw.geometry.geom_type == "LineString" else ([rw.crest_x], [rw.crest_y])
        ax.plot(xs, ys, "r", lw=2.5)
        ax.plot(rw.crest_x, rw.crest_y, "wo", ms=5, mec="k")
        prev = f" [{rw.prev_culvert}]" if isinstance(rw.get("prev_culvert", ""), str) and rw.get("prev_culvert") else ""
        ax.set_title(f"#{rw.num} [{rw.test}]{prev} ({rw.crest_x:.0f}, {rw.crest_y:.0f})\n"
                     f"h={rw.h_b:.1f} L={rw.L_b:.0f} cross={rw.cross_angle:.0f}° veg={rw.veg_frac:.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(ext[:2])
        ax.set_ylim(ext[2:])
    fig.suptitle(title, fontsize=13)
    fig.savefig(fname, dpi=75)
    plt.close(fig)


def main():
    import geopandas as gpd
    import pandas as pd
    from scipy.spatial import cKDTree

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="output folder of run_aoi.py")
    ap.add_argument("--dem", nargs="+", required=True)
    ap.add_argument("--dsm", nargs="*", default=None)
    ap.add_argument("--labels", nargs="*", default=[], help="earlier review GeoJSONs to carry labels from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--assume-crs", help="relabel inputs with this CRS (no reprojection)")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gpkg = run / "candidates.gpkg"
    cand = gpd.read_file(gpkg, layer="candidates")
    order = sorted(range(len(cand)), key=lambda i: (cand.test[i] != "A+B", cand.test[i] != "A", -cand.h_b[i]))
    cand = cand.iloc[order].reset_index(drop=True)     # same numbering as the quick-look
    cand.insert(0, "num", range(1, len(cand) + 1))
    try:
        obs = gpd.read_file(gpkg, layer="channel_obstructions")
        obs = obs.sort_values("h_b", ascending=False).reset_index(drop=True)
        obs.insert(0, "num", range(1, len(obs) + 1))
    except Exception:  # noqa: BLE001
        obs = None
    cand["prev_label"], cand["prev_culvert"] = "", ""
    if a.labels:
        L = pd.concat([gpd.read_file(f) for f in a.labels], ignore_index=True)
        L = L[L.get("what_is_it", "").astype(str).str.len() > 0]
        if len(L):
            dd, kk = cKDTree(L[["crest_x", "crest_y"]].values).query(cand[["crest_x", "crest_y"]].values)
            cand["prev_label"] = np.where(dd <= 10, L.what_is_it.values[kk], "")
            cand["prev_culvert"] = np.where(dd <= 10, L.culvert_y_n_unsure.values[kk], "")
    pts = gpd.GeoDataFrame(cand[[f for f in FIELDS if f in cand.columns]].copy(),
                           geometry=gpd.points_from_xy(cand.crest_x, cand.crest_y), crs=cand.crs)
    pts["what_is_it"], pts["culvert_y_n_unsure"] = "", ""
    pts.to_file(out / "candidates_for_review.geojson", driver="GeoJSON")

    b = tuple(cand.total_bounds) if len(cand) else None
    with __import__("rasterio").open(a.dem[0]) as s:
        full = tuple(s.bounds)
    win = dataio.read_window(a.dem, full if b is None else (b[0] - 60, b[1] - 60, b[2] + 60, b[3] + 60),
                             assume_crs=a.assume_crs)
    dsm = dataio.read_window(a.dsm, win.bounds, assume_crs=a.assume_crs).z if a.dsm else None
    gallery(cand[cand.test != "B"], win.z, dsm, win.transform, out / "gallery_test_A.png",
            "Test A candidates (80 m chips)")
    gallery(cand[cand.test == "B"], win.z, dsm, win.transform, out / "gallery_test_B.png",
            "Test B candidates (80 m chips)")
    if obs is not None and len(obs):
        ob = obs.head(36).copy()
        ob["test"] = "obstruction"
        gallery(ob, win.z, dsm, win.transform, out / "gallery_channel_obstructions.png",
                f"Channel obstructions (largest {len(ob)} of {len(obs)}): false barriers inside channels")
    n = cand.test.value_counts().to_dict()
    print(f"wrote {out}: {len(cand)} candidates {n}; carried labels: {(cand.prev_label != '').sum()}")


if __name__ == "__main__":
    main()
