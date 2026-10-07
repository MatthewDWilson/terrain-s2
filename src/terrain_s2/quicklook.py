"""Quick-look PNG: hillshade with candidates, and breach depth with flow accumulation."""
from __future__ import annotations

import numpy as np


def hillshade(z, cellsize, azimuth=315.0, altitude=35.0, vexag=2.0):
    zz = np.where(np.isfinite(z), z, np.nanmin(z)) * vexag
    gy, gx = np.gradient(zz, cellsize)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(azimuth), np.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.where(np.isfinite(z), np.clip(hs, 0, 1), np.nan)


def render(path, z, transform, breaches, candidates, breach_depth, upa, aoi_bounds=None,
           title="", min_cut=0.3, channels=None, test_a=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cs = abs(transform.a)
    ext = (transform.c, transform.c + z.shape[1] * transform.a,
           transform.f + z.shape[0] * transform.e, transform.f)
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    hs = hillshade(z, cs)
    for ax in axs:
        ax.imshow(hs, cmap="gray", extent=ext, interpolation="nearest")
        if aoi_bounds is not None:
            x0, y0, x1, y1 = aoi_bounds
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "c--", lw=1)
        ax.set_aspect("equal")
    colours = {"A": "red", "B": "orange", "A+B": "magenta"}
    ax = axs[0]
    for i, r in enumerate(sorted(candidates, key=lambda r: (r["test"] != "A+B", r["test"] != "A", -r["h_b"])), 1):
        x, y = r["geometry"].xy
        ax.plot(x, y, color=colours.get(r["test"], "red"), lw=3)
        ax.annotate(f"{i}", (r["crest_x"], r["crest_y"]), color=colours.get(r["test"], "red"),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.legend(handles=[Line2D([], [], color=c, lw=3, label=f"Test {k}") for k, c in colours.items()],
              loc="lower left", fontsize=8)
    ax.set_title(f"{title}\ncrossing candidates ({len(candidates)}), numbered")
    ax = axs[1]
    if channels is not None:
        from scipy import ndimage as ndi
        grow = max(1, int(max(z.shape) / 600) | 1)
        ch = ndi.maximum_filter(channels.astype(np.uint8), size=grow) > 0
        ax.imshow(np.where(ch, 1.0, np.nan), cmap="winter", extent=ext, alpha=0.9, interpolation="nearest")
    for r in (test_a or []):
        x, y = r["geometry"].xy
        ax.plot(x, y, color=("red" if r.get("is_candidate") else "yellow"), lw=1.2, alpha=0.9)
    ax.set_title("channel map (blue); Test A bridging paths: passed (red), failed (yellow)")
    fig.savefig(path, dpi=130)
    plt.close(fig)
