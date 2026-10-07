"""Synthetic 1 m DEM reproducing the SH12 situation, with ground truth.

Layout (x east, y north; rows run south from the top edge):

* Terrain falls to the west; nodata "sea" in the north-west corner.
* ``stream``: small channel (≈4 m wide, 1 m deep) in a valley, flowing west.
* ``river``: larger channel (≈12 m wide, 2 m deep) in a wider valley.
* ``road``: N–S road on embankment (8 m crest, 1:2 batters), graded smoothly
  across both valleys. The DSM keeps the bridge deck; the DEM does not.
    - over the stream the embankment is continuous      -> CULVERT (must detect)
    - over the river the deck is removed (a gap)        -> BRIDGE_REMOVED (must not flag)
* ``track``: 0.2 m high farm track crossing both channels  -> FORD (must not flag)
* ``mound``: round 2 m mound sitting in the stream channel -> not elongated (Test B must fail)
* LiDAR-like noise (σ = 3 cm) creating many micro-pits.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from affine import Affine


@dataclass
class Scene:
    z: np.ndarray
    transform: object
    crs: str
    truth: dict = field(default_factory=dict)   # name -> (x, y) map coordinates
    dsm: np.ndarray | None = None               # DSM: bridge deck present over the river


def _channel_surface(X, Y, yc, bed, half_bottom, depth, valley_depth, valley_width):
    d = np.abs(Y - yc)
    bank = np.clip((d - half_bottom) / 1.0, 0.0, 1.0) * depth          # 1:1 banks
    flood = np.clip(d - half_bottom - 1.0, 0.0, None) * 0.02
    valley = valley_depth * (1.0 - np.exp(-(d / valley_width) ** 2))
    return bed + bank + flood + valley


def make_scene(nx: int = 800, ny: int = 600, cellsize: float = 1.0, seed: int = 0,
               x0: float = 1_600_000.0, y0: float = 6_000_600.0) -> Scene:
    rng = np.random.default_rng(seed)
    xs = (np.arange(nx) + 0.5) * cellsize
    ys = (ny - np.arange(ny) - 0.5) * cellsize          # northing offset, top row = north
    X, Y = np.meshgrid(xs, ys)

    hills = 1.2 * np.sin(X / 170.0) * np.cos(Y / 130.0) + 0.6 * np.sin((X + Y) / 90.0)
    z = 22.0 + 0.012 * X + hills

    ys_c = 150.0 + 20.0 * np.sin(X / 60.0)
    stream = _channel_surface(X, Y, ys_c, 10.0 + 0.012 * X, 1.0, 1.0, 5.0, 45.0)
    yr_c = 430.0 + 15.0 * np.sin(X / 90.0)
    river = _channel_surface(X, Y, yr_c, 8.0 + 0.012 * X, 5.0, 2.0, 6.0, 70.0)
    z = np.minimum(z, np.minimum(stream, river))

    # Round mound in the stream channel (natural blockage, not elongated).
    mx = 150.0
    my = 150.0 + 20.0 * np.sin(mx / 60.0)
    mound = 2.2 * np.exp(-((X - mx) ** 2 + (Y - my) ** 2) / (2 * 7.0**2))
    z = np.where(mound > 0.05, np.maximum(z, 10.0 + 0.012 * mx + mound), z)

    # Farm track at x = 650: 3 m wide, 0.2 m high, following the ground (a ford at channels).
    tx = 650.0
    z = np.where(np.abs(X - tx) <= 1.5, z + 0.2, z)

    # Road: centreline x_r(y), crest graded as a heavily smoothed ground profile.
    xr = 400.0 + 10.0 * np.sin(Y[:, 0] / 200.0)
    col = np.clip(np.round(xr / cellsize - 0.5).astype(int), 0, nx - 1)
    ground = z[np.arange(ny), col]
    w = int(201 / cellsize) | 1
    pad = np.pad(ground, w // 2, mode="edge")
    crest = np.convolve(pad, np.ones(w) / w, mode="valid") + 0.5
    dx = np.abs(X - xr[:, None])
    zc = crest[:, None]
    emb = zc - np.clip(dx - 4.0, 0, None) / 2.0            # fill batters 1:2
    cut = zc + np.clip(dx - 4.0, 0, None) / 1.0            # cut batters 1:1
    road_z = np.where(dx <= 4.0, zc, np.where(emb > z, emb, np.minimum(z, cut)))
    road_z = np.where(dx <= 4.0, zc, np.where(dx <= 40.0, road_z, z))
    yr_at_road = 430.0 + 15.0 * np.sin(xr / 90.0)
    gap = np.abs(Y[:, 0] - yr_at_road) < 14.0              # bridge span removed from the DEM
    deck = np.where(gap[:, None] & (dx <= 4.0), zc + 1.0, -np.inf)   # deck ~1 m above road grade
    z = np.where(gap[:, None], z, road_z)

    noise = rng.normal(0.0, 0.03, z.shape)
    dsm = np.maximum(z, deck) + noise
    z = z + noise
    sea = (X < 60) & (Y > ny * cellsize - 80)
    z[sea] = np.nan
    dsm[sea] = np.nan

    transform = Affine(cellsize, 0.0, x0, 0.0, -cellsize, y0)   # north-up, origin top-left
    ys_at_road = 150.0 + 20.0 * np.sin(xr / 60.0)
    i_c = int(np.argmin(np.abs(Y[:, 0] - ys_at_road)))
    i_b = int(np.argmin(np.abs(Y[:, 0] - yr_at_road)))
    truth = {
        "culvert": (x0 + xr[i_c], y0 - ny * cellsize + Y[i_c, 0]),
        "bridge_removed": (x0 + xr[i_b], y0 - ny * cellsize + Y[i_b, 0]),
        "ford_stream": (x0 + tx, y0 - ny * cellsize + 150.0 + 20.0 * np.sin(tx / 60.0)),
        "ford_river": (x0 + tx, y0 - ny * cellsize + 430.0 + 15.0 * np.sin(tx / 90.0)),
        "mound": (x0 + mx, y0 - ny * cellsize + my),
    }
    return Scene(z.astype(np.float32), transform, "EPSG:2193", truth, dsm.astype(np.float32))


def make_divide_scene(nx: int = 400, ny: int = 300, cellsize: float = 1.0, seed: int = 1,
                      x0: float = 1_600_000.0, y0: float = 6_000_300.0) -> Scene:
    """The SH12 configuration: a culvert on a DEM drainage divide.

    Flat floodplain falling gently *away* from a N-S road embankment (3 m high, 8 m crest, 1:2
    batters) on both sides, so nothing ponds against the road. Ditches (1 m deep, 3 m wide):

    * ``culvert``: at y = 100 a ditch starts at each toe and drains away from the road; a pipe
      joins them under the road. No ponding, so only Test A can find it.
    * ``dead_end``: at y = 200 a ditch ends at the east toe with no channel opposite (not a crossing).
    * ``open_end``: a ditch at y = 250 ending in the open paddock, 60 m from the road (no barrier).
    """
    rng = np.random.default_rng(seed)
    xs = (np.arange(nx) + 0.5) * cellsize
    ys = (ny - np.arange(ny) - 0.5) * cellsize
    X, Y = np.meshgrid(xs, ys)
    xr = 200.0
    dxr = np.abs(X - xr)
    z = 2.0 - 0.002 * dxr                                   # falls away from the road both sides
    base_half = 4.0 + 2.0 * 3.0                             # crest half-width + batter run
    ditch = lambda yc: 1.0 * np.clip(1.0 - np.abs(Y - yc) / 1.5, 0, 1) ** 0.5  # noqa: E731
    # culvert ditches start at the toes and run to the edges
    z = z - np.where(dxr >= base_half, ditch(100.0), 0.0)
    # dead end: east side only
    z = z - np.where(X >= xr + base_half, ditch(200.0), 0.0)
    # open end: east side, from x = xr + 60 to the edge
    z = z - np.where(X >= xr + 60.0, ditch(250.0), 0.0)
    emb = 2.0 + 3.0 - np.clip(dxr - 4.0, 0, None) / 2.0
    z = np.maximum(z, emb)
    z = z + rng.normal(0.0, 0.02, z.shape)
    transform = Affine(cellsize, 0.0, x0, 0.0, -cellsize, y0)
    oy = y0 - ny * cellsize
    truth = {"culvert": (x0 + xr, oy + 100.0), "dead_end": (x0 + xr, oy + 200.0),
             "open_end": (x0 + xr + 60.0, oy + 250.0)}
    return Scene(z.astype(np.float32), transform, "EPSG:2193", truth, z.astype(np.float32))
