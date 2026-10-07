"""Floodplain mask from the DEM alone: height above nearest drainage (HAND; Nobre et al., 2016).

Drainage = wide channels (rivers) from the channel map, plus cells with D8 upstream area >=
``min_drain_upa_m2`` on the breached DEM. Floodplain = HAND <= ``max_hand_m`` (default 10 m: 5 m
fell short of the true floodplain at AOI2, where the drainage reference is itself incomplete), with small islands
removed and small holes filled. Processing (channel network, crossings) is restricted to the
floodplain plus a buffer, which removes hillside false positives and is where the information is
most critical for flood modelling. Hill-country track culverts are therefore out of scope.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def floodplain_mask(z, flw, upa, wide_channels, cellsize, max_hand_m=10.0, min_drain_upa_m2=1e5,
                    min_area_m2=1e4):
    drain = np.nan_to_num(upa) >= min_drain_upa_m2
    if wide_channels is not None:
        drain |= wide_channels
    if not drain.any():
        return np.zeros(z.shape, bool), np.full(z.shape, np.nan)
    hand = flw.hand(drain=drain, elevtn=np.nan_to_num(z, nan=-9999.0).astype(np.float32))
    hand = np.where(np.isfinite(z), hand, np.nan)
    fp = np.nan_to_num(hand, nan=np.inf) <= max_hand_m
    lab, n = ndi.label(fp)
    area = np.bincount(lab.ravel(), minlength=n + 1) * cellsize * cellsize
    keep = area >= min_area_m2
    keep[0] = False
    fp = keep[lab]
    holes = ~fp
    lab, n = ndi.label(holes)
    area = np.bincount(lab.ravel(), minlength=n + 1) * cellsize * cellsize
    small = area < min_area_m2
    small[0] = False
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    small[edge] = False                   # only enclosed holes; land running off the window stays out
    fp |= small[lab]
    return fp, hand
