"""Area averaging from 1 m to 2, 4 or 8 m (option 1 only; plan 6.2 step 3).

The Stage 1 grid is snapped to multiples of the output resolution and the LINZ 1 m grid has whole-metre
edges, so every output cell is exactly a k x k block of 1 m cells and averaging is a block mean (equivalent
to ``Resampling.average`` on aligned grids, without resampling). Cells with under ``min_valid`` (50 %) valid
input are no data. Feature-aware aggregation to model grids is Stage 3 (P-10).
"""
from __future__ import annotations

import numpy as np


def _blocks(a: np.ndarray, k: int):
    h, w = a.shape
    if h % k or w % k:
        raise ValueError(f"array shape {a.shape} is not a multiple of the block size {k}")
    return a.reshape(h // k, k, w // k, k)


def block_mean(z: np.ndarray, k: int, min_valid: float = 0.5) -> np.ndarray:
    """Mean of each k x k block over valid (finite) cells; NaN where the valid share is below ``min_valid``."""
    if k == 1:
        return np.asarray(z, np.float32).copy()
    b = _blocks(np.asarray(z, np.float64), k)
    valid = np.isfinite(b)
    n = valid.sum(axis=(1, 3))
    s = np.where(valid, b, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / n
    out[n < min_valid * k * k] = np.nan
    return out.astype(np.float32)


def block_mode(idx: np.ndarray, k: int, nodata: int = -1) -> np.ndarray:
    """Most frequent value of each block, ignoring ``nodata`` (ties: the smallest value, i.e. the higher
    priority survey). Used for ``lidar_source``."""
    if k == 1:
        return np.asarray(idx).copy()
    a = np.asarray(idx).astype(np.int64)
    b = _blocks(a, k).transpose(0, 2, 1, 3).reshape(a.shape[0] // k, a.shape[1] // k, k * k)
    vals = np.unique(a[a != nodata])
    if vals.size == 0:
        return np.full(b.shape[:2], nodata, dtype=a.dtype)
    counts = np.stack([(b == v).sum(axis=2) for v in vals], axis=-1)
    best = vals[np.argmax(counts, axis=-1)]
    return np.where(counts.max(axis=-1) > 0, best, nodata)


def aggregate(z: np.ndarray, source: np.ndarray | None, k: int, min_valid: float = 0.5):
    """(z, source) at k times the cell size; ``source`` is no data wherever ``z`` is."""
    zk = block_mean(z, k, min_valid)
    if source is None:
        return zk, None
    sk = block_mode(np.where(np.isfinite(z), source, -1), k)
    sk = np.where(np.isfinite(zk), sk, -1)
    return zk, sk
