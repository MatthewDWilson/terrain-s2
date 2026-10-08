"""Relative elevation model: continuity and values on a synthetic sloping valley."""
import numpy as np
from affine import Affine

from terrain_s2 import rem

T = Affine(1.0, 0, 0, 0, -1.0, 200.0)


def test_rem_on_a_sloping_valley_is_height_above_the_channel():
    yy, xx = np.mgrid[0:200, 0:200]
    z = 20.0 + 0.02 * (200 - yy) + 0.05 * np.abs(xx - 100.5)            # channel along x = 100, falling south
    drain = np.zeros(z.shape, bool)
    drain[:, 100] = True
    r, base, n = rem.rem(z, T, drain, sample_m=10, coarse_m=20)
    assert n == 20
    side = r[80:120, 60]                                                  # 40 m west of the channel, mid-valley
    assert np.allclose(side, 0.05 * 40.5, atol=0.05)                     # the valley slope is removed
    # within ~60 m of the window edge the k nearest samples are one-sided and the REM is biased (here up to
    # ~1 m at the edge): inside the 200 m buffer of a site or tile, so the core is not affected
    assert np.abs(r[70:130, 60] - 0.05 * 40.5).max() < 0.05
    assert np.isfinite(r).all()                                          # defined everywhere
    jumps = np.abs(np.diff(base, axis=1)).max()
    assert jumps < 0.1                                                    # smooth across columns


def test_rem_without_drainage_is_nan():
    z = np.ones((10, 10))
    r, base, n = rem.rem(z, T, np.zeros(z.shape, bool))
    assert n == 0 and np.isnan(r).all()
