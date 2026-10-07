"""Drain network: bed-profile bumps and gap bed rise on a synthetic roadside drain."""
import numpy as np
from affine import Affine

from terrain_s2 import drains


def _drain_scene(driveway=True):
    """Flat ground at 10 m; an E-W drain 1 m deep, 2 m wide, 200 m long; optionally a 6 m wide
    driveway filling the drain to ground level at x = 100 m."""
    z = np.full((41, 220), 10.0)
    z[19:22, 10:210] = 9.0
    if driveway:
        z[19:22, 97:103] = 10.0
    sk = np.zeros(z.shape, bool)
    sk[20, 10:210] = True                       # centreline straight through, as after a join
    return z + np.random.default_rng(0).normal(0, 0.01, z.shape), sk, Affine(1, 0, 0, 0, -1, 41)


def test_bump_at_driveway():
    z, sk, T = _drain_scene(True)
    b = drains.bed_bumps(sk, z, T)
    assert len(b) == 1
    assert abs(b[0]["x"] - 100) <= 4 and 0.8 < b[0]["h_b"] < 1.2 and b[0]["length"] <= 10


def test_no_bump_without_crossing():
    z, sk, T = _drain_scene(False)
    assert drains.bed_bumps(sk, z, T) == []


def test_gap_bed_rise():
    z, _, T = _drain_scene(True)
    rise, _ = drains.gap_bed_rise(z, T, (92.5, 20.5), (107.5, 20.5))   # drain ends either side
    assert 0.8 < rise < 1.2
    z2, _, _ = _drain_scene(False)
    rise2, _ = drains.gap_bed_rise(z2, T, (92.5, 20.5), (107.5, 20.5))
    assert rise2 < 0.1
