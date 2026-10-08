"""Conditioned hydrology on a synthetic valley: a channel blocked by an embankment, with and without
a culvert link, and REC2 inflow at the window edge."""
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString

from terrain_s2 import conditioning

T = Affine(1.0, 0, 0, 0, -1.0, 100.0)          # 100 x 100 cells of 1 m, y from 100 down to 0


def valley(embankment=True):
    """Valley falling south (1 cm/m), sides rising 5 cm/m, a 1 m deep channel down the middle (x = 50),
    a high north edge (so water cannot leave that way), and an east-west embankment 2 m high across the
    valley at y = 40 (rows 59-61): its lowest crest is over the channel."""
    yy, xx = np.mgrid[0:100, 0:100]
    z = 10.0 + 0.01 * (100 - yy) + 0.05 * np.abs(xx - 50.5)
    z[:, 49:52] -= 1.0
    z[0:2, :] += 5.0
    if embankment:
        z[59:62, :] += 2.0
    return z


CHANNEL_N = LineString([(50.5, 99.5), (50.5, 42.5)])      # north of the embankment
CHANNEL_S = LineString([(50.5, 37.5), (50.5, 0.5)])       # south of it
LINK = LineString([(50.5, 42.5), (50.5, 37.5)])           # the culvert through it


def test_blocked_channel_ponds_and_spills_over_the_embankment():
    c = conditioning.run(valley(), T, 2193, [CHANNEL_N, CHANNEL_S], [])
    inflow = c.ponds[c.ponds.inflow]
    assert len(inflow) == 1 and inflow.max_depth_m.iloc[0] > 1.5           # ponds up to the crest
    o = c.outlets[c.outlets.inflow].geometry.iloc[0]
    assert 37 <= o.y <= 43                                                    # spills at the embankment
    assert np.isnan(c.hand[30, 50]) or c.hand[30, 50] >= 0


def test_culvert_link_drains_the_pond():
    c = conditioning.run(valley(), T, 2193, [CHANNEL_N, CHANNEL_S], [LINK])
    assert not len(c.ponds) or not c.ponds.inflow.any()
    assert c.drain[60, 50]                                                    # the link is network
    assert c.z_conditioned[60, 50] < valley()[58, 50]                         # invert below the bed upstream
    up = c.upa.reshape(100, 100)
    assert up[65, 50] > up[55, 50] > 1000                                     # flow passes through the link
    assert np.nanmax(np.abs(c.hand[np.asarray(c.drain)])) == pytest.approx(0, abs=1e-6)


def test_rec2_inflow_seeds_upstream_area_at_the_edge():
    import geopandas as gpd
    rec2 = gpd.GeoDataFrame({"up_x": [50.5], "up_y": [150.0], "down_x": [50.5], "down_y": [50.0],
                             "upstream_area_m2": [5e6], "catchment_area_m2": [1e6]},
                            geometry=[LineString([(50.5, 150.0), (50.5, 50.0)])], crs=2193)
    c = conditioning.run(valley(False), T, 2193, [LineString([(50.5, 99.5), (50.5, 0.5)])], [], rec2=rec2)
    assert c.n_seeded == 1 and c.upa_inflow.sum() == pytest.approx(4e6)
    assert c.upa.reshape(100, 100)[5, 50] > 4e6                               # carried downstream
