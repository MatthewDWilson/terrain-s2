"""Network graph: reaches continue straight through junctions; smoothing reduces vertices;
classification is propagated along the network."""
import numpy as np
from affine import Affine

from terrain_s2 import network


def _t_junction():
    sk = np.zeros((60, 120), bool)
    sk[30, 5:115] = True                # a straight channel W-E
    for r in range(31, 55):             # a side branch joining from the south
        sk[r, 60] = True
    return sk, Affine(1, 0, 0, 0, -1, 60)


def test_reaches_continue_straight_through_junction():
    sk, T = _t_junction()
    nodes, edges, reaches = network.build_reaches(sk, T)
    lengths = sorted(round(np.hypot(*np.diff(R.coords, axis=0).T).sum()) for R in reaches)
    assert len(reaches) == 2
    assert lengths[-1] >= 105 and 20 <= lengths[0] <= 30        # through channel, side branch
    assert all(len(R.coords) <= 6 for R in reaches)              # generalised, not one node per cell


def test_smoothing_pulls_weak_reach_to_neighbours():
    import pandas as pd

    class R:
        def __init__(self, a, b):
            self.node_start, self.node_end = a, b
    reaches = [R(0, 1), R(1, 2), R(2, 3)]                        # a chain: stream - ? - stream
    F = pd.DataFrame(dict(length_m=[200, 30, 200], bed_start=[3.0, 2.0, 1.0], bed_end=[2.0, 1.0, 0.0]))
    p = network.smooth_on_graph(reaches, F, np.array([0.95, 0.3, 0.95]))
    assert p[1] > 0.5                                            # the weak drain becomes a stream
    assert p[0] > 0.9 and p[2] > 0.9


def test_clean_reaches_drops_isolated_and_short_dangling():
    class R:
        def __init__(self, a, b, length):
            self.node_start, self.node_end = a, b
            self.coords = np.array([[0.0, 0.0], [length, 0.0]])
    reaches = [R(0, 1, 200), R(1, 2, 150),     # main channel through junction 1
               R(1, 3, 10),                    # short dangling spur at junction 1 -> dropped
               R(4, 5, 30),                    # isolated short piece -> dropped
               R(6, 7, 80), R(7, 8, 10)]       # small shallow network (90 m) -> dropped when shallow
    inc = np.array([0.8, 0.8, 0.8, 0.8, 0.2, 0.2])
    kept = network.clean_reaches(reaches, incision=inc)
    assert kept == [0, 1]
    assert 4 in network.clean_reaches(reaches, incision=np.array([0.8] * 6))   # deep enough: kept


def test_floodplain_hand():
    """A valley: river at the bottom, slopes rising 1 m per 10 m. HAND <= 5 m within ~50 m."""
    from terrain_s2 import floodplain, hydro
    x = np.abs(np.arange(201) - 100.0)
    z = np.tile(10 + 0.1 * x, (150, 1)) + np.linspace(1.0, 0.0, 150)[:, None]   # draining south
    T = Affine(1, 0, 0, 0, -1, 150)
    upa, flw = hydro.flow_accumulation(z, T)
    river = np.zeros(z.shape, bool); river[:, 98:103] = True
    fp, hand = floodplain.floodplain_mask(z, flw, upa, river, 1.0, max_hand_m=5.0, min_drain_upa_m2=1e12)
    row = fp[75]
    assert row[100] and row[60] and row[140] and not row[20] and not row[180]
