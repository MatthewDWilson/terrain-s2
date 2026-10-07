import numpy as np
import pytest

from terrain_s2 import features, hydro
from terrain_s2.backend import get_backend
from terrain_s2.config import HydroParams


def test_breach_simple_dam():
    """A 1-D valley dammed by a 2 m wall: one breach, cut ≈ wall height above the pit."""
    nr, nc = 21, 60
    z = np.tile(np.linspace(10.0, 4.0, nc), (nr, 1))
    z += np.abs(np.arange(nr) - nr // 2)[:, None] * 0.5     # V valley along the row axis
    z[:, 30:33] += 3.0                                      # wall across the valley
    br = hydro.breach_least_cost(z.astype(np.float32), 1.0, HydroParams(breach_store_min_cut_m=0.1))
    assert br.n_unresolved == 0
    k = int(np.argmax(br.max_cut_m))
    assert 2.5 < br.max_cut_m[k] < 3.5
    cells, _ = br.path(k)
    assert np.all(np.isin(np.arange(30, 33), cells % nc))   # path crosses the wall


def test_breach_limit_leaves_pit():
    z = np.full((41, 41), 10.0, np.float32)
    z[20, 20] = 5.0                                         # deep pit 20 cells from any outlet
    br = hydro.breach_least_cost(z, 1.0, HydroParams(breach_max_length_m=5.0))
    assert br.n_unresolved == 1


def test_flat_pit_handled_once():
    z = np.full((30, 30), 10.0, np.float32)
    z[10:20, 10:20] = 9.0                                   # flat-bottomed pit of 100 cells
    br = hydro.breach_least_cost(z, 1.0)
    assert br.n_pits == 1 and br.n_unresolved == 0


def test_openness_v_valley():
    be = get_backend("cpu")
    x = np.abs(np.arange(101) - 50).astype(np.float32)
    z = np.tile(x, (101, 1))                                # V valley, 45° sides
    po, no = features.openness(z, np.ones_like(z, bool), 20.0, 1.0, be)
    assert po[50, 50] < po[50, 20]                          # valley floor is less open above
    assert no[50, 50] > no[50, 20] - 1e-3


def test_tophat_plane_is_zero():
    be = get_backend("cpu")
    yy, xx = np.mgrid[0:200, 0:200].astype(np.float32)
    z = 0.05 * xx + 0.02 * yy
    w, b = features.tophats(z, 21.0, 1.0, be)
    assert np.abs(w[30:-30, 30:-30]).max() < 1e-4 and np.abs(b[30:-30, 30:-30]).max() < 1e-4


def test_units_in_metres():
    assert features.odd_cells(11.0, 1.0) == 11
    assert features.odd_cells(11.0, 2.0) == 7   # 5.5 -> 6 -> odd 7
