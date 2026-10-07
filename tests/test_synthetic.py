"""Phase 2a exit criteria, reproduced on the synthetic SH12-like scene.

Real-data criteria (design §11): SH12 culvert found without external data;
Whirinaki bridge not flagged. These tests are the synthetic analogue.
"""
import math

import numpy as np
import pytest

from terrain_s2 import hydro
from terrain_s2.backend import get_backend
from terrain_s2.pipeline import run
from terrain_s2.synthetic import make_divide_scene, make_scene


@pytest.fixture(scope="module")
def scene_result():
    s = make_scene()
    return s, run(s.z, s.transform, be=get_backend("cpu"))


def _dist(r, xy):
    return math.hypot(r["crest_x"] - xy[0], r["crest_y"] - xy[1])


def test_culvert_found(scene_result):
    s, res = scene_result
    hits = [r for r in res.candidates if _dist(r, s.truth["culvert"]) < 10]
    assert len(hits) == 1
    r = hits[0]
    assert r["test_a"] and r["test_b"]
    assert 4.0 < r["h_b"] < 6.0          # embankment ≈ 4–5 m above the channel bed
    assert 15 < r["L_b"] < 45
    assert 6 <= r["crest_width"] <= 10   # synthetic crest is 8 m
    assert r["cross_angle"] > 45         # path crosses the embankment, not along it
    assert r["dep_volume"] > 1000


@pytest.mark.parametrize("name", ["bridge_removed", "ford_stream", "ford_river", "mound"])
def test_no_false_candidate(scene_result, name):
    s, res = scene_result
    assert not [r for r in res.candidates if _dist(r, s.truth[name]) < 30]


def test_only_one_candidate(scene_result):
    _, res = scene_result
    assert len(res.candidates) == 1


def test_breached_dem_drains(scene_result):
    _, res = scene_result
    assert res.breach.n_unresolved == 0
    zf, _ = hydro.fill(res.breach.z_breached)
    assert np.nanmax(zf - res.breach.z_breached) < 1e-3


def test_breach_never_raises(scene_result):
    s, res = scene_result
    ok = np.isfinite(s.z)
    assert np.all(res.breach.z_breached[ok] <= s.z[ok] + 1e-6)


def test_nodata_preserved(scene_result):
    s, res = scene_result
    nod = ~np.isfinite(s.z)
    assert nod.any()
    assert np.all(np.isnan(res.breach.z_breached[nod]))
    assert np.all(np.isnan(res.feats["openness_pos"][nod]))


def test_resolution_independence():
    """Same scene at 2 m: the culvert must still be found (thresholds are in metres)."""
    s = make_scene(nx=400, ny=300, cellsize=2.0)
    res = run(s.z, s.transform, be=get_backend("cpu"))
    assert [r for r in res.candidates if _dist(r, s.truth["culvert"]) < 15]


def test_culvert_found_by_both_tests(scene_result):
    """A stream culverted under a road: channel on both sides and a pond behind (design §4.2)."""
    s, res = scene_result
    hit = [r for r in res.candidates if _dist(r, s.truth["culvert"]) < 10]
    assert hit and hit[0]["test"] == "A+B"


@pytest.fixture(scope="module")
def divide_result():
    s = make_divide_scene()
    return s, run(s.z, s.transform, be=get_backend("cpu"), dsm=s.dsm)


def test_divide_culvert_found_by_test_a_only(divide_result):
    """The SH12 case: both ditches drain away from the road, so nothing ponds; only Test A sees it."""
    s, res = divide_result
    hit = [r for r in res.candidates if _dist(r, s.truth["culvert"]) < 10]
    assert len(hit) == 1 and hit[0]["test"] == "A"
    r = hit[0]
    # h_b is from the ditch beds (1 m below the floodplain) to the 3 m embankment crest
    assert 3.5 < r["h_b"] < 4.3 and r["cross_angle"] > 75 and r["elongation"] >= 3


@pytest.mark.parametrize("name", ["dead_end", "open_end"])
def test_divide_controls_not_flagged(divide_result, name):
    s, res = divide_result
    assert not [r for r in res.candidates if _dist(r, s.truth[name]) < 25]
