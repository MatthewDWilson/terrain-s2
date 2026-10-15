"""ML-3a step 4: label rasters (positives, ring, unsnapped corridors, unrecorded channels, flat-only negatives)."""
import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import LineString

from terrain_s2.ml import channel_rasters as CR

X0, Y0, N = 1_570_000.0, 5_200_200.0, 200
T = Affine(1.0, 0, X0, 0, -1.0, Y0)
CFG = dict(negative_weight=0.05, ignore_ring_m=3.0, labels=dict(unsnapped="ignore", flat_depth_m=0.05),
           snap=dict(corridor_m=15.0),
           councils=dict(w=dict(negatives="all", ignore_unrecorded=True),
                         a=dict(negatives="flat", ignore_unrecorded=True, negative_weight=0.02)))


def _inputs():
    snapped = LineString([(X0 + 10, Y0 - 50), (X0 + 190, Y0 - 50)])
    unsnapped = LineString([(X0 + 10, Y0 - 150), (X0 + 190, Y0 - 150)])
    lines = gpd.GeoDataFrame({"council": ["w", "w"], "snapped": [True, False],
                              "geometry_council": [snapped.wkt, unsnapped.wkt]}, geometry=[snapped, unsnapped], crs=2193)
    foot = gpd.GeoDataFrame({"council": ["w", "w"], "snapped": [True, False], "weight": [1.5, 0.5]},
                            geometry=[snapped.buffer(1.0, cap_style="flat"), unsnapped.buffer(1.0, cap_style="flat")],
                            crs=2193)
    mask = np.zeros((N, N), bool)
    mask[95:100, 20:180] = True                                     # an unrecorded physics channel
    d11 = np.zeros((N, N), np.float32)
    d11[:, 185:] = 0.3                                              # incised ground at the east edge
    valid = np.ones((N, N), bool)
    valid[:3, :3] = False
    return lines, foot, mask, d11, valid


def test_label_rules():
    lines, foot, mask, d11, valid = _inputs()
    lab, w, n = CR.label_raster((N, N), T, foot, lines, mask, d11, d11, valid, CFG, "w")
    assert lab[50, 100] == CR.POS and w[50, 100] == 1.5
    assert lab[53, 100] == CR.IGNORE and w[53, 100] == 0                 # the ring
    assert lab[150, 100] == CR.IGNORE and lab[160, 100] == CR.IGNORE     # unsnapped: corridor ignored, not positive
    assert lab[97, 100] == CR.IGNORE                                     # unrecorded physics channel
    assert lab[180, 100] == CR.NEG and w[180, 100] == np.float32(0.05)
    assert lab[0, 0] == CR.IGNORE                                        # no data
    assert lab[20, 190] == CR.NEG                                        # incised but council w: negatives everywhere
    assert n["positive"] == int((lab == CR.POS).sum()) and n["positive_weight_sum"] > 0


def test_flat_only_negatives_and_unsnapped_positive():
    lines, foot, mask, d11, valid = _inputs()
    lab, w, _ = CR.label_raster((N, N), T, foot, lines.assign(council="a"), mask, d11, d11, valid, CFG, "a")
    assert lab[20, 190] == CR.IGNORE and lab[20, 100] == CR.NEG and w[20, 100] == np.float32(0.02)
    cfg = dict(CFG, labels=dict(unsnapped="positive", flat_depth_m=0.05))
    lab2, w2, _ = CR.label_raster((N, N), T, foot, lines, mask, d11, d11, valid, cfg, "w")
    assert lab2[150, 100] == CR.POS and w2[150, 100] == 0.5
