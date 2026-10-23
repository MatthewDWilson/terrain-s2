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


def test_unsnapped_river_line_stays_positive():
    lines, foot, mask, d11, valid = _inputs()
    lines = lines.assign(width_source=["physics", "river"])
    foot = foot.assign(width_source=["physics", "river"])
    lab, w, _ = CR.label_raster((N, N), T, foot, lines, mask, d11, d11, valid, CFG, "w")
    assert lab[150, 100] == CR.POS and w[150, 100] == 0.5             # the river footprint, not an ignored corridor


def test_corrections_cleaning_and_review(tmp_path):
    """Labels v3: a council line the held-out model sees nothing on (and the physics map neither) is ignored; review
    answers override: n -> ignored, model-only y -> positive, model-only n -> hard negative."""
    import json

    import geopandas as gpd
    from terrain_s2 import dataio
    run = tmp_path / "runs" / "BX00_10000_0101"
    run.mkdir(parents=True)
    f = {"tophat_black11": np.zeros((N, N), np.float32), "tophat_black21": np.zeros((N, N), np.float32)}
    dataio.write_stack(run / "features.tif", f, T, "EPSG:2193")
    mask = np.zeros((N, N), np.float32)
    mask[48:53, :] = 1                                              # the physics map holds line a only
    dataio.write_stack(run / "hydro.tif", {"channel_mask": mask}, T, "EPSG:2193")
    a = LineString([(X0 + 10, Y0 - 50), (X0 + 190, Y0 - 50)])
    b = LineString([(X0 + 10, Y0 - 150), (X0 + 190, Y0 - 150)])
    lines = gpd.GeoDataFrame({"council": ["w", "w"], "source_id": ["a", "b"], "snapped": [True, True],
                              "width_source": ["physics", "default"], "half_width_m": [1.0, 1.0], "weight": [1.0, 1.0],
                              "geometry_council": [a.wkt, b.wkt], "class": ["drain", "drain"]},
                             geometry=[a, b], crs=2193)
    lines.to_file(run / "ml_channels.gpkg", layer="council_channels", driver="GPKG")
    prob = np.full((N, N), 0.01, np.float32)
    prob[45:56, :] = 0.9                                            # the model sees line a, not b
    dataio.write_stack(run / "ml_channel_prob.tif", {"channel_prob": prob}, T, "EPSG:2193")
    (run / "ml_channel_prob.json").write_text(json.dumps(dict(held_out=True)))
    rv = gpd.GeoDataFrame({"tile": [run.name] * 3, "kind": ["model_only", "model_only", "council_missed"],
                           "exists": ["y", "n", "unsure"]},
                          geometry=[LineString([(X0 + 20, Y0 - 100), (X0 + 80, Y0 - 100)]),
                                    LineString([(X0 + 120, Y0 - 100), (X0 + 180, Y0 - 100)]),
                                    LineString([(X0 + 20, Y0 - 30), (X0 + 40, Y0 - 30)])], crs=2193)
    (tmp_path / "manual").mkdir()
    rv.to_file(tmp_path / "manual" / "review_x.gpkg", layer="reaches", driver="GPKG")
    cfg = dict(CFG, labels=dict(CFG["labels"], reviews=str(tmp_path / "manual" / "review_*.gpkg")))
    n = CR.tile_labels(run, cfg, council="w")
    import rasterio
    with rasterio.open(run / "ml_channel_labels.tif") as s:
        lab, w = s.read(1), s.read(2)
    assert lab[50, 100] == CR.POS                                   # line a stays
    assert lab[150, 100] == CR.IGNORE and n["cleaned_pieces"] >= 3  # line b cleaned out
    assert lab[100, 50] == CR.POS and lab[100, 150] == CR.NEG and w[100, 150] == 1.0
    assert lab[30, 30] == CR.IGNORE and n["review_rows"] == 3 and n["label_version"] == 3
    # labels v5 beside them (no OSM in this bundle): the class band, and the v3 file untouched
    (tmp_path / "empty_bundle").mkdir()
    n5 = CR.tile_labels(run, dict(cfg, labels=dict(cfg["labels"], osm=True)), council="w", bundle=tmp_path / "empty_bundle")
    with rasterio.open(run / "ml_channel_labels_v5.tif") as s:
        assert list(s.descriptions) == ["label", "weight", "class"]
        lab5, cls5 = s.read(1), s.read(3)
    with rasterio.open(run / "ml_channel_labels.tif") as s:
        assert s.count == 2 and (s.read(1) == lab).all()
    assert n5["label_version"] == 5 and (lab5 == lab).all() and cls5[50, 100] == CR.CLASS["drain"]
    assert cls5[100, 50] == CR.CLASS_UNKNOWN and n5["class_cells"]["drain"] > 0    # a reviewed reach: no class known


def test_labels_v5_osm_positives_and_classes(tmp_path):
    """Labels v5: an OSM river area 4 m off the low bed and an OSM stream 5 m off its incised bed become positives
    on the bed (river weight 1, line weight 0.7), with their classes; a council drain footprint keeps class drain."""
    from terrain_s2 import dataio
    bundle = tmp_path / "tiles" / "BX00_10000_0101"
    bundle.mkdir(parents=True)
    z = np.full((N, N), 10.0, np.float32)
    z[24:37, :] = 8.0                                               # a 12 m wide river bed at rows 24-36
    dataio.write_stack(bundle / "dem.tif", {"z": z}, T, "EPSG:2193")
    river = LineString([(X0, Y0 - 26), (X0 + N, Y0 - 26)]).buffer(6.0)     # drawn 4 m north of the bed's centre
    stream = LineString([(X0, Y0 - 115), (X0 + N, Y0 - 115)])                # drawn 5 m north of the bed at row 120
    gpd.GeoDataFrame({"water_kind": ["river_area"]}, geometry=[river], crs=2193).to_file(bundle / "context.gpkg",
                                                                                         layer="osm_water", driver="GPKG")
    gpd.GeoDataFrame({"waterway": ["stream"]}, geometry=[stream], crs=2193).to_file(bundle / "context.gpkg",
                                                                                     layer="osm_waterways", driver="GPKG")
    b11 = np.zeros((N, N), np.float32)
    b11[119:122, :] = 0.5                                           # the stream's incised bed
    label = np.full((N, N), CR.NEG, np.uint8)
    weight = np.full((N, N), 0.05, np.float32)
    drain = LineString([(X0 + 20, Y0 - 170), (X0 + 180, Y0 - 170)])
    label[169:172, 20:180] = CR.POS
    weight[169:172, 20:180] = 1.0
    valid = np.ones((N, N), bool)
    cfg = dict(CFG, labels=dict(CFG["labels"], osm=True))
    rx, code, n = CR.osm_labels(tmp_path / "runs" / "x", bundle, label, weight, T, valid, b11, b11, cfg)
    assert label[30, 100] == CR.POS and weight[30, 100] == 1.0 and rx[30, 100]       # the bed, not the drawn polygon
    assert not rx[15, 100] and label[120, 100] == CR.POS and weight[120, 100] == np.float32(0.7)
    assert n["osm_lines_snapped"] == 1 and n["osm_new_positive_cells"] > 0
    foot = gpd.GeoDataFrame({"source_id": ["d"], "width_source": ["physics"]}, geometry=[drain.buffer(1.5)], crs=2193)
    lines = gpd.GeoDataFrame({"source_id": ["d"], "class": ["drain"]}, geometry=[drain], crs=2193)
    cls = CR.class_band(label, T, foot, lines, rx, code)
    assert cls[30, 100] == CR.CLASS["river"] and cls[120, 100] == CR.CLASS["stream"] and cls[170, 100] == CR.CLASS["drain"]
    assert cls[80, 100] == CR.CLASS_UNKNOWN                                          # not a positive
    assert CR.label_version(cfg) == 5 and CR.label_file(cfg) == "ml_channel_labels_v5.tif"
    assert CR.label_version(CFG) == 3 and CR.label_file(CFG) == "ml_channel_labels.tif"
