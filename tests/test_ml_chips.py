"""ML-3a step 5: chips read across tile edges, sampled by kind, with the card."""
import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from shapely.geometry import LineString

from terrain_s2 import dataio
from terrain_s2.ml import chips as C

X0, Y0, N = 1_570_000.0, 5_200_100.0, 100


def _run(root, tile, dx, value, channel_row=50):
    """A 100 m run east of X0 + dx: every feature band = value; a channel along channel_row, labelled positive."""
    run = root / "runs" / tile
    run.mkdir(parents=True)
    T = Affine(1.0, 0, X0 + dx, 0, -1.0, Y0)
    f = {b: np.full((N, N), value, np.float32) for b in C.FEATURE_BANDS}
    dataio.write_stack(run / "features.tif", f, T, "EPSG:2193")
    mask = np.zeros((N, N), np.float32)
    mask[channel_row - 1:channel_row + 2, :] = 1
    mask[80:82, 10:90] = 1                                   # an unrecorded physics channel
    h = {b: np.full((N, N), value, np.float32) for b in C.HYDRO_BANDS}
    h["channel_mask"] = mask
    dataio.write_stack(run / "hydro.tif", h, T, "EPSG:2193")
    lab = np.zeros((N, N), np.float32)
    lab[channel_row - 1:channel_row + 2, :] = 1
    lab[80:82, 10:90] = 255
    w = np.where(lab == 1, 1.0, np.where(lab == 255, 0.0, 0.05)).astype(np.float32)
    dataio.write_stack(run / "ml_channel_labels.tif", {"label": lab, "weight": w}, T, "EPSG:2193")
    bundle = root / "tiles" / tile
    bundle.mkdir(parents=True)
    gpd.GeoDataFrame(geometry=[LineString([(X0 + dx, Y0 - 20), (X0 + dx + N, Y0 - 20)])], crs=2193).to_file(
        bundle / "context.gpkg", layer="roads", driver="GPKG")
    C.ways_raster(run, bundle)
    return run


def test_mosaic_reads_across_the_edge_and_marks_missing(tmp_path):
    a = _run(tmp_path, "A", 0, 1.0)
    b = _run(tmp_path, "B", N, 2.0)
    M = C.Mosaic({"A": a, "B": b})
    x, lab, wt, ph, cl = M.read(X0 + 80, Y0 - 34, 32)              # 20 m in A, 12 m in B
    i = C.BANDS.index("relief_med5")
    assert (x[i, :, :20] == 1).all() and (x[i, :, 20:] == 2).all()
    assert (lab[15:18, :] == 1).all() and (lab[18, :] == 0).all() and wt[16, 5] == 1.0   # raster rows 49-51
    assert ph[16, 5] == 1 and ph[25, 5] == 0                            # the physics mask kept
    way = x[C.BANDS.index(C.WAY_BAND)]
    assert way[0, 0] == np.float32(14.0) or abs(float(way[0, 0]) - 14.0) <= 1.0   # the road at Y0 - 20
    x2, lab2, wt2, _, _ = M.read(X0 + 180, Y0 - 34, 32)           # 20 m in B, 12 m beyond any run
    assert (lab2[:, 20:] == 255).all() and (wt2[:, 20:] == 0).all() and np.isnan(x2[:, :, 20:]).all()
    M.close()


def test_tile_chips_kinds_files_and_card(tmp_path):
    a = _run(tmp_path, "A", 0, 1.0)
    b = _run(tmp_path, "B", N, 2.0)
    M = C.Mosaic({"A": a, "B": b})
    cfg = dict(C.DEFAULTS, size=32, per_tile=10, min_sep_m=8.0, jitter_m=4.0)
    rows, sums = C.tile_chips("A", a, M, tmp_path / "chips", cfg, seed=1)
    z = np.load(tmp_path / "chips" / "A.npz")
    assert z["x"].shape == (10, len(C.BANDS), 32, 32) and z["x"].dtype == np.float16
    kinds = pd.Series(z["kind"]).value_counts()
    assert kinds["positive"] == 5 and kinds.get("hard", 0) >= 1 and kinds.sum() == 10
    assert not np.isnan(z["x"].astype(float)).any()
    idx = pd.DataFrame(rows).assign(council="w")
    c = C.card(idx, [sums], "t", cfg)
    assert c["chips"] == 10 and c["per_council_kind"]["w"]["positive"] == 5 and len(c["mean"]) == len(C.BANDS)
    assert "| w | 5 |" in C.card_markdown(c)


def test_chips_v3_osm_band_classes_and_class_shares(tmp_path):
    """Chips version 3: the OSM band (ml_osm.tif), the v5 labels' class array, positives split by class."""
    a = _run(tmp_path, "A", 0, 1.0)
    T = Affine(1.0, 0, X0, 0, -1.0, Y0)
    lab = np.zeros((N, N), np.float32)
    lab[49:52, :] = 1                                            # a stream along row 50
    lab[19:22, :] = 1                                            # a wide river's edge along row 20 (river)
    cls = np.full((N, N), 255, np.float32)
    cls[49:52, :] = 2
    cls[19:22, :] = 1
    w = np.where(lab == 1, 1.0, 0.05).astype(np.float32)
    dataio.write_stack(a / "ml_channel_labels_v5.tif", {"label": lab, "weight": w, "class": cls}, T, "EPSG:2193")
    bundle = tmp_path / "tiles" / "A"
    gpd.GeoDataFrame({"waterway": ["stream"]}, geometry=[LineString([(X0, Y0 - 50.5), (X0 + N, Y0 - 50.5)])],
                     crs=2193).to_file(bundle / "context.gpkg", layer="osm_waterways", driver="GPKG")
    d = C.osm_raster(a, bundle)
    assert d[50, 50] == 0 and abs(float(d[20, 50]) - 30.0) < 1.5 and (a / "ml_osm.tif").exists()
    cfg = dict(C.DEFAULTS, size=32, per_tile=12, min_sep_m=6.0, jitter_m=0.0, label_file="ml_channel_labels_v5.tif",
               osm_band=True, positive_class_shares=dict(river=0.5, stream=0.5))
    bands = C.bands_for(cfg)
    assert bands == C.BANDS_V4 and len(bands) == 24
    M = C.Mosaic({"A": a}, bands=bands, label_file=cfg["label_file"])
    x, lb, wt, ph, cl = M.read(X0 + 10, Y0 - 34, 32)
    assert x.shape[0] == 24 and cl[16, 5] == 2 and cl[0, 5] == 255 and cl[5, 5] == 255
    cents = C.sample_centres(a, cfg, seed=0)
    pos = [c for c in cents if c[2] == "positive"]
    rows_ = [int(round(Y0 - y - 0.5)) for _, y, _ in pos]
    assert sum(abs(r - 20) <= 1 for r in rows_) >= 2 and sum(abs(r - 50) <= 1 for r in rows_) >= 2   # both classes
    rows, sums = C.tile_chips("A", a, M, tmp_path / "chips3", cfg, seed=1)
    z = np.load(tmp_path / "chips3" / "A.npz")
    assert z["x"].shape[1] == 24 and z["cls"].shape == z["label"].shape and list(z["bands"]) == list(bands)
    c = C.card(pd.DataFrame(rows).assign(council="w"), [sums], "t3", cfg, bands)
    assert c["class_cells"]["stream"] > 0 and c["class_cells"]["river"] > 0 and c["label_file"] == "ml_channel_labels_v5.tif"
    assert "Positive cells by class" in C.card_markdown(c)
    import pytest
    with pytest.raises(ValueError):
        C.Mosaic({"A": a}, bands=C.BANDS[::-1])
