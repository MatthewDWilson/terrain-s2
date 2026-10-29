"""The channel presence model (ml.channel_presence): features, recorded labels, fit by council, calibration on reviews."""
import pickle

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString as L

from terrain_s2.ml import channel_presence as CP


def _reaches(n, rng, real):
    rows = []
    for i in range(n):
        y = float(i * 20)
        big = real[i]
        rows.append(dict(geometry=L([(0, y), (100, y + rng.normal(0, 1))]), width_m=3.0 if big else 0.8,
                         incision_m=0.8 if big else 0.1, bend_deg_per_100m=5.0, straight_frac=0.9, corners=0,
                         bed_start=10.0, bed_end=9.5, roadside_frac=0.0 if big else 0.6,
                         channel_source="physics" if big else "ml", p_stream=0.3, **{"class": "drain"}))
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=2193)


def test_features_labels_fit_and_calibration(tmp_path):
    rng = np.random.default_rng(0)
    real = rng.random(200) < 0.5
    R = _reaches(200, rng, real)
    X = CP.features(R)
    assert list(X.columns) == list(CP.FEATURES) and X.physics.sum() == real.sum() and np.isnan(X.unet_prob).all()
    refs = gpd.GeoDataFrame(geometry=[g for g, r in zip(R.geometry, real) if r and rng.random() < 0.8], crs=2193)
    y = CP.recorded(R, refs)
    assert y.sum() <= real.sum() and (y & ~real).sum() == 0                        # only real reaches are recorded
    T = X.assign(recorded=y)
    M, rep, oof = CP.fit(T, groups=np.where(np.arange(200) < 100, "a", "b"))
    assert rep["held_out"]["auc"] > 0.8 and set(rep["folds"]) == {"a", "b"} and np.isfinite(oof).all()
    p = CP.predict(M, X)
    assert p[real].mean() > p[~real].mean() + 0.3
    rv = X.assign(real=real)
    cal = CP.calibrate(M, rv, dict(precision_target=0.9))
    assert cal["precision"] >= 0.9 and 0 < M["threshold"] < 1 and cal["reviewed"] == 200
    reviews = gpd.GeoDataFrame(dict(answer=np.where(real[:20], "y", "n")), geometry=list(R.geometry[:20]), crs=2193)
    hit = CP.match_reviews(gpd.GeoDataFrame(X.iloc[:30], geometry=list(R.geometry[:30]), crs=2193), reviews)
    assert len(hit) == 20 and (hit.real.to_numpy() == real[:20]).all()


def test_stage_a_uses_the_model(tmp_path, monkeypatch):
    from terrain_s2 import routed_network as RN
    rng = np.random.default_rng(1)
    real = rng.random(60) < 0.5
    R = _reaches(60, rng, real)
    X = CP.features(R)
    M, _, _ = CP.fit(X.assign(recorded=real))
    M["threshold"] = 0.42
    f = tmp_path / "model.pkl"
    with open(f, "wb") as h:
        pickle.dump(M, h)
    ev, thr, info = RN.presence_evidence(R.assign(tile="T"), tmp_path, dict(RN.DEFAULTS, presence_model=str(f)))
    assert info["source"] == "model" and thr == 0.42 and ev[real].mean() > ev[~real].mean()
    ev2, thr2, info2 = RN.presence_evidence(R.assign(tile="T"), tmp_path, dict(RN.DEFAULTS,
                                                                                presence_model=str(tmp_path / "none.pkl")))
    assert info2["source"] == "hand-set" and thr2 == RN.DEFAULTS["keep_p"]
    assert isinstance(pd.Series(ev2).sum(), float)
