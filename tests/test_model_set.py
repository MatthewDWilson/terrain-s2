"""Model sets (ml.model_set): the frozen ML-3c parts by name, and the scorer's channel features in the run."""
import json
import pickle

import numpy as np
import pytest

import terrain_s2.ml.channel_predict as CP
from terrain_s2.ml import apply as A
from terrain_s2.ml import model_set as MS
from terrain_s2.ml import train as TR
from terrain_s2.ml.chips import BANDS
from terrain_s2.run import network
from test_run import _write


def _fake_unet(model_dir, bands, device="auto", threshold=None, min_cells=30, model=None, meta=None):
    h, w = bands[0].shape
    prob = np.zeros((h, w), np.float32)
    prob[h // 2 - 1:h // 2 + 2, w // 4:3 * w // 4] = 0.9           # a drain only the "model" sees
    return prob, prob >= 0.5, dict(model=str(model_dir), threshold=0.5, device="cpu", cells=int((prob >= 0.5).sum()),
                                   seconds=0.0, dataset="test")


def _unet_dir(root, bands=BANDS):
    d = root / "models" / "unet_t" / "holdout"
    d.mkdir(parents=True)
    (d / "model.json").write_text(json.dumps(dict(bands=list(bands), threshold=0.5, dataset="chips_t", best_epoch=3)))
    (d / "model.pt").write_bytes(b"weights")
    return d


def _scorer(path, tab, cols):
    rng = np.random.default_rng(0)
    tab = tab.assign(struct=np.where(rng.random(len(tab)) < 0.4, 1.0, np.nan), near_positive=False, weight=1.0,
                     label_kind="council", struct_class="culvert")
    tab["n_labels"] = tab.struct.notna().astype(int)
    cfg = TR.load_config()
    cfg["model"]["max_iter"] = 10
    model = dict(structure=TR.fit(tab, cols, cfg), bridge=None, features=cols, meta=dict(name="scorer_t", code="c"))
    path.mkdir(parents=True)
    with open(path / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    return path


def _set(tmp_path, body):
    f = tmp_path / "set_t.yml"
    f.write_text(body)
    return f


def test_load_resolves_checks_and_describes(tmp_path):
    data = tmp_path / "data"
    _unet_dir(data)
    f = _set(tmp_path, "name: t\nchannel_model: models/unet_t/holdout\nchannel_min_sharpness: 0.15\n")
    s = MS.load(f, data)
    assert s["options"]["channel_model"] == str(data / "models" / "unet_t" / "holdout")
    d = MS.describe(s)
    assert d["name"] == "t" and d["rules"] == {"channel_min_sharpness": 0.15}
    assert d["models"]["channel_model"]["dataset"] == "chips_t" and len(d["models"]["channel_model"]["weights_sha"]) == 16
    with pytest.raises(ValueError, match="unknown run option"):
        MS.load(_set(tmp_path, "channel_modle: x\n"), data)
    with pytest.raises(FileNotFoundError, match="no U-Net"):
        MS.load(_set(tmp_path, "channel_model: models/nothing\n"), data)
    (data / "models" / "unet_t" / "holdout" / "model.json").write_text(json.dumps(dict(bands=["z"])))
    with pytest.raises(ValueError, match="bands are not"):
        MS.load(_set(tmp_path, "channel_model: models/unet_t/holdout\n"), data)
    with pytest.raises(FileNotFoundError, match="no model set"):
        MS.find("no_such_set")
    assert MS.find("ml3c_v1").name == "ml3c_v1.yml"                 # the shipped set


def test_model_set_runs_the_channel_model_and_a_scorer_that_needs_it(tmp_path, monkeypatch):
    d = _write(tmp_path)
    data = tmp_path / "data"
    _unet_dir(data)
    monkeypatch.setattr(CP, "channel_mask_in_run", _fake_unet)
    base = network(d / "dem.tif", d / "dsm.tif", d / "aoi.geojson", out=d / "base", buffer=50.0, rasters="none",
                   device="cpu", keep=True, channel_model="m")
    st = base.state
    tab = A.model_features(base, d, (st["ml_prob"], st["w"].transform))
    assert {"ml_channel", "ml_prob_max", "ml_sharp_max"} <= set(tab.columns) and tab.ml_prob_max.notna().all()
    # the same values as the training path, which reads ml_channel_prob.tif and network.gpkg
    from terrain_s2 import dataio
    from terrain_s2.ml import dataset
    dataio.write_stack(base.out / "ml_channel_prob.tif", {"channel_prob": st["ml_prob"]}, st["w"].transform, "EPSG:2193")
    ref = dataset.channel_features(base.out, tab.drop(columns=["ml_channel", "ml_prob_max", "ml_sharp_max"]))
    for c in ("ml_channel", "ml_prob_max", "ml_sharp_max"):
        assert np.allclose(ref[c].to_numpy(float), tab[c].to_numpy(float), equal_nan=True)
    cols = [c for c in TR.feature_columns(tab) if c not in ("x", "y")][:15] + ["ml_prob_max", "ml_sharp_max"]
    _scorer(data / "models" / "scorer_t", tab, cols)
    f = _set(tmp_path, "name: t\nchannel_model: models/unet_t/holdout\nchannel_min_sharpness: 0.0\n"
                       "crossing_model: models/scorer_t\n")
    monkeypatch.setenv("TERRAIN_DATA_DIR", str(data))
    res = network(d / "dem.tif", d / "dsm.tif", d / "aoi.geojson", out=d / "set", buffer=50.0, rasters="none",
                  device="cpu", model_set=str(f), record_sources=[])
    s = res.layers["crossings"]
    assert "p_structure" in s and s.p_structure.between(0, 1).all() and s.model_version.str.startswith("scorer_t").all()
    info = json.loads((d / "set" / "network.json").read_text())
    assert info["model_set"]["name"] == "t" and info["model_set"]["models"]["crossing_model"]["features"] == len(cols)
    assert info["channel_model"]["rules"]["min_sharpness"] == 0.0 and info["scored"]["model"].startswith("scorer_t")
    # an explicit option wins over the set's
    res2 = network(d / "dem.tif", d / "dsm.tif", d / "aoi.geojson", out=d / "set2", buffer=50.0, rasters="none",
                   device="cpu", model_set=str(f), record_sources=[], channel_min_sharpness=0.99)
    assert res2.info["channel_model"]["rules"]["min_sharpness"] == 0.99
    # the same scorer without its channel model is refused, rather than scoring with the evidence missing
    with pytest.raises(ValueError, match="channel model"):
        network(d / "dem.tif", d / "dsm.tif", d / "aoi.geojson", out=d / "bad", buffer=50.0, rasters="none",
                device="cpu", crossing_model=data / "models" / "scorer_t", record_sources=[])
    with pytest.raises(ValueError, match="no channel_model"):
        MS.load(_set(tmp_path, "crossing_model: models/scorer_t\n"), data)
