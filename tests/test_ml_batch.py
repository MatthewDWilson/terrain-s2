"""Batch runner: runs pre-built bundles, skips finished tiles, reruns on new parameters, spawns workers."""
import json

from terrain_s2.ml.batch import BatchConfig, params_hash, run_batch, status
from test_run import _write


def _bundles(tmp_path, ids):
    for t in ids:
        d = tmp_path / "bundles" / t
        d.mkdir(parents=True)
        _write(d)                                   # dem.tif, dsm.tif, aoi.geojson (synthetic scene)
    return BatchConfig(partition="unused", bundles=str(tmp_path / "bundles"), runs=str(tmp_path / "runs"),
                       params=dict(buffer=50.0, device="cpu"), build=False)


def test_batch_resumes_and_reruns_on_new_parameters(tmp_path):
    cfg = _bundles(tmp_path, ["T1"])
    r = run_batch(["T1"], cfg, log=lambda *a: None)
    assert len(r) == 1 and r[0]["ok"], r[0].get("error")
    done = json.loads((tmp_path / "runs" / "T1" / "done.json").read_text())
    from terrain_s2.ml.features import FEATURES_VERSION
    assert done["params_hash"] == params_hash(dict(buffer=50.0, rasters="none", device="cpu", features=FEATURES_VERSION))
    assert (tmp_path / "runs" / "T1" / "ml_features.parquet").exists() and (tmp_path / "runs" / "T1" / "ml_patches.npy").exists()
    assert status("T1", cfg, dict(buffer=50.0, rasters="none", device="cpu")) == "done"
    assert run_batch(["T1"], cfg, log=lambda *a: None) == []                      # skipped
    cfg.params["max_hand"] = 5.0
    assert len(run_batch(["T1"], cfg, log=lambda *a: None)) == 1                 # new parameters: rerun


def test_failure_is_recorded_and_workers_spawn(tmp_path):
    cfg = _bundles(tmp_path, ["A", "B"])
    (tmp_path / "bundles" / "missing").mkdir()
    r = run_batch(["A", "B", "missing"], cfg, workers=2, log=lambda *a: None)
    ok = {x["tile_id"]: x["ok"] for x in r}
    assert ok == {"A": True, "B": True, "missing": False}
    assert (tmp_path / "runs" / "missing" / "failed.json").exists()


def test_bundle_without_dem_or_with_elevation_errors_is_rebuilt(tmp_path):
    from terrain_s2.ml.batch import _bundle_ok
    b = tmp_path / "b"
    b.mkdir()
    (b / "site.json").write_text(json.dumps({"errors": []}))
    assert not _bundle_ok(b)                                                     # no DEM
    (b / "dem.tif").write_bytes(b"x")
    assert _bundle_ok(b)
    (b / "site.json").write_text(json.dumps({"errors": [{"step": "elevation:dsm", "error": "x"}]}))
    assert not _bundle_ok(b)
    (b / "site.json").write_text(json.dumps({"errors": [{"step": "source:osm", "error": "504"}]}))
    assert _bundle_ok(b)                                                         # a vector source is not fatal
