"""scripts/check_runs.py: issues found from the files of bundles and runs."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_runs", ROOT / "scripts" / "check_runs.py")
CR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CR)


def _tile(root, t, key_missing=False, dsm_valid=0.99, done=True, rasters="all", files=CR.RUN_FILES):
    b, r = root / "tiles" / t, root / "runs" / t
    b.mkdir(parents=True), r.mkdir(parents=True)
    errs = [dict(step="source:linz_roads", error="LINZ_API_KEY is not set (...)")] if key_missing else []
    (b / "site.json").write_text(json.dumps(dict(errors=errs, elevation=[dict(product="dem", valid_fraction=0.99),
                                                                       dict(product="dsm", valid_fraction=dsm_valid)])))
    (b / "dem.tif").write_text("x"), (b / "dsm.tif").write_text("x")
    if done:
        (r / "done.json").write_text(json.dumps(dict(crossings=10, channels_km=3.0, params=dict(rasters=rasters))))
    for f in files:
        (r / f).write_text("x")


def test_issues(tmp_path):
    _tile(tmp_path, "A")
    _tile(tmp_path, "B", key_missing=True, dsm_valid=0.80)
    _tile(tmp_path, "C", done=False)
    _tile(tmp_path, "D", files=("network.gpkg",))
    runs, bundles = tmp_path / "runs", tmp_path / "tiles"
    rows = {t: CR.check_tile(t, runs, bundles, "channels", None, 0.02) for t in "ABCD"}
    assert rows["A"]["issues"] == "not_snapped;no_label_raster"
    assert {"linz_key_missing", "dsm_gap"} <= set(rows["B"]["issues"].split(";"))
    assert "not_done" in rows["C"]["issues"]
    assert "missing_outputs" in rows["D"]["issues"] and "ml_features.parquet" in rows["D"]["missing"]
    assert CR.check_tile("A", runs, bundles, "batch", None, 0.02)["issues"] == ""


def test_stale_failed(tmp_path):
    import os
    import time
    _tile(tmp_path, "A")
    r = tmp_path / "runs" / "A"
    (r / "failed.json").write_text("{}")
    old = time.time() - 100
    os.utime(r / "failed.json", (old, old))
    row = CR.check_tile("A", tmp_path / "runs", tmp_path / "tiles", "batch", None, 0.02)
    assert row["issues"] == "stale_failed"
