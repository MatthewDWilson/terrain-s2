"""terrain_s2.run.network, the library entry point (S2-1): runs on a synthetic scene, writes the same
files as the script, returns the layers and, with keep=True, the in-memory state."""
import json

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

from terrain_s2.run import NetworkResult, NetworkRun, main, network
from terrain_s2.synthetic import make_scene


def _write(tmp_path):
    sc = make_scene(nx=500, ny=520)
    prof = dict(driver="GTiff", height=sc.z.shape[0], width=sc.z.shape[1], count=1, dtype="float32",
                crs=sc.crs, transform=sc.transform, nodata=-9999.0)
    for name, arr in (("dem", sc.z), ("dsm", sc.dsm)):
        with rasterio.open(tmp_path / f"{name}.tif", "w", **prof) as d:
            d.write(arr, 1)
    x0, y1 = sc.transform.c, sc.transform.f
    x1, y0 = x0 + 500 * sc.transform.a, y1 + 520 * sc.transform.e
    gpd.GeoDataFrame(geometry=[box(x0 + 60, y0 + 60, x1 - 60, y1 - 60)], crs=sc.crs).to_file(tmp_path / "aoi.geojson")
    return tmp_path


def test_network_entry_point(tmp_path):
    d = _write(tmp_path)
    run = network(d / "dem.tif", d / "dsm.tif", d / "aoi.geojson", out=d / "out", params={"buffer": 50.0, "rasters": "hydro"},
                  device="cpu", keep=True)
    assert isinstance(run, NetworkResult) and NetworkRun is NetworkResult
    assert run.paths["network"].exists() and run.paths["hydro"].exists() and run.perf["total_s"] > 0
    info = json.loads((d / "out" / "network.json").read_text())
    assert info["channels_km"] == run.info["channels_km"] > 0 and info["dem"] == [str(d / "dem.tif")]
    assert {"channels", "crossings"} <= set(run.layers) and (d / "out" / "hydro.tif").exists()
    assert not (d / "out" / "features.tif").exists()
    s = run.state
    assert s["w"].z.shape == s["sk"].shape and s["rem"] is not None and np.isfinite(s["rem"]).any()
    assert len(s["cr"]) >= len(run.layers["crossings"])                  # window vs written (AOI) layers
    assert network(d / "dem.tif", None, d / "aoi.geojson", out=d / "out2", buffer=50.0, rasters="none", device="cpu",
                   conditioned=False).state is None


def test_geometry_aoi_params_and_cli(tmp_path):
    """The Stage 1 module's call (eddie_terrain.tasks.ensure_network): network(dem, None, geometry, params=..., out=...)."""
    import pytest
    d = _write(tmp_path)
    g = gpd.read_file(d / "aoi.geojson").geometry.iloc[0]
    a = network(d / "dem.tif", None, g, params={"buffer": 50.0, "rasters": "none", "device": "cpu"}, out=d / "g")
    b = network(d / "dem.tif", None, d / "aoi.geojson", buffer=50.0, rasters="none", device="cpu", out=d / "p")
    assert a.info["crossings"] == b.info["crossings"] and a.info["channels_km"] == b.info["channels_km"]
    with pytest.raises(ValueError, match="unknown network option"):
        network(d / "dem.tif", None, g, params={"bufer": 50.0})
    assert main(["--dem", str(d / "dem.tif"), "--aoi", str(d / "aoi.geojson"), "--out", str(d / "cli"), "--buffer", "50",
                 "--rasters", "none", "--device", "cpu", "--no-conditioned"]) == 0
    assert (d / "cli" / "network.json").exists()
