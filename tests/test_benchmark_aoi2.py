"""AOI2 benchmark (Whirinaki upper floodplain): culvert #2, a stream crossing a road obliquely
(channels 42.6 deg and 47.3 deg off the road axis, 4.7 deg apart: the same stream continues).

Runs only when the site data are available locally (LiDAR stays out of the repository):
    TERRAIN_S2_AOI2_DIR = folder with AW27_aoi2_clip.tif, AW27_aoi2_dsm_clip.tif, roads.gpkg (optional)
Acceptance: (1) a crossing within 10 m, tier high, and high DEM-only as well; (2) the network
passes through it (through_culvert); (3) the reach through it is a stream.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

D = os.environ.get("TERRAIN_S2_AOI2_DIR")
CULVERT = (1641716.5, 6075208.5)
pytestmark = pytest.mark.skipif(not D or not Path(D, "AW27_aoi2_clip.tif").exists(), reason="AOI2 data not available")


@pytest.fixture(scope="module")
def products(tmp_path_factory):
    import geopandas as gpd
    out = tmp_path_factory.mktemp("aoi2")
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "scripts" / "run_network.py"), "--dem", str(Path(D, "AW27_aoi2_clip.tif")),
           "--dsm", str(Path(D, "AW27_aoi2_dsm_clip.tif")),
           "--aoi", str(root / "examples" / "aoi2_whirinaki_upperfloodplain.geojson"), "--out", str(out), "--device", "cpu"]
    if Path(D, "roads.gpkg").exists():
        cmd += ["--roads", str(Path(D, "roads.gpkg"))]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    g = out / "network.gpkg"
    return gpd.read_file(g, layer="channels"), gpd.read_file(g, layer="crossings")


def test_culvert_high_with_and_without_roads(products):
    from shapely.geometry import Point
    _, cr = products
    d = cr.distance(Point(*CULVERT))
    k = d.idxmin()
    assert d[k] <= 10 and cr.tier[k] == "high" and cr.tier_dem_only[k] == "high"


def test_network_through_culvert_as_stream(products):
    from shapely.geometry import Point
    ch, _ = products
    near = ch[ch.distance(Point(*CULVERT)) < 5]
    assert len(near) >= 1 and near.through_culvert.any()
    assert (near[near.through_culvert]["class"] == "stream").all()
