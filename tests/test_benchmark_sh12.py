"""SH12 benchmark (Whirinaki, Northland): the stream culverted under SH12 on a DEM drainage divide.

Runs only when the site data are available locally (LiDAR stays out of the repository):
    TERRAIN_S2_SH12_DIR = folder with AW27_clip.tif, AW27_dsm_clip.tif, roads.gpkg (optional)
Acceptance: (1) a crossing within 10 m of the culvert, tier high; (2) the network passes through
the culvert (one reach, flagged through_culvert); (3) the reaches either side are streams;
(4) the SH12 bridge is classed as a bridge.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

D = os.environ.get("TERRAIN_S2_SH12_DIR")
CULVERT = (1642255.6, 6075854.7)
BRIDGE = (1642170.4, 6075760.4)
pytestmark = pytest.mark.skipif(not D or not Path(D, "AW27_clip.tif").exists(), reason="SH12 data not available")


@pytest.fixture(scope="module")
def products(tmp_path_factory):
    import geopandas as gpd
    out = tmp_path_factory.mktemp("sh12")
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "scripts" / "run_network.py"), "--dem", str(Path(D, "AW27_clip.tif")),
           "--dsm", str(Path(D, "AW27_dsm_clip.tif")), "--aoi", str(root / "examples" / "aoi_whirinaki_sh12.geojson"),
           "--out", str(out), "--device", "cpu"]
    if Path(D, "roads.gpkg").exists():
        cmd += ["--roads", str(Path(D, "roads.gpkg"))]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    g = out / "network.gpkg"
    return gpd.read_file(g, layer="channels"), gpd.read_file(g, layer="crossings")


def _pt(xy):
    from shapely.geometry import Point
    return Point(*xy)


def test_culvert_found_high(products):
    _, cr = products
    d = cr.distance(_pt(CULVERT))
    assert d.min() <= 10 and cr.tier[d.idxmin()] == "high"


def test_network_through_culvert(products):
    ch, _ = products
    near = ch[ch.distance(_pt(CULVERT)) < 5]
    assert len(near) >= 1 and near.through_culvert.any()


def test_streams_either_side(products):
    ch, _ = products
    for dx, dy in ((-20, 20), (20, -20)):
        p = _pt((CULVERT[0] + dx, CULVERT[1] + dy))
        s = ch[ch.distance(p) < 25].copy()
        s["d"] = s.distance(p)
        assert s.sort_values("d")["class"].iloc[0] == "stream"


def test_bridge(products):
    _, cr = products
    d = cr.distance(_pt(BRIDGE))
    assert d.min() <= 10 and cr.tier[d.idxmin()] in ("bridge", "bridge_or_open")
