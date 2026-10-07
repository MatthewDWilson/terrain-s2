"""GDAL-stack tests (rasterio). If these fail and the others pass, the environment's
GDAL/PROJ DLLs are broken or shadowed: run `python scripts/check_env.py`."""
import warnings

import numpy as np
import pytest

import rasterio  # noqa: E402  (deliberately not skipped: a broken GDAL stack must fail loudly)
from affine import Affine  # noqa: E402

from terrain_s2 import dataio  # noqa: E402


def _write(path, z, transform, nodata=-9999.0, crs="EPSG:2193"):
    with rasterio.open(path, "w", driver="GTiff", height=z.shape[0], width=z.shape[1], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=nodata) as d:
        d.write(np.where(np.isfinite(z), z, nodata).astype(np.float32), 1)


def test_mosaic_across_tile_boundary(tmp_path):
    """A window straddling two tiles is read seamlessly, on the source grid."""
    yy, xx = np.mgrid[0:100, 0:200].astype(np.float32)
    z = xx * 0.1 + yy * 0.01
    t_w = Affine(1, 0, 1_600_000, 0, -1, 6_000_100)
    t_e = Affine(1, 0, 1_600_100, 0, -1, 6_000_100)
    _write(tmp_path / "w.tif", z[:, :100], t_w)
    _write(tmp_path / "e.tif", z[:, 100:], t_e)
    w = dataio.read_window([tmp_path / "w.tif", tmp_path / "e.tif"],
                           (1_600_080.3, 6_000_020.0, 1_600_120.0, 6_000_080.0), buffer_m=10.0)
    assert w.uncovered_m == 0
    assert w.transform.c == 1_600_070 and w.transform.f == 6_000_090   # snapped outward
    assert np.isfinite(w.z).all()
    np.testing.assert_allclose(w.z, z[10:90, 70:130], atol=1e-5)       # no seam, no shift


def test_nodata_and_uncovered_warning(tmp_path):
    z = np.ones((50, 50), np.float32)
    z[:10, :10] = np.nan
    _write(tmp_path / "a.tif", z, Affine(1, 0, 0, 0, -1, 50))
    # pytest.warns records every warning in its block (overriding filters) and re-emits the
    # unmatched ones, so silence rasterio's known affine deprecation inside it.
    with pytest.warns(UserWarning, match="beyond the input tiles"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Use `@` matmul", category=PendingDeprecationWarning)
            w = dataio.read_window([tmp_path / "a.tif"], (10, 10, 40, 40), buffer_m=20.0)
    assert w.uncovered_m == 10
    assert np.isnan(w.z).sum() > 0 and np.nanmax(w.z) == 1.0


def test_rejects_geographic_crs(tmp_path):
    p = tmp_path / "g.tif"
    with rasterio.open(p, "w", driver="GTiff", height=10, width=10, count=1, dtype="float32",
                       crs="EPSG:4326", transform=Affine(0.001, 0, 173, 0, -0.001, -35)) as d:
        d.write(np.zeros((1, 10, 10), np.float32))
    with pytest.raises(ValueError, match="projected"):
        dataio.read_window([p], (173.001, -35.009, 173.009, -35.001))


NZTM_T = Affine(1, 0, 1_570_000, 0, -1, 5_200_100)


def test_compound_nztm_nzvd2016(tmp_path):
    """NZTM 2000 + NZVD2016 (compound) is accepted: only the horizontal part must be metric."""
    _write(tmp_path / "c.tif", np.ones((100, 100), np.float32), NZTM_T, crs="EPSG:2193+7839")
    w = dataio.read_window([tmp_path / "c.tif"], (1_570_010, 5_200_010, 1_570_050, 5_200_050))
    assert w.z.shape == (40, 40)
    assert dataio.horizontal_crs(w.crs).to_epsg() == 2193


def test_mislabelled_epsg9528_refused_then_relabelled(tmp_path):
    """EPSG:9528 is NZGD2000 + NZVD2016 (geographic). A file labelled so but holding NZTM metres
    is refused with a hint, and read correctly with assume_crs (relabel, no reprojection)."""
    _write(tmp_path / "m.tif", np.ones((100, 100), np.float32), NZTM_T, crs="EPSG:9528")
    b = (1_570_010, 5_200_010, 1_570_050, 5_200_050)
    with pytest.raises(ValueError, match="metadata is wrong"):
        dataio.read_window([tmp_path / "m.tif"], b)
    w = dataio.read_window([tmp_path / "m.tif"], b, assume_crs="EPSG:2193+7839")
    assert w.z.shape == (40, 40) and np.all(w.z == 1)
    assert w.transform.c == 1_570_010                    # unchanged coordinates: no reprojection
    assert dataio.horizontal_crs(w.crs).to_epsg() == 2193


def test_tiles_differing_only_in_vertical_label(tmp_path):
    """EPSG:2193 and EPSG:2193+7839 tiles share the horizontal CRS and mosaic together."""
    yy, xx = np.mgrid[0:100, 0:200].astype(np.float32)
    z = xx * 0.1 + yy * 0.01
    _write(tmp_path / "w.tif", z[:, :100], Affine(1, 0, 1_600_000, 0, -1, 6_000_100), crs="EPSG:2193")
    _write(tmp_path / "e.tif", z[:, 100:], Affine(1, 0, 1_600_100, 0, -1, 6_000_100), crs="EPSG:2193+7839")
    w = dataio.read_window([tmp_path / "w.tif", tmp_path / "e.tif"], (1_600_080, 6_000_020, 1_600_120, 6_000_080))
    np.testing.assert_allclose(w.z, z[20:80, 80:120], atol=1e-5)


def test_clip_window_script_with_mislabelled_tile(tmp_path):
    """End to end: clip_window.py on an EPSG:9528-labelled NZTM tile with --assume-crs."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    _write(tmp_path / "t.tif", np.ones((100, 100), np.float32), NZTM_T, crs="EPSG:9528")
    aoi = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::2193"}},
           "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[
               [1_570_030, 5_200_030], [1_570_060, 5_200_030], [1_570_060, 5_200_060], [1_570_030, 5_200_060],
               [1_570_030, 5_200_030]]]}}]}
    (tmp_path / "aoi.geojson").write_text(json.dumps(aoi))
    script = Path(__file__).resolve().parents[1] / "scripts" / "clip_window.py"
    r = subprocess.run([sys.executable, str(script), "--aoi", str(tmp_path / "aoi.geojson"), "--buffer", "10",
                        "--src", str(tmp_path / "t.tif"), "--out", str(tmp_path / "clips"),
                        "--assume-crs", "EPSG:2193+7839"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with rasterio.open(tmp_path / "clips" / "t_clip.tif") as s:
        assert s.width == 50 and s.height == 50 and s.crs.is_projected


def _aoi(path, coords, crs_name=None):
    import json
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [coords + [coords[0]]]}}]}
    if crs_name:
        fc["crs"] = {"type": "name", "properties": {"name": crs_name}}
    path.write_text(json.dumps(fc))
    return path


NZTM_BOX = [[1_570_030, 5_200_030], [1_570_060, 5_200_030], [1_570_060, 5_200_060], [1_570_030, 5_200_060]]


@pytest.mark.parametrize("crs_name", [None, "urn:ogc:def:crs:EPSG::9528"])
def test_aoi_metres_labelled_geographic_refused_then_relabelled(tmp_path, crs_name):
    """An AOI with NZTM coordinates but no CRS (read as WGS84) or labelled EPSG:9528 is refused with
    a hint, and read correctly with aoi_crs."""
    from pyproj import CRS
    f = _aoi(tmp_path / "a.geojson", NZTM_BOX, crs_name)
    with pytest.raises(ValueError, match="--aoi-crs"):
        dataio.read_aoi(f, CRS.from_epsg(2193))
    a = dataio.read_aoi(f, CRS.from_epsg(2193), aoi_crs="EPSG:2193")
    np.testing.assert_allclose(a.total_bounds, [1_570_030, 5_200_030, 1_570_060, 5_200_060])


def test_window_sanity_checks(tmp_path):
    """Non-finite, non-overlapping or huge windows fail with a clear message, before allocating."""
    _write(tmp_path / "t.tif", np.ones((100, 100), np.float32), NZTM_T)
    with pytest.raises(ValueError, match="not finite"):
        dataio.read_window([tmp_path / "t.tif"], (np.inf, 5_200_010, np.inf, 5_200_050))
    with pytest.raises(ValueError, match="does not overlap"):
        dataio.read_window([tmp_path / "t.tif"], (100.0, 100.0, 200.0, 200.0))
    with pytest.raises(ValueError, match="cells"):
        dataio.read_window([tmp_path / "t.tif"], (1_570_010, 5_200_010, 1_570_050, 5_200_050), buffer_m=1e8)


def _geographic_tile(path, crs="EPSG:9528"):
    """A geographic (NZGD2000) tile near Christchurch holding a plane defined in NZTM metres, at
    the LDS export pixel size (1.03e-5 deg)."""
    from pyproj import Transformer
    lon0, lat0, d = 172.6159, -43.3257, 1.03e-5
    n = 400
    lon = lon0 + (np.arange(n) + 0.5) * d
    lat = lat0 - (np.arange(n) + 0.5) * d
    LON, LAT = np.meshgrid(lon, lat)
    E, N = Transformer.from_crs(4167, 2193, always_xy=True).transform(LON, LAT)
    z = (0.001 * (E - 1_570_000) + 0.002 * (N - 5_195_000)).astype(np.float32)
    _write(path, z, Affine(d, 0, lon0, 0, -d, lat0), crs=crs)
    cx, cy = Transformer.from_crs(4167, 2193, always_xy=True).transform(lon0 + n * d / 2, lat0 - n * d / 2)
    return round(cx), round(cy)                    # whole metres, so AOI bounds sit on the 1 m grid


def test_relabelling_real_geographic_data_refused(tmp_path):
    _geographic_tile(tmp_path / "g.tif")
    with pytest.raises(ValueError, match="really is geographic"):
        dataio.read_window([tmp_path / "g.tif"], (1_570_000, 5_200_000, 1_570_050, 5_200_050),
                           assume_crs="EPSG:2193+7839")


def test_geographic_refused_without_reprojection(tmp_path):
    _geographic_tile(tmp_path / "g.tif")
    with pytest.raises(ValueError, match="reproject-to"):
        dataio.read_window([tmp_path / "g.tif"], (172.62, -43.33, 172.621, -43.329))


def test_reproject_geographic_tile_to_nztm(tmp_path):
    """EPSG:9528 (NZGD2000 + NZVD2016) tile warped to a 1 m NZTM grid: values match the NZTM
    plane; the output keeps NZVD2016 as its vertical label."""
    cx, cy = _geographic_tile(tmp_path / "g.tif")
    b = (cx - 50, cy - 50, cx + 50, cy + 50)
    with pytest.warns(UserWarning, match="resampling smooths"):
        w = dataio.read_window([tmp_path / "g.tif"], b, reproject_to="EPSG:2193", resolution=1.0)
    assert w.transform.a == 1.0 and w.transform.e == -1.0 and w.z.shape == (100, 100)
    assert dataio.horizontal_crs(w.crs).to_epsg() == 2193
    assert dataio.vertical_crs(w.crs).to_epsg() == 7839
    r, c = np.mgrid[0:100, 0:100]
    E = w.transform.c + (c + 0.5)
    N = w.transform.f - (r + 0.5)
    expect = 0.001 * (E - 1_570_000) + 0.002 * (N - 5_195_000)
    ok = np.isfinite(w.z)
    assert ok.mean() > 0.99
    assert np.abs(w.z[ok] - expect[ok]).max() < 1e-3      # a plane survives bilinear warping


def test_clip_window_reproject_and_name_collision(tmp_path):
    """DEM and DSM tiles with the same file name in different folders get distinct clip names."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    (tmp_path / "dem").mkdir()
    (tmp_path / "dsm").mkdir()
    cx, cy = _geographic_tile(tmp_path / "dem" / "92N58.tif")
    _geographic_tile(tmp_path / "dsm" / "92N58.tif")
    box = [[cx - 30, cy - 30], [cx + 30, cy - 30], [cx + 30, cy + 30], [cx - 30, cy + 30], [cx - 30, cy - 30]]
    aoi = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::2193"}},
           "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [box]}}]}
    (tmp_path / "aoi.geojson").write_text(json.dumps(aoi))
    script = Path(__file__).resolve().parents[1] / "scripts" / "clip_window.py"
    r = subprocess.run([sys.executable, str(script), "--aoi", str(tmp_path / "aoi.geojson"), "--buffer", "10",
                        "--src", str(tmp_path / "dem" / "92N58.tif"), "--src", str(tmp_path / "dsm" / "92N58.tif"),
                        "--out", str(tmp_path / "clips"), "--reproject-to", "EPSG:2193"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    outs = sorted(p.name for p in (tmp_path / "clips").glob("*.tif"))
    assert outs == ["dem_92N58_clip.tif", "dsm_92N58_clip.tif"]
    with rasterio.open(tmp_path / "clips" / outs[0]) as s:
        assert s.res == (1.0, 1.0) and s.width == 80 and dataio.horizontal_crs(s.crs).to_epsg() == 2193
