"""Channel network v2, stage C: OSM guidance placed on the LiDAR (guidance), candidate links (connectivity), and both
in the run (osm_guidance, c_outputs)."""
import json

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from terrain_s2 import connectivity as C
from terrain_s2 import guidance as G
from terrain_s2.run import network
from terrain_s2.synthetic import make_scene
from test_run import _write

NX, NY = 500, 520


def _lines(sc, which, shift_m=0.0, step=5.0):
    """The scene's river or stream centreline in map coordinates, shifted north by ``shift_m``."""
    x0, ytop = sc.transform.c, sc.transform.f
    ybot = ytop - NY
    xs = np.arange(5.0, NX - 5.0, step)
    yc = 430.0 + 15.0 * np.sin(xs / 90.0) if which == "river" else 150.0 + 20.0 * np.sin(xs / 60.0)
    return LineString(np.column_stack([x0 + xs, ybot + yc + shift_m]))


def test_river_extent_moves_a_misplaced_area_onto_the_bed():
    sc = make_scene(nx=NX, ny=NY)
    true = _lines(sc, "river")
    area = _lines(sc, "river", shift_m=7.0).buffer(5.0)              # the OSM polygon, 7 m off the bed
    areas = gpd.GeoDataFrame({"water_kind": ["river_area"]}, geometry=[area], crs=2193)
    m = G.river_extent(areas, sc.z, sc.transform)
    from rasterio.features import rasterize
    bed = rasterize([(true.buffer(3.0), 1)], out_shape=sc.z.shape, transform=sc.transform).astype(bool)
    far = ~rasterize([(true.buffer(25.0), 1)], out_shape=sc.z.shape, transform=sc.transform).astype(bool)
    assert m[bed].mean() > 0.9 and m[far].mean() < 0.01
    assert not G.river_extent(areas.iloc[0:0], sc.z, sc.transform).any()


def test_snap_waterways_puts_an_offset_stream_on_its_bed():
    sc = make_scene(nx=NX, ny=NY)
    from terrain_s2.backend import get_backend
    from terrain_s2.features import feature_stack
    feats = feature_stack(sc.z.astype(np.float32), 1.0, get_backend("cpu"))
    lines = gpd.GeoDataFrame({"waterway": ["stream", "footway"]},
                             geometry=[_lines(sc, "stream", shift_m=6.0), _lines(sc, "stream")], crs=2193)
    S = G.snap_waterways(lines, feats, sc.transform)
    assert len(S) == 1 and S.snapped.iloc[0] and "geometry_council" not in S
    true = _lines(sc, "stream", step=1.0)
    d = [true.distance(S.geometry.iloc[0].interpolate(t)) for t in np.linspace(50, S.geometry.iloc[0].length - 50, 40)]
    assert np.median(d) < 2.0


def test_candidate_links_carry_their_evidence(tmp_path):
    from shapely.geometry import box
    run = tmp_path / "run"
    run.mkdir()
    L = LineString
    ch = [L([(0, 500), (300, 500)]), L([(304, 500), (400, 500)]), L([(320, 520), (320, 600)])]
    gpd.GeoDataFrame({"channel_source": ["physics"] * 3, "class": ["drain"] * 3}, geometry=ch, crs=2193).to_file(
        run / "network.gpkg", layer="channels", driver="GPKG")
    aoi = box(0, 0, 1000, 1000)
    s, comps = C.measure(run, aoi)
    from affine import Affine
    T = Affine(1, 0, 0, 0, -1, 1000)
    z = np.full((1000, 1000), 10.0, np.float32)
    z[:, 301:303] = 11.0                                              # a 1 m bank across the 4 m gap
    roads = gpd.GeoDataFrame(geometry=[L([(302, 400), (302, 450)])], crs=2193)
    cr = gpd.GeoDataFrame({"tier": ["high"]}, geometry=[L([(300, 495), (304, 505)]).centroid], crs=2193)
    links = C.candidate_links(run / "network.gpkg", comps, z, T, crossings=cr, roads=roads)
    gap = links[links.length_m.round(1) == 4.0]
    assert len(gap) == 1 and gap.kind.iloc[0] == "end_end"           # the 4 m gap, found once from both sides
    assert gap.rise_m.iloc[0] == pytest.approx(1.0, abs=0.05) and gap.crossing_tier.iloc[0] == "high"
    assert not gap.crosses_road.iloc[0] and {gap.from_anchor.iloc[0], gap.to_anchor.iloc[0]} == {"edge", ""}
    assert (links.length_m <= 30.0).all() and links.geometry.apply(lambda g: g.length > 0).all()
    k = links.geometry.apply(lambda g: tuple(sorted(map(tuple, np.round(g.coords, 1))))).tolist()
    assert len(k) == len(set(k))                                      # each pair once


def _context(d, sc):
    river_area = _lines(sc, "river", shift_m=7.0).buffer(5.0)
    stream = _lines(sc, "stream", shift_m=6.0)
    gpd.GeoDataFrame({"water_kind": ["river_area"], "natural": ["water"], "water": ["river"]}, geometry=[river_area],
                     crs=2193).to_file(d / "context.gpkg", layer="osm_water", driver="GPKG")
    gpd.GeoDataFrame({"waterway": ["stream"]}, geometry=[stream], crs=2193).to_file(d / "context.gpkg",
                                                                                     layer="osm_waterways", driver="GPKG")


def test_run_with_osm_guidance_and_c_outputs(tmp_path):
    d = _write(tmp_path)
    sc = make_scene(nx=NX, ny=NY)
    _context(d, sc)
    res = network(d / "dem.tif", d / "dsm.tif", d / "aoi.geojson", out=d / "c", buffer=50.0, rasters="none",
                  device="cpu", osm_guidance=True, c_outputs=True)
    info = json.loads((d / "c" / "network.json").read_text())
    g = info["guidance"]
    assert g["river_areas"] == 1 and g["river_extent_km2"] > 0 and g["snapped"] >= 1 and set(g["by_kind"]) == {"stream"}
    ch = res.layers["channels"]
    assert "osm_share" in ch and (ch.osm_share > 0.5).any() and "stream" in set(ch.osm_waterway.dropna())
    c = info["c_outputs"]
    assert c["connectivity"]["components"] >= 1 and "links" in c and "channel_prob" not in c   # no U-Net here
    import pyogrio
    layers = {n for n, _ in pyogrio.list_layers(d / "c" / "network.gpkg")}
    assert ("links" in layers) == (c["links"] > 0)
    base = network(d / "dem.tif", d / "dsm.tif", d / "aoi.geojson", out=d / "base", buffer=50.0, rasters="none",
                   device="cpu")
    assert "guidance" not in base.info and "osm_share" not in base.layers["channels"]
