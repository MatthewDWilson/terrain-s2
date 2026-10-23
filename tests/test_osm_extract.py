"""OSM from a local extract: the .osm(.pbf) converted once, then read per window like Overpass."""
from pathlib import Path

from terrain_s2.acquire import osm

OSM = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
 <node id="1" lat="-43.30" lon="172.60"/><node id="2" lat="-43.30" lon="172.61"/>
 <node id="3" lat="-43.301" lon="172.605"/><node id="4" lat="-43.299" lon="172.605"/>
 <node id="5" lat="-43.302" lon="172.602"/><node id="6" lat="-43.302" lon="172.603"/>
 <node id="7" lat="-43.303" lon="172.603"/><node id="8" lat="-43.303" lon="172.602"/>
 <node id="9" lat="-45.0" lon="170.0"/><node id="10" lat="-45.0" lon="170.01"/>
 <way id="100"><nd ref="1"/><nd ref="2"/><tag k="highway" v="residential"/><tag k="surface" v="paved"/></way>
 <way id="101"><nd ref="3"/><nd ref="4"/><tag k="waterway" v="drain"/><tag k="tunnel" v="culvert"/><tag k="layer" v="-1"/></way>
 <way id="102"><nd ref="5"/><nd ref="6"/><nd ref="7"/><nd ref="8"/><nd ref="5"/><tag k="building" v="yes"/></way>
 <way id="103"><nd ref="9"/><nd ref="10"/><tag k="railway" v="rail"/></way>
 <way id="104"><nd ref="1"/><nd ref="3"/><tag k="power" v="line"/></way>
</osm>
"""


def test_extract_and_read_window(tmp_path):
    src = tmp_path / "t.osm"
    src.write_text(OSM)
    info = osm.extract_pbf(src, tmp_path / "osm" / "x.gpkg", log=lambda m: None)
    assert info["counts"] == dict(ways=3, buildings=1)
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, 2193, always_xy=True)
    x0, y0 = tr.transform(172.598, -43.305)
    x1, y1 = tr.transform(172.612, -43.297)
    parts, meta = osm.from_extract(tmp_path / "osm" / "x.gpkg", [x0, y0, x1, y1])
    assert set(parts) == {"road", "culvert", "building"}                  # the railway is 200 km away
    c = parts["culvert"].iloc[0]
    assert c.tunnel == "culvert" and c.layer == "-1" and c.waterway == "drain"
    assert parts["road"].iloc[0].surface == "paved" and meta["count"] == 3 and meta["extract_date"]
    assert parts["road"].crs.to_epsg() == 2193


def test_bundle_fetch_uses_the_extract(tmp_path):
    from pyproj import Transformer

    from terrain_s2.acquire.build import _fetch
    src = tmp_path / "t.osm"
    src.write_text(OSM)
    osm.extract_pbf(src, tmp_path / "x.gpkg", log=lambda m: None)
    tr = Transformer.from_crs(4326, 2193, always_xy=True)
    x0, y0 = tr.transform(172.598, -43.305)
    x1, y1 = tr.transform(172.612, -43.297)

    class NoNet:
        store = None

        def get(self, *a, **k):
            raise AssertionError("Overpass called")
    got = _fetch(NoNet(), None, "osm", {"kind": "osm", "extract": str(tmp_path / "x.gpkg")}, [x0, y0, x1, y1], False)
    assert set(got["layers"]) == {"osm_roads", "osm_culverts", "osm_buildings"} and got["meta"]["extract"]


WATER = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
 <node id="1" lat="-43.300" lon="172.600"/><node id="2" lat="-43.300" lon="172.602"/>
 <node id="3" lat="-43.301" lon="172.602"/><node id="4" lat="-43.301" lon="172.600"/>
 <node id="5" lat="-43.302" lon="172.600"/><node id="6" lat="-43.302" lon="172.602"/>
 <node id="7" lat="-43.303" lon="172.602"/><node id="8" lat="-43.303" lon="172.600"/>
 <node id="9" lat="-43.304" lon="172.600"/><node id="10" lat="-43.304" lon="172.602"/>
 <node id="11" lat="-43.305" lon="172.602"/><node id="12" lat="-43.305" lon="172.600"/>
 <node id="13" lat="-43.306" lon="172.600"/><node id="14" lat="-43.306" lon="172.610"/>
 <way id="200"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/><tag k="natural" v="water"/><tag k="water" v="river"/></way>
 <way id="201"><nd ref="5"/><nd ref="6"/><nd ref="7"/><nd ref="8"/><nd ref="5"/></way>
 <relation id="300"><member type="way" ref="201" role="outer"/><tag k="type" v="multipolygon"/><tag k="natural" v="water"/><tag k="water" v="lake"/></relation>
 <way id="202"><nd ref="9"/><nd ref="10"/><nd ref="11"/><nd ref="12"/><nd ref="9"/><tag k="natural" v="wetland"/><tag k="wetland" v="tidalflat"/></way>
 <way id="203"><nd ref="13"/><nd ref="14"/><tag k="natural" v="coastline"/></way>
 <way id="204"><nd ref="1"/><nd ref="4"/><tag k="waterway" v="river"/></way>
</osm>
"""


def test_extract_water_areas_and_coastline(tmp_path):
    src = tmp_path / "w.osm"
    src.write_text(WATER)
    info = osm.extract_pbf(src, tmp_path / "w.gpkg", log=lambda m: None)
    assert info["counts"]["water"] == 3 and info["counts"]["ways"] == 2
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, 2193, always_xy=True)
    x0, y0 = tr.transform(172.598, -43.308)
    x1, y1 = tr.transform(172.612, -43.298)
    parts, meta = osm.from_extract(tmp_path / "w.gpkg", [x0, y0, x1, y1])
    assert meta["water_layer"] and set(parts) >= {"water", "coastline", "waterway"}
    assert sorted(parts["water"].water_kind) == ["lake", "river_area", "tidal"]
    assert parts["water"].geom_type.isin(["Polygon", "MultiPolygon"]).all()
    assert osm.water_kind({"waterway": "riverbank"}) == "river_area" and osm.water_kind({"natural": "mud"}) == "tidal"
    assert osm.water_kind({"natural": "wetland", "wetland": "swamp"}) is None
