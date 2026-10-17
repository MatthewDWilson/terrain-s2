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
