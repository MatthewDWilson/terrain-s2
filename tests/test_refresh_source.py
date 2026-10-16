"""acquire.build.refresh_source: a failed source fetched again into a built bundle."""
import json

import geopandas as gpd
import pyogrio
from pyproj import Transformer
from shapely.geometry import LineString

from terrain_s2.acquire.build import refresh_source

WINDOW = [1570000.0, 5200000.0, 1571000.0, 5201000.0]


class FakeHttp:
    store = None

    def get(self, url, params, source, method="GET", data=None):
        tr = Transformer.from_crs(2193, 4326, always_xy=True)
        pts = [tr.transform(1570100, 5200500), tr.transform(1570900, 5200500)]
        way = dict(type="way", id=7, tags=dict(highway="service"), geometry=[dict(lon=x, lat=y) for x, y in pts])
        return json.dumps(dict(elements=[way]))


def test_refresh_osm_into_bundle(tmp_path):
    b = tmp_path / "BX00"
    b.mkdir()
    (b / "site.json").write_text(json.dumps(dict(site=dict(window=WINDOW), sources=[dict(name="linz_roads")],
                                                 errors=[dict(step="source:osm", error="Overpass failed"),
                                                         dict(step="source:other", error="x")])))
    gpd.GeoDataFrame({"source": ["linz_tracks"]}, geometry=[LineString([(1570000, 5200100), (1571000, 5200100)])],
                     crs=2193).to_file(b / "context.gpkg", layer="tracks", driver="GPKG")
    refresh_source(b, "osm", {"osm": {"kind": "osm"}}, tmp_path, http=FakeHttp(), log=lambda m: None)
    layers = {n for n, _ in pyogrio.list_layers(b / "context.gpkg")}
    assert {"tracks", "osm_roads"} <= layers
    assert len(gpd.read_file(b / "context.gpkg", layer="osm_roads")) == 1
    assert (b / "roads.gpkg").exists()                                    # no LINZ roads: OSM roads become roads
    prov = json.loads((b / "site.json").read_text())
    assert [e["step"] for e in prov["errors"]] == ["source:other"]
    assert any(s["name"] == "osm" and "refreshed" in s for s in prov["sources"])
