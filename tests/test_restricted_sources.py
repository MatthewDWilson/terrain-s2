"""S2-11: restricted sources may train models but never reach the chain's inputs, products, the committed registry
or the coverage index."""
import geopandas as gpd
import pyogrio
import pytest
import yaml
from shapely.geometry import LineString, box

from terrain_s2.acquire import build as B
from terrain_s2.acquire import registry
from terrain_s2.acquire.site import load_sources


def _registry(tmp_path, tracked, local=None):
    (tmp_path / "sources.yml").write_text(yaml.safe_dump({"sources": tracked}))
    if local is not None:
        (tmp_path / "sources.local.yml").write_text(yaml.safe_dump({"sources": local}))
    return tmp_path / "sources.yml"


def test_restricted_only_in_the_local_file(tmp_path):
    with pytest.raises(ValueError, match="sources.local.yml"):
        load_sources(_registry(tmp_path, {"s": {"kind": "arcgis", "restricted": True}}))
    s = load_sources(_registry(tmp_path, {"a": {"kind": "osm"}}, {"s": {"kind": "arcgis", "restricted": True}}))
    assert set(s) == {"a", "s"} and s["s"]["restricted"]


def test_restricted_never_resolved_automatically():
    w = (1570000, 5200000, 1571000, 5201000)
    lonlat = box(*registry._lonlat(w))
    index = {"a": (lonlat, {})}
    srcs = {"a": {"kind": "osm"}, "s": {"kind": "arcgis", "restricted": True, "layer": "stopbanks"}}
    assert registry.resolve(srcs, index, w) == ["a"]          # and no "not in the coverage index" for s


def test_restricted_layers_kept_apart(tmp_path, monkeypatch):
    line = gpd.GeoDataFrame({"source": ["s"]}, geometry=[LineString([(1570100, 5200100), (1570900, 5200900)])], crs=2193)

    def fake_fetch(http, store, name, spec, window, dry_run):
        return dict(meta=dict(count=1), layers={spec["layer"]: line.assign(source=name)})
    monkeypatch.setattr(B, "_fetch", fake_fetch)

    class Site:
        id, role, bounds, buffer_m, aoi_path, notes = "X", "test", (1570000, 5200000, 1571000, 5201000), 0, None, ""
        window, elevation, layers, exclude = (1570000, 5200000, 1571000, 5201000), None, None, ()
        sources = ["open", "secret"]
    class Http:
        store = None
    srcs = {"open": {"kind": "arcgis", "layer": "channels"},
            "secret": {"kind": "arcgis", "layer": "stopbanks", "restricted": True}}
    prov = B.build_site(Site(), srcs, tmp_path, tmp_path / "cache", http=Http(), log=lambda m: None)
    out = tmp_path / "X"
    assert {n for n, _ in pyogrio.list_layers(out / "labels.gpkg")} == {"channels"}
    assert {n for n, _ in pyogrio.list_layers(out / B.RESTRICTED_FILE)} == {"stopbanks"}
    assert any(s["name"] == "secret" and s["restricted"] for s in prov["sources"])
    srcs["secret"]["layer"] = "roads"                           # a restricted source may not feed the chain
    prov = B.build_site(Site(), srcs, tmp_path, tmp_path / "cache", http=Http(), log=lambda m: None)
    assert any("label layers" in e["error"] for e in prov["errors"])
    assert not (out / "roads.gpkg").exists() and not (out / B.RESTRICTED_FILE).exists()
