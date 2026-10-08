"""W6: the ``terrain`` CLI and the regression harness."""
import json

import pytest

from terrain_s2 import cli
from terrain_s2.io import contract as C

from stage1_fakes import FakeHttp, nz_profile, stac
from test_stage1 import X0, Y1, _surveys

AOI = f"POLYGON(({X0 + 8} {Y1 - 152}, {X0 + 152} {Y1 - 152}, {X0 + 152} {Y1 - 8}, {X0 + 8} {Y1 - 8}, {X0 + 8} {Y1 - 152}))"


@pytest.fixture
def fake_source(tmp_path, monkeypatch):
    from terrain_s2.stage1 import source
    http = FakeHttp(stac(tmp_path, _surveys()))
    prof = nz_profile(national="new-zealand/new-zealand/dem_1m/2193/collection.json")
    monkeypatch.setattr(source, "get_source", lambda profile, **kw: source.InterimLinzSource(
        prof, http=http, cache_dir=tmp_path / "stac"))
    for k in [k for k in __import__("os").environ if k.startswith("TERRAIN_")]:
        monkeypatch.delenv(k)
    return tmp_path


def _dem(tmp, *extra):
    out = tmp / f"r{len(list(tmp.glob('r*.json')))}.json"
    rc = cli.main(["dem", "--aoi", AOI, "--aoi-crs", "EPSG:2193", "--buffer", "0", "--data-dir", str(tmp / "data"),
                   "--out-json", str(out), "--quiet", *extra])
    assert rc == 0
    return json.loads(out.read_text())


def test_cli_dem_and_registry(fake_source, capsys):
    tmp = fake_source
    r = _dem(tmp, "--resolution", "8")
    assert r["resolution"] == 8 and not r["reused"] and r["paths"]["netcdf"].endswith("dem_8m.nc")
    assert _dem(tmp, "--resolution", "8")["reused"]
    reg = _dem(tmp, "--resolution", "8", "--register", "--db", f"sqlite:///{tmp / 'r.sqlite'}")
    assert reg["product_id"] == 1 and reg["generator_key"] == r["generator_key"]


def test_cli_compare_harness(fake_source, tmp_path):
    tmp = fake_source
    a = _dem(tmp, "--resolution", "1", "--product-dir", str(tmp / "runA"))
    b = _dem(tmp, "--resolution", "1", "--product-dir", str(tmp / "runB"))
    nb = b["paths"]["netcdf"]
    p = C.read_dem(nb)                                  # perturb B to see the harness report it
    p.z[:10, :10] += 0.5
    C.write_dem(nb, p, cog=False, provenance=False)
    out = tmp / "report.md"
    assert cli.main(["compare", str(tmp / "runA"), nb, "--out", str(out)]) == 0
    rep = json.loads(out.with_suffix(".json").read_text())
    assert rep["z_diff"]["n"] == 144 * 144 and rep["z_diff"]["max_abs"] == pytest.approx(0.5)
    assert rep["z_diff"]["share_abs_gt_0.2"] == pytest.approx(100 / 144 ** 2)
    assert rep["provenance_a"]["generator_key"] == a["generator_key"]
    assert "| all |" in out.read_text()


def test_cli_condition_not_yet(capsys):
    assert cli.main(["condition", "--help"]) in (0, 2)


def test_stage1_input_resolves_to_cog(fake_source):
    tmp = fake_source
    r = _dem(tmp, "--resolution", "1")
    folder = str(__import__("pathlib").Path(r["paths"]["netcdf"]).parent)
    assert cli._stage1_input(folder) == r["paths"]["cog"]
    assert cli._stage1_input("plain.tif") == "plain.tif"
