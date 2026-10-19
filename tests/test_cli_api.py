"""``terrain api``: the CLI and client against a stand-in for eddie_terrain's blueprint (a local HTTP server
with the same routes and responses)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("shapely")
pytest.importorskip("pyproj")

from terrain_s2 import cli
from terrain_s2.client.api import TerrainApi, TerrainApiError

WKT = "POLYGON ((173.4603 -35.4633, 173.4710 -35.4633, 173.4710 -35.4570, 173.4603 -35.4570, 173.4603 -35.4633))"


class Fake(BaseHTTPRequestHandler):
    posts, polls = [], {}

    def log_message(self, *a):
        pass

    def _send(self, code, d):
        b = json.dumps(d).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/terrain/dem":
            return self._send(404, {"error": "no route"})
        if body["params"].get("resolution") == 3:
            return self._send(400, {"error": "resolution must be one of (1, 2, 4, 8), got 3"})
        Fake.posts.append(body)
        return self._send(202, {"taskId": "fail" if body["params"].get("backend") == "geofabrics" else "t1"})

    def do_GET(self):
        if self.path.startswith("/terrain/tasks/"):
            tid = self.path.rsplit("/", 1)[1]
            n = Fake.polls[tid] = Fake.polls.get(tid, 0) + 1
            if tid == "fail":
                return self._send(200, {"taskId": tid, "status": "FAILURE", "error": "Traceback ...\nNoPointClouds: none"})
            return self._send(200, {"taskId": tid, "status": "SUCCESS", "result": 7} if n > 1
                              else {"taskId": tid, "status": "PENDING"})
        if self.path == "/terrain/products/7":
            return self._send(200, {"id": 7, "resolution": 1, "paths": {"netcdf": "/terrain_products/k/g/dem_1m.nc"}})
        return self._send(404, {"error": "no terrain product"})


@pytest.fixture
def server():
    Fake.posts, Fake.polls = [], {}
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_client_dem_waits_and_returns_the_row(server):
    api = TerrainApi(server)
    row = api.dem(WKT, {"resolution": 1})
    assert row["id"] == 7 and Fake.polls["t1"] == 2
    with pytest.raises(TerrainApiError, match="HTTP 400: resolution must be one of"):
        api.submit(WKT, {"resolution": 3})
    with pytest.raises(TerrainApiError, match="failed:\nTraceback"):
        api.wait(api.submit(WKT, {"backend": "geofabrics"}), poll_s=0)
    with pytest.raises(TerrainApiError, match="is the EDDIE backend running"):
        TerrainApi("http://127.0.0.1:9", timeout_s=2).task("x")


def test_cli_api(server, tmp_path, capsys):
    aoi = tmp_path / "a.geojson"
    aoi.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                   "geometry": {"type": "Polygon", "coordinates": [[[173.4603, -35.4633], [173.471, -35.4633],
                                                                    [173.471, -35.457], [173.4603, -35.4633]]]}}]}))
    rc = cli.main(["api", "dem", "--aoi", str(aoi), "--resolution", "1", "--rivers", "--param", "buffer_m=200",
                   "--url", server, "--quiet", "--out-json", str(tmp_path / "row.json")])
    assert rc == 0 and json.loads(capsys.readouterr().out)["id"] == 7
    sent = Fake.posts[-1]
    assert sent["params"] == {"resolution": 1, "rivers": True, "buffer_m": 200}     # only what was given
    assert sent["aoi"].startswith("POLYGON ((173.4603 -35.4633")                      # WKT in EPSG:4326
    assert json.loads((tmp_path / "row.json").read_text())["paths"]["netcdf"].endswith("dem_1m.nc")
    assert cli.main(["api", "dem", "--aoi", WKT, "--no-wait", "--url", server, "--quiet"]) == 0
    assert json.loads(capsys.readouterr().out) == {"taskId": "t1"}
    assert cli.main(["api", "product", "7", "--url", server]) == 0
    capsys.readouterr()
    assert cli.main(["api", "product", "8", "--url", server]) == 1
    assert "HTTP 404: no terrain product" in capsys.readouterr().err
    assert cli.main(["api", "task", "fail", "--url", server]) == 0
    assert "NoPointClouds" in capsys.readouterr().err
