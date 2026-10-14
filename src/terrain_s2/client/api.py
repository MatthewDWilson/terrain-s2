"""Client for an EDDIE instance's terrain API (``eddie_terrain``'s blueprint), standard library only.

    from terrain_s2.client.api import TerrainApi
    api = TerrainApi("http://localhost:5500")
    row = api.dem("POLYGON((...))", {"backend": "raster", "resolution": 1})   # waits; returns the product row
    api.submit(wkt, params) -> task id;  api.task(id) -> {"status", "result" | "error"};  api.product(id)

The routes are ``POST /terrain/dem``, ``GET /terrain/tasks/<id>`` and ``GET /terrain/products/<id>``. Under v5
the core may mount blueprints below ``/api`` (D9): give that in the URL (``http://host:5000/api``). The base URL
defaults to ``TERRAIN_API_URL``, else ``http://localhost:5500`` (the workstation's slot 5).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://localhost:5500"


class TerrainApiError(RuntimeError):
    """The API refused a request, or a task failed (the message carries the API's error text)."""


class TerrainApi:
    def __init__(self, url: str | None = None, timeout_s: float = 30.0):
        self.url = (url or os.environ.get("TERRAIN_API_URL") or DEFAULT_URL).rstrip("/")
        self.timeout_s = timeout_s

    def _call(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{self.url}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                d = json.loads(e.read() or b"{}")
            except ValueError:
                d = {}
            raise TerrainApiError(f"{method} {path}: HTTP {e.code}: {d.get('error') or e.reason}") from None
        except urllib.error.URLError as e:
            raise TerrainApiError(f"{method} {self.url}{path}: {e.reason} (is the EDDIE backend running? "
                                  "set --url or TERRAIN_API_URL)") from None

    def submit(self, aoi_wkt: str, params: dict | None = None) -> str:
        """Send ``ensure_dem`` for an AOI (WKT, EPSG:4326); returns the task id. Parameters are validated by
        the API before the task is sent."""
        _, d = self._call("POST", "/terrain/dem", {"aoi": aoi_wkt, "params": params or {}})
        if not d.get("taskId"):
            raise TerrainApiError(f"POST /terrain/dem: no task id in {d}")
        return d["taskId"]

    def task(self, task_id: str) -> dict:
        return self._call("GET", f"/terrain/tasks/{task_id}")[1]

    def product(self, product_id: int) -> dict:
        return self._call("GET", f"/terrain/products/{int(product_id)}")[1]

    def wait(self, task_id: str, timeout_s: float = 6 * 3600, poll_s: float = 3.0, progress=None) -> int:
        """Poll until the task finishes; returns the product id. Raises with the traceback on failure."""
        t0 = time.time()
        while True:
            st = self.task(task_id)
            status = st.get("status")
            if status == "SUCCESS":
                return int(st["result"])
            if status == "FAILURE":
                raise TerrainApiError(f"task {task_id} failed:\n{st.get('error', '')}")
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"task {task_id} still {status} after {timeout_s:.0f} s")
            if progress:
                progress(status, time.time() - t0)
            time.sleep(poll_s)

    def dem(self, aoi_wkt: str, params: dict | None = None, timeout_s: float = 6 * 3600, progress=None) -> dict:
        """Submit, wait and return the product's registry row (paths are as the worker sees them, e.g.
        ``/terrain_products/...``)."""
        return self.product(self.wait(self.submit(aoi_wkt, params), timeout_s=timeout_s, progress=progress))


def aoi_wkt(aoi: str, aoi_crs: str | None = None) -> str:
    """An AOI as WKT in EPSG:4326 from a vector file, WKT or GeoJSON (``aoi_crs`` for WKT not in 4326)."""
    from ..stage1.aoi import to_working
    return to_working(aoi, "EPSG:4326", aoi_crs).wkt
