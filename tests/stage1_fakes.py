"""Fakes for Stage 1 tests: a static STAC catalogue of local COGs served through a fake HTTP layer."""
from __future__ import annotations

import json

import numpy as np

ROOT = "https://fake-stac.example/"


class FakeHttp:
    def __init__(self, docs: dict, binary: dict | None = None):
        self.docs, self.binary, self.calls = docs, binary or {}, []

    def json(self, url, params=None, source=None, **kw):
        self.calls.append((url, params))
        if url not in self.docs:
            raise IOError(f"no fake document for {url}")
        d = self.docs[url]
        return d(params) if callable(d) else json.loads(json.dumps(d))

    def get(self, url, params=None, source=None, ext="json", **kw):
        self.calls.append((url, params))
        if url in self.binary:
            return self.binary[url]
        return json.dumps(self.json(url, params)).encode()


def write_cog(path, z, x0, y1, crs="EPSG:2193", nodata=-9999.0):
    import rasterio
    from affine import Affine
    z = np.where(np.isfinite(z), z, nodata).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=z.shape[0], width=z.shape[1], count=1, dtype="float32",
                       crs=crs, transform=Affine(1, 0, x0, 0, -1, y1), nodata=nodata, tiled=True,
                       blockxsize=64, blockysize=64) as d:
        d.write(z, 1)
    return path


def _ll(b):
    from pyproj import Transformer
    return list(Transformer.from_crs(2193, 4326, always_xy=True).transform_bounds(*b))


def stac(tmp, surveys, root=ROOT):
    """``surveys``: [dict(region, slug, end, tiles=[(id, x0, y0, x1, y1, value_fn)], national=False)].
    Writes COGs under ``tmp`` and returns the fake documents (url -> JSON)."""
    docs, links = {}, []
    for s in surveys:
        rel = f"{s['region']}/{s['slug']}/dem_1m/2193/collection.json"
        curl = root + rel
        links.append({"rel": "child", "href": "./" + rel})
        items, xs = [], []
        for tid, x0, y0, x1, y1, fn in s["tiles"]:
            yy, xx = np.mgrid[y1 - 0.5:y0:-1, x0 + 0.5:x1:1]
            z = fn(xx, yy).astype(np.float32)
            p = write_cog(tmp / f"{s['slug']}_{tid}.tif", z, x0, y1)
            iurl = root + f"{s['region']}/{s['slug']}/dem_1m/2193/{tid}.json"
            docs[iurl] = {"id": tid, "bbox": _ll((x0, y0, x1, y1)),
                          "assets": {"visual": {"href": "file://" + str(p), "type": "image/tiff; application=geotiff",
                                                "file:checksum": f"1220{tid}{s['slug']}"}}}
            items.append({"rel": "item", "href": f"./{tid}.json"})
            xs.append((x0, y0, x1, y1))
        bb = (min(b[0] for b in xs), min(b[1] for b in xs), max(b[2] for b in xs), max(b[3] for b in xs))
        docs[curl] = {"id": s["slug"], "linz:slug": s["slug"], "title": s["slug"], "updated": s.get("updated", "2025-01-01T00:00:00Z"),
                      "linz:geospatial_category": "dem", "extent": {"spatial": {"bbox": [_ll(bb)]},
                      "temporal": {"interval": [[s.get("start", "2010-01-01T00:00:00Z"), s["end"]]]}},
                      "links": items}
    docs[root + "catalog.json"] = {"links": links}
    return docs


def nz_profile(national=None):
    from terrain_s2.stage1.profile import load
    p = json.loads(json.dumps(load("nz")))
    p["elevation"]["stac_root"] = ROOT
    if national is not None:
        p["elevation"]["national"] = {"dem": national}
    return p
