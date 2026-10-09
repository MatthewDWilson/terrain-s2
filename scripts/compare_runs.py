#!/usr/bin/env python
"""Compare two run_network output folders exactly: every layer (rows, columns, attributes, geometry),
every raster band, rasters.json and network.json (timings and paths excepted). For refactors that must
not change outputs (e.g. S2-1) and for CPU vs GPU runs.

    python scripts/compare_runs.py D:/eddie_data/runs/a D:/eddie_data/runs/b
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def compare(a, b, log=print):
    import geopandas as gpd
    import pyogrio
    import rasterio
    a, b = Path(a), Path(b)
    ok = True
    la = {n for n, _ in pyogrio.list_layers(a / "network.gpkg")}
    lb = {n for n, _ in pyogrio.list_layers(b / "network.gpkg")}
    if la != lb:
        log(f"layers only in one: {sorted(la ^ lb)}")
        ok = False
    for n in sorted(la & lb):
        x, y = gpd.read_file(a / "network.gpkg", layer=n), gpd.read_file(b / "network.gpkg", layer=n)
        same = (len(x) == len(y) and list(x.columns) == list(y.columns)
                and x.drop(columns="geometry").equals(y.drop(columns="geometry"))
                and all(g1.equals_exact(g2, 0) for g1, g2 in zip(x.geometry, y.geometry)))
        log(f"layer {n}: {len(x)} / {len(y)} rows, {'identical' if same else 'DIFFERENT'}")
        ok &= same
    for f in ("features.tif", "hydro.tif", "conditioned.tif"):
        pa, pb = a / f, b / f
        if not (pa.exists() or pb.exists()):
            continue
        if not (pa.exists() and pb.exists()):
            log(f"{f}: only in one")
            ok = False
            continue
        with rasterio.open(pa) as p, rasterio.open(pb) as q:
            same = p.descriptions == q.descriptions and p.transform == q.transform and np.array_equal(
                p.read(), q.read(), equal_nan=True)
            if not same and p.descriptions == q.descriptions and p.shape == q.shape:
                d = np.nanmax(np.abs(p.read().astype(float) - q.read().astype(float)), axis=(1, 2))
                log(f"{f}: max |difference| per band " + ", ".join(f"{n} {v:.3g}" for n, v in zip(p.descriptions, d)))
        log(f"{f}: {'identical' if same else 'DIFFERENT'}")
        ok &= same
    ia, ib = (json.loads((d / "network.json").read_text()) for d in (a, b))
    for d in (ia, ib):
        for k in ("seconds", "perf", "dem", "dsm", "roads", "aoi", "stream_model", "device"):
            d.pop(k, None)
    if ia != ib:
        log("network.json differs: " + json.dumps({k: (ia.get(k), ib.get(k)) for k in set(ia) | set(ib) if ia.get(k) != ib.get(k)}))
        ok = False
    ra, rb = a / "rasters.json", b / "rasters.json"
    if ra.exists() and rb.exists() and ra.read_text() != rb.read_text():
        log("rasters.json differs")
        ok = False
    log("IDENTICAL" if ok else "DIFFERENT")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    a = ap.parse_args()
    sys.exit(0 if compare(a.a, a.b) else 1)


if __name__ == "__main__":
    main()
