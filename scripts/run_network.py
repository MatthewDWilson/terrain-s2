#!/usr/bin/env python
"""Drain/stream network and crossing products for an AOI (DESIGN_NOTES §4i).

    python scripts/run_network.py --dem dem.tif [--dsm dsm.tif] --aoi aoi.geojson \
        [--roads roads.gpkg] [--stream-model stream_model.joblib] --out out/site

Writes ``network.gpkg`` with layers:
  channels   reaches (chained through junctions, smoothed, generalised): class (stream/drain),
             p_stream (after smoothing along the network), p_stream_raw, class_confidence,
             through_culvert, and the reach features
  crossings  culvert candidates: x, y, sources (gap / bump / testA / testB / road_x_drain /
             road_end_pair), dem_h_b, deck_m (DSM deck over the crossing), on_stream
             (drain-bed rise, m), veg_frac, road_dist_m, tier (with roads if given), tier_dem_only
  repairs    continuity joins (gaps whose bed rises < 0.15 m): DEM-conditioning repairs
Roads are optional; without them the road-based sources are absent and tier = tier_dem_only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scipy import ndimage as ndi  # noqa: E402

from terrain_s2 import channels, dataio, drains, floodplain, hydro, network, pipeline, products  # noqa: E402
from terrain_s2.backend import get_backend  # noqa: E402
from terrain_s2.config import Params  # noqa: E402


def _burn(sk, geom, T):
    from skimage.draw import line
    pts = np.array(geom.coords)
    cc, rr = drains._to_cr(T, pts[:, 0], pts[:, 1])
    for i in range(len(pts) - 1):
        r_, c_ = line(int(rr[i]), int(cc[i]), int(rr[i + 1]), int(cc[i + 1]))
        ok = (r_ >= 0) & (r_ < sk.shape[0]) & (c_ >= 0) & (c_ < sk.shape[1])
        sk[r_[ok], c_[ok]] = True


def main():
    import geopandas as gpd
    from shapely.geometry import LineString, box
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", nargs="+", required=True)
    ap.add_argument("--dsm", nargs="*", default=None)
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--aoi-crs")
    ap.add_argument("--roads", help="optional road centrelines (any vector format)")
    ap.add_argument("--stream-model", help="optional trained stream/drain model (joblib)")
    ap.add_argument("--buffer", type=float, default=200.0)
    ap.add_argument("--max-hand", type=float, default=10.0, help="floodplain: height above nearest drainage (m); <= 0 disables")
    ap.add_argument("--no-clean", action="store_true", help="keep isolated pieces and short dangling reaches")
    ap.add_argument("--assume-crs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    hcrs = dataio.processing_crs(a.dem[0], a.assume_crs)
    aoi = dataio.read_aoi(a.aoi, hcrs, a.aoi_crs)
    b = tuple(aoi.total_bounds)
    w = dataio.read_window(a.dem, b, a.buffer, assume_crs=a.assume_crs)
    dsm = dataio.read_window(a.dsm, b, a.buffer, assume_crs=a.assume_crs).z if a.dsm else None
    P = Params(); P.candidates.hard_gates = False; cs = abs(w.transform.a)
    roads = gpd.read_file(a.roads).to_crs(hcrs) if a.roads else None
    if roads is not None:
        roads = roads[roads.intersects(box(*w.bounds))]
    # Tests A and B (ungated) and the shared features
    res = pipeline.run(w.z, w.transform, P, get_backend(a.device), dsm=dsm)
    strong = channels.channel_mask(res.feats, cs, P.channels, w.z, res.upa, filter_fragments=False)
    # floodplain (HAND, DEM only): restrict the network and crossings to it (plus a buffer)
    fp = fp_buf = None
    if a.max_hand > 0:
        _, flw = hydro.flow_accumulation(res.breach.z_breached, w.transform)
        wide = (ndi.distance_transform_edt(strong) * cs) >= 4.0
        fp, hand = floodplain.floodplain_mask(w.z, flw, res.upa, wide, cs, a.max_hand)
        fp_buf = ndi.binary_dilation(fp, iterations=int(round(50 / cs)))
        strong &= fp_buf
    net = drains.build_drain_network(strong, res.feats, w.z, w.transform, dsm)
    if fp_buf is not None:
        net["skeleton"] &= fp_buf
    # culvert edges: accepted Test A paths become part of the network, so channel type and flow
    # continue through culverts (e.g. SH12, a culvert on a DEM drainage divide)
    sk = net["skeleton"].copy()
    culvert_edges = [r for r in res.test_a if r["test_a"] and r["kind"] == "crossing" and r["h_b"] >= 0.3]
    for r in culvert_edges:
        _burn(sk, r["geometry"], w.transform)
    # rivers: one centreline per wide channel instead of lines along both banks
    rpolys, riv = network.river_polygons(net["mask"], w.transform)
    if riv.any():
        sk = network.river_centrelines(riv, sk, w.transform)
    nodes, edges, reaches = network.build_reaches(sk, w.transform)
    rd = rdir = None
    if roads is not None and len(roads):
        rd, rdir = products._road_rasters(roads, sk.shape, w.transform)
    Fr = network.reach_features(reaches, net["mask"], res.feats, w.z, w.transform, rd, rdir, riv)
    # cleaning: drop isolated pieces < 50 m and dangling reaches < 15 m (keep culvert links)
    if not a.no_clean:
        through = [i for i, R in enumerate(reaches)
                   if any(LineString(R.coords).distance(r["geometry"].centroid) < 2 for r in culvert_edges)]
        kept = network.clean_reaches(reaches, keep=through, incision=Fr.incision_m.values)
        n_removed = len(reaches) - len(kept)
        reaches = [reaches[i] for i in kept]
        Fr = Fr.iloc[kept].reset_index(drop=True)
    else:
        n_removed = 0
    if a.stream_model:
        import joblib
        p0 = joblib.load(a.stream_model).predict_proba(network.design_matrix(Fr))[:, 1]
    else:
        p0 = np.asarray(network.stream_prior(Fr), float)
    p = network.smooth_on_graph(reaches, Fr, np.asarray(p0, float))
    ch = gpd.GeoDataFrame(Fr, geometry=[LineString(R.coords) for R in reaches], crs=hcrs)
    ch["p_stream_raw"] = p0
    ch["p_stream"] = p
    ch["class"] = np.where(p >= 0.5, "stream", "drain")
    ch["class_confidence"] = np.maximum(p, 1 - p)
    ch["through_culvert"] = [any(LineString(R.coords).distance(r["geometry"].centroid) < 2 for r in culvert_edges)
                             for R in reaches]
    ch = ch[ch.length > 0]
    C = products.crossings(net, w.z, w.transform, dsm, roads)
    C = products.merge_test_candidates(C, res.candidates)
    C = products.add_context(C, w.z, dsm, w.transform, ch, roads)
    # crossings on Test A paths accepted into the network as culvert links count as network evidence
    links = [r["geometry"] for r in culvert_edges]
    from shapely.geometry import Point
    C["network_link"] = [any(g.distance(Point(x, y)) <= 8 for g in links) for x, y in zip(C.x, C.y)] if links else False
    C = products.finalise_crossings(C, roads is not None)
    cr = gpd.GeoDataFrame(C, geometry=gpd.points_from_xy(C.x, C.y), crs=hcrs)
    # crossings must sit on the cleaned network (or come from Test A / Test B evidence)
    if len(cr) and len(ch):
        on_net = cr.geometry.apply(lambda p: ch.distance(p).min() <= 10)
        tests = cr.sources.astype(str).str.contains("testA|testB")
        cr = cr[on_net | tests]
    if fp_buf is not None and len(cr):
        # judged by the land around the crossing (the 50 m buffered floodplain): an embankment crest
        # can stand more than max_hand above the drainage while both its channels are on the floodplain
        cc_, rr_ = drains._to_cr(w.transform, cr.x.values, cr.y.values)
        cr = cr[fp_buf[rr_.astype(int), cc_.astype(int)]]
    rep = gpd.GeoDataFrame([g for g in net["gaps"] if g["kind"] == "continuity"],
                           geometry=[LineString([(g["x0"], g["y0"]), (g["x1"], g["y1"])])
                                     for g in net["gaps"] if g["kind"] == "continuity"], crs=hcrs)
    core = aoi.union_all()
    gpkg = out / "network.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    riv_g = gpd.GeoDataFrame(rpolys, crs=hcrs) if rpolys else gpd.GeoDataFrame(geometry=[], crs=hcrs)
    if len(riv_g):
        riv_g["area_m2"] = riv_g.area
    fp_g = gpd.GeoDataFrame(geometry=[], crs=hcrs)
    if fp is not None and fp.any():
        from rasterio.features import shapes
        from shapely.geometry import shape
        fp_g = gpd.GeoDataFrame(geometry=[shape(g) for g, v in shapes(fp.astype(np.uint8), mask=fp, transform=w.transform) if v == 1], crs=hcrs)
    for name, g in (("channels", ch), ("crossings", cr), ("repairs", rep), ("rivers", riv_g), ("floodplain", fp_g)):
        g = g[g.intersects(core)]
        if len(g):
            g.to_file(gpkg, layer=name, driver="GPKG")
    info = dict(dem=a.dem, dsm=a.dsm, roads=a.roads, stream_model=a.stream_model, aoi=a.aoi,
                seconds=round(time.perf_counter() - t0, 1),
                channels_km=round(float(ch[ch.intersects(core)].length.sum() / 1000), 2),
                crossings={k: int(v) for k, v in cr[cr.intersects(core)].tier.value_counts().items()},
                repairs=int(rep.intersects(core).sum()) if len(rep) else 0,
                reaches_removed_by_cleaning=int(n_removed), rivers=len(riv_g),
                floodplain_frac=round(float(fp.mean()), 3) if fp is not None else None)
    (out / "network.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
