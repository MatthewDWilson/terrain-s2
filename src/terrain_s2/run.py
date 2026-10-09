"""Library entry point for the Stage 2 network chain (integration plan S2-1).

``network(...)`` runs the chain that ``scripts/run_network.py`` ran (the script is now a thin wrapper)
and returns a ``NetworkRun``; the outputs are unchanged.

Writes ``<out>/network.gpkg`` with layers:
  channels   reaches (chained through junctions, smoothed, generalised): class (stream/drain),
             p_stream (after smoothing along the network), p_stream_raw, class_confidence,
             through_culvert, and the reach features
  crossings  culvert candidates: x, y, sources (gap / bump / testA / testB / road_x_drain /
             road_end_pair), dem_h_b, deck_m (DSM deck over the crossing), on_stream
             (drain-bed rise, m), veg_frac, road_dist_m, tier (with roads if given), tier_dem_only
  repairs    continuity joins (gaps whose bed rises < 0.15 m): DEM-conditioning repairs
Roads are optional; without them the road-based sources are absent and tier = tier_dem_only.
Also ``network.json`` (summary, timings, peak memory) and, with ``rasters``, ``features.tif``,
``hydro.tif``, ``conditioned.tif`` and ``rasters.json``.

    from terrain_s2.run import network
    run = network("dem.tif", "aoi.geojson", "out/site", dsm="dsm.tif", roads="roads.gpkg", rasters="none")
    run.info["crossings"], run.layers["crossings"]

``keep=True`` also returns the in-memory state (DEM window, pipeline result, network, rasters) as
``run.state``, for feature extraction while the arrays are in memory (ML-0). It holds the run's memory
(~220 bytes per cell), so it is off by default.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import ndimage as ndi

from . import channels, dataio, drains, floodplain, pipeline, products
from . import network as network_mod
from .backend import get_backend
from .config import Params
from .perf import StageTimer


@dataclass
class NetworkRun:
    info: dict                                  # what network.json holds
    out: Path
    layers: dict = field(default_factory=dict)  # name -> GeoDataFrame, as written (cropped to the AOI)
    state: dict | None = None                   # keep=True: w, dsm, res, net, sk, fp, rem, rd, cond, ch, cr, ...


def _paths(p):
    if p is None:
        return None
    if isinstance(p, (str, Path)):
        return [str(p)]
    return [str(x) for x in p]


def _burn(sk, geom, T):
    from skimage.draw import line
    pts = np.array(geom.coords)
    cc, rr = drains._to_cr(T, pts[:, 0], pts[:, 1])
    for i in range(len(pts) - 1):
        r_, c_ = line(int(rr[i]), int(cc[i]), int(rr[i + 1]), int(cc[i + 1]))
        ok = (r_ >= 0) & (r_ < sk.shape[0]) & (c_ >= 0) & (c_ < sk.shape[1])
        sk[r_[ok], c_[ok]] = True


def _near_any(geoms, targets, d, strict=False):
    """Boolean per geometry: within ``d`` of any target (< d if ``strict``), by spatial index rather
    than all pairs (the pairwise loops grew with the square of the area: ~64 x for a full LINZ tile)."""
    from shapely import STRtree, distance
    out = np.zeros(len(geoms), bool)
    if not len(geoms) or not len(targets):
        return out
    gi, ti = STRtree(targets).query(geoms, predicate="dwithin", distance=d)
    if strict and len(gi):
        keep = distance(np.asarray(geoms, object)[gi], np.asarray(targets, object)[ti]) < d
        gi = gi[keep]
    out[gi] = True
    return out


def _conditioned_pass(a, w, hcrs, ch, culvert_edges, rep, riv, cr, rd):
    """Second hydrology pass on the DEM conditioned with the mapped network (terrain_s2.conditioning)."""
    import geopandas as gpd
    from terrain_s2 import conditioning
    rec2 = None
    ctx = Path(a.context) if a.context else Path(a.dem[0]).with_name("context.gpkg")
    if ctx.exists():
        import pyogrio
        if "rivers_rec2" in {n for n, _ in pyogrio.list_layers(ctx)}:
            rec2 = gpd.read_file(ctx, layer="rivers_rec2").to_crs(hcrs)
    links = [r["geometry"] for r in culvert_edges] + (list(rep.geometry) if len(rep) else [])
    return conditioning.run(w.z, w.transform, hcrs, list(ch.geometry), links, river_mask=riv, rec2=rec2,
                            window_bounds=w.bounds, candidates=cr if len(cr) else None, rd=rd, burn_m=a.burn_m,
                            height_model=a.height_model, rem_lines=list(ch.geometry[ch["class"] == "stream"]))


def _cond_summary(cond, core):
    P = cond.ponds[cond.ponds.intersects(core)] if len(cond.ponds) else cond.ponds
    by = P.status.value_counts().to_dict() if len(P) else {}
    O = cond.outlets[cond.outlets.intersects(core)] if cond.outlets is not None and len(cond.outlets) else []
    return dict(rec2_inflows=cond.n_seeded, residual_ponds=int(len(P)), pond_outlets=int(len(O)),
                ponds_by_status={k: int(v) for k, v in by.items()},
                network_cells=int(cond.drain.sum()), height_defined_frac=round(float(np.isfinite(cond.hand).mean()), 3),
                rem_basis=cond.rem_basis, rem_samples=cond.rem_samples)


def _write_rasters(out, which, w, core_bounds, res, net, sk, fp, hand, dsm, rd, cond=None, height_model="rem"):
    """features.tif and hydro.tif on the window grid, cropped to the AOI (tiles then mosaic without
    overlap), float32, deflate; rasters.json gives each band's units and meaning."""
    import rasterio.windows
    from terrain_s2 import bands as B
    win = rasterio.windows.from_bounds(*core_bounds, transform=w.transform).round_offsets().round_lengths()
    r0, c0 = max(int(win.row_off), 0), max(int(win.col_off), 0)
    r1, c1 = min(r0 + int(win.height), w.z.shape[0]), min(c0 + int(win.width), w.z.shape[1])
    tr = rasterio.windows.transform(rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0), w.transform)
    crop = lambda a: np.asarray(a)[r0:r1, c0:c1]  # noqa: E731
    valid = np.isfinite(w.z)
    meta = {}
    if which == "all":
        f = {k: crop(v) for k, v in res.feats.items()}
        dataio.write_stack(out / "features.tif", f, tr, w.crs)
        meta["features.tif"] = B.describe(list(f), B.FEATURES)
    h = {}
    if hand is not None:
        h["rem_m" if height_model == "rem" else "hand_m"] = hand
        h["floodplain"] = np.where(valid, fp.astype(np.float32), np.nan)
    h["log10_upstream_area_m2"] = np.where(valid, np.log10(np.maximum(res.upa, 1.0)), np.nan)
    h["breach_depth_m"] = np.where(valid, w.z - res.breach.z_breached, np.nan)
    h["depression_depth_m"] = np.where(valid, res.dep.depth, np.nan)
    h["channel_mask"] = net["mask"].astype(np.float32)
    h["channel_centreline"] = sk.astype(np.float32)
    if dsm is not None:
        h["dsm_minus_dem_m"] = dsm - w.z
    if rd is not None:
        h["road_distance_m"] = rd
    h = {k: crop(v) for k, v in h.items()}
    dataio.write_stack(out / "hydro.tif", h, tr, w.crs)
    meta["hydro.tif"] = B.describe(list(h), B.HYDRO)
    if cond is not None:
        c = {"z_conditioned_m": cond.z_conditioned, ("rem_m" if height_model == "rem" else "hand_m"): cond.hand,
             "log10_upstream_area_m2": np.where(valid, np.log10(np.maximum(cond.upa, 1.0)), np.nan),
             "upstream_inflow_m2": cond.upa_inflow.astype(np.float32),
             "residual_depression_m": cond.residual, "network": cond.drain.astype(np.float32)}
        c = {k: crop(v) for k, v in c.items()}
        dataio.write_stack(out / "conditioned.tif", c, tr, w.crs)
        meta["conditioned.tif"] = B.describe(list(c), B.CONDITIONED)
    (out / "rasters.json").write_text(json.dumps(meta, indent=2))


def network(dem, aoi, out, *, dsm=None, roads=None, stream_model=None, buffer=200.0, max_hand=10.0, clean=True,
            aoi_crs=None, assume_crs=None, device="auto", height_model="rem", conditioned=True, burn_m=0.25,
            context=None, rasters="all", profile=False, keep=False) -> NetworkRun:
    """Run the network chain for one AOI. Arguments mirror ``scripts/run_network.py`` (``clean`` and
    ``conditioned`` negate ``--no-clean`` and ``--no-conditioned``); ``dem`` and ``dsm`` are a path or a
    list of paths (mosaicked)."""
    a = SimpleNamespace(dem=_paths(dem), dsm=_paths(dsm), aoi=str(aoi), out=str(out), roads=str(roads) if roads else None,
                        stream_model=str(stream_model) if stream_model else None, buffer=buffer, max_hand=max_hand,
                        no_clean=not clean, aoi_crs=aoi_crs, assume_crs=assume_crs, device=device,
                        height_model=height_model, no_conditioned=not conditioned, burn_m=burn_m,
                        context=str(context) if context else None, rasters=rasters, profile=profile)
    import geopandas as gpd
    from shapely.geometry import LineString, box
    prof = None
    if a.profile:
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    T = StageTimer(sample=not a.profile)
    hcrs = dataio.processing_crs(a.dem[0], a.assume_crs)
    aoi = dataio.read_aoi(a.aoi, hcrs, a.aoi_crs)
    b = tuple(aoi.total_bounds)
    w = dataio.read_window(a.dem, b, a.buffer, assume_crs=a.assume_crs)
    dsm = dataio.read_window(a.dsm, b, a.buffer, assume_crs=a.assume_crs).z if a.dsm else None
    T.mark("read")
    P = Params(); P.candidates.hard_gates = False; cs = abs(w.transform.a)
    roads = gpd.read_file(a.roads).to_crs(hcrs) if a.roads else None
    if roads is not None:
        roads = roads[roads.intersects(box(*w.bounds))]
    # Tests A and B (ungated) and the shared features
    T.mark("read_roads")
    res = pipeline.run(w.z, w.transform, P, get_backend(a.device), dsm=dsm)
    T.mark("pipeline")
    T.add(res.timings, "pipeline.")
    strong = channels.channel_mask(res.feats, cs, P.channels, w.z, res.upa, filter_fragments=False)
    # floodplain (HAND, DEM only): restrict the network and crossings to it (plus a buffer)
    fp = fp_buf = None
    if a.max_hand > 0:
        flw = res.flw                      # routed once in pipeline.run (was recomputed here: fill + D8 again)
        wide = (ndi.distance_transform_edt(strong) * cs) >= 4.0
        fp, hand = floodplain.floodplain_mask(w.z, flw, res.upa, wide, cs, a.max_hand, transform=w.transform,
                                              height_model=a.height_model)
        fp_buf = ndi.binary_dilation(fp, iterations=int(round(50 / cs)))
        strong &= fp_buf
    T.mark("floodplain")
    net = drains.build_drain_network(strong, res.feats, w.z, w.transform, dsm)
    T.mark("drain_network")
    if fp_buf is not None:
        net["skeleton"] &= fp_buf
    # culvert edges: accepted Test A paths become part of the network, so channel type and flow
    # continue through culverts (e.g. SH12, a culvert on a DEM drainage divide)
    sk = net["skeleton"].copy()
    culvert_edges = [r for r in res.test_a if r["test_a"] and r["kind"] == "crossing" and r["h_b"] >= 0.3]
    for r in culvert_edges:
        _burn(sk, r["geometry"], w.transform)
    # rivers: one centreline per wide channel instead of lines along both banks
    rpolys, riv = network_mod.river_polygons(net["mask"], w.transform)
    if riv.any():
        sk = network_mod.river_centrelines(riv, sk, w.transform)
    T.mark("culverts_rivers")
    nodes, edges, reaches = network_mod.build_reaches(sk, w.transform)
    T.mark("reaches")
    rd = rdir = None
    if roads is not None and len(roads):
        rd, rdir = products._road_rasters(roads, sk.shape, w.transform)
    Fr = network_mod.reach_features(reaches, net["mask"], res.feats, w.z, w.transform, rd, rdir, riv)
    T.mark("reach_features")
    # cleaning: drop isolated pieces < 50 m and dangling reaches < 15 m (keep culvert links)
    if not a.no_clean:
        through = list(np.flatnonzero(_near_any([LineString(R.coords) for R in reaches],
                                                [r["geometry"].centroid for r in culvert_edges], 2.0, strict=True)))
        kept = network_mod.clean_reaches(reaches, keep=through, incision=Fr.incision_m.values)
        n_removed = len(reaches) - len(kept)
        reaches = [reaches[i] for i in kept]
        Fr = Fr.iloc[kept].reset_index(drop=True)
    else:
        n_removed = 0
    T.mark("cleaning")
    if a.stream_model:
        import joblib
        p0 = joblib.load(a.stream_model).predict_proba(network_mod.design_matrix(Fr))[:, 1]
    else:
        p0 = np.asarray(network_mod.stream_prior(Fr), float)
    p = network_mod.smooth_on_graph(reaches, Fr, np.asarray(p0, float))
    ch = gpd.GeoDataFrame(Fr, geometry=[LineString(R.coords) for R in reaches], crs=hcrs)
    ch["p_stream_raw"] = p0
    ch["p_stream"] = p
    ch["class"] = np.where(p >= 0.5, "stream", "drain")
    ch["class_confidence"] = np.maximum(p, 1 - p)
    ch["through_culvert"] = _near_any([LineString(R.coords) for R in reaches], [r["geometry"].centroid for r in culvert_edges],
                                      2.0, strict=True)
    ch = ch[ch.length > 0]
    T.mark("classify_reaches")
    C = products.crossings(net, w.z, w.transform, dsm, roads)
    T.mark("crossings")
    C = products.merge_test_candidates(C, res.candidates)
    C = products.add_context(C, w.z, dsm, w.transform, ch, roads)
    # crossings on Test A paths accepted into the network as culvert links count as network evidence
    links = [r["geometry"] for r in culvert_edges]
    from shapely.geometry import Point
    C["network_link"] = _near_any([Point(x, y) for x, y in zip(C.x, C.y)], links, 8.0) if links else False
    C = products.finalise_crossings(C, roads is not None)
    T.mark("crossing_context")
    cr = gpd.GeoDataFrame(C, geometry=gpd.points_from_xy(C.x, C.y), crs=hcrs)
    # crossings must sit on the cleaned network (or come from Test A / Test B evidence)
    if len(cr) and len(ch):
        on_net = _near_any(list(cr.geometry), list(ch.geometry), 10.0)
        tests = cr.sources.astype(str).str.contains("testA|testB")
        cr = cr[on_net | tests]
    if fp_buf is not None and len(cr):
        # judged by the land around the crossing (the 50 m buffered floodplain): an embankment crest
        # can stand more than max_hand above the drainage while both its channels are on the floodplain
        cc_, rr_ = drains._to_cr(w.transform, cr.x.values, cr.y.values)
        cr = cr[fp_buf[rr_.astype(int), cc_.astype(int)]]
    T.mark("crossing_filters")
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
    cond = None
    if not a.no_conditioned:
        T.mark("write")
        cond = _conditioned_pass(a, w, hcrs, ch, culvert_edges, rep, riv, cr, rd)
        for name, g in (("residual_ponds", cond.ponds), ("pond_outlets", cond.outlets)):
            g = g[g.intersects(core)] if len(g) else g
            if len(g):
                g.to_file(gpkg, layer=name, driver="GPKG")
        T.mark("conditioned_hydrology")
    if a.rasters != "none":
        _write_rasters(out, a.rasters, w, aoi.total_bounds, res, net, sk, fp, hand if fp is not None else None, dsm, rd,
                       cond, a.height_model)
    T.mark("write")
    T.stop()
    info = dict(dem=a.dem, dsm=a.dsm, roads=a.roads, stream_model=a.stream_model, aoi=a.aoi,
                seconds=round(time.perf_counter() - t0, 1),
                channels_km=round(float(ch[ch.intersects(core)].length.sum() / 1000), 2),
                crossings={k: int(v) for k, v in cr[cr.intersects(core)].tier.value_counts().items()},
                repairs=int(rep.intersects(core).sum()) if len(rep) else 0,
                reaches_removed_by_cleaning=int(n_removed), rivers=len(riv_g),
                floodplain_frac=round(float(fp.mean()), 3) if fp is not None else None,
                cells=int(w.z.size), device=res.device,
                conditioned=_cond_summary(cond, core) if cond is not None else None, perf=T.report())
    (out / "network.json").write_text(json.dumps(info, indent=2))
    if prof is not None:
        import io
        import pstats
        prof.disable()
        prof.dump_stats(out / "profile.prof")
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(40)
        pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(30)
        (out / "profile_top.txt").write_text(buf.getvalue())
    written = {}
    if gpkg.exists():
        import pyogrio
        written = {n: gpd.read_file(gpkg, layer=n) for n, _ in pyogrio.list_layers(gpkg)}
    state = None
    if keep:
        state = dict(w=w, dsm=dsm, res=res, net=net, sk=sk, fp=fp, rem=hand if fp is not None else None, rd=rd,
                     cond=cond, ch=ch, cr=cr, rep=rep, culvert_edges=culvert_edges, aoi=aoi, crs=hcrs)
    return NetworkRun(info=info, out=out, layers=written, state=state)
