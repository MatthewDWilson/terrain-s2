"""Library entry point for the Stage 2 network chain (integration plan S2-1).

``network(...)`` runs the chain that ``scripts/run_network.py`` ran (the script is now a thin wrapper)
and returns a ``NetworkRun``; the outputs are unchanged.

Writes ``<out>/network.gpkg`` with up to seven layers (a layer with nothing in the AOI is not written):
  channels        reaches (chained through junctions, smoothed, generalised): class (stream/drain), p_stream
                  (after smoothing along the network), p_stream_raw, class_confidence, through_culvert, and the
                  reach features
  crossings       culvert and bridge candidates: x, y, sources (gap / bump / testA / testB / road_x_drain /
                  road_end_pair), dem_h_b, deck_m (DSM deck over the crossing), on_stream (within 10 m of a stream
                  reach), veg_frac, road_dist_m, tier (with roads if given), tier_dem_only; with a crossing model
                  (ML-1) also p_structure (a ranking score), p_bridge, model_version, recorded and record, and
                  recorded structures with no candidate added (tier "recorded")
  repairs         continuity joins (gaps whose bed rises < 0.15 m): DEM-conditioning repairs
  rivers          wide-channel polygons (channels at least 8 m wide), with area_m2
  floodplain      the floodplain (REM below its threshold), as polygons
  residual_ponds  depressions left after the conditioned hydrology pass (not with ``no_conditioned``)
  pond_outlets    their outlets
Roads are optional; without them the road-based sources are absent and tier = tier_dem_only.
Also ``network.json`` (summary, timings, peak memory) and, with ``rasters``, ``features.tif``,
``hydro.tif``, ``conditioned.tif`` and ``rasters.json`` (band names in ``bands.py``).

    from terrain_s2.run import network
    res = network("dem.tif", "dsm.tif", "aoi.geojson", roads="roads.gpkg", out="out/site", rasters="none")
    res.info["crossings"], res.layers["crossings"], res.paths["network"]

``keep=True`` also returns the in-memory state (DEM window, pipeline result, network, rasters) as
``res.state``, for feature extraction while the arrays are in memory (ML-0). It holds the run's memory
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
class NetworkResult:
    info: dict                                  # what network.json holds (summary, conditioned pass, perf)
    out: Path
    layers: dict = field(default_factory=dict)  # name -> GeoDataFrame, as written (cropped to the AOI)
    paths: dict = field(default_factory=dict)   # network, summary, features, hydro, conditioned, rasters_json
    provenance: dict | None = None              # S2-13 (provenance.json) when it lands
    state: dict | None = None                   # keep=True: w, dsm, res, net, sk, fp, rem, rd, cond, ch, cr, ...

    @property
    def perf(self) -> dict:
        return self.info.get("perf", {})


NetworkRun = NetworkResult                      # the name in the first S2-1 commit (aa975d8)

OPTIONS = dict(stream_model=None, buffer=200.0, max_hand=10.0, clean=True, aoi_crs=None, assume_crs=None,
               device="auto", height_model="rem", conditioned=True, burn_m=0.25, context=None, rasters="all",
               profile=False, keep=False, crossing_model=None, record_sources=None)


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


def network(dem, dsm=None, aoi=None, roads=None, params=None, out=None, **options) -> NetworkResult:
    """Run the network chain for one AOI (integration plan S2-1).

    ``dem``, ``dsm``: a path or a list of paths (mosaicked). ``aoi``: a path, a shapely geometry, a GeoSeries
    or a GeoDataFrame (a geometry without a CRS is taken in ``aoi_crs``, else the DEM's CRS). ``roads``: a
    vector path. ``params`` and ``options`` set the run options (``OPTIONS``: the script's flags, ``clean``
    and ``conditioned`` negating ``--no-clean`` and ``--no-conditioned``); ``options`` win. ``out``: the
    output folder (a temporary folder if None)."""
    opts = dict(OPTIONS)
    for src in (params or {}), options:
        bad = sorted(set(src) - set(OPTIONS))
        if bad:
            raise ValueError(f"unknown network option(s) {bad}; known: {sorted(OPTIONS)}")
        opts.update(src)
    if aoi is None:
        raise ValueError("network() needs an aoi (path, geometry or GeoDataFrame)")
    if out is None:
        import tempfile
        out = tempfile.mkdtemp(prefix="terrain_network_")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    aoi_path = _aoi_path(aoi, out, opts["aoi_crs"], _paths(dem)[0], opts["assume_crs"])
    a = SimpleNamespace(dem=_paths(dem), dsm=_paths(dsm), aoi=aoi_path, out=str(out), roads=str(roads) if roads else None,
                        stream_model=str(opts["stream_model"]) if opts["stream_model"] else None, buffer=opts["buffer"],
                        max_hand=opts["max_hand"], no_clean=not opts["clean"], aoi_crs=opts["aoi_crs"],
                        assume_crs=opts["assume_crs"], device=opts["device"], height_model=opts["height_model"],
                        no_conditioned=not opts["conditioned"], burn_m=opts["burn_m"],
                        context=str(opts["context"]) if opts["context"] else None, rasters=opts["rasters"],
                        profile=opts["profile"])
    keep = opts["keep"]
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
                cells=int(w.z.size), device=res.device, gpu_pool_gb=res.gpu_pool_gb,
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
    if opts.get("crossing_model"):                       # ML-1 step 9: p_structure, p_bridge, recorded structures
        from .ml.apply import score_run
        tmp = NetworkResult(info=info, out=out, layers=written,
                            state=dict(w=w, dsm=dsm, res=res, net=net, sk=sk, fp=fp, rem=hand if fp is not None else None,
                                       rd=rd, cond=cond, ch=ch, cr=cr, aoi=aoi, crs=hcrs))
        rs = opts.get("record_sources")
        if rs is None:
            from .ml.labels import load_config as _lc
            lc = Path(__file__).resolve().parents[2] / "ml" / "config" / "labels.yml"
            c = _lc(lc if lc.exists() else None)
            rs = list(c.get("council_sources") or []) + list(c.get("asset_sources") or [])
        scored = score_run(tmp, opts["crossing_model"], Path(a.dem[0]).parent, rs)
        scored.to_file(gpkg, layer="crossings", driver="GPKG")
        written["crossings"] = scored
        info["scored"] = dict(model=scored["model_version"].iloc[0] if len(scored) else None,
                              recorded=int(scored["recorded"].sum()), added_records=int((scored["tier"] == "recorded").sum()))
        (out / "network.json").write_text(json.dumps(info, indent=2))
    paths = {k: out / f for k, f in (("network", "network.gpkg"), ("summary", "network.json"), ("features", "features.tif"),
                                     ("hydro", "hydro.tif"), ("conditioned", "conditioned.tif"),
                                     ("rasters_json", "rasters.json")) if (out / f).exists()}
    state = None
    if keep:
        state = dict(w=w, dsm=dsm, res=res, net=net, sk=sk, fp=fp, rem=hand if fp is not None else None, rd=rd,
                     cond=cond, ch=ch, cr=cr, rep=rep, culvert_edges=culvert_edges, aoi=aoi, crs=hcrs)
    return NetworkResult(info=info, out=out, layers=written, paths=paths, state=state)


def _aoi_path(aoi, out, aoi_crs, dem0, assume_crs):
    """A path for dataio.read_aoi: as given, or the geometry written to <out>/aoi_input.geojson."""
    if isinstance(aoi, (str, Path)):
        return str(aoi)
    import geopandas as gpd
    if isinstance(aoi, gpd.GeoDataFrame):
        g = aoi
    elif isinstance(aoi, gpd.GeoSeries):
        g = gpd.GeoDataFrame(geometry=aoi)
    else:
        g = gpd.GeoDataFrame(geometry=[aoi])
    if g.crs is None:
        g = g.set_crs(aoi_crs or dataio.processing_crs(dem0, assume_crs))
    f = Path(out) / "aoi_input.geojson"
    g[["geometry"]].to_file(f, driver="GeoJSON")
    return str(f)


def main(argv=None):
    """The command line (``scripts/run_network.py`` and ``terrain network``)."""
    import argparse
    import sys
    ap = argparse.ArgumentParser(prog="run_network", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    ap.add_argument("--profile", action="store_true", help="cProfile the run: <out>/profile.prof and profile_top.txt")
    ap.add_argument("--height-model", choices=["rem", "hand"], default="rem",
                    help="height above drainage for the floodplain and rasters: relative elevation model (IDW of channel "
                         "elevations, continuous) or HAND (nearest drainage along flow); default rem")
    ap.add_argument("--no-conditioned", action="store_true",
                    help="skip the conditioned hydrology pass (conditioned.tif, residual_ponds)")
    ap.add_argument("--burn-m", type=float, default=0.25, help="channel burn below the local bed, conditioned pass (m)")
    ap.add_argument("--context", help="context.gpkg with rivers_rec2 to seed upstream area at the window edge "
                                      "(default: context.gpkg beside the DEM, if present)")
    ap.add_argument("--rasters", choices=["all", "hydro", "none"], default="all",
                    help="write features.tif (DEM feature stack) and hydro.tif (HAND, upstream area, ...), cropped "
                         "to the AOI, with rasters.json describing the bands (all: both; hydro: hydro.tif only)")
    ap.add_argument("--crossing-model", help="score the crossings with a trained scorer (model.pkl or its folder): "
                                             "p_structure, p_bridge, recorded structures (ML-1 step 9)")
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    res = network(a.dem, a.dsm, a.aoi, roads=a.roads, out=a.out, stream_model=a.stream_model, buffer=a.buffer,
                  max_hand=a.max_hand, clean=not a.no_clean, aoi_crs=a.aoi_crs, assume_crs=a.assume_crs,
                  device=a.device, height_model=a.height_model, conditioned=not a.no_conditioned, burn_m=a.burn_m,
                  context=a.context, rasters=a.rasters, profile=a.profile, crossing_model=a.crossing_model)
    print(json.dumps(res.info))
    return 0
