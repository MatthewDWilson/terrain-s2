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
With ``channel_model`` (ML-3c: a U-Net run folder, ``ml/unet.py``), its channels join the physics channel map
before the network is built, and the candidates are those of the merged network plus those of the physics
network alone (so no evidence is lost where the model closes a gap at a culvert); ``channel_min_sharpness`` and
``channel_max_urban`` drop the model's cells that are smudged or urban (``channel_predict.sharpness_map``,
``building_cover``), or, without ``channel_min_sharpness``, ``channel_min_contrast``, ``channel_max_width_m`` and
``channel_max_urban`` drop whole components (``component_quality``), before they join: ``channels`` gains channel_source (physics / ml), ``crossings`` ml_channel (on a reach
only the model mapped), ``hydro.tif`` the band ml_channel_prob, and ``network.json`` a channel_model summary.
With ``model_set`` (a name in ``ml/model_sets/`` or a path, ``ml.model_set``), the channel model, its filter and
the scorer come from the set (options given explicitly win), and ``network.json`` records the set and its models.
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
               profile=False, keep=False, crossing_model=None, record_sources=None, channel_model=None,
               channel_threshold=None, channel_min_cells=30, channel_min_contrast=None, channel_max_width_m=None,
               channel_max_urban=None, channel_min_sharpness=None, model_set=None, osm_guidance=False,
               c_outputs=False)


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


GUIDE_CODES = dict(river=1, stream=2, canal=3, drain=4)


def _osm_guidance(a, w, res, riv, rpolys, sk, fp_buf):
    """Stage C's OSM guidance (``guidance``): river extents from OSM river areas refined on the DEM join the rivers
    (their polygons are redrawn from the union); snapped OSM waterway lines join the skeleton (within the floodplain
    restriction, as the rest of the network). Returns (polygons, river mask, skeleton, the burned OSM path, its tag
    codes, a summary for network.json)."""
    from . import guidance as G
    T_ = w.transform
    h, wd = w.z.shape
    bounds = (T_.c, T_.f + h * T_.e, T_.c + wd * T_.a, T_.f)
    ctx = a.context or Path(a.dem[0]).with_name("context.gpkg")
    g = G.load(ctx, bounds)
    rx = G.river_extent(g["areas"], w.z, T_)
    n_dem = int(riv.sum())
    riv = riv | rx
    if rx.any():
        rpolys = G.polygons(riv, T_)
    S = G.snap_waterways(g["lines"], res.feats, T_, river_mask=riv)
    n_lines = int(len(S))
    if len(S):
        S = S[S.snapped.astype(bool)]
    path = np.zeros(sk.shape, bool)
    kind = np.zeros(sk.shape, np.uint8)
    for geom, ww in zip(S.geometry, S.waterway):
        for part in getattr(geom, "geoms", [geom]):
            if part.geom_type != "LineString" or part.length <= 0:
                continue
            tmp = np.zeros(sk.shape, bool)
            _burn(tmp, part, T_)
            kind[tmp & (kind == 0)] = GUIDE_CODES.get(ww, 4)
            path |= tmp
    if fp_buf is not None:
        path &= fp_buf
        kind[~fp_buf] = 0
    sk = sk | path
    cs = abs(T_.a)
    info = dict(source=g["source"], river_areas=int((g["areas"].get("water_kind") == "river_area").sum())
                if len(g["areas"]) and "water_kind" in g["areas"] else 0,
                river_extent_km2=round(float(rx.sum()) * cs * cs / 1e6, 4), river_cells_dem=n_dem,
                river_cells=int(riv.sum()), waterway_lines=n_lines, snapped=int(len(S)),
                snapped_km=round(float(S.length.sum()) / 1000, 2) if len(S) else 0.0,
                by_kind={k: int(v) for k, v in S.waterway.value_counts().items()} if len(S) else {})
    return rpolys, riv, sk, path, kind, info


def _c_outputs(out, gpkg, core, w, ml_prob, mask, roads, cr, crs):
    """Stage C's stored outputs for stages B and A: ``channel_prob.tif`` (the U-Net's probability over the whole
    window, 0-250 as uint8, 255 = no data; a COG) and the ``links`` layer of ``network.gpkg`` (candidate links between
    the network's pieces with their evidence, ``connectivity.candidate_links``); returns a summary with the run's
    connectivity measures."""
    from . import connectivity
    info = {}
    if ml_prob is not None:
        import rasterio
        q = np.where(np.isfinite(ml_prob), np.clip(np.round(ml_prob * 250), 0, 250), 255).astype(np.uint8)
        prof = dict(driver="COG", height=q.shape[0], width=q.shape[1], count=1, dtype="uint8", crs=crs,
                    transform=w.transform, nodata=255, compress="deflate")
        with rasterio.open(out / "channel_prob.tif", "w", **prof) as f:
            f.write(q, 1)
        info["channel_prob"] = "channel_prob.tif"
    summ, comps = connectivity.measure(out, core)
    info["connectivity"] = summ
    if comps is not None:
        links = connectivity.candidate_links(gpkg, comps, w.z, w.transform, prob=ml_prob, mask=mask, crossings=cr,
                                             roads=roads)
        if len(links):
            links.to_file(gpkg, layer="links", driver="GPKG")
        info["links"] = int(len(links))
    return info


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


def _write_rasters(out, which, w, core_bounds, res, net, sk, fp, hand, dsm, rd, cond=None, height_model="rem",
                   ml_prob=None):
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
    if ml_prob is not None:
        h["ml_channel_prob"] = ml_prob
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
    explicit = {k for src in ((params or {}), options) for k, v in src.items() if v is not None}
    model_set = None
    if opts.get("model_set"):                       # the frozen ML parts by name (ml/model_sets/); explicit options win
        from .ml import model_set as MS
        model_set = MS.apply(opts, explicit)
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
    # ML-3c: the channel U-Net's channels join the physics channel map (union), so the network, its classes and
    # the crossing tests (A, B, road crossings, road-end pairs) see them too; reaches the model alone found are
    # flagged channel_source = "ml"
    ml_only = ml_prob = ml_info = None
    if opts.get("channel_model"):
        from .ml import channel_predict as CP
        ctx = a.context or (str(Path(a.roads).with_name("context.gpkg")) if a.roads else None)
        ways = CP.ways_distance(w.z.shape, w.transform, hcrs, roads, ctx)
        bands_in = CP.run_inputs(res.feats, w.z, res.upa, res.breach.z_breached, res.dep.depth,
                                 hand if fp is not None else None, dsm, ways)
        ml_prob, ml_mask, ml_info = CP.channel_mask_in_run(opts["channel_model"], bands_in, a.device,
                                                           opts.get("channel_threshold"), opts.get("channel_min_cells", 30))
        del bands_in, ways
        if opts.get("channel_min_sharpness") is not None or (opts.get("channel_max_urban") is not None and
                                                            opts.get("channel_min_contrast") is None and
                                                            opts.get("channel_max_width_m") is None):
            # cell by cell: sharp enough (CP.sharpness_map) and not urban, fragments dropped
            cover = CP.building_cover(w.z.shape, w.transform, hcrs, ctx) if opts.get("channel_max_urban") is not None else None
            before = int(ml_mask.sum())
            ml_mask = CP.keep_sharp(ml_mask, CP.sharpness_map(ml_prob, w.transform), cover,
                                    opts.get("channel_min_sharpness"), opts.get("channel_max_urban"),
                                    opts.get("channel_min_cells", 30))
            ml_info.update(cells_dropped=before - int(ml_mask.sum()),
                           rules=dict(min_sharpness=opts.get("channel_min_sharpness"), max_urban=opts.get("channel_max_urban")))
            del cover
        elif any(opts.get(k) is not None for k in ("channel_min_contrast", "channel_max_width_m", "channel_max_urban")):
            # drop the model's smudged components (braid-plain flow paths) and urban ones (piped drainage): the
            # second review found 6 of 89 confident model-only reaches real on an urban, an HBRC and a braid-plain tile
            cover = CP.building_cover(w.z.shape, w.transform, hcrs, ctx) if opts.get("channel_max_urban") is not None else None
            lab_c, qt = CP.component_quality(ml_prob, ml_mask, w.transform, cover)
            ml_mask, dropped = CP.keep_components(lab_c, qt, opts.get("channel_min_contrast"),
                                                  opts.get("channel_max_width_m"), opts.get("channel_max_urban"))
            ml_info.update(components=int(len(qt)), components_dropped=dropped,
                           rules=dict(min_contrast=opts.get("channel_min_contrast"),
                                      max_width_m=opts.get("channel_max_width_m"), max_urban=opts.get("channel_max_urban")))
            del lab_c, cover
        if fp_buf is not None:
            ml_mask &= fp_buf
        ml_only = ml_mask & ~strong
        ml_info.update(ml_only_cells=int(ml_only.sum()), physics_cells=int(strong.sum()))
        strong_physics = strong
        strong = strong | ml_mask
        T.mark("channel_model")
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
    osm_path = osm_kind = guidance_info = None
    if opts["osm_guidance"]:                 # network v2, stage C: OSM rivers and waterways, placed on the LiDAR
        rpolys, riv, sk, osm_path, osm_kind, guidance_info = _osm_guidance(a, w, res, riv, rpolys, sk, fp_buf)
        T.mark("osm_guidance")
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
    if ml_only is not None:                      # ML-3c: a reach mostly on cells only the U-Net mapped
        near_ml = ndi.binary_dilation(ml_only, iterations=1)
        ch["channel_source"] = ["ml" if len(R.cells) and near_ml[R.cells[:, 0], R.cells[:, 1]].mean() > 0.5 else "physics"
                                for R in reaches]
    if osm_path is not None:                     # share of each reach on a snapped OSM waterway, and its tag
        near_o = ndi.binary_dilation(osm_path, iterations=2)
        names = (None, "river", "stream", "canal", "drain")
        share, tag = [], []
        for R in reaches:
            if not len(R.cells):
                share.append(0.0); tag.append(None)
                continue
            share.append(round(float(near_o[R.cells[:, 0], R.cells[:, 1]].mean()), 3))
            k = osm_kind[R.cells[:, 0], R.cells[:, 1]]
            k = k[k > 0]
            tag.append(names[int(np.bincount(k).argmax())] if len(k) else None)
        ch["osm_share"], ch["osm_waterway"] = share, tag
    ch = ch[ch.length > 0]
    T.mark("classify_reaches")
    C = products.crossings(net, w.z, w.transform, dsm, roads)
    if ml_only is not None:
        # the candidates of the physics network as well: the U-Net draws channels through culverts (council lines
        # run through them), so on the merged network a gap at a culvert becomes a continuous drain and its gap
        # evidence was lost (6 ECan tiles, 20 Oct: gap -24 %, bump +20 %). With both, the channel model only adds
        # candidates and evidence (merged within 8 m below, keeping every source).
        import pandas as pd
        net_p = drains.build_drain_network(strong_physics, res.feats, w.z, w.transform, dsm)
        if fp_buf is not None:
            net_p["skeleton"] &= fp_buf
        Cp = products.crossings(net_p, w.z, w.transform, dsm, roads)
        ml_info["candidates_merged_network"], ml_info["candidates_physics_network"] = int(len(C)), int(len(Cp))
        C = pd.concat([C, Cp], ignore_index=True)
        del net_p
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
    if ml_only is not None and len(cr):           # ML-3c: candidates on a reach only the U-Net mapped
        mlr = list(ch[ch.channel_source == "ml"].geometry) if "channel_source" in ch else []
        cr["ml_channel"] = _near_any(list(cr.geometry), mlr, 10.0) if mlr else False
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
                       cond, a.height_model, ml_prob)
    T.mark("write")
    c_info = None
    if opts["c_outputs"]:                       # network v2, stage C: what stages B and A read
        c_info = _c_outputs(out, gpkg, core, w, ml_prob, net["mask"], roads, cr, hcrs)
        T.mark("c_outputs")
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
    if guidance_info is not None:
        info["guidance"] = guidance_info
    if c_info is not None:
        info["c_outputs"] = c_info
    if model_set is not None:
        info["model_set"] = model_set
    if ml_info is not None:
        chc = ch[ch.intersects(core)]
        mlc = chc[chc.channel_source == "ml"] if "channel_source" in chc else chc.iloc[0:0]
        crc = cr[cr.intersects(core)]
        info["channel_model"] = dict(ml_info, ml_reaches=int(len(mlc)), ml_km=round(float(mlc.length.sum() / 1000), 2),
                                     crossings_on_ml=int(crc.ml_channel.sum()) if "ml_channel" in crc else 0)
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
                                       rd=rd, cond=cond, ch=ch, cr=cr, aoi=aoi, crs=hcrs, ml_prob=ml_prob))
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
                     cond=cond, ch=ch, cr=cr, rep=rep, culvert_edges=culvert_edges, aoi=aoi, crs=hcrs, ml_prob=ml_prob)
    return NetworkResult(info=info, out=out, layers=written, paths=paths, state=state)


TILE_DEFAULTS = dict(buffer=200.0, rasters="none")


def tile(tile_id: str, model_set=None, out=None, bundles=None, partition=None, build: bool = True, params=None,
         sources=None, index=None, cache=None, mirror: bool = True, log=print, **options) -> NetworkResult:
    """One partition tile by Stage 2's own route (integration step 2): the tile's bundle (``tiles.ensure_bundle``:
    built if missing, with the DEM, DSM, roads, context and records the model set needs), then ``network`` on
    it with ``model_set`` (e.g. ``"ml3c_v1"``). ``out``: default ``<TERRAIN_DATA_DIR>/network/<tile>``. Options as
    ``network`` (``TILE_DEFAULTS``: the bundle's 200 m buffer, no rasters; ``options`` win over ``params``).
    ``network.json`` gains ``tile`` (id, bundle, the bundle's build errors)."""
    from . import tiles
    from .acquire.elevation import default_data_dir
    opts = dict(TILE_DEFAULTS, **(params or {}), **options)
    if model_set is not None:
        opts["model_set"] = model_set
    bundle, errors = tiles.ensure_bundle(tile_id, bundles, partition, opts["buffer"], sources, index, cache, mirror,
                                         build, log)
    inp = tiles.bundle_inputs(bundle)
    out = Path(out or default_data_dir() / "network" / tile_id)
    res = network(inp["dem"], inp["dsm"], inp["aoi"], roads=inp["roads"], out=out, params=opts)
    res.info["tile"] = dict(id=tile_id, bundle=str(bundle), build_errors=errors)
    (out / "network.json").write_text(json.dumps(res.info, indent=2, default=str))
    return res


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
    ap.add_argument("--model-set", help="the frozen ML parts by name (ml/model_sets/<name>.yml) or path: channel "
                                        "model, its filter and the scorer; explicit options win")
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    res = network(a.dem, a.dsm, a.aoi, roads=a.roads, out=a.out, stream_model=a.stream_model, buffer=a.buffer,
                  max_hand=a.max_hand, clean=not a.no_clean, aoi_crs=a.aoi_crs, assume_crs=a.assume_crs,
                  device=a.device, height_model=a.height_model, conditioned=not a.no_conditioned, burn_m=a.burn_m,
                  context=a.context, rasters=a.rasters, profile=a.profile, crossing_model=a.crossing_model,
                  model_set=a.model_set)
    print(json.dumps(res.info))
    return 0
