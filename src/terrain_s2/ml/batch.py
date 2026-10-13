"""ML-0 step 2: build and run many tiles, in parallel and resumably.

Per tile: the bundle (``acquire.build.build_site`` for the partition tile: elevation through the shared reader,
labels and context through ``sources: auto``), then the network chain (``run.network``). Each finished tile
gets ``<runs>/<tile>/done.json`` with the parameters, their hash and the code version; a tile whose
``done.json`` has the same parameter hash is skipped (``rerun=True`` forces it; ``rerun_older=True`` also reruns
tiles done by another code version).

With ``features`` (default), the run keeps its arrays and ``ml.features.extract`` writes the candidate
features and oriented patches (step 4) before they are freed; ``FEATURES_VERSION`` is in the hash.

Workers are processes (Windows spawns them, so everything here is importable). ``gpu_workers`` of them run the
feature stage on the GPU (``device="gpu"``), the rest on the CPU; size ``workers`` by peak memory (~8.7 GB per
LINZ tile, ~2.5-3.2 GB per quadrant, DESIGN_NOTES 7j and 11). Rasters are off by default (``rasters="none"``):
step 4 samples what the model needs while the arrays are in memory.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_PARAMS = dict(buffer=200.0, rasters="none")


@dataclass
class BatchConfig:
    partition: str
    bundles: str
    runs: str
    cache: str | None = None
    sources: str | None = None
    index: str | None = None
    params: dict = field(default_factory=dict)        # run.network options (buffer, rasters, height_model, ...)
    build: bool = True                                # (re)build the bundle if it has no site.json
    features: bool = True                             # step 4: ml_features.parquet and ml_patches.npy
    rerun: bool = False
    rerun_older: bool = False
    mirror: bool = True


def params_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _hashed(params: dict, cfg) -> dict:
    """What a tile's done.json hash covers: the network options and, with features, their version."""
    from .features import FEATURES_VERSION
    return dict(params, features=FEATURES_VERSION) if getattr(cfg, "features", False) else dict(params)


def _version() -> str:
    try:
        from .. import __version__
        return str(__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


def status(tile_id: str, cfg: BatchConfig, params: dict) -> str:
    """'todo', 'done' or 'stale' (done by another code version)."""
    f = Path(cfg.runs) / tile_id / "done.json"
    if not f.exists():
        return "todo"
    d = json.loads(f.read_text())
    if d.get("params_hash") != params_hash(_hashed(params, cfg)):
        return "todo"
    return "done" if d.get("version") == _version() else "stale"


def run_tile(tile_id: str, cfg: BatchConfig, device: str = "cpu", log=print) -> dict:
    """Bundle and network for one tile; returns a record (also written as done.json when it succeeds)."""
    from ..acquire import registry
    from ..acquire.build import build_site
    from ..acquire.elevation import default_data_dir
    from ..acquire.site import load_sources, tile_site
    from ..run import network
    root = Path(__file__).resolve().parents[3]
    params = dict(DEFAULT_PARAMS, **cfg.params)
    t0 = time.perf_counter()
    rec = dict(tile_id=tile_id, device=device, params=params, params_hash=params_hash(_hashed(params, cfg)),
               version=_version(), features=cfg.features)
    bundle = Path(cfg.bundles) / tile_id
    try:
        if cfg.build and not (bundle / "site.json").exists():
            sources_path = cfg.sources or str(root / "sites" / "sources.yml")
            idx_path = Path(cfg.index) if cfg.index else registry.default_index_path(sources_path)
            site = tile_site(tile_id, cfg.partition, params.get("buffer", 200.0))
            prov = build_site(site, load_sources(sources_path), cfg.bundles, cfg.cache or str(default_data_dir()),
                              index=registry.load_index(idx_path), mirror=cfg.mirror, log=log)
            rec["build_errors"] = prov["errors"]
            if any(e["step"].startswith("elevation") for e in prov["errors"]):
                raise RuntimeError(f"elevation failed: {prov['errors']}")
        rec["build_s"] = round(time.perf_counter() - t0, 1)
        dsm = bundle / "dsm.tif"
        res = network(bundle / "dem.tif", dsm if dsm.exists() else None, bundle / "aoi.geojson",
                      roads=bundle / "roads.gpkg" if (bundle / "roads.gpkg").exists() else None,
                      out=Path(cfg.runs) / tile_id, params=dict(params, device=device), keep=cfg.features)
        if cfg.features:
            from .features import extract
            t1 = time.perf_counter()
            tab, _, _ = extract(res, Path(cfg.runs) / tile_id)
            rec.update(features_s=round(time.perf_counter() - t1, 1), n_features=int(tab.shape[1]))
        rec.update(ok=True, seconds=round(time.perf_counter() - t0, 1), crossings=res.info.get("crossings"),
                   channels_km=res.info.get("channels_km"), peak_rss_gb=res.perf.get("peak_rss_gb"))
        del res
        (Path(cfg.runs) / tile_id / "done.json").write_text(json.dumps(rec, indent=1, default=str))
    except Exception as e:  # noqa: BLE001 - recorded; the batch carries on
        rec.update(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:],
                   seconds=round(time.perf_counter() - t0, 1))
        d = Path(cfg.runs) / tile_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "failed.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


def _worker(q, results, cfg_dict, device):
    cfg = BatchConfig(**cfg_dict)
    while True:
        tile_id = q.get()
        if tile_id is None:
            return
        results.put(run_tile(tile_id, cfg, device, log=lambda *a: None))


def run_batch(tile_ids, cfg: BatchConfig, workers: int = 1, gpu_workers: int = 0, log=print) -> list:
    """Run the tiles not yet done, ``workers`` processes (``gpu_workers`` of them on the GPU). Returns the
    records of the tiles run, in completion order; failures carry ``error`` and leave ``failed.json``."""
    params = dict(DEFAULT_PARAMS, **cfg.params)
    todo = []
    for t in tile_ids:
        s = status(t, cfg, params)
        if s == "todo" or cfg.rerun or (s == "stale" and cfg.rerun_older):
            todo.append(t)
    log(f"{len(tile_ids)} tiles: {len(todo)} to run, {len(tile_ids) - len(todo)} done "
        f"({workers} workers, {gpu_workers} on the GPU)")
    if not todo:
        return []
    out = []
    if workers <= 1:
        for i, t in enumerate(todo):
            r = run_tile(t, cfg, "gpu" if gpu_workers else "cpu", log=lambda *a: None)
            out.append(r)
            log(_line(i + 1, len(todo), r))
        return out
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q, results = ctx.Queue(), ctx.Queue()
    for t in todo:
        q.put(t)
    procs = []
    for i in range(min(workers, len(todo))):
        q.put(None)
        p = ctx.Process(target=_worker, args=(q, results, asdict(cfg), "gpu" if i < gpu_workers else "cpu"), daemon=True)
        p.start()
        procs.append(p)
    import queue
    while len(out) < len(todo):
        try:
            r = results.get(timeout=30)
        except queue.Empty:
            if not any(p.is_alive() for p in procs):          # a worker died (e.g. out of memory)
                done = {r_["tile_id"] for r_ in out}
                for t in todo:
                    if t not in done:
                        out.append(dict(tile_id=t, ok=False, device="?", seconds=0,
                                        error="worker exited without a result (out of memory?)"))
                        log(_line(len(out), len(todo), out[-1]))
                break
            continue
        out.append(r)
        log(_line(len(out), len(todo), r))
    for p in procs:
        p.join(timeout=60)
    return out


def _line(i, n, r):
    if r.get("ok"):
        c = r.get("crossings") or {}
        return (f"[{i}/{n}] {r['tile_id']} {r['device']}: {r['seconds']} s, {r.get('channels_km')} km, "
                f"{sum(c.values())} crossings, peak {r.get('peak_rss_gb')} GB")
    return f"[{i}/{n}] {r['tile_id']} {r['device']}: FAILED {r.get('error')}"
