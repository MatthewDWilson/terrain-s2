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
    from . import code_version
    return code_version()


def status(tile_id: str, cfg: BatchConfig, params: dict) -> str:
    """'todo', 'done' or 'stale' (done by another code version)."""
    f = Path(cfg.runs) / tile_id / "done.json"
    if not f.exists():
        return "todo"
    d = json.loads(f.read_text())
    if d.get("params_hash") != params_hash(_hashed(params, cfg)):
        return "todo"
    return "done" if d.get("version") == _version() else "stale"


def _label_cfg():
    from .labels import load_config
    f = Path(__file__).resolve().parents[3] / "ml" / "config" / "labels.yml"
    return load_config(f if f.exists() else None)


def _bundle_ok(bundle: Path) -> bool:
    """A usable bundle (``tiles.bundle_ok``)."""
    from ..tiles import bundle_ok
    return bundle_ok(bundle)


def run_tile(tile_id: str, cfg: BatchConfig, device: str = "cpu", log=print) -> dict:
    """Bundle and network for one tile; returns a record (also written as done.json when it succeeds)."""
    from ..run import network
    from ..tiles import ensure_bundle
    params = dict(DEFAULT_PARAMS, **cfg.params)
    t0 = time.perf_counter()
    rec = dict(tile_id=tile_id, device=device, params=params, params_hash=params_hash(_hashed(params, cfg)),
               version=_version(), features=cfg.features)
    bundle = Path(cfg.bundles) / tile_id
    try:
        if cfg.build and not _bundle_ok(bundle):
            _, errs = ensure_bundle(tile_id, cfg.bundles, cfg.partition, params.get("buffer", 200.0), cfg.sources,
                                    cfg.index, cfg.cache, cfg.mirror, log=log)
            rec["build_errors"] = errs
        rec["build_s"] = round(time.perf_counter() - t0, 1)
        dsm = bundle / "dsm.tif"
        run_params = dict(params, device=device)
        cm = run_params.get("channel_model")
        if cm and (Path(cm) / "oof_map.json").exists():  # ML-3c: each tile with the model that never trained on it
            fold = json.loads((Path(cm) / "oof_map.json").read_text()).get(tile_id)
            if fold is None:
                raise RuntimeError(f"{tile_id} is not in {cm}/oof_map.json (no held-out model for it)")
            run_params["channel_model"] = str(Path(cm) / fold)
            rec["channel_model_fold"] = fold
        res = network(bundle / "dem.tif", dsm if dsm.exists() else None, bundle / "aoi.geojson",
                      roads=bundle / "roads.gpkg" if (bundle / "roads.gpkg").exists() else None,
                      out=Path(cfg.runs) / tile_id, params=run_params, keep=cfg.features)
        if cfg.features:
            from .features import extract
            t1 = time.perf_counter()
            tab, _, _ = extract(res, Path(cfg.runs) / tile_id)
            rec.update(features_s=round(time.perf_counter() - t1, 1), n_features=int(tab.shape[1]))
            try:                                  # why labelled structures get no candidate (not in the hash)
                from . import probes
                pr = probes.write(res, bundle, Path(cfg.runs) / tile_id, _label_cfg())
                rec["probes"] = {k: int(v) for k, v in pr.reason.value_counts().items()}
            except Exception as e:  # noqa: BLE001 - diagnostics only
                rec["probes_error"] = f"{type(e).__name__}: {e}"
        rec.update(ok=True, seconds=round(time.perf_counter() - t0, 1), crossings=res.info.get("crossings"),
                   channels_km=res.info.get("channels_km"), peak_rss_gb=res.perf.get("peak_rss_gb"),
                   gpu_pool_gb=res.info.get("gpu_pool_gb"))
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
    gpu_workers = min(gpu_workers, workers)
    if gpu_workers:
        try:                                  # a GPU worker without a usable GPU would fail every tile
            from ..backend import get_backend
            get_backend("gpu")
        except Exception as e:  # noqa: BLE001
            log(f"WARNING: no usable GPU ({type(e).__name__}: {e}); running every worker on the CPU")
            gpu_workers = 0
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
        g = f", GPU {r['gpu_pool_gb']} GB" if r.get("gpu_pool_gb") is not None else ""
        return (f"[{i}/{n}] {r['tile_id']} {r['device']}: {r['seconds']} s, {r.get('channels_km')} km, "
                f"{sum(c.values())} crossings, peak {r.get('peak_rss_gb')} GB{g}")
    return f"[{i}/{n}] {r['tile_id']} {r['device']}: FAILED {r.get('error')}"
