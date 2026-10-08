"""Phase 2a pipeline on in-memory arrays (no I/O), with per-stage timings."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import channels, crossings, features, hydro
from .backend import Backend, get_backend
from .config import Params


@dataclass
class Result:
    feats: dict
    breach: hydro.BreachResult
    dep: hydro.Depressions
    upa: np.ndarray
    breaches: list
    candidates: list
    network: object = None
    test_a: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    device: str = "cpu"
    flw: object = None          # D8 flow directions on the breached DEM (reuse: do not route twice)


def run(z: np.ndarray, transform, params: Params | None = None, be: Backend | None = None,
        dsm: np.ndarray | None = None, core=None, source_version: str = "") -> Result:
    params = params or Params()
    be = be or get_backend("auto")
    cs = abs(transform.a)
    if not np.isclose(abs(transform.a), abs(transform.e)):
        raise ValueError("square cells required")
    t = {}

    t0 = time.perf_counter()
    feats = features.feature_stack(z, cs, be, params.features)
    t["features"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    br = hydro.breach_least_cost(z, cs, params.hydro)
    t["breach"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dep = hydro.depressions(z, cs)
    t["depressions"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    upa, flw = hydro.flow_accumulation(br.z_breached, transform)
    t["flow_accumulation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    net = channels.build_network(z, feats, cs, params.channels, upa=upa)
    t["channel_network"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ctx = crossings.Context(z=z, transform=transform, breach=br, dep=dep, upa=upa,
                            idxs_ds=flw.idxs_ds, idxs_us_main=flw.idxs_us_main, feats=feats,
                            dsm=dsm, core=core, source_version=source_version, net=net)
    allb, b_cands = crossings.candidates(ctx, params.candidates)
    t["test_b"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    a_recs = crossings.test_a(ctx, params.candidates)
    cands = crossings.combine(b_cands, a_recs, params.candidates.merge_distance_m)
    t["test_a"] = time.perf_counter() - t0
    t["total"] = sum(t.values())
    return Result(feats=feats, breach=br, dep=dep, upa=upa, breaches=allb, candidates=cands,
                  timings=t, device=be.name, network=net, test_a=a_recs, flw=flw)
