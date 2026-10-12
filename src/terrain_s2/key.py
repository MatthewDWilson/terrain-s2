"""Generator key and provenance (integration plan 6.5).

Key = the first 16 hex characters of the SHA-256 of canonical JSON (sorted keys, normalised numbers) of the
key components. The full, unhashed components are stored beside it as ``generator_info``. Changing any
component gives a new product; repeating an identical request reuses the stored one.

Components:

========================  ===========================================  ===========================================
Component                 Stage 1                                      Stage 2 adds
========================  ===========================================  ===========================================
``request``               product, profile, backend, resolution,       resolution fixed at 1
                          land_source, buffer_m
``code``                  terrain_s2 version + commit (+ geofabrics    terrain_s2 version + commit
                          and geoapis versions)
``source_version``        STAC item ids + ``file:checksum`` +          the input's key (Input A) or the STAC
                          collection ``updated`` (raster);             checksums (Input B); recorded-structure
                          OpenTopography names + ids + mapping (GF)    snapshots
``configuration``         lidar_classes (+ template sha256 for GF)     ``Params`` hash, conditioning configuration
                                                                       hash, model versions
========================  ===========================================  ===========================================

``config_key`` hashes everything except ``source_version``: two products with the same ``config_key`` differ only
in their inputs, so a larger stored product can serve a smaller AOI if the inputs under that AOI are unchanged
(see :mod:`terrain_s2.store`).

The key is the ``input_version`` for v5 D3 (``find_cached_result``), with ``process_id = "terrain.dem"``
(Stage 1) or ``"terrain.network"`` (Stage 2).
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import math
import platform
from pathlib import Path

KEY_LENGTH = 16
SCHEMA = 1
PROCESS_STAGE1 = "terrain.dem"
PROCESS_STAGE2 = "terrain.network"


def normalise(obj):
    """JSON-ready, canonical form: integral floats become ints, other floats keep 12 significant digits;
    tuples, sets, numpy scalars and arrays, paths and dataclasses become plain JSON types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return normalise(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): normalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalise(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((normalise(v) for v in obj), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, Path):
        return obj.as_posix()
    if hasattr(obj, "tolist") and not isinstance(obj, (int, float)):   # numpy scalar or array
        return normalise(obj.tolist())
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return str(obj)
        if obj == int(obj) and abs(obj) < 2 ** 53:
            return int(obj)
        return float(f"{obj:.12g}")
    if isinstance(obj, (_dt.date, _dt.datetime)):
        return obj.isoformat()
    return str(obj)


def canonical(obj) -> str:
    return json.dumps(normalise(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(obj, n: int = KEY_LENGTH) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()[:n]


def generator_key(components: dict) -> tuple[str, dict]:
    """(key, info): ``info`` is the normalised components plus ``config_key`` and the schema number."""
    info = normalise(dict(components, schema=SCHEMA))
    info.pop("config_key", None)
    key = digest(info)
    info["config_key"] = config_key(info)
    return key, info


def config_key(components: dict) -> str:
    c = {k: v for k, v in normalise(components).items() if k not in ("source_version", "config_key")}
    return digest(c)


def code_version(geofabrics: bool = False) -> dict:
    """terrain_s2 version and commit; with ``geofabrics``, the installed geofabrics and geoapis versions
    (read from package metadata, without importing them)."""
    from . import __version__, git_commit
    out = {"terrain_s2": __version__, "commit": git_commit()}
    if geofabrics:
        from importlib.metadata import PackageNotFoundError, version
        for pkg in ("geofabrics", "geoapis"):
            try:
                out[pkg] = version(pkg)
            except PackageNotFoundError:
                out[pkg] = None
    return out


def stage1_components(*, product: str, profile: str, backend: str, resolution: int, land_source: str,
                      buffer_m: float, source_version, configuration: dict, code: dict | None = None) -> dict:
    return {"process": PROCESS_STAGE1,
            "request": {"product": product, "profile": profile, "backend": backend, "resolution": resolution,
                        "land_source": land_source, "buffer_m": buffer_m},
            "code": code if code is not None else code_version(geofabrics=backend == "geofabrics"),
            "source_version": source_version, "configuration": configuration}


def stage2_components(*, source_version, params, conditioning: dict | None = None, model_versions: dict | None = None,
                      recorded: dict | None = None, profile: str = "nz", code: dict | None = None) -> dict:
    """For S2-13. ``source_version``: the Stage 1 input's generator key (Input A) or the STAC checksums
    (Input B). ``params`` (a dataclass or dict) and ``conditioning`` are hashed, and kept in full in the
    provenance rather than in the key info."""
    return {"process": PROCESS_STAGE2,
            "request": {"product": "network", "profile": profile, "resolution": 1},
            "code": code if code is not None else code_version(),
            "source_version": {"input": source_version, "recorded": recorded or {}},
            "configuration": {"params": digest(params, 64), "conditioning": digest(conditioning or {}, 64),
                              "models": model_versions or {}}}


def file_sha256(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def provenance(key: str, info: dict, *, inputs: list | dict, parameters: dict | None = None,
               timings: dict | None = None, extra: dict | None = None) -> dict:
    """The provenance record (``<product>.provenance.json`` and the netCDF ``terrain_provenance``):
    inputs with checksums, the generator key and its unhashed components, package version and commit,
    parameters and timings."""
    code = info.get("code", {})
    return normalise({"generator_key": key, "generator_info": info, "inputs": inputs,
                      "parameters": parameters or {}, "timings": timings or {},
                      "package": {"terrain_s2": code.get("terrain_s2"), "commit": code.get("commit")},
                      "python": platform.python_version(),
                      "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                      **(extra or {})})
