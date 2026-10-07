#!/usr/bin/env python
"""Diagnose the terrain-s2 environment, especially Windows DLL conflicts.

    conda activate terrain-s2
    python scripts/check_env.py

Standard library only, so it runs even when rasterio/GDAL is broken. Checks:
 1. each key import in a fresh interpreter (import order matters for DLL conflicts);
 2. activation: is <env>/Library/bin on PATH, and ahead of everything else?
 3. same-named DLLs elsewhere on PATH that shadow the environment's own
    (the usual cause of "DLL load failed ... The specified procedure could not be found");
 4. binary packages installed by pip into the conda environment;
 5. GDAL/PROJ environment variables pointing outside the environment.
"""
from __future__ import annotations

import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

IMPORTS = [
    ("numpy", "import numpy"),
    ("scipy.ndimage", "import scipy.ndimage"),
    ("numba", "import numba"),
    ("pyflwdir", "import pyflwdir"),
    ("rasterio (alone)", "import rasterio"),
    ("rasterio._warp (alone)", "import rasterio._warp"),
    ("numpy then rasterio", "import numpy, rasterio"),
    ("scipy, pyflwdir then rasterio", "import scipy.ndimage, pyflwdir, rasterio"),
    ("pyogrio", "import pyogrio"),
    ("shapely", "import shapely"),
    ("geopandas", "import geopandas"),
    ("matplotlib", "import matplotlib"),
    ("whitebox_workflows (optional)", "import whitebox_workflows"),
    ("cupy (GPU env only)", "import cupy; cupy.cuda.runtime.getDeviceCount()"),
]
WATCH = ("gdal", "proj", "geos", "sqlite3", "spatialite", "tiff", "curl", "ssl", "crypto", "zlib",
         "zstd", "lzma", "xml2", "iconv", "hdf5", "netcdf", "png", "jpeg", "webp", "lerc", "deflate",
         "openjp2", "expat", "freexl", "kea", "pq", "arrow", "parquet", "blosc")


def head(title):
    print(f"\n== {title} " + "=" * max(0, 70 - len(title)))


def check_imports():
    head("1. Imports, each in a fresh interpreter")
    for label, code in IMPORTS:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  OK    {label}")
        else:
            last = (r.stderr.strip().splitlines() or ["?"])[-1]
            print(f"  FAIL  {label}: {last}")


def env_dirs(prefix: Path):
    return [prefix, prefix / "Library" / "bin", prefix / "Library" / "mingw-w64" / "bin",
            prefix / "Scripts", prefix / "DLLs"]


def check_path(prefix: Path):
    head("2. Activation and PATH order")
    print(f"  sys.executable : {sys.executable}")
    print(f"  CONDA_PREFIX   : {os.environ.get('CONDA_PREFIX', '(not set: environment not activated?)')}")
    path = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    if os.name != "nt":
        print("  (Library\\bin check is Windows-only; skipped)")
        return path
    lib = str(prefix / "Library" / "bin").lower()
    idx = [i for i, p in enumerate(path) if os.path.normcase(p).rstrip("\\/") == os.path.normcase(lib)]
    if not idx:
        print(f"  PROBLEM: {lib} is not on PATH. Activate with `conda activate`, not by calling python.exe directly.")
    else:
        before = [p for p in path[:idx[0]] if Path(p).resolve() not in {d.resolve() for d in env_dirs(prefix) if d.exists()}]
        print(f"  Library\\bin is PATH entry #{idx[0]}; entries before it outside the env: {len(before)}")
        for p in before:
            print(f"      {p}")
    return path


def check_shadowing(prefix: Path, path):
    head("3. DLLs on PATH that shadow the environment's own")
    if os.name != "nt":
        print("  (Windows-only check; skipped)")
        return
    libbin = prefix / "Library" / "bin"
    own = {f.name.lower() for f in libbin.glob("*.dll")} if libbin.exists() else set()
    env_set = {os.path.normcase(str(d)) for d in env_dirs(prefix)}
    hits = {}
    for p in path:
        if os.path.normcase(p.rstrip("\\/")) in env_set or not os.path.isdir(p):
            continue
        try:
            names = {f.lower() for f in os.listdir(p) if f.lower().endswith(".dll")}
        except OSError:
            continue
        for n in sorted(names & own):
            if any(w in n for w in WATCH):
                hits.setdefault(p, []).append(n)
    if not hits:
        print("  none found")
    for p, names in hits.items():
        print(f"  {p}\n      {', '.join(names)}")
    if hits:
        print("  -> These directories contain GDAL-family DLLs with the same names as the environment's."
              "\n     Remove them from PATH (or test with a minimal PATH, see README).")


def check_pip_binaries():
    head("4. Binary packages installed by pip (not conda)")
    found = False
    for d in metadata.distributions():
        try:
            inst = (d.read_text("INSTALLER") or "").strip()
        except Exception:
            inst = ""
        if inst != "pip":
            continue
        files = d.files or []
        binaries = [str(f) for f in files if str(f).lower().endswith((".pyd", ".dll", ".so"))]
        if binaries:
            found = True
            note = "  (expected; self-contained)" if d.metadata["Name"].lower() == "whitebox-workflows" else "  <-- suspect"
            print(f"  {d.metadata['Name']} {d.version}: {len(binaries)} binary files{note}")
    if not found:
        print("  none")


def check_vars(prefix: Path):
    head("5. GDAL/PROJ variables")
    for v in ("GDAL_DATA", "GDAL_DRIVER_PATH", "PROJ_LIB", "PROJ_DATA", "GDAL_HOME", "PYTHONPATH"):
        val = os.environ.get(v)
        if val is None:
            continue
        outside = not os.path.normcase(val).startswith(os.path.normcase(str(prefix)))
        print(f"  {v}={val}" + ("   <-- outside this environment" if outside and v != "PYTHONPATH" else ""))
        if v == "PYTHONPATH":
            print("      <-- PYTHONPATH can make Python import packages from another installation")


def main():
    prefix = Path(sys.prefix)
    print(f"Python {sys.version.split()[0]} at {prefix}")
    check_imports()
    path = check_path(prefix)
    check_shadowing(prefix, path)
    check_pip_binaries()
    check_vars(prefix)


if __name__ == "__main__":
    main()
