#!/usr/bin/env python
"""Find which DLL breaks an extension import on Windows ("DLL load failed ... The specified
procedure could not be found", 0xc0000139). Standard library only.

    conda activate terrain-s2
    python scripts/diagnose_dll.py                  # default target: rasterio._warp
    python scripts/diagnose_dll.py --target C:\\path\\to\\something.pyd

1. Static: parse the PE import table of the target and, recursively, of every DLL it needs.
   For each dependency, list every copy on the search path (environment, System32, PATH) and
   check that each copy exports the functions required.
     - the environment's own copy is missing functions  -> mismatch inside the environment
     - a copy elsewhere is missing functions            -> shadowing risk, if Windows picks it
2. Runtime: in a fresh interpreter, list DLLs loaded at start-up (injected by other software),
   after `import numpy`, and after the target import. Flags DLLs whose name matches an
   environment DLL but that were loaded from elsewhere (Windows then reuses that copy).
3. Versions of the Microsoft C++ runtime (msvcp140 etc.) in the environment and in System32.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

SYSTEM_PREFIXES = ("api-ms-win-", "ext-ms-")
BENIGN = {"ucrtbase.dll"}       # Windows' own UCRT; conda ships a copy too
PACKAGES = ("python", "rasterio", "libgdal", "libgdal-core", "gdal", "proj", "libcurl", "libsqlite",
            "openssl", "libzlib", "zstd", "libtiff", "pyogrio", "shapely", "geos", "numpy",
            "vc", "vc14_runtime", "vcomp14", "ucrt")
VC_RUNTIME = ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140.dll",
              "vcruntime140_1.dll", "concrt140.dll", "vcomp140.dll")


# --------------------------------------------------------------------------- PE parsing
def _cstr(data: bytes, off: int) -> str:
    end = data.index(b"\0", off)
    return data[off:end].decode("ascii", "replace")


def pe_info(path) -> dict:
    """Imports ({dll_lower: set(function names)}), exports (set), and 64-bit flag of a PE file."""
    data = Path(path).read_bytes()
    if data[:2] != b"MZ":
        raise ValueError(f"{path}: not a PE file")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError(f"{path}: bad PE signature")
    coff = pe + 4
    machine, nsec = struct.unpack_from("<HH", data, coff)
    opt_size = struct.unpack_from("<H", data, coff + 16)[0]
    opt = coff + 20
    pe32p = struct.unpack_from("<H", data, opt)[0] == 0x20B
    dd = opt + (112 if pe32p else 96)
    ndd = struct.unpack_from("<I", data, dd - 4)[0]
    dirs = [struct.unpack_from("<II", data, dd + 8 * i) for i in range(min(ndd, 16))]
    secs = []
    for i in range(nsec):
        _, vsize, va, rawsize, raw = struct.unpack_from("<8sIIII", data, opt + opt_size + 40 * i)
        secs.append((va, max(vsize, rawsize), raw))

    def off(rva):
        for va, size, raw in secs:
            if va <= rva < va + size:
                return raw + rva - va
        raise ValueError(f"RVA {rva:#x} outside sections")

    imports: dict[str, set] = {}
    irva = dirs[1][0] if len(dirs) > 1 else 0
    if irva:
        p = off(irva)
        tsize, flag = (8, 1 << 63) if pe32p else (4, 1 << 31)
        fmt = "<Q" if pe32p else "<I"
        while True:
            oft, _, _, name_rva, ft = struct.unpack_from("<IIIII", data, p)
            if not (oft or name_rva or ft):
                break
            dll = _cstr(data, off(name_rva)).lower()
            names = imports.setdefault(dll, set())
            t = off(oft or ft)
            while True:
                v = struct.unpack_from(fmt, data, t)[0]
                if v == 0:
                    break
                if v & flag:
                    names.add(f"#ordinal{v & 0xFFFF}")
                else:
                    names.add(_cstr(data, off(v & 0x7FFFFFFF) + 2))
                t += tsize
            p += 20
    exports = set()
    erva = dirs[0][0] if dirs else 0
    if erva:
        e = off(erva)
        nnames, names_rva = struct.unpack_from("<I", data, e + 24)[0], struct.unpack_from("<I", data, e + 32)[0]
        if nnames:
            n0 = off(names_rva)
            for i in range(nnames):
                exports.add(_cstr(data, off(struct.unpack_from("<I", data, n0 + 4 * i)[0])))
    return {"imports": imports, "exports": exports, "x64": machine == 0x8664}


# --------------------------------------------------------------------------- search path
UNREADABLE: list[str] = []


def _isdir(p: Path) -> bool:
    try:
        return p.is_dir()
    except OSError:                  # e.g. a PATH entry in another user's profile (WinError 5)
        UNREADABLE.append(str(p))
        return False


def _isfile(p: Path) -> bool:
    try:
        return p.is_file()
    except OSError:
        return False


def known_dlls() -> set:
    if os.name != "nt":
        return set()
    import winreg
    out = set()
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\Session Manager\KnownDLLs") as k:
            i = 0
            while True:
                try:
                    _, v, _ = winreg.EnumValue(k, i)
                    out.add(str(v).lower())
                    i += 1
                except OSError:
                    break
    except OSError:
        pass
    return out


def env_dirs(prefix: Path) -> list[Path]:
    return [prefix, prefix / "Library" / "bin", prefix / "Library" / "mingw-w64" / "bin",
            prefix / "Library" / "usr" / "bin", prefix / "bin", prefix / "DLLs"]


def search_dirs(prefix: Path, target_dir: Path, system32: Path | None) -> list[tuple[str, Path]]:
    dirs = [("target dir", target_dir)] + [("env", d) for d in env_dirs(prefix)]
    if system32:
        dirs.append(("System32", system32))
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if p:
            dirs.append(("PATH", Path(p)))
    seen, out = set(), []
    for label, d in dirs:
        key = os.path.normcase(str(d)).rstrip("\\/")
        if key in seen or not _isdir(d):
            continue
        seen.add(key)
        out.append((label, d))
    return out


def find_copies(name: str, dirs) -> list[tuple[str, Path]]:
    out = []
    for label, d in dirs:
        f = d / name
        if _isfile(f):
            out.append((label, f))
    return out


def in_env(path: Path, prefix: Path) -> bool:
    return os.path.normcase(str(path)).startswith(os.path.normcase(str(prefix)))


def analyse(target: Path, prefix: Path, dirs, known: set, max_modules: int = 400) -> dict:
    """Walk the dependency graph using the environment's copies, checking every copy found."""
    report = {"not_found": [], "env_missing": [], "shadow_risk": [], "modules": 0, "errors": []}
    cache: dict[str, dict] = {}

    def info(p: Path):
        k = str(p)
        if k not in cache:
            try:
                cache[k] = pe_info(p)
            except Exception as e:  # noqa: BLE001
                cache[k] = {"imports": {}, "exports": set(), "error": str(e)}
                report["errors"].append(f"{p}: {e}")
        return cache[k]

    queue, done = [target], set()
    while queue and len(done) < max_modules:
        mod = queue.pop(0)
        if str(mod).lower() in done:
            continue
        done.add(str(mod).lower())
        for dll, funcs in sorted(info(mod)["imports"].items()):
            if dll.startswith(SYSTEM_PREFIXES) or dll in known:
                continue
            copies = find_copies(dll, dirs)
            if not copies:
                report["not_found"].append((mod.name, dll))
                continue
            env_copies = [c for c in copies if in_env(c[1], prefix) or c[0] == "target dir"]
            preferred = (env_copies or copies)[0][1]
            for label, path in copies:
                missing = sorted(f for f in funcs if not f.startswith("#") and f not in info(path)["exports"])
                if not missing:
                    continue
                entry = (mod.name, dll, str(path), len(missing), missing[:3])
                if path == preferred and (env_copies and path == env_copies[0][1]):
                    report["env_missing"].append(entry)
                else:
                    report["shadow_risk"].append(entry + (label,))
            if not str(preferred).lower().startswith(os.path.normcase(os.environ.get("SystemRoot", "C:\\Windows")).lower()):
                queue.append(preferred)
    report["modules"] = len(done)
    return report


# --------------------------------------------------------------------------- runtime probe
def loaded_modules() -> list[str]:
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32EnumProcessModulesEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_void_p),
                                            wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
    k32.K32GetModuleFileNameExW.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.LPWSTR, wintypes.DWORD]
    h = k32.GetCurrentProcess()
    arr = (ctypes.c_void_p * 4096)()
    need = wintypes.DWORD()
    k32.K32EnumProcessModulesEx(h, arr, ctypes.sizeof(arr), ctypes.byref(need), 3)
    out = []
    for m in arr[: need.value // ctypes.sizeof(ctypes.c_void_p)]:
        buf = ctypes.create_unicode_buffer(1024)
        k32.K32GetModuleFileNameExW(h, m, buf, 1024)
        out.append(buf.value)
    return out


def runtime_probe(module: str):
    stages = {"start": loaded_modules()}
    try:
        import numpy  # noqa: F401
        stages["after numpy"] = loaded_modules()
    except Exception as e:  # noqa: BLE001
        stages["numpy error"] = repr(e)
    err = None
    try:
        __import__(module)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    stages[f"after {module}"] = loaded_modules()
    print(json.dumps({"stages": stages, "error": err}))


def file_version(path) -> str:
    import ctypes
    ver = ctypes.WinDLL("version")
    size = ver.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        return "?"
    buf = ctypes.create_string_buffer(size)
    ver.GetFileVersionInfoW(str(path), 0, size, buf)
    p, n = ctypes.c_void_p(), ctypes.c_uint()
    if not ver.VerQueryValueW(buf, "\\", ctypes.byref(p), ctypes.byref(n)):
        return "?"
    ffi = ctypes.cast(p, ctypes.POINTER(ctypes.c_uint32 * 13)).contents
    return f"{ffi[2] >> 16}.{ffi[2] & 0xFFFF}.{ffi[3] >> 16}.{ffi[3] & 0xFFFF}"


# --------------------------------------------------------------------------- main
def default_target(module: str) -> Path:
    import importlib.util
    pkg, _, sub = module.partition(".")
    spec = importlib.util.find_spec(pkg)       # does not execute the package
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"package {pkg} not found")
    d = Path(list(spec.submodule_search_locations)[0])
    hits = sorted(d.glob(f"{sub or '__init__'}*.pyd"))
    if not hits:
        raise SystemExit(f"no {sub}*.pyd in {d}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="rasterio._warp", help="extension module to test")
    ap.add_argument("--target", help="a .pyd/.dll file to analyse instead of --module")
    ap.add_argument("--runtime-probe", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.runtime_probe:
        runtime_probe(a.module)
        return
    if os.name != "nt":
        raise SystemExit("Windows only (the PE parser itself is portable: import pe_info)")

    prefix = Path(sys.prefix)
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    target = Path(a.target) if a.target else default_target(a.module)
    print(f"Python {sys.version.split()[0]}  env {prefix}\nTarget {target}")
    known = known_dlls()
    dirs = search_dirs(prefix, target.parent, system32)

    print("\n== 0. Packages " + "=" * 55)
    conda = os.environ.get("CONDA_EXE")
    if conda:
        res = subprocess.run([conda, "list", "-p", str(prefix), "--json"], capture_output=True, text=True)
        try:
            for pk in json.loads(res.stdout):
                if pk["name"] in PACKAGES:
                    print(f"  {pk['name']:14s} {pk['version']:12s} {pk.get('build_string', ''):28s} {pk.get('channel', '')}")
        except Exception:  # noqa: BLE001
            print("  (conda list failed)")
    else:
        print("  (CONDA_EXE not set: activate the environment first)")

    for u in sorted(set(UNREADABLE)):
        print(f"  note: PATH entry not readable by you (skipped): {u}")

    print("\n== 1. Static analysis of the dependency graph " + "=" * 24)
    r = analyse(target, prefix, dirs, known)
    print(f"  modules analysed: {r['modules']}")
    for mod, dll in r["not_found"]:
        print(f"  NOT FOUND   {dll}  (needed by {mod})")
    for mod, dll, path, n, ex in r["env_missing"]:
        print(f"  ENV COPY INCOMPLETE  {path} lacks {n} function(s) needed by {mod}, e.g. {', '.join(ex)}")
    for mod, dll, path, n, ex, label in r["shadow_risk"]:
        print(f"  SHADOW RISK [{label}]  {path} lacks {n} function(s) needed by {mod}, e.g. {', '.join(ex)}")
    for e in r["errors"][:5]:
        print(f"  (could not parse {e})")
    if not (r["not_found"] or r["env_missing"] or r["shadow_risk"]):
        print("  every copy of every dependency provides the functions needed")

    print("\n== 2. Runtime: DLLs actually loaded in a fresh interpreter " + "=" * 11)
    res = subprocess.run([sys.executable, __file__, "--runtime-probe", "--module", a.module],
                         capture_output=True, text=True)
    try:
        probe = json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        print(f"  probe failed (exit {res.returncode}): {res.stderr.strip()[-500:]}")
        probe = None
    if probe:
        print(f"  import {a.module}: {'OK' if not probe['error'] else probe['error']}")
        env_names = {}
        for d in env_dirs(prefix):
            if _isdir(d):
                for f in d.glob("*.dll"):
                    env_names.setdefault(f.name.lower(), f)
        win = os.path.normcase(os.environ.get("SystemRoot", r"C:\Windows")).lower()
        reported = set()
        for stage, mods in probe["stages"].items():
            if not isinstance(mods, list):
                continue
            for m in mods:
                mp = Path(m)
                low = os.path.normcase(m).lower()
                if in_env(mp, prefix) or low in reported:
                    continue
                name = mp.name.lower()
                if name in known or name.startswith(SYSTEM_PREFIXES) or name in BENIGN:
                    continue
                if name in env_names:
                    print(f"  CONFLICT [{stage}] {m}\n      is loaded instead of the environment's {env_names[name]}")
                    reported.add(low)
                elif not low.startswith(win) and mp.suffix.lower() in (".dll", ".pyd"):
                    print(f"  foreign module [{stage}] {m}")
                    reported.add(low)
        if not reported:
            print("  no DLLs from outside the environment and Windows were loaded")

    print("\n== 3. Microsoft C++ runtime versions " + "=" * 33)
    for n in VC_RUNTIME:
        for label, p in find_copies(n, [("env", d) for d in env_dirs(prefix)] + [("System32", system32)]):
            print(f"  {n:20s} {file_version(p):16s} [{label}] {p}")

    print("\nInterpretation: ENV COPY INCOMPLETE -> reinstall/pin inside the environment;"
          "\nCONFLICT or SHADOW RISK -> another program's DLL wins: remove it from PATH, or update it"
          "\n(for msvcp140/vcruntime140 in System32: install the current Microsoft Visual C++ x64 redistributable).")


if __name__ == "__main__":
    main()
