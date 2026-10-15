#!/usr/bin/env python
"""Run scripts/probe_source.py over the candidate council sources and save each output, for the wave-3 review.

    python scripts/probe_candidates.py                                  # every candidate, then the extra probes
    python scripts/probe_candidates.py --only waikato_lds_drains hauraki_land_drains
    python scripts/probe_candidates.py --skip-done                      # keep outputs already written
    python scripts/probe_candidates.py --no-extras                      # candidates only
    python scripts/probe_candidates.py --tile CE09_5000_0505            # window = one LINZ tile instead of the layer

Candidates come from sites/candidates/sources_candidates.yml (--sources). Each is probed with --fields --values
over the whole layer (its own extent) unless --tile/--bbox is given, so the value counts cover every feature the
entry's ``where`` selects. Outputs go to sites/candidates/probes/<name>.txt (--out), with the command at the top,
and a one-line status per probe is printed. A probe that fails or times out is recorded and the run continues.

The extra probes look at layers that could not be read from the drafting thread (servers that refuse automated
fetches) or that need comparing with what is already enabled: the WRC flood-protection asset service, the HBRC
open-data drains layer against hbrc_channels, the Waimakariri water-race layer, and a few ArcGIS Online searches.
This script only reads; it changes no source file.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_source.py"
SOURCES = ROOT / "sites" / "sources.yml"

# name -> probe_source.py arguments (besides the window); for layers that need a look but are not candidates yet
EXTRAS = {
    "x_wrc_flood_protection_layers": [
        "--url", "https://maps.waikatoregion.govt.nz/arcgis/rest/services/WRCMAPS/"
                 "Flood_protection_and_land_drainage_assets/MapServer/0", "--layers", "--fields", "--values"],
    "x_hbrc_drains_item": ["--sources", str(SOURCES), "--source", "hbrc_drains", "--fields", "--values"],
    "x_hbrc_channels_current": ["--sources", str(SOURCES), "--source", "hbrc_channels", "--values"],
    "x_hbrc_assetmanagement_layers": [
        "--url", "https://gis.hbrc.govt.nz/server/rest/services/ExternalServices/AssetManagement/MapServer/3",
        "--layers"],
    "x_waimakariri_drainage_service_layers": [
        "--url", "https://gisservices.waimakariri.govt.nz/arcgis/rest/services/Drainage_Areas_Water_Races/"
                 "Drainage_Areas_and_Water_Races/MapServer/0", "--layers", "--fields", "--values"],
    "x_search_tasman": ["--search", "Tasman District Council stormwater drains"],
    "x_search_tararua": ["--search", "Tararua stormwater open drains"],
    "x_search_greater_wellington_drains": ["--search", "Greater Wellington drainage scheme drains Wairarapa"],
    "x_search_hastings_drains": ["--search", "Hastings District Council open drains"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=str(ROOT / "sites" / "candidates" / "sources_candidates.yml"))
    ap.add_argument("--out", default=str(ROOT / "sites" / "candidates" / "probes"))
    ap.add_argument("--only", nargs="+", metavar="NAME", help="probe only these candidates / extras")
    ap.add_argument("--skip-done", action="store_true", help="skip probes whose output file exists and succeeded")
    ap.add_argument("--no-extras", action="store_true")
    ap.add_argument("--tile", help="LINZ tile as the window (passed to probe_source.py)")
    ap.add_argument("--bbox", help="xmin,ymin,xmax,ymax in EPSG:2193 as the window")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per probe")
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from terrain_s2.acquire.site import load_sources
    cands = load_sources(a.sources)
    window = ["--tile", a.tile] if a.tile else ["--bbox", a.bbox] if a.bbox else []
    jobs = {n: ["--sources", a.sources, "--source", n, "--fields", "--values", *window] for n in cands}
    if not a.no_extras:
        jobs.update({n: (args if "--search" in args else [*args, *window]) for n, args in EXTRAS.items()})
    if a.only:
        unknown = [n for n in a.only if n not in jobs]
        if unknown:
            sys.exit(f"unknown name(s): {', '.join(unknown)}; known: {', '.join(jobs)}")
        jobs = {n: jobs[n] for n in a.only}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, args in jobs.items():
        f = out / f"{name}.txt"
        if a.skip_done and f.exists() and f.read_text(encoding="utf-8", errors="replace").startswith("# OK"):
            print(f"{name}: skipped (done)")
            continue
        cmd = [sys.executable, str(PROBE), *args]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout, cwd=ROOT,
                               encoding="utf-8", errors="replace")
            status = "OK" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
            body = r.stdout + (f"\n--- stderr ---\n{r.stderr}" if r.stderr.strip() else "")
        except subprocess.TimeoutExpired as e:
            status = f"TIMEOUT after {a.timeout} s"
            body = (e.stdout or "") if isinstance(e.stdout, str) else ""
        dt = time.time() - t0
        cmdline = " ".join(f'"{c}"' if " " in c else c for c in ["python", *map(str, cmd[1:])])
        f.write_text(f"# {status} in {dt:.0f} s\n# {cmdline}\n\n{body}", encoding="utf-8")
        line = f"{name}: {status} ({dt:.0f} s)"
        print(line, flush=True)
        summary.append(line)
    (out / "_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
