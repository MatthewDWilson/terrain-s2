#!/usr/bin/env python
"""Run a plan of long steps unattended, in order, resumably (the weekend runs; ``terrain_hpc.sh pipeline`` detaches it).

    python scripts/pipeline.py --plan ml/pipelines/weekend_2026_10_31.yml
    python scripts/pipeline.py --plan ... --dry-run          # the commands, with the values filled in
    python scripts/pipeline.py --plan ... --from unet_v5     # run that step and the ones after it again

A plan (YAML): ``name``; ``vars`` (text substituted as ``{var}`` into the commands, besides ``{data}`` =
TERRAIN_DATA_DIR, ``{ts2}`` = the repository, ``{partition}``, ``{work}`` = ``<data>/pipelines/<name>``, and ``{sizing}``
= the batch runner's ``--workers N --gpu-workers G`` from memory, cores and GPU memory, as ``terrain_hpc.sh``);
``manifests`` ({name: [csv, ...]}: merged into ``{work}/<name>.csv`` first); ``min_free_gb`` (stop before starting when
the data disk has less); ``steps``, each with ``name``, ``run`` (a shell command, run in the repository with the
environment's python), optionally ``after`` (steps it needs: if one failed, it is skipped), ``done_if`` (a path: when it
exists the step counts as done), ``retries`` (extra attempts, e.g. for a batch, which resumes) and ``reports`` (paths
copied to ``{work}/reports`` when it succeeds).

Each step logs to ``{work}/<step>.log``; ``{work}/status.json`` says where it stands; a step that succeeded leaves
``{work}/<step>.done`` and is skipped next time. A failed step does not stop the plan: the steps that do not need it
go on. At the end ``{work}/summary.md`` lists every step (status, time, log) with the last lines of its log.
The steps run with bash: plans are for WSL or Linux (``terrain_hpc.sh pipeline``), not Windows itself.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GB_PER_TILE = 12
GB_GPU_PER_WORKER = 8


def sizing():
    """``--workers N --gpu-workers G`` as terrain_hpc.sh sizes a batch."""
    mem = 0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                mem = int(line.split()[1]) // 1048576
    except OSError:
        mem = 16
    cores = os.cpu_count() or 2
    w = max(1, min(mem * 85 // 100 // GB_PER_TILE, cores // 2))
    g = 0
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        g = min(int(int(out[0]) / 1024) // GB_GPU_PER_WORKER, w) if out else 0
    except (OSError, ValueError, subprocess.SubprocessError):
        g = 0
    return f"--workers {w} --gpu-workers {g}"


def fill(text, values):
    out = str(text)
    for _ in range(3):                                  # vars may refer to vars
        for k, v in values.items():
            out = out.replace("{" + k + "}", str(v))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--from", dest="from_step", help="run this step and the ones after it again")
    ap.add_argument("--only", help="run only this step")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    import yaml

    from terrain_s2.acquire.elevation import default_data_dir
    plan = yaml.safe_load(Path(a.plan).read_text())
    data = default_data_dir()
    work = data / "pipelines" / plan["name"]
    work.mkdir(parents=True, exist_ok=True)
    values = dict(data=str(data), ts2=str(ROOT), partition=str(data / "partition" / "nz_tiles.gpkg"), work=str(work),
                  sizing=sizing())
    for k, v in (plan.get("vars") or {}).items():
        values[k] = fill(v, values)
    steps = plan["steps"]
    names = [s["name"] for s in steps]
    if a.from_step:
        if a.from_step not in names:
            sys.exit(f"no step {a.from_step}: {names}")
        for s in steps[names.index(a.from_step):]:
            (work / f"{s['name']}.done").unlink(missing_ok=True)
    for mname, files in (plan.get("manifests") or {}).items():
        import pandas as pd
        out = work / f"{mname}.csv"
        pd.concat([pd.read_csv(ROOT / f) for f in files], ignore_index=True).drop_duplicates("tile_id").to_csv(out, index=False)
        values[f"manifest_{mname}"] = str(out)
    if a.dry_run:
        for s in steps:
            print(f"[{s['name']}] after {s.get('after', [])}\n  {fill(s['run'], values)}")
        return
    free = shutil.disk_usage(data).free / 1e9
    if plan.get("min_free_gb") and free < float(plan["min_free_gb"]):
        sys.exit(f"only {free:.0f} GB free on {data}; the plan needs {plan['min_free_gb']} GB")
    status = {}
    sf = work / "status.json"

    def save(cur=None):
        sf.write_text(json.dumps(dict(plan=plan["name"], updated=time.strftime("%F %T"), current=cur, free_gb=round(
            shutil.disk_usage(data).free / 1e9), steps=status), indent=1))
    t_all = time.time()
    for s in steps:
        n = s["name"]
        if a.only and n != a.only:
            continue
        done_mark = work / f"{n}.done"
        di = fill(s["done_if"], values) if s.get("done_if") else None
        if done_mark.exists() or (di and Path(di).exists() and not a.from_step and not a.only):
            status[n] = dict(status="done before", log=str(work / f"{n}.log"))
            done_mark.touch()
            save()
            continue
        failed = [d for d in s.get("after", []) if status.get(d, {}).get("status") not in ("done", "done before")]
        if failed and not a.only:
            status[n] = dict(status="skipped", reason=f"needs {', '.join(failed)}")
            save()
            continue
        cmd = fill(s["run"], values)
        log = work / f"{n}.log"
        t0 = time.time()
        rc = None
        for attempt in range(1 + int(s.get("retries", 0))):
            status[n] = dict(status="running", attempt=attempt + 1, started=time.strftime("%F %T"), log=str(log),
                             command=cmd)
            save(n)
            with open(log, "a") as f:
                f.write(f"\n==== {time.strftime('%F %T')} attempt {attempt + 1}: {cmd}\n")
                f.flush()
                rc = subprocess.run(cmd, shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                    executable="/bin/bash").returncode
            if rc == 0:
                break
            time.sleep(30)
        took = round((time.time() - t0) / 60, 1)
        if rc == 0:
            done_mark.touch()
            rep = []
            for r in s.get("reports", []):
                src = Path(fill(r, values))
                if src.exists():
                    dst = work / "reports" / f"{n}__{src.name}"
                    dst.parent.mkdir(exist_ok=True)
                    shutil.copy2(src, dst)
                    rep.append(str(dst))
            status[n] = dict(status="done", minutes=took, log=str(log), reports=rep)
        else:
            status[n] = dict(status="failed", exit=rc, minutes=took, log=str(log))
        save()
    save()
    lines = [f"# Pipeline {plan['name']}", "", f"Finished {time.strftime('%F %T')} after "
             f"{(time.time() - t_all) / 3600:.1f} h; {round(shutil.disk_usage(data).free / 1e9)} GB free.", "",
             "| step | status | minutes | log |", "|---|---|---|---|"]
    for s in steps:
        st = status.get(s["name"], {})
        lines.append(f"| {s['name']} | {st.get('status', 'not run')} {st.get('reason', '')} | {st.get('minutes', '')} | "
                     f"{st.get('log', '')} |")
    for s in steps:
        lg = work / f"{s['name']}.log"
        if lg.exists():
            tail = lg.read_text(errors="replace").splitlines()[-15:]
            lines += ["", f"## {s['name']}", "", "```", *tail, "```"]
    (work / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"-> {work / 'summary.md'}")


if __name__ == "__main__":
    main()
