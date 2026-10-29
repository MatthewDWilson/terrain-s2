"""The unattended plan runner (scripts/pipeline.py): order, dependencies, retries, resuming, reports, the summary."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pipeline", ROOT / "scripts" / "pipeline.py")
PL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PL)


def _plan(tmp_path, steps):
    import yaml
    f = tmp_path / "plan.yml"
    f.write_text(yaml.safe_dump(dict(name="t", vars=dict(x="hello {data}"),
                                     manifests=dict(m=["ml/manifests/wave3.csv"]), steps=steps)))
    return f


def _run(monkeypatch, plan, *extra):
    monkeypatch.setattr(sys, "argv", ["pipeline.py", "--plan", str(plan), *extra])
    PL.main()


@pytest.mark.skipif(os.name == "nt", reason="plans run their steps with bash, in WSL (terrain_hpc.sh pipeline)")
def test_plan_runs_in_order_skips_dependents_and_resumes(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("TERRAIN_DATA_DIR", str(data))
    out = tmp_path / "o.txt"
    steps = [dict(name="a", run=f"echo {{x}} >> {out}", reports=[str(out)]),
             dict(name="bad", run="exit 3", retries=1),
             dict(name="c", after=["bad"], run=f"echo c >> {out}"),
             dict(name="d", after=["a"], run=f"wc -l {{manifest_m}} >> {out}"),
             dict(name="e", run="echo e", done_if=str(out))]
    plan = _plan(tmp_path, steps)
    monkeypatch.setattr(PL.time, "sleep", lambda s: None)
    _run(monkeypatch, plan)
    work = data / "pipelines" / "t"
    st = json.loads((work / "status.json").read_text())["steps"]
    assert st["a"]["status"] == "done" and st["bad"]["status"] == "failed" and st["c"]["status"] == "skipped"
    assert st["d"]["status"] == "done" and st["e"]["status"] == "done before"
    txt = out.read_text().splitlines()
    assert txt[0] == f"hello {data}" and len(txt) == 2 and "c" not in txt
    assert (work / "bad.log").read_text().count("attempt") == 2 and (work / "reports" / "a__o.txt").exists()
    assert len(pd.read_csv(work / "m.csv")) == len(pd.read_csv(ROOT / "ml" / "manifests" / "wave3.csv"))
    assert "| bad | failed" in (work / "summary.md").read_text()
    _run(monkeypatch, plan)                                       # again: a and d are done; only bad runs (and fails)
    assert len(out.read_text().splitlines()) == 2
    _run(monkeypatch, plan, "--from", "d")                        # d and after run again
    assert len(out.read_text().splitlines()) == 3


def test_dry_run_and_sizing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TERRAIN_DATA_DIR", str(tmp_path / "data"))
    plan = _plan(tmp_path, [dict(name="a", run="echo {sizing} {work}")])
    _run(monkeypatch, plan, "--dry-run")
    o = capsys.readouterr().out
    work = str(tmp_path / "data" / "pipelines" / "t")
    assert "--workers" in o and work in o and not (tmp_path / "data" / "pipelines" / "t" / "a.done").exists()


def test_the_weekend_plan_fills():
    import yaml
    plan = yaml.safe_load((ROOT / "ml" / "pipelines" / "weekend_2026_10_31.yml").read_text())
    names = [s["name"] for s in plan["steps"]]
    assert len(names) == len(set(names))
    for s in plan["steps"]:
        assert set(s.get("after", [])) <= set(names[:names.index(s["name"])])     # dependencies come first
        for w in ("python scripts/", "mkdir"):
            if w in s["run"]:
                script = s["run"].split("python ")[-1].split()[0]
                assert (ROOT / script).exists(), script
