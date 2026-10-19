"""environment-gpu.yml stays in step with environment.yml: the same packages plus CUDA and CuPy."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _deps(f):
    out = set()
    for d in yaml.safe_load((ROOT / f).read_text())["dependencies"]:
        out.add(d if isinstance(d, str) else "pip:" + ",".join(sorted(d["pip"])))
    return out


def test_gpu_environment_in_step():
    cpu, gpu = _deps("environment.yml"), _deps("environment-gpu.yml")
    assert cpu <= gpu, f"missing from environment-gpu.yml: {sorted(cpu - gpu)}"
    assert {d.split(">")[0].split("=")[0] for d in gpu - cpu} == {"cuda-version", "cupy"}
