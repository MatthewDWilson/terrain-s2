"""Stage 1 profiles (``profiles/<name>.yml``): CRS, sources and layers for a country or region."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).parent / "profiles"


@lru_cache(maxsize=8)
def load(name: str = "nz") -> dict:
    import yaml
    p = Path(name)
    path = p if p.suffix in (".yml", ".yaml") and p.exists() else HERE / f"{name}.yml"
    if not path.exists():
        raise ValueError(f"unknown profile {name!r} (no {path})")
    d = yaml.safe_load(path.read_text())
    for k in ("name", "crs", "elevation"):
        if k not in d:
            raise ValueError(f"profile {path}: missing {k!r}")
    return d


def crs_string(profile: dict) -> str:
    c = profile["crs"]
    return f"EPSG:{c['horizontal']}+{c['vertical']}" if c.get("vertical") else f"EPSG:{c['horizontal']}"


def horizontal_epsg(profile: dict) -> int:
    return int(profile["crs"]["horizontal"])
