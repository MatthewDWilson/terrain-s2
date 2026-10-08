"""Site definitions (``sites/<id>.yml``): one file per site, the unit of train/test separation.

    id: canterbury1
    role: test                        # train | validation | test
    aoi: examples/aoi_canterbury1.geojson     # or bounds: [xmin, ymin, xmax, ymax] (EPSG:2193)
    buffer_m: 200
    elevation: {region: canterbury, products: [dem, dsm], survey: canterbury_2020-2023}  # survey optional
    sources: auto                     # every source whose coverage meets the window (sources_coverage.geojson)
    layers: [crossings, channels, roads]   # optional, with auto: only sources providing these
    exclude: [osm]                    # optional, with auto
    # or an explicit list of names from sources.yml (or inline definitions):
    # sources: [waimakariri_culverts, waimakariri_channels, linz_roads, osm]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Site:
    id: str
    bounds: tuple                     # EPSG:2193 xmin, ymin, xmax, ymax of the AOI
    buffer_m: float = 200.0
    role: str = "train"
    elevation: dict = field(default_factory=dict)
    sources: list | str = "auto"
    layers: list | None = None
    exclude: list = field(default_factory=list)
    aoi_path: str | None = None
    notes: str = ""

    @property
    def window(self):
        b = self.buffer_m
        x0, y0, x1, y1 = self.bounds
        return (x0 - b, y0 - b, x1 + b, y1 + b)


def load_site(path, root=None) -> Site:
    import yaml
    path = Path(path)
    d = yaml.safe_load(path.read_text())
    if not isinstance(d, dict) or "id" not in d or not ("aoi" in d or "bounds" in d):
        raise ValueError(f"{path}: not a site definition (needs id, and aoi or bounds)")
    root = Path(root) if root else path.parent.parent
    aoi = d.get("aoi")
    if aoi:
        import geopandas as gpd
        p = Path(aoi) if Path(aoi).is_absolute() else root / aoi
        g = gpd.read_file(p)
        g = g.to_crs(2193) if g.crs else g.set_crs(2193)
        bounds = tuple(float(v) for v in g.total_bounds)
    else:
        bounds = tuple(float(v) for v in d["bounds"])
    if d.get("role", "train") not in ("train", "validation", "test"):
        raise ValueError(f"{path}: role must be train, validation or test")
    return Site(id=d["id"], bounds=bounds, buffer_m=float(d.get("buffer_m", 200)), role=d.get("role", "train"),
                elevation=d.get("elevation", {}), sources=d.get("sources") or "auto", layers=d.get("layers"),
                exclude=d.get("exclude") or [], aoi_path=aoi, notes=d.get("notes", ""))


def load_sources(path) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text())["sources"]
