"""Point-cloud discovery for the ``geofabrics`` backend (plan 6.3 step 2): OpenTopography ``otCatalog``.

Replaces NewZeaLiDAR's scrapy crawl and its ``dataset`` table. Datasets intersecting the buffered AOI are
ranked newest first (survey end, then publication date, then name) into GeoFabrics' ``dataset_mapping.lidar``
(1 = highest priority). Profile overrides: ``geofabrics.exclude`` (never used) and ``elevation.priority``
does not apply here; ``geofabrics.priority`` (optional list of dataset names) goes ahead of newest-first.

The dataset name GeoFabrics needs is ``Dataset.alternateName`` (geoapis uses it as the ``pc-bulk`` prefix).

Open check (W4): whether ``detail=true`` returns survey dates. The parser reads, in order, ``temporalCoverage``
(ISO ``start/end``), then ``dateCreated``; publication from ``datePublished``. Datasets with no dates rank
after dated ones (by name), and the record says so (``date_source``), so the check can be answered from the
stored response. Every response is kept in the snapshot store.

Note: geoapis 0.3.4 sends ``inlcude_federated`` (misspelt), so the API applies its own default there; this
module sends ``include_federated=false``, since only OpenTopography-hosted point clouds can be fetched from
``pc-bulk`` (upstream issue, plan 13.2 item 6).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

OT_CATALOG = "https://portal.opentopography.org/API/otCatalog"
OT_BULK = "https://opentopography.s3.sdsc.edu/pc-bulk/"


class NoPointClouds(LookupError):
    """No point-cloud dataset covers the requested area."""


@dataclass
class Dataset:
    name: str                       # alternateName: the pc-bulk prefix
    title: str | None
    id: str | None                  # OpenTopography identifier, e.g. OT.012020.2193.1
    url: str | None
    survey_end: str | None
    survey_start: str | None
    published: str | None
    date_source: str | None
    hosted: bool
    raw: dict = field(default_factory=dict, repr=False)

    def record(self) -> dict:
        return {"name": self.name, "title": self.title, "id": self.id, "url": self.url,
                "survey_start": self.survey_start, "survey_end": self.survey_end, "published": self.published,
                "date_source": self.date_source}


_DATE = re.compile(r"\d{4}(-\d{2}(-\d{2})?)?")


def _date(s) -> str | None:
    if not s:
        return None
    m = _DATE.search(str(s))
    return m.group(0) if m else None


def _interval(tc) -> tuple[str | None, str | None]:
    if not tc:
        return None, None
    if isinstance(tc, dict):
        tc = tc.get("value") or tc.get("@value") or ""
    parts = str(tc).replace("..", "").split("/")
    start = _date(parts[0]) if parts else None
    end = _date(parts[-1]) if len(parts) > 1 else start
    return start, end


def parse(response: dict) -> list[Dataset]:
    """Datasets from an ``otCatalog`` JSON response (schema.org ``Dataset`` records)."""
    out = []
    for entry in response.get("Datasets", []) or []:
        d = entry.get("Dataset", entry)
        name = d.get("alternateName")
        if not name:
            continue
        ident = d.get("identifier")
        if isinstance(ident, dict):
            ident = ident.get("value") or ident.get("propertyID")
        start, end = _interval(d.get("temporalCoverage"))
        src = "temporalCoverage" if end else None
        if not end and d.get("dateCreated"):
            end, src = _date(d.get("dateCreated")), "dateCreated"
        url = d.get("url") or d.get("@id")
        hosted = bool(ident and str(ident).startswith("OT.")) or bool(url and "/meta/OT." in str(url))
        prov = d.get("provider") or {}
        if isinstance(prov, dict) and prov.get("name") and "opentopography" not in str(prov.get("name")).lower():
            hosted = False
        out.append(Dataset(name=name, title=d.get("name"), id=ident, url=url, survey_start=start, survey_end=end,
                           published=_date(d.get("datePublished")), date_source=src, hosted=hosted, raw=d))
    return out


def rank(datasets: list[Dataset], priority=(), exclude=(), hosted_only: bool = True) -> list[Dataset]:
    """Newest first: priority names, then by survey end, publication date and name; undated last."""
    ds = [d for d in datasets if d.name not in set(exclude) and (d.hosted or not hosted_only)]
    seen, uniq = set(), []
    for d in ds:
        if d.name not in seen:
            seen.add(d.name)
            uniq.append(d)
    pr = list(priority)

    def key(d):
        if d.name in pr:
            return (0, pr.index(d.name), "", "", d.name)
        dated = 1 if d.survey_end else 2
        return (dated, 0, _desc(d.survey_end), _desc(d.published), d.name)
    return sorted(uniq, key=key)


def _desc(s):
    return "".join(chr(0x10FFFF - ord(c)) for c in (s or ""))


def mapping(datasets: list[Dataset]) -> dict:
    """GeoFabrics ``dataset_mapping.lidar``: {name: rank}, 1 = highest."""
    return {d.name: i + 1 for i, d in enumerate(datasets)}


def query(http, bounds_lonlat, *, detail: bool = True, source: str = "opentopography-otcatalog") -> dict:
    x0, y0, x1, y1 = bounds_lonlat
    params = {"productFormat": "PointCloud", "minx": x0, "miny": y0, "maxx": x1, "maxy": y1,
              "detail": str(bool(detail)).lower(), "outputFormat": "json", "include_federated": "false"}
    return http.json(OT_CATALOG, params, source=source)


def discover(http, bounds, crs, profile: dict) -> tuple[list[Dataset], dict]:
    """(ranked datasets, raw response) for ``bounds`` in ``crs``. Raises :class:`NoPointClouds` if none."""
    from .aoi import lonlat_bounds
    resp = query(http, lonlat_bounds(bounds, crs))
    gf = profile.get("geofabrics", {})
    ranked = rank(parse(resp), priority=gf.get("priority") or (), exclude=gf.get("exclude") or (),
                  hosted_only=gf.get("hosted_only", True))
    if not ranked:
        raise NoPointClouds(f"no OpenTopography point cloud covers {tuple(round(b) for b in bounds)} ({crs})")
    return ranked, resp


def tile_index_url(name: str) -> str:
    return f"{OT_BULK}{name}/{name}_TileIndex.zip"


def fetch_tile_indexes(http, datasets, downloads_dir) -> list:
    """Download each dataset's tile index to where geoapis puts it (``<downloads>/lidar/<name>/``), so the
    land ``coverage`` polygon can be built before GeoFabrics runs. Returns the local paths."""
    from pathlib import Path
    paths = []
    for d in datasets:
        p = Path(downloads_dir) / "lidar" / d.name / f"{d.name}_TileIndex.zip"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(http.get(tile_index_url(d.name), source=f"ot-tileindex-{d.name}", ext="zip"))
        paths.append(p)
    return paths
