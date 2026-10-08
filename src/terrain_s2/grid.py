"""Tile grids for partitioning a country (or any region) into processing units.

NZ profile: the LINZ map-sheet grid in NZTM2000 (EPSG:2193), the grid LINZ names its elevation
tiles by. A 1:50k sheet (Topo50 layout) is 24 km wide and 36 km high, named by a two-letter row
(AS, AT, ... AZ, BA, ... skipping I and O) and a two-digit column (``BW24``). Sheets are divided
into n x n tiles numbered row then column from the top left:

    scale   n     tile size (m)   id
    10000   5     4800 x 7200     BW24_10000_0402   (LINZ 1 m DEM/DSM COG tiles)
    5000    10    2400 x 3600     BW24_5000_0703
    1000    50    480 x 720       BW24_1000_3512
    500     100   240 x 360       BW24_500_070012

Quadrants of a 1:10k tile (2.4 x 3.6 km, the test processing unit) are named by the tile and a
cardinal suffix, e.g. ``BW24_10000_0402_NW`` / ``_NE`` / ``_SW`` / ``_SE``; they coincide with 1:5k tiles
but keep the LINZ tile readable in the name.

The origin (E 1,012,000, N 6,234,000) was fitted to LINZ tile names: it reproduces every tile the
four test sites use (SH12, AOI2: AW27_10000_0202/0302; canterbury1/2: BW24_10000_0401/0402/0501), and
``scripts/partition.py --check`` compares it with every cached LINZ tile footprint.

Other regions: ``RegularGrid`` (an origin, a tile size and a CRS), until a region profile names
its own published tiling (USGS 3DEP 1 km tiles, IGN LiDAR HD 1 km dalles).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"                       # A-Z without I and O
SHEET_W, SHEET_H = 24_000.0, 36_000.0
ORIGIN_E, ORIGIN_N = 1_012_000.0, 6_234_000.0
FIRST_ROW = LETTERS.index("A") * len(LETTERS) + LETTERS.index("S")   # sheets start at row AS
SUBDIV = {50000: 1, 10000: 5, 5000: 10, 1000: 50, 500: 100}
ID_RE = re.compile(r"^([A-Z]{2})(\d{2})(?:_(\d+)_(\d+))?$")
QUADRANTS = ("NW", "NE", "SW", "SE")


def _row_code(i: int) -> str:
    k = i + FIRST_ROW
    return LETTERS[k // len(LETTERS)] + LETTERS[k % len(LETTERS)]


def _row_index(code: str) -> int:
    return LETTERS.index(code[0]) * len(LETTERS) + LETTERS.index(code[1]) - FIRST_ROW


def _digits(scale: int) -> int:
    return 3 if SUBDIV[scale] >= 100 else 2


@dataclass(frozen=True)
class NZSheetGrid:
    """The LINZ map-sheet grid at one scale (see the module docstring)."""
    scale: int = 5000
    crs: str = "EPSG:2193"

    def __post_init__(self):
        if self.scale not in SUBDIV:
            raise ValueError(f"scale must be one of {sorted(SUBDIV)}")

    @property
    def size(self):
        n = SUBDIV[self.scale]
        return SHEET_W / n, SHEET_H / n

    def tile_id(self, e: float, n: float) -> str:
        col = math.floor((e - ORIGIN_E) / SHEET_W)
        row = math.floor((ORIGIN_N - n) / SHEET_H)
        sheet = f"{_row_code(row)}{col + 1:02d}"
        if self.scale == 50000:
            return sheet
        k, (w, h) = SUBDIV[self.scale], self.size
        x0, y_top = ORIGIN_E + col * SHEET_W, ORIGIN_N - row * SHEET_H
        c = min(int((e - x0) // w), k - 1) + 1
        r = min(int((y_top - n) // h), k - 1) + 1
        d = _digits(self.scale)
        return f"{sheet}_{self.scale}_{r:0{d}d}{c:0{d}d}"

    def bounds(self, tile_id: str):
        return tile_bounds(tile_id)

    def tiles(self, bounds) -> list[str]:
        """Ids of the tiles that intersect ``bounds`` (xmin, ymin, xmax, ymax), top-left first.
        Sheets are whole multiples of tiles, so tiles can be counted from the grid origin."""
        x0, y0, x1, y1 = bounds
        w, h = self.size
        c0, c1 = math.floor((x0 - ORIGIN_E) / w), math.ceil((x1 - ORIGIN_E) / w)
        r0, r1 = math.floor((ORIGIN_N - y1) / h), math.ceil((ORIGIN_N - y0) / h)
        return [self.tile_id(ORIGIN_E + (c + 0.5) * w, ORIGIN_N - (r + 0.5) * h) for r in range(r0, r1) for c in range(c0, c1)]


def parse(tile_id: str):
    """(sheet, row index, column index, scale, tile row, tile column) of a LINZ tile id."""
    m = ID_RE.match(tile_id)
    if not m:
        raise ValueError(f"not a LINZ map-sheet tile id: {tile_id!r}")
    rows, col, scale, rc = m.groups()
    if scale is None:
        return f"{rows}{col}", _row_index(rows), int(col) - 1, 50000, 1, 1
    scale = int(scale)
    if scale not in SUBDIV or scale == 50000 or len(rc) != 2 * _digits(scale):
        raise ValueError(f"unsupported tile id {tile_id!r}")
    d = _digits(scale)
    r, c = int(rc[:d]), int(rc[d:])
    if not (1 <= r <= SUBDIV[scale] and 1 <= c <= SUBDIV[scale]):
        raise ValueError(f"tile row/column out of range in {tile_id!r}")
    return f"{rows}{col}", _row_index(rows), int(col) - 1, scale, r, c


def split_quadrant(tile_id: str):
    """(LINZ tile id, quadrant or None): ``BW24_10000_0402_NW`` -> (``BW24_10000_0402``, ``NW``)."""
    base, _, q = tile_id.rpartition("_")
    if q in QUADRANTS:
        if parse(base)[3] != 10000:
            raise ValueError(f"quadrants are defined for 1:10k tiles only: {tile_id!r}")
        return base, q
    return tile_id, None


def quadrants(tile_id: str) -> list[str]:
    """The four quadrant ids of a 1:10k tile, NW, NE, SW, SE."""
    if parse(tile_id)[3] != 10000:
        raise ValueError(f"quadrants are defined for 1:10k tiles only: {tile_id!r}")
    return [f"{tile_id}_{q}" for q in QUADRANTS]


def quadrant_id(e: float, n: float) -> str:
    """The quadrant containing a point (same edge rule as tile_id: a shared edge belongs to the
    tile to the right and below)."""
    t = NZSheetGrid(10000).tile_id(e, n)
    x0, y0, x1, y1 = tile_bounds(t)
    return f"{t}_{'N' if n > (y0 + y1) / 2 else 'S'}{'E' if e >= (x0 + x1) / 2 else 'W'}"


def tile_bounds(tile_id: str):
    """(xmin, ymin, xmax, ymax) in EPSG:2193 of a LINZ map-sheet tile id at any supported scale,
    or of a 1:10k quadrant (``..._NW``)."""
    base, q = split_quadrant(tile_id)
    if q:
        x0, y0, x1, y1 = tile_bounds(base)
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        return (x0 if q[1] == "W" else xm, ym if q[0] == "N" else y0, xm if q[1] == "W" else x1, y1 if q[0] == "N" else ym)
    _, row, col, scale, r, c = parse(tile_id)
    k = SUBDIV[scale]
    w, h = SHEET_W / k, SHEET_H / k
    x0 = ORIGIN_E + col * SHEET_W + (c - 1) * w
    y1 = ORIGIN_N - row * SHEET_H - (r - 1) * h
    return (x0, y1 - h, x0 + w, y1)


def parent(tile_id: str, scale: int) -> str:
    """The tile at a coarser ``scale`` that contains this one (a quadrant's parent at 1:10k is its tile)."""
    x0, y0, x1, y1 = tile_bounds(tile_id)
    return NZSheetGrid(scale).tile_id((x0 + x1) / 2, (y0 + y1) / 2)


def children(tile_id: str, scale: int) -> list[str]:
    """The tiles at a finer ``scale`` inside this one (the grids nest exactly)."""
    if SUBDIV[scale] % SUBDIV[parse(tile_id)[3]]:
        raise ValueError(f"1:{scale} tiles do not nest in {tile_id}")
    x0, y0, x1, y1 = tile_bounds(tile_id)
    g = NZSheetGrid(scale)
    w, h = g.size
    return [g.tile_id(x0 + (i + 0.5) * w, y1 - (j + 0.5) * h)
            for j in range(round((y1 - y0) / h)) for i in range(round((x1 - x0) / w))]


@dataclass(frozen=True)
class RegularGrid:
    """A plain grid for regions without a published tiling: ids ``<prefix>_<col>_<row>``."""
    origin_x: float
    origin_y: float
    size_x: float
    size_y: float
    crs: str
    prefix: str = "T"

    def tile_id(self, x: float, y: float) -> str:
        return f"{self.prefix}_{math.floor((x - self.origin_x) / self.size_x)}_{math.floor((y - self.origin_y) / self.size_y)}"

    def bounds(self, tile_id: str):
        _, c, r = tile_id.rsplit("_", 2)
        x0, y0 = self.origin_x + int(c) * self.size_x, self.origin_y + int(r) * self.size_y
        return (x0, y0, x0 + self.size_x, y0 + self.size_y)

    def tiles(self, bounds) -> list[str]:
        x0, y0, x1, y1 = bounds
        c0, c1 = math.floor((x0 - self.origin_x) / self.size_x), math.ceil((x1 - self.origin_x) / self.size_x)
        r0, r1 = math.floor((y0 - self.origin_y) / self.size_y), math.ceil((y1 - self.origin_y) / self.size_y)
        return [f"{self.prefix}_{c}_{r}" for r in range(r1 - 1, r0 - 1, -1) for c in range(c0, c1)]
