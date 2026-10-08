"""LINZ map-sheet grid: tile names the four test sites were actually served (site.json, 8 Oct 2026)."""
import pytest

from terrain_s2 import grid
from terrain_s2.grid import NZSheetGrid, RegularGrid

SITES = {   # window (EPSG:2193) -> LINZ 1:10k tiles in the bundle's provenance
    "sh12": ((1641569.6, 6075281.0, 1642937.6, 6076373.0), {"AW27_10000_0202", "AW27_10000_0302"}),
    "aoi2": ((1641042.5, 6074212.5, 1642268.5, 6075459.5), {"AW27_10000_0302"}),
    "canterbury1": ((1570444.7, 5198416.6, 1573131.4, 5200182.5), {"BW24_10000_0402"}),
    "canterbury2": ((1565402.0, 5196720.0, 1568089.6, 5198486.0), {"BW24_10000_0401", "BW24_10000_0501"}),
}


@pytest.mark.parametrize("site", sorted(SITES))
def test_tiles_match_linz_names(site):
    window, expected = SITES[site]
    assert set(NZSheetGrid(10000).tiles(window)) == expected


def test_sheet_letters_skip_i_and_o():
    assert grid._row_code(0) == "AS" and grid._row_code(7) == "AZ" and grid._row_code(8) == "BA"
    assert grid._row_code(16) == "BJ"                     # BH, then BJ (no BI)
    assert grid._row_index("BW") == 28 and grid._row_index("AW") == 4


def test_bounds_round_trip_and_nesting():
    b = grid.tile_bounds("BW24_10000_0402")
    assert b == (1568800.0, 5197200.0, 1573600.0, 5204400.0)
    assert NZSheetGrid(10000).tile_id(*(((b[0] + b[2]) / 2), ((b[1] + b[3]) / 2))) == "BW24_10000_0402"
    kids = grid.children("BW24_10000_0402", 5000)
    assert len(kids) == 4 and all(grid.parent(k, 10000) == "BW24_10000_0402" for k in kids)
    assert kids[0] == "BW24_5000_0703"
    assert len(grid.children("BW24", 1000)) == 2500 and grid.parent("BW24_1000_3512", 50000) == "BW24"
    assert grid.tile_bounds("BW24_500_070012")[2] - grid.tile_bounds("BW24_500_070012")[0] == 240


def test_edges_belong_to_the_tile_to_the_right_and_below():
    g = NZSheetGrid(10000)
    assert g.tile_id(1568800.0, 5204400.0) == "BW24_10000_0402"       # top edge of row 04 -> row 04
    assert g.tile_id(1568800.0, 5197200.0) == "BW24_10000_0502"       # bottom edge of row 04 -> row 05


def test_bad_ids_are_rejected():
    for bad in ("BW24_10000_402", "BI24", "bw24", "BW24_2500_0101"):
        with pytest.raises(ValueError):
            grid.parse(bad)


def test_regular_grid():
    g = RegularGrid(0, 0, 1000, 1000, "EPSG:2154", "FR")
    assert g.tile_id(1500, 2500) == "FR_1_2" and g.bounds("FR_1_2") == (1000, 2000, 2000, 3000)
    assert g.tiles((900, 900, 1100, 1100)) == ["FR_0_1", "FR_1_1", "FR_0_0", "FR_1_0"]


def test_quadrants_name_the_linz_tile_and_coincide_with_5k_tiles():
    qs = grid.quadrants("BW24_10000_0402")
    assert qs == ["BW24_10000_0402_NW", "BW24_10000_0402_NE", "BW24_10000_0402_SW", "BW24_10000_0402_SE"]
    five = {grid.tile_bounds(t) for t in grid.children("BW24_10000_0402", 5000)}
    assert {grid.tile_bounds(q) for q in qs} == five
    x0, y0, x1, y1 = grid.tile_bounds("BW24_10000_0402_NW")
    assert (x1 - x0, y1 - y0) == (2400.0, 3600.0) and grid.parent("BW24_10000_0402_NW", 10000) == "BW24_10000_0402"
    assert grid.quadrant_id((x0 + x1) / 2, (y0 + y1) / 2) == "BW24_10000_0402_NW"
    assert grid.quadrant_id(x1, y0) == "BW24_10000_0402_SE"              # the tile's centre point: SE by the edge rule
    for bad in ("BW24_5000_0703_NW", "BW24_10000_0402_XX"):
        with pytest.raises(ValueError):
            grid.tile_bounds(bad)
