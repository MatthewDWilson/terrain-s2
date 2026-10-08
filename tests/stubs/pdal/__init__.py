"""Test stub for python-pdal, which needs the PDAL library (conda-forge) and is not installable in every test
environment. GeoFabrics imports it at module level; the coarse-DEM path used in tests never calls it. Only
added to the path when the real package is missing (tests/test_stage1_gf.py)."""
