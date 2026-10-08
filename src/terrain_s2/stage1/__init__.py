"""Stage 1: one DEM product per AOI and parameter set, in the section 4.3 contract.

Backends:

- ``raster`` (:mod:`.raster`): reads the LINZ 1 m DEM COGs and writes the contract itself (no GeoFabrics);
- ``geofabrics`` (:mod:`.gf`): fills a GeoFabrics 1.1.30 instruction template and runs its runner, then
  finishes the netCDF to the contract.

Entry point: :func:`terrain_s2.stage1.run.dem`. Stage 2 never imports this package.
"""
