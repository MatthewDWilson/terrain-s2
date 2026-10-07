"""Automated assembly of site data (R2/M1): LiDAR from the LINZ nz-elevation bucket, council
inventories (ArcGIS REST), roads/rail/tracks (LINZ Topo50 WFS, OpenStreetMap Overpass, KiwiRail).

Every remote result is stored in a snapshot store with its URL, parameters, retrieval time and
content hash, so any score can name the data it came from (see the R2/R3 design document).
"""
