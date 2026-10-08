"""DEM product contract (integration plan 4.3): a GeoFabrics-compatible netCDF, a COG copy, provenance.

netCDF (the EDDIE contract):

- variables ``z`` (m, float32, NaN = no data), ``data_source`` (GeoFabrics codes 0-9, -1 = no data) and
  ``lidar_source`` (survey index, -1 = none, with the mapping in its ``mapping`` attribute); ``zo`` when
  roughness is requested; ``modification_source`` on Stage 2 DEMs (plan 8.2);
- CF coordinate attributes on ``x`` and ``y`` and a ``spatial_ref`` grid mapping holding the compound CRS
  (e.g. EPSG:2193+7839), so that GDAL reads ``netcdf:"<file>":z`` with the right transform and CRS;
- global ``description`` ending in the integer resolution (FReDT parses the last token with ``int()``);
- global ``terrain_provenance`` (JSON), beside GeoFabrics' own ``geofabrics_instructions`` in that route.

COG copy: ``z`` only, deflate, 512 x 512 internal tiles, beside every 1 m product. Stage 2 and GIS users
read it; netCDF stays the EDDIE contract.

Provenance: ``<product>.provenance.json`` (see :mod:`terrain_s2.key`).

This module is part of the light base install: numpy, xarray, netCDF4 and rioxarray (which brings
rasterio; the calls used here work with rasterio 1.3.8).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# GeoFabrics' DemBase.SOURCE_CLASSIFICATION (1.1.30); codes 0-5 unchanged since 0.10.23.
DATA_SOURCE = {"interpolated": 0, "LiDAR": 1, "ocean bathymetry": 2, "rivers and fans": 3, "waterways": 4,
               "coarse DEM": 5, "patch": 6, "stopbanks": 7, "masked feature": 8, "lakes": 9, "no data": -1}
# Stage 2 modifications (plan 8.2). 5 is reserved for bed estimation (ML-4).
MODIFICATION_SOURCE = {"unmodified": 0, "bridge deck removed": 1, "culvert burned": 2,
                       "channel obstruction breached": 3, "continuity repair": 4, "bed estimation": 5}
NO_DATA = -1
PRODUCTS = ("dem", "geofabric", "dem_honest", "dem_burned")
CONTRACT_VERSION = "1"


@dataclass
class DemProduct:
    """One DEM product on a regular north-up grid. Arrays are (rows, cols), row 0 at the top."""
    z: np.ndarray
    transform: tuple                     # affine (a, b, c, d, e, f) as from Affine[:6]
    crs: str                             # e.g. "EPSG:2193+7839"
    resolution: int
    product: str = "dem"
    generator: str = "terrain_s2"        # description prefix, e.g. "terrain_s2:raster"
    data_source: np.ndarray | None = None
    lidar_source: np.ndarray | None = None
    lidar_mapping: dict = field(default_factory=dict)     # {survey name: index}; -1 = no LiDAR
    zo: np.ndarray | None = None
    modification_source: np.ndarray | None = None
    provenance: dict = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.z.shape

    def bounds(self):
        a, _, c, _, e, f = self.transform[:6]
        h, w = self.z.shape
        return (c, f + e * h, c + a * w, f)


def check_resolution(resolution) -> int:
    """Integer metres only: FReDT parses the resolution from ``description`` with ``int()``."""
    if isinstance(resolution, bool):
        raise ValueError(f"resolution must be an integer number of metres, got {resolution!r}")
    try:
        f = float(resolution)
    except (TypeError, ValueError):
        raise ValueError(f"resolution must be an integer number of metres, got {resolution!r}") from None
    if not math.isfinite(f) or f <= 0 or f != int(f):
        raise ValueError(f"resolution must be a positive integer number of metres, got {resolution!r}")
    return int(f)


def description(generator: str, resolution) -> str:
    """GeoFabrics' format: ``<library>:<class> resolution <n>``; the last token is the integer resolution."""
    return f"{generator} resolution {check_resolution(resolution)}"


def resolution_from_description(desc: str) -> int:
    """What FReDT does: the last token of ``description``, with ``int()``."""
    return int(str(desc).split()[-1])


def gdal_name(path, var: str = "z") -> str:
    """GDAL subdataset name for one variable: ``netcdf:"<path>":z``. ``rasterio.open(path)`` on the file
    itself returns 0 bands."""
    return f'netcdf:"{Path(path)}":{var}'


def _coords(transform, shape):
    a, _, c, _, e, f = transform[:6]
    if transform[1] != 0 or transform[3] != 0:
        raise ValueError("rotated grids are not supported")
    h, w = shape
    x = c + a * (np.arange(w) + 0.5)
    y = f + e * (np.arange(h) + 0.5)
    return x, y


def to_dataset(p: DemProduct):
    """An xarray Dataset in the contract layout (CF coordinates, grid mapping, attributes)."""
    import rioxarray  # noqa: F401 - registers the .rio accessor
    import xarray as xr
    from affine import Affine

    if p.product not in PRODUCTS:
        raise ValueError(f"product must be one of {PRODUCTS}, got {p.product!r}")
    res = check_resolution(p.resolution)
    tr = Affine(*p.transform[:6])
    if not (math.isclose(abs(tr.a), res) and math.isclose(abs(tr.e), res)):
        raise ValueError(f"grid cell size ({tr.a}, {-tr.e}) does not match resolution {res}")
    if tr.e > 0:
        raise ValueError("the grid must be north-up (negative y step)")
    x, y = _coords(tr, p.z.shape)
    z = np.asarray(p.z, dtype=np.float32)
    valid = np.isfinite(z)
    ds_source = (np.where(valid, DATA_SOURCE["coarse DEM"], NO_DATA) if p.data_source is None
                 else np.asarray(p.data_source)).astype(np.float32)
    lidar = (np.full(z.shape, NO_DATA) if p.lidar_source is None else np.asarray(p.lidar_source)).astype(np.float32)
    mapping = dict(p.lidar_mapping)
    mapping.setdefault("no LiDAR", NO_DATA)
    v = {"z": (("y", "x"), z, {"units": "m", "long_name": "ground elevation"}),
         "data_source": (("y", "x"), ds_source, {"units": "", "long_name": "source data classification",
                                                 "mapping": json.dumps(DATA_SOURCE)}),
         "lidar_source": (("y", "x"), lidar, {"units": "", "long_name": "source lidar ID",
                                              "mapping": json.dumps(mapping)})}
    if p.zo is not None:
        v["zo"] = (("y", "x"), np.asarray(p.zo, np.float32), {"units": "m", "long_name": "Roughness length"})
    if p.modification_source is not None:
        v["modification_source"] = (("y", "x"), np.asarray(p.modification_source, np.float32),
                                    {"units": "", "long_name": "DEM modification",
                                     "mapping": json.dumps(MODIFICATION_SOURCE)})
    attrs = {"title": f"EDDIE terrain {p.product}", "source": p.generator,
             "description": description(p.generator, res), "Conventions": "CF-1.8",
             "terrain_product": p.product, "terrain_contract": CONTRACT_VERSION,
             "terrain_provenance": json.dumps(p.provenance, sort_keys=True, default=str)}
    attrs.update(p.attrs)
    ds = xr.Dataset(v, coords={"x": ("x", x), "y": ("y", y)}, attrs=attrs)
    return write_conventions(ds, p.crs, tr)


def write_conventions(ds, crs, transform=None):
    """Add CF coordinate attributes, the compound CRS as ``spatial_ref`` and grid-mapping links in place.

    Used on our own products and on GeoFabrics output (plan 6.3 step 7), whose netCDF may lack them.
    """
    import rioxarray  # noqa: F401
    from pyproj import CRS
    c = CRS.from_user_input(crs)
    ds["x"].attrs.update(standard_name="projection_x_coordinate", long_name="x coordinate of projection",
                         units="m", axis="X")
    ds["y"].attrs.update(standard_name="projection_y_coordinate", long_name="y coordinate of projection",
                         units="m", axis="Y")
    ds = ds.rio.write_crs(c.to_wkt(), grid_mapping_name="spatial_ref")
    ds["spatial_ref"].attrs["crs_wkt"] = c.to_wkt()
    ds["spatial_ref"].attrs["spatial_ref"] = c.to_wkt()
    if transform is not None:
        ds = ds.rio.write_transform(transform, grid_mapping_name="spatial_ref")
    for name in ds.data_vars:
        if name != "spatial_ref" and ds[name].dims == ("y", "x"):
            ds[name].attrs["grid_mapping"] = "spatial_ref"
            ds[name].encoding.pop("grid_mapping", None)
    return ds


def write_dem(path, p: DemProduct, cog: bool | None = None, provenance: bool = True) -> dict:
    """Write the netCDF contract, the COG copy (default: at 1 m only) and the provenance JSON.

    Returns ``{"netcdf": ..., "cog": ..., "provenance": ...}`` (absent entries omitted).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = to_dataset(p)
    # NaN fill everywhere: -1 in data_source / lidar_source is a value (GeoFabrics' "no data" code), not masked.
    enc = {n: {"zlib": True, "complevel": 4, "_FillValue": np.nan} for n in ds.data_vars if n != "spatial_ref"}
    tmp = path.with_name(path.name + ".part")
    ds.to_netcdf(tmp, format="NETCDF4", engine="netcdf4", encoding=enc)
    tmp.replace(path)
    out = {"netcdf": str(path)}
    if cog if cog is not None else check_resolution(p.resolution) == 1:
        out["cog"] = str(write_cog(cog_path(path), p.z, p.transform, p.crs))
    if provenance:
        out["provenance"] = str(write_provenance(provenance_path(path), p.provenance))
    return out


def cog_path(nc_path) -> Path:
    return Path(nc_path).with_suffix(".tif")


def provenance_path(nc_path) -> Path:
    p = Path(nc_path)
    return p.with_name(p.stem + ".provenance.json")


def write_provenance(path, prov: dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps(prov, indent=1, sort_keys=True, default=str))
    return path


def write_cog(path, z, transform, crs, blocksize: int = 512) -> Path:
    """``z`` as a Cloud-Optimised GeoTIFF (deflate, 512 x 512 tiles, float32, NaN no data)."""
    import rasterio
    from affine import Affine
    from rasterio.crs import CRS
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    z = np.asarray(z, dtype=np.float32)
    prof = dict(height=z.shape[0], width=z.shape[1], count=1, dtype="float32", nodata=np.nan,
                crs=CRS.from_wkt(_wkt(crs)), transform=Affine(*transform[:6]))
    tmp = path.with_name(path.name + ".part.tif")
    try:
        with rasterio.open(tmp, "w", driver="COG", compress="DEFLATE", predictor="YES",
                           blocksize=blocksize, BIGTIFF="IF_SAFER", **prof) as d:
            d.write(z, 1)
    except Exception:  # noqa: BLE001 - very old GDAL without the COG driver: tiled GTiff
        with rasterio.open(tmp, "w", driver="GTiff", compress="deflate", predictor=3, tiled=True,
                           blockxsize=blocksize, blockysize=blocksize, BIGTIFF="IF_SAFER", **prof) as d:
            d.write(z, 1)
    tmp.replace(path)
    return path


def _wkt(crs) -> str:
    from pyproj import CRS
    return CRS.from_user_input(crs).to_wkt()


def open_dem(path, decode: bool = True):
    """The netCDF contract as an xarray Dataset (lazy; close it, or use as a context manager)."""
    import xarray as xr
    return xr.open_dataset(path, engine="netcdf4", decode_coords="all", mask_and_scale=decode)


def read_dem(path) -> DemProduct:
    """Read a contract netCDF (ours or GeoFabrics') back into a :class:`DemProduct` (loads the arrays)."""
    with open_dem(path) as ds:
        x, y = ds["x"].values, ds["y"].values
        if len(x) < 2 or len(y) < 2:
            raise ValueError(f"{path}: grid needs at least 2 x 2 cells")
        a = float(x[1] - x[0])
        e = float(y[1] - y[0])
        z = ds["z"].values.astype(np.float32)
        ds_src = ds["data_source"].values if "data_source" in ds else None
        lid = ds["lidar_source"].values if "lidar_source" in ds else None
        zo = ds["zo"].values if "zo" in ds else None
        mod = ds["modification_source"].values if "modification_source" in ds else None
        if e > 0:                      # bottom-up storage: flip to north-up
            z = z[::-1]
            ds_src, lid, zo, mod = (None if v is None else v[::-1] for v in (ds_src, lid, zo, mod))
            e = -e
            y = y[::-1]
        transform = (a, 0.0, float(x[0] - a / 2), 0.0, e, float(y[0] - e / 2))
        gm = ds["spatial_ref"].attrs if "spatial_ref" in ds else {}
        wkt = gm.get("crs_wkt") or gm.get("spatial_ref")
        crs = crs_string(wkt) if wkt else None
        try:
            lm = json.loads(ds["lidar_source"].attrs.get("mapping", "{}")) if lid is not None else {}
        except json.JSONDecodeError:   # GeoFabrics writes a Python dict repr
            import ast
            lm = ast.literal_eval(ds["lidar_source"].attrs.get("mapping", "{}"))
        try:
            prov = json.loads(ds.attrs.get("terrain_provenance", "{}"))
        except json.JSONDecodeError:
            prov = {}
        desc = ds.attrs.get("description", "")
        return DemProduct(z=z, transform=transform, crs=crs, resolution=resolution_from_description(desc),
                          product=ds.attrs.get("terrain_product", "dem"), generator=str(desc).rsplit(" resolution", 1)[0],
                          data_source=ds_src, lidar_source=lid, lidar_mapping=lm, zo=zo, modification_source=mod,
                          provenance=prov, attrs={k: v for k, v in ds.attrs.items()
                                                  if k not in ("description", "terrain_provenance")})


def finish_netcdf(path, crs, provenance: dict, extra_attrs: dict | None = None) -> None:
    """Bring a netCDF written by another tool (GeoFabrics) up to the contract, in place: CF coordinate
    attributes, ``spatial_ref`` with the compound CRS, and ``terrain_provenance``. The ``description`` and
    ``geofabrics_instructions`` attributes are kept as GeoFabrics wrote them."""
    path = Path(path)
    with open_dem(path, decode=False) as ds:
        ds = ds.load()
    if "description" not in ds.attrs:
        raise ValueError(f"{path}: no 'description' attribute (expected '... resolution <n>')")
    resolution_from_description(ds.attrs["description"])
    ds = write_conventions(ds, crs)
    ds.attrs["terrain_provenance"] = json.dumps(provenance, sort_keys=True, default=str)
    ds.attrs["terrain_contract"] = CONTRACT_VERSION
    ds.attrs.update(extra_attrs or {})
    for v in ds.variables.values():          # re-encode as read
        v.encoding = {k: val for k, val in v.encoding.items() if k in ("_FillValue", "dtype", "zlib", "complevel")}
    tmp = path.with_name(path.name + ".part")
    ds.to_netcdf(tmp, format="NETCDF4", engine="netcdf4")
    tmp.replace(path)


def valid_footprint(z, transform, crs=None, decimate: int | None = None, max_cells: int = 25_000_000):
    """Polygon (shapely) of the cells holding data. Above ``max_cells`` the mask is decimated by 8
    (plan 6.3 step 6), which may miss slivers narrower than 8 cells."""
    import rasterio.features
    from affine import Affine
    from shapely.geometry import shape
    from shapely.ops import unary_union
    valid = np.isfinite(np.asarray(z))
    tr = Affine(*transform[:6])
    k = decimate if decimate is not None else (8 if valid.size > max_cells else 1)
    if k > 1:
        h, w = valid.shape
        H, W = -(-h // k), -(-w // k)
        pad = np.zeros((H * k, W * k), bool)
        pad[:h, :w] = valid
        valid = pad.reshape(H, k, W, k).any(axis=(1, 3))
        tr = tr * Affine.scale(k)
    shapes = rasterio.features.shapes(valid.astype(np.uint8), mask=valid, transform=tr)
    geoms = [shape(g) for g, v in shapes if v == 1]
    return unary_union(geoms) if geoms else None


def write_extents(path, geom, crs) -> Path:
    """The valid-data footprint as GeoJSON (replaces GeoFabrics' ``raw_dem_extents``, gone since 0.10.24)."""
    import geopandas as gpd
    path = Path(path)
    gdf = gpd.GeoDataFrame(geometry=[geom] if geom is not None else [], crs=_horizontal(crs))
    gdf.to_file(path, driver="GeoJSON")
    return path


def crs_string(crs) -> str:
    """Short form of a CRS: ``EPSG:2193+7839`` for a compound of two EPSG CRSs, else ``EPSG:n`` or WKT."""
    from pyproj import CRS
    c = CRS.from_user_input(crs)
    if c.is_compound and len(c.sub_crs_list) == 2:
        h, v = (s.to_epsg() for s in c.sub_crs_list)
        if h and v:
            return f"EPSG:{h}+{v}"
    e = c.to_epsg()
    return f"EPSG:{e}" if e else c.to_wkt()


def _horizontal(crs):
    from pyproj import CRS
    c = CRS.from_user_input(crs)
    return c.sub_crs_list[0] if c.is_compound else c
