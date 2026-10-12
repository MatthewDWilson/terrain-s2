"""PyWPS processes (v5 D4): ``terrain.dem``. ``terrain.network`` follows when Stage 2 provides its entry point.

Unused in v4 (the core serves no plugin processes there). The process sends ``ensure_dem`` to the terrain queue
and waits for the product id, so the identifier doubles as the D3 cache ``process_id``.
"""
from pywps import LiteralInput, LiteralOutput, Process


class TerrainDemProcess(Process):
    def __init__(self):
        inputs = [
            LiteralInput("aoi", "Area of interest (WKT, EPSG:4326)", data_type="string"),
            LiteralInput("resolution", "Resolution in metres (1, 2, 4 or 8)", data_type="integer", min_occurs=0),
            LiteralInput("backend", "raster or geofabrics", data_type="string", min_occurs=0,
                         allowed_values=["raster", "geofabrics"]),
            LiteralInput("product", "dem or geofabric (adds roughness zo)", data_type="string", min_occurs=0,
                         allowed_values=["dem", "geofabric"]),
            LiteralInput("land_source", "coverage, topo50, topo50_mangrove or file", data_type="string",
                         min_occurs=0, allowed_values=["coverage", "topo50", "topo50_mangrove", "file"]),
            LiteralInput("buffer_m", "AOI buffer in metres", data_type="float", min_occurs=0),
        ]
        outputs = [LiteralOutput("product_id", "terrain_product id", data_type="integer"),
                   LiteralOutput("netcdf", "Path of the DEM product (netCDF)", data_type="string")]
        super().__init__(self._handler, identifier="terrain.dem", title="Terrain DEM (Stage 1)",
                         abstract="A DEM at an integer resolution from the LINZ 1 m DEM or from point clouds "
                                  "through GeoFabrics, in the GeoFabrics-compatible netCDF contract.",
                         version="1", inputs=inputs, outputs=outputs, store_supported=True, status_supported=True)

    @staticmethod
    def _handler(req, response):
        from eddie.tasks import app as celery_app

        from . import QUEUE
        from .tables import get_store
        aoi = req.inputs["aoi"][0].data
        params = {k: req.inputs[k][0].data for k in ("resolution", "backend", "product", "land_source", "buffer_m")
                  if k in req.inputs}
        pid = celery_app.send_task("eddie_terrain.tasks.ensure_dem", args=[aoi, params], queue=QUEUE).get()
        response.outputs["product_id"].data = int(pid)
        response.outputs["netcdf"].data = get_store().get(int(pid)).netcdf
        return response


processes = [TerrainDemProcess()]
