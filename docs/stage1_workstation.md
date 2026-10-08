# Stage 1 and eddie_terrain: workstation steps (W3, W4, W5, W6, W9 exits)

The sandbox cannot reach nz-elevation, OpenTopography or LINZ, and it has no PostGIS or Docker daemon. These
steps finish the exits the sandbox could not check. The commands are PowerShell, run from the terrain-s2
checkout unless stated otherwise.

Data live outside the repository and outside OneDrive. Replace `D:\terrain_data` below with your data folder.

## 0. Apply the branch

The work comes as one mailbox file of commits on `439f01d`.

```powershell
git fetch origin
git switch main
git pull
git switch -c stage1/w1-w9
git am '\\file\Usersm$\mwi104\Home\Downloads\stage1-w1-w9.mbox'
git log --oneline -8
git push -u origin stage1/w1-w9
# optional: version numbers read 0.1.x instead of 0.0.1.devN
git tag v0.1.0
git push origin v0.1.0
```

If `git am` stops on a conflict (if the Stage 2 agent changed a shared file in the meantime): resolve it,
`git add` the file, then `git am --continue`.

## 1. Environments

**Stage 2 development environment, for the raster backend and the tests.** Add the Stage 1 base packages and
GDAL's netCDF driver, which conda-forge ships separately.

```powershell
conda activate terrain-s2
mamba install -c conda-forge xarray netcdf4 rioxarray sqlalchemy libgdal-netcdf psycopg2
pip install --no-deps -e .
pip install --no-deps -e .\eddie
python -c "from rasterio.drivers import raster_driver_extensions as d; print('netCDF driver:', d().get('nc'))"
pytest
```

Expect `netCDF driver: netCDF`, and 127 passed. The skips are the GPU and Whitebox tests, plus the GeoFabrics
runner test if GeoFabrics is absent.

**Terrain-worker environment, for the geofabrics backend.** This is the environment the image uses, so solving
it here also checks risk 1 in plan section 14.

```powershell
mamba env create -f environment-worker.yml
conda activate terrain-worker
pip install --no-deps "geofabrics==1.1.30" "osmpythontools>=0.3.5"
pip install --no-deps -e .
pip install --no-deps -e .\eddie
python -c "import pdal; from geofabrics import runner; import importlib.metadata as m; print(m.version('geofabrics'))"
pytest tests\test_stage1_gf.py
```

If the environment does not solve, send me the error. The fallback is two images (GeoFabrics; Stage 2) sharing
the queue prefix.

## 2. W3: raster backend on live data

```powershell
conda activate terrain-s2
$env:TERRAIN_DATA_DIR = 'D:\terrain_data'
New-Item -ItemType Directory -Force out | Out-Null
terrain dem --aoi examples\aoi_whirinaki_sh12.geojson --resolution 1 --out-json out\sh12_raster_1m.json
terrain dem --aoi examples\aoi_whirinaki_sh12.geojson --resolution 8 --out-json out\sh12_raster_8m.json
Get-Content out\sh12_raster_8m.json
```

- **Check:**
  - `gap_fraction` is 0 or near it.
  - The `*.provenance.json` beside the netCDF lists the Northland survey, its item ids and `file:checksum`
    values.
  - The 1 m run has a `.tif` COG copy.
- **Repeat check:** run the 8 m command again. It prints `"reused": true` and takes about a second.
- **Survey edge (exit):** I need an AOI that straddles two LINZ surveys. Send me one, or run this on any you
  choose:

  ```powershell
  terrain dem --aoi <edge_aoi.geojson> --resolution 1 --out-json out\edge_1m.json
  ```

  The provenance should list both surveys and `gap_fraction` should be 0. `lidar_source` in the netCDF shows
  which survey supplied each cell.

The first run in a region lists every collection's metadata and builds item indexes, so it takes minutes. Later
runs use the cache in `D:\terrain_data\stac`. To pick up new surveys, delete `D:\terrain_data\stac\json`, or
send `refresh_catalogue` on a deployment.

## 3. W4: GeoFabrics backend, and the otCatalog date check

```powershell
conda activate terrain-worker
$env:TERRAIN_DATA_DIR = 'D:\terrain_data'
$env:TERRAIN_GF_MEMORY_LIMIT = '8GiB'
$env:TERRAIN_GF_CORES = '4'
terrain dem --aoi examples\aoi_whirinaki_sh12.geojson --backend geofabrics --resolution 8 --out-json out\sh12_gf_8m.json
terrain dem --aoi examples\aoi_whirinaki_sh12.geojson --backend geofabrics --resolution 1 --out-json out\sh12_gf_1m.json
terrain dem --aoi examples\aoi_whirinaki_sh12.geojson --backend geofabrics --product geofabric --resolution 8 --out-json out\sh12_geofabric_8m.json
```

- **Check:**
  - `instructions.json` beside each product has `dataset_mapping.lidar` in newest-first order.
  - The geofabric product has a `zo` variable.
  - `data_source` is 1 (LiDAR) or 0 (interpolated) on land.
  - With the `coverage` land source, the `*_land.geojson` follows the survey's tile index.
- **otCatalog date check (open item, plan 14).** Does `detail=true` return survey dates?

  ```powershell
  $q = @{productFormat='PointCloud'; minx=173.4603; miny=-35.4633; maxx=173.4710; maxy=-35.4570
         detail='true'; outputFormat='json'; include_federated='false'}
  $r = Invoke-RestMethod 'https://portal.opentopography.org/API/otCatalog' -Body $q
  $r.Datasets | ForEach-Object { $_.Dataset | Select-Object alternateName, temporalCoverage, dateCreated, datePublished }
  $r | ConvertTo-Json -Depth 20 | Set-Content out\otcatalog_sh12.json
  ```

  Send me `out\otcatalog_sh12.json`. If `temporalCoverage` is empty, I'll read the dates from the dataset
  metadata page instead. The parser already records which field it used (`date_source` in the provenance
  inputs).

## 4. W5: registry on PostGIS

Use any PostGIS database. An EDDIE deployment's `db_postgres` on port 5431 works.

```powershell
$env:TERRAIN_DB_URL = 'postgresql://postgres:<password>@localhost:5431/db'
conda activate terrain-worker
terrain dem --aoi examples\aoi_whirinaki_sh12.geojson --resolution 8 --register
terrain dem --aoi examples\aoi_whirinaki_sh12.geojson --resolution 8 --register
terrain dem --aoi examples\aoi_whirinaki_sh12.geojson --resolution 4 --register
terrain dem --aoi examples\aoi_whirinaki_sh12.geojson --resolution 8 --backend geofabrics --register
psql $env:TERRAIN_DB_URL -c "select id, product, resolution, generator_key, config_key, parent_id, created_at from terrain_product order by id"
```

**Exit:** the first two calls print the same `product_id`. Changing the resolution, backend or land source
gives a new row.

**Shim (what FReDT calls):**

```powershell
$env:TERRAIN_RESOLUTION = '8'
python -c "import geopandas as g; from terrain_s2.client.compat import get_dem_by_geometry as f; a = g.read_file('examples/aoi_whirinaki_sh12.geojson'); print(f(None, g.GeoDataFrame(geometry=[a.geometry.iloc[0].envelope], crs=a.crs)))"
```

## 5. W6: GeoFabrics regression harness

Run the same request twice into two product folders, then compare them.

```powershell
terrain dem --aoi examples\aoi_whirinaki_sh12.geojson --backend geofabrics --resolution 8 --product-dir out\harness\A
terrain dem --aoi examples\aoi_whirinaki_sh12.geojson --backend geofabrics --resolution 8 --product-dir out\harness\B
terrain compare out\harness\A out\harness\B --out out\harness\sh12_report.md
```

**Exit:** the report shows `max_abs` 0 (or float noise) and no instruction differences.

- **For a pin bump or template edit:** run B in an environment with the change.
- **For W7:** compare R-b with R-a, e.g. `--product-dir out\w7\raster` against `out\w7\gf`.

## 6. W9 and plan 9.3: worker image and the stack test

Docker Desktop, in a checkout of the EDDIE core with its usual `.env`, `api_keys.env` and
`.env.docker-override`.

```powershell
git clone https://github.com/GeospatialResearch/Digital-Twins eddie-core
cd eddie-core
git checkout v4.0.0
$env:TERRAIN_S2_DIR = '<full path to the terrain-s2 checkout>'
$env:TERRAIN_VERSION = git -C $env:TERRAIN_S2_DIR describe --tags --always
$env:TERRAIN_COMMIT = git -C $env:TERRAIN_S2_DIR rev-parse HEAD
$env:EDDIE_REF = 'v4.0.0'
docker compose -f docker-compose.yml -f "$env:TERRAIN_S2_DIR\compose\terrain-worker.yml" build terrain_worker
docker compose -f docker-compose.yml -f "$env:TERRAIN_S2_DIR\compose\terrain-worker.yml" up -d db_postgres message_broker terrain_worker
docker compose -f docker-compose.yml -f "$env:TERRAIN_S2_DIR\compose\terrain-worker.yml" ps
```

The build prints `geofabrics 1.1.30` and the three `eddie_terrain` task names; it stops if either check
fails. `ps` should show `terrain_worker` with no published ports.

**Send `ensure_dem` by name (SH12, WKT in EPSG:4326), twice:**

```powershell
$wkt = 'POLYGON ((173.4603 -35.4633, 173.4710 -35.4633, 173.4710 -35.4570, 173.4603 -35.4570, 173.4603 -35.4633))'
$py = "from eddie.tasks import app; r = app.send_task('eddie_terrain.tasks.ensure_dem', args=['$wkt', {'resolution': 8}], queue='terrain'); print(r.get(timeout=7200))"
docker compose -f docker-compose.yml -f "$env:TERRAIN_S2_DIR\compose\terrain-worker.yml" exec terrain_worker python -c $py
docker compose -f docker-compose.yml -f "$env:TERRAIN_S2_DIR\compose\terrain-worker.yml" exec terrain_worker python -c $py
docker compose exec db_postgres psql -U postgres -d db -c "select id, product, resolution, generator_key, paths->>'netcdf' from terrain_product"
```

**Exit:** both calls print the same id; the second returns in seconds (a cache hit). The row's netCDF is
under `/stored_data/terrain/`.

Repeat with `git checkout v5-integration` and `$env:EDDIE_REF = 'v5-integration'`.

Send me the outputs, or any failure, and I'll fold them into DESIGN_NOTES and the W7 report.
