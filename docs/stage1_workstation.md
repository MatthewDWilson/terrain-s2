# Stage 1 and eddie_terrain on the workstation: WSL and Docker (slot 5)

The terrain module runs as a fifth instance in the `eddie` WSL distro, next to slots 1–3 of
`EDDIE_dev_env_install_guide.md`, with the same conventions:

- a pinned clone on a local branch `instance`;
- a git-excluded overlay and `.env`;
- loopback-only ports;
- a step script that starts from a clean environment.

Everything that runs is a Linux image, as in production. This replaces the PowerShell steps of the earlier
version of this document. The Windows conda environments are now optional (section 10).

These steps finish the exits the sandbox could not check: W3, W4 and W5 on live data, W6, W9, and plan 9.3
against core v5-integration and v4.0.0.

## 0. What gets built

| Part | Where | Content |
|---|---|---|
| `~/eddie/instances/terrain` | core `Digital-Twins`, pinned to `v5-integration` (`5cb3880`); `v4.0.0` available | Postgres/PostGIS, Redis, GeoServer, Terria, backend, core worker. Built with Step 5's core harness, plus `terrain-s2` (base install) and `eddie_terrain` in the backend and core-worker images |
| terrain worker | `eddie-terrain-terrain` (from `terrain-s2/docker/terrain-worker.Dockerfile`) | GeoFabrics 1.1.30, PDAL, Stage 1 and the Stage 2 libraries; Celery queue `terrain`; no published port |
| `~/eddie/dev/terrain-s2` | WSL mirror of the GitHub branch `stage1/w1-w9` | Source for the images. Never edited: `build` fast-forwards it from GitHub. Git work (patches, commits, pushes) stays in your Windows clone |
| `~/eddie/data/terrain/cache` | bind mount at `/terrain_data` | STAC catalogue and item indexes, vector snapshots, OpenTopography LAZ tiles |
| `~/eddie/data/terrain/products` | bind mount at `/terrain_products` | One folder per generator key; readable from Windows at `\\wsl.localhost\eddie\home\<user>\eddie\data\terrain\products` |

Slot 5 ports, all on `127.0.0.1`: Terria 3501, API 5500, GeoServer 8580, Postgres 5445.

**Flow of a request.** `POST /terrain/dem` (backend) → task `eddie_terrain.tasks.ensure_dem` on queue `terrain` →
the terrain worker makes or reuses the product → a `terrain_product` row in the instance's Postgres →
`GET /terrain/tasks/<id>` returns the product id, and `GET /terrain/products/<id>` returns its paths.

**One change from the integration plan (9.2).** The terrain worker runs Celery's **threads** pool with
concurrency 1, not prefork. GeoFabrics starts a dask cluster with worker processes, and a prefork pool child is
daemonic, so it may not start child processes ("Nanny failed to start"). I reproduced this in the sandbox.
FReDT's worker uses threads too.

## 1. Prerequisites

- Steps 3–5 of the install guide are done:
  - the distro;
  - Docker integration for `eddie`;
  - `~/eddie/secrets.env`, which needs `LINZ_API_KEY` only for the Topo50 land sources.
- About 25 GB of free Docker disk for the images, plus space for products and LAZ tiles.
  `~/eddie/data/terrain/cache` grows with every new survey area.
- GitHub access from the distro (the Credential Manager set up in install guide §3.3), to clone `terrain-s2`.

## 2. Working pattern: git in PowerShell, scripts in WSL

Git work stays where it is: your Windows clone (`D:\EDDIE_Terrain\terrain-s2`), in PowerShell. WSL only builds and
runs. The WSL copy (`~/eddie/dev/terrain-s2`) is a mirror of the GitHub branch, which `build` brings up to date.
The images are built from that mirror rather than from `/mnt/d/...`, for two reasons:
- the Windows checkout has CRLF line endings (`core.autocrlf=true`), which would break the shell scripts in the
  images;
- builds are much faster on WSL's own filesystem.

So the cycle for every change I send is:

**In PowerShell (git):**

```powershell
cd D:\EDDIE_Terrain\terrain-s2
git switch stage1/w1-w9
git pull --ff-only
git am '\\file\Usersm$\mwi104\Home\Downloads\<patch>.mbox'
git push
Copy-Item '\\file\Usersm$\mwi104\Home\Downloads\eddie_step8_terrain.sh' D:\EDDIE\    # when I send a new script
```

**In WSL (`wsl -d eddie`; build and run):**

```bash
tr -d '\r' < /mnt/d/EDDIE/eddie_step8_terrain.sh > ~/eddie/scripts/eddie_step8_terrain.sh   # when the script changed
S=~/eddie/scripts/eddie_step8_terrain.sh
bash $S build                  # fast-forwards the mirror from GitHub, then builds
cd ~/eddie/instances/terrain && docker compose up -d
```

If `build` reports that the mirror has local commits or changes, something was committed or edited in WSL. Discard
it with `git -C ~/eddie/dev/terrain-s2 reset --hard origin/stage1/w1-w9`.

If you use the `edc` wrapper, add `terrain` → `~/eddie/instances/terrain` to its instance list. Otherwise run
compose commands from `~/eddie/instances/terrain` in a fresh shell. The install guide's rule applies here too:
never source an instance's `.env` into your shell.

## 3. Set up, build, start, check

```bash
S=~/eddie/scripts/eddie_step8_terrain.sh
bash $S setup 2>&1 | tee /mnt/d/EDDIE/eddie_step8_setup_report.txt
bash $S build                                    # first build ~20-40 min; full log: /mnt/d/EDDIE/eddie_step8_build.log
cd ~/eddie/instances/terrain && docker compose up -d
sleep 120
bash $S bootstrap                                # once per fresh instance: GeoServer stores (core finding F2)
bash $S check 2>&1 | tee /mnt/d/EDDIE/eddie_step8_check_report.txt
```

`bootstrap` prints the catalog status before and after: 500 then 200. Without it, Terria (http://localhost:3501)
shows "Error loading map configuration". That is the empty-instance behaviour of the core (finding F2), which
slot 1 also needed. Terrain products themselves aren't published to GeoServer.

**`setup` should show:**
- `terrain-s2 stage1/w1-w9 @ <sha>` and a `TERRAIN_VERSION`;
- `pinned -> 5cb388065 (v5-integration)`;
- `config OK`;
- seven services, with ports only on `127.0.0.1` and `(none)` for `terrain_worker`;
- the two bind mounts;
- `git status clean`.

**`build` prints `geofabrics 1.1.30`, the `terrain_s2` version and commit, and the three `eddie_terrain` task names
from the terrain worker's build-time checks.** It stops with an error if any check fails. If it fails, send me
`/mnt/d/EDDIE/eddie_step8_build.log`. The likeliest failure is the conda solve of `environment-worker.yml` on
linux-64 (plan risk 1).

**`check` should show:**
- all seven containers running (Terria `healthy`);
- `/health-check` returning 200, and `POST /terrain/dem {}` returning **400** (the blueprint is registered);
- in the terrain worker:
  - `geofabrics 1.1.30`, `pdal`, and `GDAL … netCDF driver: netCDF`;
  - both mounts writable;
  - `1 pong`, the three tasks, and queue `terrain`;
- core worker routes `{'eddie_terrain.tasks.*': {'queue': 'terrain'}}`.

## 4. Stage 1 end to end: raster backend (W3, W5)

```bash
bash $S e2e raster 8 2>&1 | tee /mnt/d/EDDIE/eddie_step8_e2e_raster8.txt
bash $S e2e raster 1 2>&1 | tee /mnt/d/EDDIE/eddie_step8_e2e_raster1.txt
```

Each run does the following:
- **Sends the request through the API:** the SH12 AOI (Whirinaki; WGS84 bounding box), `backend=raster`.
- **Follows the task**, printing one line every 15 s with the latest worker log line.
- **Inspects the product on the host:**
  - `gdalinfo` of `NETCDF:"…":z` (grid, compound CRS with NZVD2016, statistics);
  - the provenance (surveys used, gap fraction, timings, code version and commit).
- **Repeats the identical request**, which should return the same product id in a few seconds (a cache hit).
- **Lists the `terrain_product` rows.**
- **Saves everything** in `D:\EDDIE\terrain_e2e_<backend>_<res>m_<time>\`.

Expected results:
- **Gap fraction** 0, or near it.
- **Inputs:** the Northland survey (item ids and `file:checksum` in the provenance).
- **COG:** a `.tif` copy beside the 1 m product.
- **Registry:** one row per resolution, with different keys.

**If a run fails,** the script prints the end of the task's traceback and saves all of it as `task_error.txt` in
the results folder. For any task id, `bash $S task <id>` shows its status, the traceback and the worker's log lines
for it.

The first run in a region lists the LINZ collections and builds their item indexes, so it takes minutes. Later
runs read the cache. To pick up newly published surveys, delete `~/eddie/data/terrain/cache/stac/json`.

**Survey edge (W3 exit).** I still need an AOI that straddles two LINZ surveys. Send me one, or run it through the
API yourself:

```bash
curl -s -H 'Content-Type: application/json' -d '{"aoi":"POLYGON((...))","params":{"resolution":1}}' http://localhost:5500/terrain/dem
```

The provenance should then list both surveys, with `gap_fraction` 0.

## 5. GeoFabrics backend (W4), and the otCatalog date check

```bash
bash $S e2e geofabrics 8           2>&1 | tee /mnt/d/EDDIE/eddie_step8_e2e_gf8.txt
bash $S e2e geofabrics 1           2>&1 | tee /mnt/d/EDDIE/eddie_step8_e2e_gf1.txt
bash $S e2e geofabrics 8 geofabric 2>&1 | tee /mnt/d/EDDIE/eddie_step8_e2e_geofabric8.txt
bash $S otcatalog
```

Check the following in the product folder:
- **`instructions.json`:** `dataset_mapping.lidar` is newest first.
- **The geofabric product:** has a `zo` variable.
- **`data_source`:** 1 (LiDAR) or 0 (interpolated) on land.
- **`*_land.geojson`:** follows the survey tile index (the `coverage` land source).

The first GeoFabrics run downloads LAZ tiles into the cache, and dask uses `TERRAIN_GF_CORES` workers of
`TERRAIN_GF_MEMORY_LIMIT` each (4 × 8 GiB in `.env`; the container limit is 48 GB). Adjust both in `.env`, then run
`docker compose up -d terrain_worker`.

**`otcatalog`** saves the OpenTopography response for SH12 to `D:\EDDIE\otcatalog_sh12_<date>.json` and prints each
dataset's name and dates. Send me the file. It answers whether `detail=true` returns survey dates (an open item).

## 6. Tests inside the worker image

```bash
bash $S test                              # full suite, with real PDAL and GeoFabrics
bash $S test tests/test_stage1_gf.py      # or any subset
```

The suite runs from the working clone, mounted read-only. Unlike in the sandbox, the GeoFabrics runner test runs
against real PDAL here.

## 7. Regression harness (W6)

Run the same request twice into two product folders, then compare them:

```bash
cd ~/eddie/instances/terrain
W='POLYGON ((173.4603 -35.4633, 173.4710 -35.4633, 173.4710 -35.4570, 173.4603 -35.4570, 173.4603 -35.4633))'
docker compose exec terrain_worker terrain dem --aoi "$W" --backend geofabrics --resolution 8 --product-dir /terrain_products/harness/A --quiet
docker compose exec terrain_worker terrain dem --aoi "$W" --backend geofabrics --resolution 8 --product-dir /terrain_products/harness/B --quiet
docker compose exec terrain_worker terrain compare /terrain_products/harness/A /terrain_products/harness/B --out /terrain_products/harness/sh12_report.md
```

**Exit:** `max_abs` is 0 (or float noise), and there are no instruction differences. The report is in
`~/eddie/data/terrain/products/harness/`.

- **For a pin bump or template edit:** produce B after rebuilding with the change.
- **For W7:** compare a raster and a GeoFabrics product the same way.

## 8. v4 compatibility (plan 9.3)

```bash
bash $S core v4 && bash $S setup && bash $S build
cd ~/eddie/instances/terrain && docker compose up -d && sleep 120
bash $S check && bash $S e2e raster 8
bash $S core v5 && bash $S setup && bash $S build && docker compose up -d     # back to the target
```

Both the backend harness and the terrain worker image are rebuilt against the selected core (`EDDIE_REF` in
`.env`).

## 9. Development loop, and the Stage 2 agent

- **Change, commit and push from the Windows clone** (section 2), then run `bash $S build` in WSL. That updates the
  mirror, refreshes `TERRAIN_VERSION` and `TERRAIN_COMMIT`, and rebuilds. Then run `docker compose up -d`.
- **Only pushed commits are built.** The generator key records the commit, so a new commit gives new products (by
  design: plan 6.5).
- **Stage 2 in the same worker.** The terrain worker already has the Stage 2 libraries (`[detect]`). When S2-1 adds
  `terrain_s2.run.network`, `eddie_terrain.tasks.ensure_network` (Input B) works in this slot without further
  setup. The Stage 2 agent's W12 work lands in the same clone and branch layout (patches, as agreed).
- **GPU.** Stage 2's CuPy path needs a GPU reservation for `terrain_worker` (as the core worker has). Add it when
  Stage 2 needs it; Stage 1 doesn't.

## 10. Windows conda environments (optional)

The `terrain-s2` conda environment on Windows still runs the tests and the raster backend from PowerShell, and
remains useful for Stage 2 development on the workstation GPU. It isn't needed for Stage 1 or the module. The
earlier Windows `terrain-worker` environment is no longer needed: the worker image uses the same
`environment-worker.yml` on linux-64.

## 11. Troubleshooting

| Symptom | Cause | Resolution |
|---|---|---|
| `setup`: `!! python pin not applied` / `redis fix not applied` | The core's `environment.yml` changed at the pinned revision | Send me the setup report; the harness `sed` patterns need updating |
| `build`: conda solve error in the terrain worker stage | `environment-worker.yml` doesn't solve on linux-64 | Send the build log |
| `build`: `TERRAIN_VERSION ... is required` | `.env` lacks the version | `bash $S setup` (or `build`, which refreshes it) |
| `check`: a mount `writable: touch: … Permission denied` | `TERRAIN_UID`/`TERRAIN_GID` in `.env` don't match the owner of `~/eddie/data/terrain` | Fix ownership (`sudo chown -R $(id -u):$(id -g) ~/eddie/data/terrain`) or the values, then `bash $S build` |
| Task `FAILURE`: `Nanny failed to start` | A prefork pool (not this image's default) | Keep the entrypoint's `-P threads` |
| Task stays `PENDING` | Terrain worker not running, or not on queue `terrain` | `docker compose logs terrain_worker`; `check` lists the queues |
| `e2e`: `POST /terrain/dem failed: …400…` | A parameter was refused by validation | The message names it |
| `bootstrap`: `PermissionError … /stored_data/geoserver/data` | The `stored_data` volume was first created by the terrain worker (before 12 Oct the worker mounted it), so it has the wrong owner | `docker compose down`, `docker volume rm eddie-terrain_stored_data`, `docker compose up -d`, then `bootstrap` again. The worker no longer mounts `stored_data` |
| Task `FAILURE` within ~15 s: `GET https://nz-elevation.s3-ap-southeast-2.amazonaws.com/catalog.json failed after 3 attempts` | The terrain worker can't reach the LINZ elevation bucket (proxy, firewall, DNS) | `check` prints the status from inside the worker (expect 200). Compare with `curl -I` of the same URL in WSL. If WSL works and the container doesn't, send both outputs |
| Task `FAILURE`: `ModuleNotFoundError: No module named 'psycopg'` | The worker image had SQLAlchemy 2.1, which reads the core's `postgresql://` URL as psycopg 3 | Fixed 12 Oct (SQLAlchemy pinned to 2.0, as the core): `bash $S build`. `check` now shows `registry: terrain_product ready on postgresql://…` |
| Worker log shows the banner and nothing else | The v5 core's `setup_logging()` stops the `celery` logger propagating (core finding F3) | Images built from 12 Oct set `TERRAIN_CELERY_LOGS=1` in the terrain worker, which restores the worker's own lines. Rebuild with `bash $S build` |
| Another instance's names or ports appear | Shell variables from another `.env` | Fresh terminal; use the script or `edc` |

## 12. What to send back

- `eddie_step8_setup_report.txt`, `eddie_step8_check_report.txt`, and the build log if it failed.
- The `e2e` outputs for raster 8 m and 1 m, and GeoFabrics 8 m and 1 m (the folders in `D:\EDDIE\`).
- `otcatalog_sh12_<date>.json`.
- A survey-edge AOI.
- `bash $S test` output.
