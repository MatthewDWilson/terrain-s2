#!/usr/bin/env bash
# eddie_step8_terrain.sh - Step 8: terrain instance (slot 5) at ~/eddie/instances/terrain.
# The EDDIE core (pinned), with eddie_terrain installed in its backend and worker, plus the terrain worker
# (GeoFabrics 1.1.30 + Stage 1 + Stage 2 libraries) on Celery queue `terrain`. Same conventions as Steps 5-7:
# pinned clone on a local branch 'instance', git-excluded overlay and .env, loopback ports, clean environment.
#
#   setup  (default)  clone/pin the core, clone terrain-s2 (if absent), write harness, overlay and .env, validate
#   build             update the WSL mirror of terrain-s2 from GitHub (fast-forward only), refresh the version in
#                     .env, then build all images (log to /mnt/d/EDDIE). Git work itself stays in the Windows clone.
#   check             after 'docker compose up -d': containers, endpoints, terrain routes, worker toolchain, mounts
#   e2e [raster|geofabrics] [resolution] [dem|geofabric]
#                     POST /terrain/dem for the SH12 AOI, follow the task, inspect the product, repeat (cache hit)
#   task <task id>    status of a terrain task; on failure the end of its traceback (full text to /mnt/d/EDDIE)
#   bootstrap         one-off per fresh instance: create the GeoServer workspaces and stores (core finding F2:
#                     otherwise /terria-catalog.json returns 500 and Terria shows "Error loading map configuration")
#   test              run the terrain-s2 test suite inside the terrain worker image (real PDAL and GeoFabrics)
#   otcatalog         save the OpenTopography catalogue response for SH12 (the W4 survey-date check)
#   core v4|v5        re-pin the core (v4.0.0 or v5-integration); then 'build' and 'docker compose up -d'
#
# Writes only untracked files in the instance (listed in .git/info/exclude), ~/eddie/data/terrain, and (on first
# setup) the clone ~/eddie/dev/terrain-s2. Existing .env / api_keys.env are kept; only the TERRAIN_* block is updated.
# Run inside the 'eddie' distro:
#   tr -d '\r' < /mnt/d/EDDIE/eddie_step8_terrain.sh > ~/eddie/scripts/eddie_step8_terrain.sh
#   bash ~/eddie/scripts/eddie_step8_terrain.sh [mode] 2>&1 | tee /mnt/d/EDDIE/eddie_step8_<mode>_report.txt
set -uo pipefail

# Start from a clean environment: variables exported in the calling shell (e.g. another instance's .env:
# COMPOSE_PROJECT_NAME, ports, passwords) take priority over this instance's .env in Docker Compose.
if [ -z "${EDDIE_CLEAN_ENV:-}" ]; then
    exec env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" TERM="${TERM:-xterm}" EDDIE_CLEAN_ENV=1 bash "$0" "$@"
fi

INST="$HOME/eddie/instances/terrain"
TS2="$HOME/eddie/dev/terrain-s2"
DATA="$HOME/eddie/data/terrain"
SECRETS="$HOME/eddie/secrets.env"
PROJECT=eddie-terrain
P_WWW=3501; P_API=5500; P_GS=8580; P_DB=5445   # slot 5, all bound to 127.0.0.1
TS2_URL="https://github.com/MatthewDWilson/terrain-s2.git"
TS2_BRANCH="stage1/w1-w9"
CORE_URL="https://github.com/GeospatialResearch/Digital-Twins.git"
CORE_V5=5cb38806535bde7fbdeda4dc019091a87d4cfdfe    # v5-integration (8 Oct 2026), the module's target
CORE_V4=9c538811b81353601a34311511ddbc0e9bde4e2c    # v4.0.0, FReDT's pin
# SH12 test site (Whirinaki), WGS84 bounding box of examples/aoi_whirinaki_sh12.geojson
SH12_WKT='POLYGON ((173.4603 -35.4633, 173.4710 -35.4633, 173.4710 -35.4570, 173.4603 -35.4570, 173.4603 -35.4633))'

say()  { printf '\n==== %s\n' "$1"; }
note() { printf '  %s\n' "$1"; }

task_error() {  # task_error JSON [FILE]: a failed task's traceback as text (last 20 lines here; all of it to FILE)
    # The backend returns it as text; older eddie_terrain versions return the Python tuple of lines.
    local txt
    txt=$(echo "$1" | jq -r '.error // empty' | python3 -c '
import ast, sys
t = sys.stdin.read().strip()
try:
    v = ast.literal_eval(t)
    t = "\n".join(v) if isinstance(v, (tuple, list)) else str(v)
except Exception:
    pass
print(t.rstrip())')
    [ -n "${2:-}" ] && printf '%s\n' "$txt" > "$2"
    printf '%s\n' "$txt" | grep -v '^\s*[~^ ]*$' | tail -20 | cut -c1-240 | sed 's/^/    /'
}

exclude_local() {  # keep instance-local files out of git status, without touching tracked files
    local ex; ex="$(git -C "$1" rev-parse --path-format=absolute --git-path info/exclude)"
    for p in 'docker-compose.local.yml' '.env' 'api_keys.env' '.local/'; do
        grep -qxF "$p" "$ex" 2>/dev/null || echo "$p" >> "$ex"
    done
}

set_env() {  # set_env KEY VALUE: replace the line in .env, or append it
    local f="$INST/.env"
    if grep -q "^$1=" "$f"; then sed -i "s|^$1=.*|$1=$2|" "$f"; else printf '%s=%s\n' "$1" "$2" >> "$f"; fi
}

refresh_version() {  # the terrain-s2 version and commit are recorded in every generator key
    local v c
    v="$(git -C "$TS2" describe --tags --always --dirty 2>/dev/null)"
    c="$(git -C "$TS2" rev-parse HEAD 2>/dev/null)"
    [ -n "$c" ] || { note "!! cannot read the terrain-s2 commit in $TS2"; exit 1; }
    # setuptools-scm needs a PEP 440 version: a tag gives one; otherwise 0.0.1.dev0+g<sha>
    case "$v" in v[0-9]*) v="${v#v}"; v="${v%%-dirty}"; [[ "$v" == *-* ]] && v="${v%%-*}.post${v#*-}"; v="${v//-g/+g}"; v="${v//-/.}" ;;
                 *) v="0.0.1.dev0+g${c:0:9}" ;; esac
    [ -n "$(git -C "$TS2" status --porcelain 2>/dev/null)" ] && note "!! terrain-s2 has uncommitted changes: they are built, but the key records the commit only"
    set_env TERRAIN_VERSION "$v"
    set_env TERRAIN_COMMIT "$c"
    note "terrain-s2 $TS2_BRANCH @ ${c:0:9} -> TERRAIN_VERSION=$v"
}

# ------------------------------------------------------------------------------------------------ setup
setup() {
    [ -f "$SECRETS" ] || { note "!! $SECRETS missing - run Step 5 setup first"; exit 1; }
    set -a; . "$SECRETS"; set +a
    [ -n "${LINZ_API_KEY:-}" ] || note "LINZ_API_KEY empty: only land_source=topo50/topo50_mangrove need it"

    say "terrain-s2 working clone ($TS2)"
    if [ -d "$TS2/.git" ]; then
        note "exists: $(git -C "$TS2" rev-parse --abbrev-ref HEAD) @ $(git -C "$TS2" rev-parse --short HEAD) ('build' updates it from GitHub)"
    else
        mkdir -p "$(dirname "$TS2")"
        git clone --quiet --branch "$TS2_BRANCH" "$TS2_URL" "$TS2" && note "cloned $TS2_BRANCH"
    fi
    git -C "$TS2" fetch --quiet origin 2>/dev/null && \
        note "behind origin/$TS2_BRANCH by $(git -C "$TS2" rev-list --count HEAD..origin/$TS2_BRANCH 2>/dev/null || echo ?) commit(s)"

    say "Core instance ($INST)"
    if [ -d "$INST/.git" ]; then note "exists"; else git clone --quiet "$CORE_URL" "$INST" && note "cloned"; fi
    git -C "$INST" fetch --quiet origin
    if git -C "$INST" rev-parse --verify --quiet instance >/dev/null; then
        note "branch 'instance' @ $(git -C "$INST" rev-parse --short HEAD) (left as is; 'core v4|v5' re-pins)"
    else
        git -C "$INST" switch --quiet -c instance "$CORE_V5" && note "pinned -> ${CORE_V5:0:9} (v5-integration)"
    fi
    exclude_local "$INST"
    CORE_SHA="$(git -C "$INST" rev-parse HEAD)"

    say "Data folders ($DATA)"
    mkdir -p "$DATA/cache" "$DATA/products"
    note "cache    -> /terrain_data      (STAC and item indexes, snapshots, LAZ tiles)"
    note "products -> /terrain_products  (one folder per generator key)"

    say "Core harness (.local/): Step 5's core harness + eddie_terrain in the backend and worker images"
    mkdir -p "$INST/.local"
    sed -E -e 's/^(\s*- python)>=3\.13\s*$/\1=3.13  # harness pin: core floats >=3.13 -> 3.14 (no fiona wheels)/' \
           -e 's/^(\s*- )redis==5\.0\.3\s*$/\1redis-py==5.0.3  # harness fix: conda-forge "redis" is the server, "redis-py" the client/' \
        "$INST/environment.yml" > "$INST/.local/environment.yml"
    grep -q -- '- python=3.13' "$INST/.local/environment.yml" || { note "!! python pin not applied - core environment.yml changed"; exit 1; }
    grep -q -- '- redis-py==5.0.3' "$INST/.local/environment.yml" || { note "!! redis fix not applied - core environment.yml changed"; exit 1; }
    # terrain-s2 base install (light): what eddie_terrain needs in the backend to validate requests
    sed -i -E 's/^(\s*)- pip\s*$/\1- xarray\n\1- rioxarray\n\1- netcdf4\n\1- pyyaml\n&/' "$INST/.local/environment.yml"
    grep -q -- '- rioxarray' "$INST/.local/environment.yml" && note "environment: python 3.13, redis-py, + xarray rioxarray netcdf4 pyyaml"
    cat > "$INST/.local/core.Dockerfile" <<'EOF'
# .local/core.Dockerfile - core harness for the terrain slot (local only, not committed).
# Step 5's vanilla-core harness (see eddie_step5_core.sh for the five documented differences from core's own
# src/Dockerfile), plus terrain-s2 (base install) and eddie_terrain from the 'terrain' build context, so the backend
# registers /terrain/* and the core worker knows the terrain task names. GeoFabrics is NOT in this image: terrain
# work runs on the separate terrain worker.
FROM continuumio/miniconda3:26.1.1 AS build
ARG TERRAIN_VERSION
ARG TERRAIN_COMMIT
WORKDIR /app
RUN conda install -c conda-forge mamba
COPY .local/environment.yml environment.yml
RUN mamba env create -f environment.yml
SHELL ["conda", "run", "-n", "digitaltwin", "/bin/bash", "-c"]
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install . && python -c "import eddie; print('eddie importable from', eddie.__file__)"
COPY --from=terrain pyproject.toml README.md /terrain/
COPY --from=terrain src /terrain/src
COPY --from=terrain eddie /terrain/eddie
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TERRAIN_S2=${TERRAIN_VERSION} \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_EDDIE_TERRAIN=${TERRAIN_VERSION}
RUN echo "${TERRAIN_COMMIT}" > /terrain/src/terrain_s2/_commit.txt \
 && pip install --no-deps /terrain /terrain/eddie \
 && python -c "import terrain_s2, eddie_terrain; print('terrain_s2', terrain_s2.__version__, terrain_s2.git_commit())"
RUN <<BUILD
    conda-pack --ignore-missing-files -n digitaltwin -o /tmp/env.tar
    mkdir /venv
    cd /venv
    tar xf /tmp/env.tar
    rm /tmp/env.tar
    /venv/bin/conda-unpack
BUILD

FROM lparkinson/bg_flood:v0.9 AS runtime
WORKDIR /app
USER root
ADD --chmod=555 https://github.com/gruntwork-io/health-checker/releases/download/v0.0.8/health-checker_linux_amd64 \
 /usr/local/bin/health-checker
RUN <<RUNTIME
    apt-get update
    apt-get install -y --no-install-recommends acl ca-certificates curl gettext-base
    rm -fr /var/lib/apt/lists/*
    apt-get purge -y ca-certificates
    mkdir /stored_data
    setfacl -R -d -m u:nonroot:rwx /stored_data
    setfacl -R -m u:nonroot:rwx /stored_data
    mkdir /stored_data/geoserver
    for NEW_DIRECTORY in "outputs" "workdir" "logs"
    do
        mkdir "$NEW_DIRECTORY"
        setfacl -R -d -m u:nonroot:rwx "$NEW_DIRECTORY"
        setfacl -R -m u:nonroot:rwx "$NEW_DIRECTORY"
    done
RUNTIME
COPY --chown=root:root --chmod=555 --from=build /venv /venv
COPY --chown=root:root --chmod=444 selected_polygon.geojson .
COPY --chown=root:root --chmod=555 src src/
COPY --chown=nonroot:nonroot --chmod=644 src/pywps.cfg src/
COPY --chown=root:root --chmod=555 .local/backend_entrypoint.sh .local/celery_worker_entrypoint.sh /app/harness/
USER nonroot
EOF
    cat > "$INST/.local/backend_entrypoint.sh" <<'EOF'
#!/bin/bash
# Harness copy of src/backend_entrypoint.sh with the app path corrected to eddie.app
ENV_VARS_TO_FILL='$BACKEND_HOST,$BACKEND_PORT'
SUBBED=$(envsubst "$ENV_VARS_TO_FILL" < src/pywps.cfg)
echo "$SUBBED" > src/pywps.cfg
source /venv/bin/activate
exec gunicorn --bind 0.0.0.0:5000 --timeout 600 -k gevent eddie.app:app
EOF
    cat > "$INST/.local/celery_worker_entrypoint.sh" <<'EOF'
#!/bin/bash
# Harness copy of src/celery_worker_entrypoint.sh with the task module corrected to eddie.tasks
source /venv/bin/activate
health-checker --listener 0.0.0.0:5001 --log-level error --script-timeout 10 \
  --script "celery -A eddie.tasks inspect ping" &
exec celery -A eddie.tasks worker -P threads --loglevel=INFO
EOF
    note "written: core.Dockerfile, environment.yml, backend/celery_worker entrypoints"

    say "Compose overlay (docker-compose.local.yml)"
    cat > "$INST/docker-compose.local.yml" <<EOF
# docker-compose.local.yml - slot 5, terrain. Local only (in .git/info/exclude).
# Merged after the core's docker-compose.yml and terrain-s2's compose/terrain-worker.yml (COMPOSE_FILE in .env).
# Unique container names; published ports bound to 127.0.0.1; Redis, worker health ports and the terrain worker
# not published. Needs Compose >= 2.24.4 for !override / !reset.
services:
  db_postgres:
    container_name: ${PROJECT}-db
    restart: unless-stopped
    ports: !override
      - "127.0.0.1:${P_DB}:5432"
  backend:
    container_name: ${PROJECT}-backend
    build:
      dockerfile: .local/core.Dockerfile
      additional_contexts:
        terrain: \${TERRAIN_S2_DIR}
      args:
        TERRAIN_VERSION: \${TERRAIN_VERSION}
        TERRAIN_COMMIT: \${TERRAIN_COMMIT}
    entrypoint: ["/app/harness/backend_entrypoint.sh"]
    volumes:
      - ${DATA}/products:/terrain_products:ro   # registry paths resolve here too (read-only)
    ports: !override
      - "127.0.0.1:${P_API}:5000"
  celery_worker:
    container_name: ${PROJECT}-worker
    restart: unless-stopped
    build:
      dockerfile: .local/core.Dockerfile
      additional_contexts:
        terrain: \${TERRAIN_S2_DIR}
      args:
        TERRAIN_VERSION: \${TERRAIN_VERSION}
        TERRAIN_COMMIT: \${TERRAIN_COMMIT}
    entrypoint: ["/app/harness/celery_worker_entrypoint.sh"]
    volumes:
      - ${DATA}/products:/terrain_products:ro
    depends_on:
      - backend    # create after backend: avoids first-start race on the nested geoserver_data mount
    ports: !reset []
  terrain_worker:
    container_name: ${PROJECT}-terrain
    restart: unless-stopped
    environment:
      TERRAIN_PRODUCT_DIR: /terrain_products
    volumes: !override
      - ${DATA}/cache:/terrain_data
      - ${DATA}/products:/terrain_products
  geoserver:
    container_name: ${PROJECT}-geoserver
    restart: unless-stopped
    ports: !override
      - "127.0.0.1:${P_GS}:8080"
  terria_map:
    container_name: ${PROJECT}-www
    ports: !override
      - "127.0.0.1:${P_WWW}:3001"
    healthcheck:
      # shipped check runs curl, which the TerriaMap image does not contain (always unhealthy); node is present
      test: ["CMD", "node", "-e", "require('http').get('http://localhost:3001/',r=>process.exit(r.statusCode<400?0:1)).on('error',()=>process.exit(1))"]
  message_broker:
    container_name: ${PROJECT}-redis
    restart: unless-stopped
    ports: !reset []
EOF
    note "written"

    say "Instance .env and api_keys.env"
    if [ -f "$INST/.env" ]; then
        note ".env exists - kept (only the TERRAIN_* block below is updated)"
    else
        umask 077
        cat > "$INST/.env" <<EOF
# .env - slot 5, terrain. Generated $(date -I). Local only.
# Host-side values for compose interpolation; container-side hosts and paths come from .env.docker-override.
COMPOSE_PROJECT_NAME=${PROJECT}
COMPOSE_FILE=docker-compose.yml:${TS2}/compose/terrain-worker.yml:docker-compose.local.yml
POSTGRES_HOST=localhost
POSTGRES_PORT=${P_DB}
POSTGRES_DB=db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=$(openssl rand -hex 16)
MESSAGE_BROKER_HOST=localhost
GEOSERVER_HOST=http://localhost
GEOSERVER_PORT=${P_GS}
GEOSERVER_ADMIN_NAME=admin
GEOSERVER_ADMIN_PASSWORD=$(openssl rand -hex 12)
BACKEND_HOST=http://localhost
BACKEND_PORT=${P_API}
WWW_PORT=${P_WWW}
CESIUM_ACCESS_TOKEN=${CESIUM_ACCESS_TOKEN:-}
LINZ_BASEMAPS_API_KEY=${LINZ_BASEMAPS_API_KEY:-}
DATA_DIR_GEOSERVER=/stored_data/geoserver
# Terrain request defaults (terrain_s2/settings.py; edit freely)
TERRAIN_BACKEND=raster
TERRAIN_RESOLUTION=8
TERRAIN_LAND_SOURCE=coverage
TERRAIN_SPATIAL_REUSE=false
TERRAIN_AOI_MODE=bbox
TERRAIN_GF_MEMORY_LIMIT=8GiB
TERRAIN_GF_CORES=4
TERRAIN_WORKER_CONCURRENCY=1
TERRAIN_WORKER_MEMORY=48g
# Managed by eddie_step8_terrain.sh (setup / build / core) - do not edit below
EOF
        note ".env generated (random passwords)"
    fi
    set_env TERRAIN_S2_DIR "$TS2"
    set_env EDDIE_REF "$CORE_SHA"
    # the terrain worker runs as your user, so the bind-mounted cache and products stay yours (root -> 1000)
    local uid gid; uid=$(id -u); gid=$(id -g); [ "$uid" = 0 ] && uid=1000 && gid=1000
    set_env TERRAIN_UID "$uid"
    set_env TERRAIN_GID "$gid"
    refresh_version
    note "core for the terrain worker image: EDDIE_REF=${CORE_SHA:0:9} (same as this instance)"
    if [ -f "$INST/api_keys.env" ]; then note "api_keys.env exists - left unchanged"
    else
        umask 077
        printf 'STATSNZ_API_KEY=%s\nLINZ_API_KEY=%s\nMFE_API_KEY=%s\n' \
            "${STATSNZ_API_KEY:-}" "${LINZ_API_KEY:-}" "${MFE_API_KEY:-}" > "$INST/api_keys.env"
        note "api_keys.env generated from secrets.env"
    fi

    say "Validate merged compose config"
    cd "$INST" || exit 1
    if docker compose config --quiet; then note "config OK"; else note "!! config invalid - see error above"; exit 1; fi
    docker compose config --format json | jq -r '
      .services | to_entries[] |
      "    \(.key)\t\(.value.container_name // "-")\t" +
      ([.value.ports[]? | "\(.host_ip // "ALL-INTERFACES"):\(.published)->\(.target)"] | join(", ") | if . == "" then "(none)" else . end)'
    if docker compose config --format json | jq -e '[.services[].ports[]? | select(.host_ip != "127.0.0.1")] | length > 0' >/dev/null; then
        note "!! some port is not bound to 127.0.0.1"
    else note "all published ports bound to 127.0.0.1"; fi
    note "terrain worker mounts:"
    docker compose config --format json | jq -r '.services.terrain_worker.volumes[] | "    \(.source) -> \(.target)"'
    note "terrain worker build args: $(docker compose config --format json | jq -c '.services.terrain_worker.build.args')"
    for n in $(docker compose config --format json | jq -r '.services[].container_name // empty'); do
        docker ps -a --format '{{.Names}}' | grep -qx "$n" && note "!! name clash: $n exists" || true
    done
    [ -z "$(git -C "$INST" status --short)" ] && note "git status clean (all local files excluded)" \
        || git -C "$INST" status --short | sed 's/^/  git status: /'

    say "Next"
    note "bash ~/eddie/scripts/eddie_step8_terrain.sh build     # first build: core images + terrain worker (conda env), ~20-40 min"
    note "cd $INST && docker compose up -d"
    note "sleep 120; bash ~/eddie/scripts/eddie_step8_terrain.sh bootstrap   # once per fresh instance"
    note "bash ~/eddie/scripts/eddie_step8_terrain.sh check 2>&1 | tee /mnt/d/EDDIE/eddie_step8_check_report.txt"
}

# ------------------------------------------------------------------------------------------------ build
sync_mirror() {  # the WSL clone mirrors GitHub; git work (commits, git am, push) happens in the Windows clone
    git -C "$TS2" fetch --quiet origin || { note "!! git fetch failed in $TS2"; exit 1; }
    local behind ahead
    behind=$(git -C "$TS2" rev-list --count HEAD.."origin/$TS2_BRANCH" 2>/dev/null || echo 0)
    ahead=$(git -C "$TS2" rev-list --count "origin/$TS2_BRANCH"..HEAD 2>/dev/null || echo 0)
    if [ "$ahead" != 0 ] || [ -n "$(git -C "$TS2" status --porcelain)" ]; then
        note "!! $TS2 has local commits or changes; it should only mirror origin/$TS2_BRANCH"
        note "   inspect: git -C $TS2 status; to discard them: git -C $TS2 reset --hard origin/$TS2_BRANCH"
        exit 1
    fi
    if [ "$behind" != 0 ]; then
        git -C "$TS2" merge --quiet --ff-only "origin/$TS2_BRANCH" && note "updated from GitHub: $behind new commit(s)"
    else
        note "up to date with origin/$TS2_BRANCH (did you push from Windows?)"
    fi
    git -C "$TS2" log -1 --format='  now at %h %s' 
}

build() {
    cd "$INST" || exit 1
    say "terrain-s2 mirror ($TS2)"; sync_mirror
    say "terrain-s2 version"; refresh_version
    say "Build (all images; log: /mnt/d/EDDIE/eddie_step8_build.log)"
    docker compose --progress=plain build 2>&1 | tee /mnt/d/EDDIE/eddie_step8_build.log | \
        grep -E "^#[0-9]+ (DONE|ERROR)|geofabrics 1|terrain_s2 |eddie_terrain|ERROR|failed" | tail -40
    st=${PIPESTATUS[0]}
    [ "$st" = 0 ] && note "build OK - next: docker compose up -d" || note "!! build failed (exit $st) - send /mnt/d/EDDIE/eddie_step8_build.log"
}

# ------------------------------------------------------------------------------------------------ check
check() {
    cd "$INST" || exit 1
    say "Containers"
    docker compose ps -a --format 'table {{.Service}}\t{{.Name}}\t{{.State}}\t{{.Status}}\t{{.Ports}}'
    say "HTTP endpoints"
    for u in "http://localhost:${P_API}/" "http://localhost:${P_API}/health-check" \
             "http://localhost:${P_GS}/geoserver/web/" "http://localhost:${P_WWW}/"; do
        printf '  %-50s %s\n' "$u" "$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$u")"
    done
    say "Terrain routes (backend has eddie_terrain)"
    printf '  POST /terrain/dem {}           -> %s (expect 400: aoi required)\n' \
        "$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -H 'Content-Type: application/json' -d '{}' "http://localhost:${P_API}/terrain/dem")"
    printf '  GET  /terrain/products/999999  -> %s (expect 404)\n' \
        "$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "http://localhost:${P_API}/terrain/products/999999")"
    say "Terrain worker"
    docker compose exec -T terrain_worker bash -c '
        source /venv/bin/activate 2>/dev/null
        echo "  user: $(id -un) uid $(id -u); python $(python --version 2>&1 | cut -d" " -f2)"
        python - <<PY 2>&1 | sed "s/^/  /"
import importlib.metadata as m, rasterio, terrain_s2
from rasterio.drivers import raster_driver_extensions as d
print("geofabrics", m.version("geofabrics"), "| geoapis", m.version("geoapis"), "| pdal", m.version("pdal"))
print("terrain_s2", terrain_s2.__version__, "| commit", terrain_s2.git_commit())
print("GDAL", rasterio.__gdal_version__, "| netCDF driver:", d().get("nc"))
PY
        for d in /terrain_data /terrain_products; do
            echo "  $d writable: $(touch "$d/.w" 2>&1 && rm "$d/.w" && echo yes)"
        done
        echo "  LINZ elevation STAC from the worker: $(curl -s -o /dev/null -w "%{http_code}" --max-time 20 https://nz-elevation.s3-ap-southeast-2.amazonaws.com/catalog.json) (expect 200)"
        echo "  ping: $(celery -A eddie.tasks inspect ping -d terrain@$HOSTNAME 2>&1 | grep -c pong) pong"
        celery -A eddie.tasks inspect registered -d terrain@$HOSTNAME 2>/dev/null | grep -o "eddie_terrain[a-z_.]*" | sed "s/^/  task: /"
        echo "  queues: $(celery -A eddie.tasks inspect active_queues -d terrain@$HOSTNAME 2>/dev/null | grep -oE "name.: .[a-z_]+" | sort -u | tr "\n" " ")"
    ' 2>&1
    say "Core worker knows the terrain tasks (routes to queue terrain)"
    docker compose exec -T celery_worker bash -c 'source /venv/bin/activate && python -c "import eddie.tasks as t; print(\"  routes:\", t.app.conf.task_routes)"' 2>&1 | tail -1
    say "Recent errors in logs (last 300 lines per service)"
    for s in backend celery_worker terrain_worker; do
        n=$(docker compose logs --tail 300 "$s" 2>&1 | grep -ciE 'error|traceback|exception')
        printf '  %-14s %s matching lines\n' "$s" "$n"
        docker compose logs --tail 300 "$s" 2>&1 | grep -iE 'error|traceback|exception' | tail -5 | cut -c1-220 | sed 's/^/      /'
    done
    say "Images and disk"
    docker images --format '{{.Repository}}:{{.Tag}}  {{.Size}}' | grep -E "^${PROJECT}" | sed 's/^/  /'
    du -sh "$DATA"/* 2>/dev/null | sed 's/^/  /'
}

# ------------------------------------------------------------------------------------------------ bootstrap
bootstrap() {
    # As eddie_step5_core.sh bootstrap: serve_static_files() creates the main PostGIS store in every catalog
    # workspace, which a module's base-data load would normally do. Core's src/static/geo holds only a readme.
    cd "$INST" || exit 1
    say "Initialise GeoServer workspaces and stores"
    printf '  catalog before: HTTP %s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 "http://localhost:${P_API}/terria-catalog.json")"
    docker compose exec -T backend bash -c 'source /venv/bin/activate && python - <<PY
import pathlib
from eddie.digitaltwin.setup_environment import get_database
from eddie.digitaltwin.data_to_db import serve_static_files
with get_database().connect() as conn:
    serve_static_files(conn, pathlib.Path("./src/static/geo"))
print("stores initialised")
PY' 2>&1 | tail -8 | sed 's/^/  /'
    printf '  catalog after:  HTTP %s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 "http://localhost:${P_API}/terria-catalog.json")"
}

# ------------------------------------------------------------------------------------------------ e2e
e2e() {
    local backend="${1:-raster}" res="${2:-8}" product="${3:-dem}"
    cd "$INST" || exit 1
    local stamp out api="http://localhost:${P_API}"
    stamp="$(date +%Y%m%d_%H%M)"; out="/mnt/d/EDDIE/terrain_e2e_${backend}_${res}m_${stamp}"; mkdir -p "$out"
    local body; body=$(jq -nc --arg aoi "$SH12_WKT" --arg b "$backend" --argjson r "$res" --arg p "$product" \
        '{aoi: $aoi, params: {backend: $b, resolution: $r, product: $p}}')
    say "Request: SH12, backend=$backend, resolution=${res} m, product=$product"
    note "results: $out"
    docker compose logs --since 1s -f --no-color terrain_worker > "$out/terrain_worker.log" 2>&1 &
    local logpid=$!
    run_once() {  # -> prints "product_id seconds"
        local t0 tid st pid
        t0=$(date +%s)
        local resp; resp=$(curl -s --max-time 30 -H 'Content-Type: application/json' -d "$body" "$api/terrain/dem")
        tid=$(echo "$resp" | jq -r '.taskId // empty' 2>/dev/null)
        [ -n "$tid" ] || { note "!! POST /terrain/dem failed: ${resp:-no response}"; return 1; }
        note "task $tid"
        while :; do
            st=$(curl -s --max-time 30 "$api/terrain/tasks/$tid")
            case "$(echo "$st" | jq -r .status)" in
                SUCCESS) pid=$(echo "$st" | jq -r .result); break ;;
                FAILURE) note "!! task failed after $(( $(date +%s) - t0 )) s; end of the traceback (all of it: $out/task_error.txt):"
                         task_error "$st" "$out/task_error.txt"; return 1 ;;
            esac
            sleep 15
            printf '  %4ds  %-8s %s\n' $(( $(date +%s) - t0 )) "$(echo "$st" | jq -r .status)" \
                "$(grep -v '^\s*$' "$out/terrain_worker.log" | tail -1 | sed -E 's/^[^|]*\| //' | cut -c1-110)"
        done
        echo "$pid $(( $(date +%s) - t0 ))" > "$out/.last"
    }
    run_once || { kill $logpid 2>/dev/null; exit 1; }
    read -r pid secs < "$out/.last"
    note "product $pid in ${secs} s"
    curl -s "$api/terrain/products/$pid" | tee "$out/product.json" | jq -r '
        "  key \(.generator_key)  resolution \(.resolution)  parent \(.parent_id)",
        (.paths | to_entries[] | "    \(.key): \(.value)")'
    local nc; nc=$(jq -r '.paths.netcdf' "$out/product.json" | sed "s#^/terrain_products#${DATA}/products#")
    say "Product on the host: $nc"
    gdalinfo "NETCDF:\"$nc\":z" 2>&1 | grep -E '^Size is|^Origin|^Pixel Size|NZVD2016|ID\["EPSG"' | head -6 | sed 's/^/  /'
    gdalinfo -stats "NETCDF:\"$nc\":z" 2>/dev/null | grep -E 'STATISTICS_(MINIMUM|MAXIMUM|VALID_PERCENT)' | sed 's/^ */  /'
    local prov="${nc%.nc}.provenance.json"
    [ -f "$prov" ] && jq -r '"  gap fraction: \(.gap_fraction)", "  inputs: \([.inputs[] | (.survey // .name)] | join(", "))",
        "  timings: \(.timings)", "  code: \(.generator_info.code)"' "$prov"
    cp "$prov" "$out/" 2>/dev/null; ls -la "$(dirname "$nc")" > "$out/product_folder.txt"
    say "Repeat the identical request (expect the same product id, in seconds)"
    run_once && { read -r pid2 secs2 < "$out/.last"; note "product $pid2 in ${secs2} s $([ "$pid2" = "$pid" ] && echo '(cache hit)' || echo '!! different product')"; }
    kill $logpid 2>/dev/null
    say "Registry"
    docker compose exec -T db_postgres psql -U postgres -d db -c \
        "select id, product, resolution, left(generator_key,16) as key, parent_id, created_at from terrain_product order by id" 2>&1 | sed 's/^/  /'
    note "worker log during the run: $out/terrain_worker.log ($(grep -ciE 'error|traceback' "$out/terrain_worker.log") error lines)"
}

# ------------------------------------------------------------------------------------------------ task
task_status() {
    local tid="${1:-}"; [ -n "$tid" ] || { note "usage: task <task id>"; exit 2; }
    local st; st=$(curl -s --max-time 30 "http://localhost:${P_API}/terrain/tasks/$tid")
    say "Task $tid: $(echo "$st" | jq -r .status)"
    case "$(echo "$st" | jq -r .status)" in
        SUCCESS) note "result: $(echo "$st" | jq -c .result)" ;;
        FAILURE) note "end of the traceback (all of it: /mnt/d/EDDIE/terrain_task_${tid:0:8}_error.txt):"
                 task_error "$st" "/mnt/d/EDDIE/terrain_task_${tid:0:8}_error.txt" ;;
    esac
    say "Terrain worker log lines for this task"
    (cd "$INST" && docker compose logs --no-color terrain_worker 2>&1 | grep -F -A40 "$tid" | tail -45 | cut -c1-220 | sed 's/^/  /')
}

# ------------------------------------------------------------------------------------------------ test
test_suite() {
    cd "$INST" || exit 1
    say "terrain-s2 tests inside the terrain worker image (source mounted read-only from $TS2)"
    [ $# -gt 0 ] || set -- tests eddie/tests
    docker compose run --rm --no-deps --entrypoint python -v "$TS2":/src:ro -w /src -e PYTHONDONTWRITEBYTECODE=1 \
        terrain_worker -m pytest -q -p no:cacheprovider "$@" 2>&1 | tail -25
}

# ------------------------------------------------------------------------------------------------ otcatalog
otcatalog() {
    local out="/mnt/d/EDDIE/otcatalog_sh12_$(date +%Y%m%d).json"
    say "otCatalog (detail=true) for SH12 -> $out"
    curl -s --max-time 60 -G 'https://portal.opentopography.org/API/otCatalog' \
        --data-urlencode productFormat=PointCloud --data-urlencode minx=173.4603 --data-urlencode miny=-35.4633 \
        --data-urlencode maxx=173.4710 --data-urlencode maxy=-35.4570 --data-urlencode detail=true \
        --data-urlencode outputFormat=json --data-urlencode include_federated=false > "$out"
    jq -r '.Datasets[]?.Dataset | "  \(.alternateName)\t\(.temporalCoverage // "-")\tcreated \(.dateCreated // "-")\tpublished \(.datePublished // "-")"' "$out" \
        || note "!! not JSON - see $out"
}

# ------------------------------------------------------------------------------------------------ core
core() {
    local sha
    case "${1:-}" in v5) sha=$CORE_V5 ;; v4) sha=$CORE_V4 ;; *) note "usage: core v4|v5"; exit 2 ;; esac
    cd "$INST" || exit 1
    git fetch --quiet origin
    git switch --quiet instance && git reset --quiet --hard "$sha" && note "instance -> ${sha:0:9} ($1)"
    set_env EDDIE_REF "$sha"
    note "next: bash ~/eddie/scripts/eddie_step8_terrain.sh setup && bash ~/eddie/scripts/eddie_step8_terrain.sh build && docker compose up -d"
}

case "${1:-setup}" in
    setup) setup ;;
    build) build ;;
    check) check ;;
    bootstrap) bootstrap ;;
    e2e) shift; e2e "$@" ;;
    task) shift; task_status "$@" ;;
    test) shift; test_suite "$@" ;;
    otcatalog) otcatalog ;;
    core) shift; core "$@" ;;
    *) echo "usage: $0 [setup|build|check|bootstrap|e2e [raster|geofabrics] [res] [dem|geofabric]|task <id>|test|otcatalog|core v4|v5]"; exit 2 ;;
esac
