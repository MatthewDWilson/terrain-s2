# Terrain worker (integration plan 9.2): a Celery worker on queue `terrain` with GeoFabrics 1.1.30, Stage 2 and
# the EDDIE core at the deployment's tag. Build from the terrain-s2 repository root:
#
#   docker build -f docker/terrain-worker.Dockerfile \
#     --build-arg EDDIE_REF=v4.0.0 --build-arg TERRAIN_VERSION=$(git describe --tags --always) \
#     --build-arg TERRAIN_COMMIT=$(git rev-parse HEAD) \
#     -t eddie-terrain-worker .
#
# EDDIE_REF: a core tag or commit (v4.0.0 for FReDT today; a v5 tag later). TERRAIN_VERSION feeds setuptools-scm,
# and TERRAIN_COMMIT (required) give the version and commit recorded in every generator key, since the build
# context has no .git.

FROM mambaorg/micromamba:2.3.3 AS build
ARG EDDIE_REF=v4.0.0
ARG TERRAIN_VERSION
ARG TERRAIN_COMMIT
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY environment-worker.yml .
RUN micromamba create -y -n worker -f environment-worker.yml && micromamba clean -a -y
SHELL ["micromamba", "run", "-n", "worker", "/bin/bash", "-c"]
COPY pyproject.toml README.md ./
COPY src src
COPY eddie eddie
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TERRAIN_S2=${TERRAIN_VERSION} \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_EDDIE_TERRAIN=${TERRAIN_VERSION}
RUN <<BUILD
    set -euo pipefail
    # The generator key includes the code version and commit (plan 6.5); the context has no .git, so both are
    # build arguments and must be given.
    test -n "${TERRAIN_VERSION:-}" || { echo "--build-arg TERRAIN_VERSION=\$(git describe --tags --always) is required"; exit 1; }
    test -n "${TERRAIN_COMMIT:-}" || { echo "--build-arg TERRAIN_COMMIT=\$(git rev-parse HEAD) is required"; exit 1; }
    echo "${TERRAIN_COMMIT}" > src/terrain_s2/_commit.txt
    pip install --no-deps "geofabrics==1.1.30" "osmpythontools>=0.3.5"
    pip install --no-deps "eddie @ git+https://github.com/GeospatialResearch/Digital-Twins@${EDDIE_REF}"
    pip install --no-deps . ./eddie
    # Build-time checks: the pin, and that the library, GeoFabrics and the module import together
    python -c "import importlib.metadata as m; v = m.version('geofabrics'); print('geofabrics', v); assert v == '1.1.30', v"
    python -c "import pdal, rasterio, terrain_s2, eddie_terrain; from geofabrics import runner; print('terrain_s2', terrain_s2.__version__, 'GDAL', rasterio.__gdal_version__)"
    # the worker command imports the core's tasks module, which discovers this module (dummy settings: the core
    # reads its configuration at import)
    STATSNZ_API_KEY=x LINZ_API_KEY=x MFE_API_KEY=x DATA_DIR_GEOSERVER=/tmp POSTGRES_PASSWORD=x \
        python -c "import eddie.tasks as t; n = [k for k in t.app.tasks if k.startswith('eddie_terrain')]; print(n); assert len(n) == 3"
    conda-pack -n worker -o /tmp/env.tar --ignore-missing-files
    mkdir /venv && tar xf /tmp/env.tar -C /venv && rm /tmp/env.tar
    /venv/bin/conda-unpack
BUILD

FROM debian:bookworm-slim AS runtime
ADD --chmod=555 https://github.com/gruntwork-io/health-checker/releases/download/v0.0.8/health-checker_linux_amd64 \
    /usr/local/bin/health-checker
RUN <<SETUP
    set -e
    apt-get update
    apt-get install -y --no-install-recommends ca-certificates curl acl
    rm -rf /var/lib/apt/lists/*
    useradd --create-home --uid 1000 nonroot
    for d in /stored_data /stored_data/terrain /terrain_data; do
        mkdir -p "$d"; setfacl -R -d -m u:nonroot:rwx "$d"; setfacl -R -m u:nonroot:rwx "$d"
    done
SETUP
COPY --from=build /venv /venv
COPY --chmod=555 docker/terrain_worker_entrypoint.sh /app/terrain_worker_entrypoint.sh
WORKDIR /app
ENV PATH=/venv/bin:$PATH \
    TERRAIN_DATA_DIR=/terrain_data \
    TERRAIN_PRODUCT_DIR=/stored_data/terrain \
    MPLBACKEND=Agg
USER nonroot
ENTRYPOINT ["/app/terrain_worker_entrypoint.sh"]
