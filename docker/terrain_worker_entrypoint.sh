#!/bin/bash
# Terrain worker: Celery on queue `terrain` only, prefork (GeoFabrics starts a dask cluster per task), with the
# same health-checker pattern as the core and FReDT workers (port 5001 inside the container; no host port).
set -e
source /venv/bin/activate 2>/dev/null || true
NAME="terrain@${HOSTNAME:-worker}"
health-checker --listener 0.0.0.0:5001 --log-level error --script-timeout 20 \
    --script "celery -A eddie.tasks inspect ping -d ${NAME}" &
exec celery -A eddie.tasks worker -Q terrain -n "${NAME}" -P prefork \
    --concurrency "${TERRAIN_WORKER_CONCURRENCY:-1}" --max-tasks-per-child "${TERRAIN_WORKER_MAX_TASKS:-10}" \
    --loglevel=INFO
