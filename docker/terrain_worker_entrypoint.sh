#!/bin/bash
# Terrain worker: Celery on queue `terrain` only, with the same health-checker pattern as the core and FReDT
# workers (port 5001 inside the container; no host port).
#
# Pool: threads, one task at a time. GeoFabrics starts a dask LocalCluster with worker *processes* for each stage,
# and a prefork pool child is a daemonic process, which may not have children ("daemonic processes are not allowed
# to have children" -> "Nanny failed to start"; reproduced, DESIGN_NOTES section 10). In a thread of the main
# worker process the cluster starts normally, and the main thread keeps answering health pings during long runs.
# Scale out with more containers, not with concurrency (each GeoFabrics run binds dask's dashboard port).
set -e
source /venv/bin/activate 2>/dev/null || true
NAME="terrain@${HOSTNAME:-worker}"
health-checker --listener 0.0.0.0:5001 --log-level error --script-timeout 20 \
    --script "celery -A eddie.tasks inspect ping -d ${NAME}" &
exec celery -A eddie.tasks worker -Q terrain -n "${NAME}" -P threads \
    --concurrency "${TERRAIN_WORKER_CONCURRENCY:-1}" --prefetch-multiplier 1 --loglevel=INFO
