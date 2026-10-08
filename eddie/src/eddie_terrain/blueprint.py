"""REST endpoints for terrain (no ``/wps``: it would collide with FReDT's in v4; v5 serves ``processes``).

- ``POST /terrain/dem``: body ``{"aoi": "<WKT, EPSG:4326>", "params": {...}}``; sends ``ensure_dem`` to the
  terrain queue and returns ``202 {"taskId": ...}``. Parameters are validated before sending (400 if not).
- ``GET /terrain/products/<id>``: the registry row (paths, key, resolution, AOI, extent).
- ``GET /terrain/tasks/<id>``: task state, and the product id when finished.

Under v5 the core mounts blueprints below ``/api`` (D9); the routes are relative, so they work there too.
"""
from http.client import ACCEPTED, BAD_REQUEST, NOT_FOUND, OK

from flask import Blueprint, jsonify, make_response, request

blueprint = Blueprint("eddie_terrain", __name__)


def _error(msg, code=BAD_REQUEST):
    return make_response(jsonify({"error": str(msg)}), code)


@blueprint.route("/terrain/dem", methods=["POST"])
def post_dem():
    body = request.get_json(silent=True) or {}
    aoi = body.get("aoi")
    params = body.get("params") or {}
    if not isinstance(aoi, str) or not aoi.strip():
        return _error("'aoi' (WKT in EPSG:4326) is required")
    if not isinstance(params, dict):
        return _error("'params' must be an object")
    try:
        from .tasks import aoi_geometry, request_settings
        s, mode = request_settings(params)
        aoi_geometry(aoi, s, mode)
    except (ValueError, TypeError, KeyError) as e:
        return _error(e)
    from eddie.tasks import app as celery_app
    from . import QUEUE
    task = celery_app.send_task("eddie_terrain.tasks.ensure_dem", args=[aoi, params], queue=QUEUE)
    return make_response(jsonify({"taskId": task.id}), ACCEPTED)


@blueprint.route("/terrain/products/<int:product_id>", methods=["GET"])
def get_product(product_id: int):
    from .tables import get_store
    row = get_store().get(product_id)
    if row is None:
        return _error(f"no terrain product {product_id}", NOT_FOUND)
    return make_response(jsonify(row.to_dict()), OK)


@blueprint.route("/terrain/tasks/<task_id>", methods=["GET"])
def get_task(task_id: str):
    from eddie.tasks import app as celery_app
    r = celery_app.AsyncResult(task_id)
    out = {"taskId": task_id, "status": r.state}
    if r.successful():
        out["result"] = r.result
    elif r.failed():
        out["error"] = str(r.result)
    return make_response(jsonify(out), OK)
