"""EDDIE terrain module (``eddie_terrain``): thin tasks, blueprint, tables and configuration over ``terrain_s2``.

All computing happens in the library, on a dedicated ``terrain`` Celery queue (the terrain worker). The module
runs on the v4 plugin mechanism (``eddie_`` prefix: core ``app.py`` imports ``eddie_terrain.blueprint`` and core
``tasks.py`` imports ``eddie_terrain.tasks``) and already carries the v5 attributes (``blueprint``, ``tasks``,
``processes``, ``Config``; entry point ``eddie.plugins: terrain``).

Nothing heavy is imported here: v4 core imports every plugin's package and ``tasks`` in every container, so the
attributes are resolved lazily (PEP 562) and the tasks import the library only inside their bodies. There is no
start-up hook: nothing crawls or downloads when a worker starts.
"""
from importlib import import_module

PROCESS_DEM = "terrain.dem"
PROCESS_NETWORK = "terrain.network"
QUEUE = "terrain"

_LAZY = {"blueprint": ".blueprint", "tasks": ".tasks", "processes": ".processes", "tables": ".tables",
         "config": ".config"}


def __getattr__(name):
    if name in _LAZY:
        mod = import_module(_LAZY[name], __name__)
        if name == "processes":
            return mod.processes
        return mod
    if name == "Config":
        return import_module(".config", __name__).TerrainConfig
    raise AttributeError(name)


def _version():
    try:
        from importlib.metadata import version
        return version("eddie-terrain")
    except Exception:  # noqa: BLE001 - source checkout
        return "0.0.0+unknown"


__version__ = _version()
