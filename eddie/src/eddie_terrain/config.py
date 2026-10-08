"""``TerrainConfig``: the ``TERRAIN_*`` settings as an EDDIE ``EnvVariable`` subclass (v5 ``Config`` attribute).

The values are read by :func:`terrain_s2.settings.from_env`; this class exposes them in EDDIE's configuration
style and documents them for deployments. Defaults are the library's (see ``terrain_s2.settings``).
"""
from eddie.config import EnvVariable


class TerrainConfig(EnvVariable):  # pylint: disable=too-few-public-methods
    """Terrain settings. ``settings()`` returns the library's validated :class:`terrain_s2.settings.Settings`."""

    TERRAIN_BACKEND = EnvVariable._get_env_variable("TERRAIN_BACKEND", default="raster")
    TERRAIN_RESOLUTION = int(EnvVariable._get_env_variable("TERRAIN_RESOLUTION", default="8"))
    TERRAIN_PRODUCT = EnvVariable._get_env_variable("TERRAIN_PRODUCT", default="dem")
    TERRAIN_LAND_SOURCE = EnvVariable._get_env_variable("TERRAIN_LAND_SOURCE", default="coverage")
    TERRAIN_LIDAR_CLASSES = EnvVariable._get_env_variable("TERRAIN_LIDAR_CLASSES", default="2")
    TERRAIN_BUFFER_M = float(EnvVariable._get_env_variable("TERRAIN_BUFFER_M", default="10"))
    TERRAIN_SPATIAL_REUSE = EnvVariable._get_bool_env_variable("TERRAIN_SPATIAL_REUSE", default=False)
    TERRAIN_PROFILE = EnvVariable._get_env_variable("TERRAIN_PROFILE", default="nz")
    TERRAIN_DATA_DIR = EnvVariable._get_env_variable("TERRAIN_DATA_DIR", default="/terrain_data")
    TERRAIN_PRODUCT_DIR = EnvVariable._get_env_variable("TERRAIN_PRODUCT_DIR", default="/stored_data/terrain")
    TERRAIN_AOI_MODE = EnvVariable._get_env_variable("TERRAIN_AOI_MODE", default="bbox")

    @staticmethod
    def settings(**overrides):
        """Validated library settings: ``TERRAIN_*`` from the environment, with the deployment defaults above for
        the data and product folders, then ``overrides`` (request parameters)."""
        import os

        from terrain_s2.settings import from_env
        base = {}
        if not os.environ.get("TERRAIN_DATA_DIR"):
            base["data_dir"] = TerrainConfig.TERRAIN_DATA_DIR
        if not os.environ.get("TERRAIN_PRODUCT_DIR"):
            base["product_dir"] = TerrainConfig.TERRAIN_PRODUCT_DIR
        base.update(overrides)
        from pathlib import Path
        for k in ("data_dir", "product_dir"):
            if k in base and base[k] is not None:
                base[k] = Path(base[k])
        return from_env(**base)
