"""Environment for importing the EDDIE core (v4.0.0 or v5-integration) in tests: the core reads its settings
at import. Skips these tests when the core is not installed."""
import os

import pytest

CORE_ENV = {"STATSNZ_API_KEY": "x", "LINZ_API_KEY": "x", "MFE_API_KEY": "x", "DATA_DIR_GEOSERVER": "/tmp/geoserver",
            "POSTGRES_PASSWORD": "x", "MESSAGE_BROKER_HOST": "localhost"}


def pytest_configure(config):
    for k, v in CORE_ENV.items():
        os.environ.setdefault(k, v)
    # PyWPS builds a ConfigParser from os.environ and fails on names that differ only in case
    # (e.g. NO_PROXY and no_proxy, as some sandboxes set): keep the lower-case one.
    seen = {}
    for k in list(os.environ):
        lk = k.lower()
        if lk in seen and k != lk:
            os.environ.pop(k)
        elif lk in seen:
            os.environ.pop(seen[lk])
        seen[lk] = k


def pytest_collection_modifyitems(config, items):
    try:
        import eddie  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(reason="EDDIE core (eddie) not installed")
        for it in items:
            if "eddie/tests" in str(it.fspath).replace("\\", "/"):
                it.add_marker(skip)
