"""EDDIE terrain library: Stage 1 DEMs and Stage 2 detection of crossings, channels and embankments.

Research prototype (Phase 2a). Package name is a working name pending TR-5.

The version comes from git tags through setuptools-scm (S2-14). An installed package carries it in
``_version.py``; a source checkout run without installing (tests and scripts use ``src/`` directly)
asks git, and falls back to ``0.0.0+unknown``.
"""


def _version() -> str:
    try:
        from ._version import version
        return version
    except ImportError:
        pass
    try:
        from importlib.metadata import PackageNotFoundError, version as _v
        try:
            return _v("terrain-s2")
        except PackageNotFoundError:
            pass
    except ImportError:  # pragma: no cover
        pass
    try:
        from pathlib import Path

        from setuptools_scm import get_version
        return get_version(root=str(Path(__file__).resolve().parents[2]), fallback_version="0.0.0+unknown")
    except Exception:  # noqa: BLE001 - setuptools-scm absent or not a git checkout
        return "0.0.0+unknown"


__version__ = _version()


def git_commit() -> str | None:
    """The git commit of a source checkout, or None (installed wheels record it in the version)."""
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None
