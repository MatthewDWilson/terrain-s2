"""Machine-learning chain (DESIGN_NOTES sections 8 and 11): ML-0 dataset builder (``select``, ``batch``,
``labels``), then the candidate scorer (ML-1)."""


def code_version() -> str:
    """The code as it is now: ``git describe`` of the checkout when there is one (an editable install's
    recorded version is fixed at install time, e.g. 0.1.1.dev16+ga6a32dcb3 for every wave-1 tile), else the
    package version."""
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    if (root / ".git").exists():
        try:
            r = subprocess.run(["git", "-C", str(root), "describe", "--tags", "--always", "--dirty"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:  # noqa: BLE001
            pass
    from .. import __version__
    return str(__version__)
