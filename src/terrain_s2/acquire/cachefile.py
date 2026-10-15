"""JSON cache files that parallel workers can share.

Wave 2 (15 Oct 2026): seven tiles failed every retry with "Expecting value: line 1 column 1 (char 0)", an empty or
half-written cache file (two workers writing the same STAC cache entry, or a worker stopped mid-write) that every
later read trusted. Writes now go to a temporary file in the same folder and are renamed into place; a cache file
that does not parse is removed and treated as missing, so it is fetched again.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path


def read(path):
    """The cached object, or None if the file is missing or unreadable (an unreadable file is removed)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        try:
            path.unlink()
        except OSError:
            pass
        return None


def write(path, obj):
    """Write ``obj`` as JSON atomically (a unique temporary file, then a rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(obj))
    try:
        os.replace(tmp, path)
    except PermissionError:                     # Windows: another worker holds it; theirs is as good
        tmp.unlink(missing_ok=True)
