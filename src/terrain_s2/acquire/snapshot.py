"""Snapshot store: every fetched payload kept with its provenance, keyed by content hash."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path


class SnapshotStore:
    """``root/<source>/<sha256[:16]>.<ext>`` plus ``root/manifest.jsonl`` (one line per fetch)."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, source: str, content: bytes, url: str, params: dict | None = None, ext: str = "json",
            extra: dict | None = None) -> dict:
        h = hashlib.sha256(content).hexdigest()
        d = self.root / _safe(source)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{h[:16]}.{ext}"
        if not path.exists():
            path.write_bytes(content)
        rec = dict(source=source, url=url, params=params or {}, sha256=h, bytes=len(content),
                   path=str(path.relative_to(self.root)),
                   retrieved=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), **(extra or {}))
        with open(self.root / "manifest.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def path(self, rec: dict) -> Path:
        return self.root / rec["path"]


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
