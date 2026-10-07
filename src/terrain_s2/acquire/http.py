"""HTTP with retries, a polite User-Agent and optional snapshotting. One place to mock in tests."""
from __future__ import annotations

import json
import time

USER_AGENT = "terrain-s2/0.1 (EDDIE, University of Canterbury; research use)"


class Http:
    def __init__(self, store=None, retries: int = 3, timeout: float = 60.0, backoff: float = 2.0):
        import requests
        self.s = requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self.store, self.retries, self.timeout, self.backoff = store, retries, timeout, backoff

    def get(self, url: str, params: dict | None = None, source: str | None = None, ext: str = "json",
            method: str = "GET", data: dict | None = None) -> bytes:
        last = None
        for k in range(self.retries):
            try:
                r = (self.s.post(url, data=data, timeout=self.timeout) if method == "POST"
                     else self.s.get(url, params=params, timeout=self.timeout))
                if r.status_code in (429, 500, 502, 503, 504):
                    raise IOError(f"HTTP {r.status_code} from {url}")
                r.raise_for_status()
                if source and self.store is not None:
                    self.store.put(source, r.content, url, params or data, ext)
                return r.content
            except Exception as e:  # noqa: BLE001 - retried, then re-raised
                last = e
                time.sleep(self.backoff * (k + 1))
        raise IOError(f"GET {url} failed after {self.retries} attempts: {last}")

    def json(self, url: str, params: dict | None = None, source: str | None = None, **kw):
        return json.loads(self.get(url, params, source, "json", **kw))
