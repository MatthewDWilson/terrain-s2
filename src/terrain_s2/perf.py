"""Stage timings and peak memory for a run (for profiling and for sizing machines).

    T = StageTimer()          # starts sampling resident memory (psutil, if installed) every 0.25 s
    ...; T.mark("read")       # time since the previous mark, and peak RSS so far
    T.report()                # {stage: seconds}, peak_rss_gb, cpu count; T.stop()
"""
from __future__ import annotations

import os
import threading
import time


class StageTimer:
    def __init__(self, interval: float = 0.25, sample: bool = True):
        """``sample=False``: no sampling thread (under cProfile, whose thread accounting it distorts);
        memory is then read at each mark only."""
        self.t0 = self.last = time.perf_counter()
        self.stages, self.rss_at = {}, {}
        self.peak = 0
        self._stop = threading.Event()
        try:
            import psutil
            self._proc = psutil.Process(os.getpid())
        except ImportError:                                   # timings still work
            self._proc = None
        if self._proc:
            self.peak = self._proc.memory_info().rss
        if self._proc and sample:
            self._thread = threading.Thread(target=self._sample, args=(interval,), daemon=True)
            self._thread.start()

    def _sample(self, interval):
        while not self._stop.wait(interval):
            try:
                self.peak = max(self.peak, self._proc.memory_info().rss)
            except Exception:  # noqa: BLE001
                return

    def mark(self, name: str, extra: dict | None = None):
        now = time.perf_counter()
        self.stages[name] = self.stages.get(name, 0.0) + now - self.last
        self.last = now
        if self._proc:
            self.peak = max(self.peak, self._proc.memory_info().rss)
            self.rss_at[name] = round(self.peak / 2 ** 30, 2)

    def add(self, timings: dict, prefix: str = ""):
        """Merge timings measured elsewhere (e.g. pipeline.Result.timings) without moving the clock."""
        for k, v in timings.items():
            if k != "total":
                self.stages[prefix + k] = round(v, 2)

    def stop(self):
        self._stop.set()

    def report(self) -> dict:
        return dict(stages={k: round(v, 2) for k, v in self.stages.items()},
                    total_s=round(time.perf_counter() - self.t0, 1),
                    peak_rss_gb=round(self.peak / 2 ** 30, 2) if self._proc else None,
                    peak_rss_gb_after=self.rss_at or None, cpu_count=os.cpu_count())
