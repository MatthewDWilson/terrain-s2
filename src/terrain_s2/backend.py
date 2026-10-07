"""Array backend selection: NumPy/SciPy on CPU, CuPy/cupyx on GPU.

Feature code is written once against ``be.xp`` (array module) and ``be.ndi``
(ndimage module). cupyx.scipy.ndimage mirrors scipy.ndimage for every filter
used here, so the same code runs on either device. Hydrological steps are
sequential (priority queues) and always run on CPU with Numba.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

import numpy as np


@dataclass(frozen=True)
class Backend:
    name: str          # "cpu" or "gpu"
    xp: ModuleType     # numpy or cupy
    ndi: ModuleType    # scipy.ndimage or cupyx.scipy.ndimage

    def asarray(self, a, dtype=None):
        return self.xp.asarray(a, dtype=dtype)

    def to_numpy(self, a) -> np.ndarray:
        if self.name == "gpu":
            return self.xp.asnumpy(a)
        return np.asarray(a)

    def sync(self):
        if self.name == "gpu":
            self.xp.cuda.Stream.null.synchronize()


def _cpu() -> Backend:
    from scipy import ndimage
    return Backend("cpu", np, ndimage)


def _gpu() -> Backend:
    import cupy
    from cupyx.scipy import ndimage
    cupy.cuda.runtime.getDeviceCount()  # raises if no usable device
    return Backend("gpu", cupy, ndimage)


def get_backend(device: str = "auto") -> Backend:
    """Return a backend. ``device`` is "cpu", "gpu" or "auto" (GPU if usable)."""
    device = device.lower()
    if device == "cpu":
        return _cpu()
    if device == "gpu":
        return _gpu()
    if device == "auto":
        try:
            return _gpu()
        except Exception:
            return _cpu()
    raise ValueError(f"unknown device {device!r}")
