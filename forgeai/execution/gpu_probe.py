"""Production GPU utilization probe via NVML (Section 3 hot-path consumer).

``pynvml`` is imported only inside ``_nvml_init_once`` and ``_sync_read_gpu_fraction``,
so ``import forgeai.execution.gpu_probe`` succeeds when ``nvidia-ml-py`` is absent.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Final

_LOG: Final[logging.Logger] = logging.getLogger(__name__)
_CACHE_TTL_SEC: Final[float] = 0.1
_NVML_READY: bool | None = None


class ProductionGpuLoadProbe:
    """Average utilization across visible GPUs; 100ms read cache.

    ``read_gpu_load`` is safe to await concurrently; each instance shares one
    cache timestamp and value pair (last writer wins — acceptable for gauges).
    """

    __slots__ = ("_cached_load", "_cached_mono", "_nvml_ok")

    def __init__(self) -> None:
        self._cached_load = 0.0
        self._cached_mono = 0.0
        self._nvml_ok = _nvml_init_once()

    async def read_gpu_load(self) -> float:
        """Return fraction in ``[0.0, 1.0]``; degrades to ``0.0`` on NVML errors."""

        now = time.monotonic()
        if now - self._cached_mono < _CACHE_TTL_SEC:
            return self._cached_load
        load = float(await asyncio.to_thread(_sync_read_gpu_fraction, self._nvml_ok))
        self._cached_load = load
        self._cached_mono = now
        return load


def _nvml_init_once() -> bool:
    global _NVML_READY
    if _NVML_READY is not None:
        return _NVML_READY
    try:
        import pynvml

        pynvml.nvmlInit()
        _NVML_READY = True
    except Exception:
        _LOG.warning("gpu_probe_nvml_unavailable using_zero_load", exc_info=True)
        _NVML_READY = False
    return _NVML_READY


def _sync_read_gpu_fraction(nvml_ok: bool) -> float:
    if not nvml_ok:
        return 0.0
    try:
        import pynvml

        n = pynvml.nvmlDeviceGetCount()
        if n == 0:
            return 0.0
        total = 0.0
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            total += float(util.gpu) / 100.0
        return min(1.0, total / float(n))
    except Exception:
        _LOG.warning("gpu_probe_nvml_read_failed using_zero_load", exc_info=True)
        return 0.0


__all__ = ["ProductionGpuLoadProbe"]
