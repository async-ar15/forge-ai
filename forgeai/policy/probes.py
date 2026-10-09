"""Dependency-injected probes for queue depth and GPU load on the policy hot path.

Does not own: execution gRPC servers, NVML, or cluster autoscaler signals.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class AsyncBloomRedisClient(Protocol):
    """Async Redis subset for Bloom ``BF.EXISTS`` (structurally implemented)."""

    async def execute_command(self, *args: object, **kwargs: object) -> object: ...


@runtime_checkable
class QueueDepthProbe(Protocol):
    """Reads execution-queue depth observed at decision time (Section 3)."""

    async def read_queue_depth(self) -> int:
        """Return a non-negative queue depth snapshot."""


@runtime_checkable
class GpuLoadProbe(Protocol):
    """Reads normalized GPU utilization in ``[0.0, 1.0]`` (Section 3)."""

    async def read_gpu_load(self) -> float:
        """Return GPU load fraction in the closed unit interval."""


class LocalQueueDepthProbe:
    """Test double: in-memory queue depth controlled by the test harness."""

    def __init__(self, initial_depth: int = 0) -> None:
        self._depth = initial_depth

    def set_depth(self, value: int) -> None:
        """Replace the depth returned by subsequent probe reads."""

        self._depth = value

    async def read_queue_depth(self) -> int:
        """Return the configured depth."""

        return self._depth


class LocalGpuLoadProbe:
    """Test double: static GPU load for unit tests."""

    def __init__(self, load: float = 0.0) -> None:
        self._load = load

    def set_load(self, value: float) -> None:
        """Replace the load returned by subsequent probe reads."""

        self._load = value

    async def read_gpu_load(self) -> float:
        """Return the configured load."""

        return self._load


__all__ = [
    "AsyncBloomRedisClient",
    "GpuLoadProbe",
    "LocalGpuLoadProbe",
    "LocalQueueDepthProbe",
    "QueueDepthProbe",
]
