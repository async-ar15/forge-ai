"""Production queue-depth probe backed by resident vLLM engines (Section 3)."""

from __future__ import annotations

from forgeai.execution.engine_pool import EnginePool


class ProductionQueueDepthProbe:
    """Aggregate ``EnginePool`` scheduler depth for policy feature extraction.

    ``read_queue_depth`` is safe to await concurrently; the pool aggregates
    atomically visible integer depths (best-effort across shards).
    """

    __slots__ = ("_pool",)

    def __init__(self, pool: EnginePool) -> None:
        self._pool = pool

    async def read_queue_depth(self) -> int:
        """Return a non-negative depth across all loaded execution shards."""

        return await self._pool.total_queue_depth()


__all__ = ["ProductionQueueDepthProbe"]
