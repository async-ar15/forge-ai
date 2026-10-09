"""``ProductionQueueDepthProbe`` contract for cold start and empty pools."""

from __future__ import annotations

import pytest
from forgeai.execution.engine_pool import EnginePool
from forgeai.execution.queue_depth_probe import ProductionQueueDepthProbe


@pytest.mark.asyncio
async def test_queue_depth_probe_empty_pool_returns_zero_int() -> None:
    """Feature extractor may run before any model load; depth must be ``0`` int."""

    pool = EnginePool()
    probe = ProductionQueueDepthProbe(pool)
    depth = await probe.read_queue_depth()
    assert depth == 0
    assert type(depth) is int
