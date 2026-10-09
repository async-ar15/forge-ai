"""Bloom filter service (Redis commands mocked)."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest
from forgeai.retrieval.bloom_filter import BloomFilterService, bloom_item_from_query


@pytest.mark.asyncio
async def test_record_query_calls_bf_add() -> None:
    r = AsyncMock()
    svc = BloomFilterService(r, bloom_key="my-bloom")
    await svc.record_query("  Hello World ")
    item = bloom_item_from_query("  Hello World ")
    r.execute_command.assert_awaited_once_with("BF.ADD", "my-bloom", item)


@pytest.mark.asyncio
async def test_check_query_calls_bf_exists() -> None:
    r = AsyncMock()
    r.execute_command = AsyncMock(return_value=1)
    svc = BloomFilterService(r, bloom_key="k")
    item = bloom_item_from_query("x")
    ok = await svc.check_query("x")
    assert ok is True
    r.execute_command.assert_awaited_with("BF.EXISTS", "k", item)


@pytest.mark.asyncio
async def test_check_query_redis_failure_false(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="forgeai.retrieval.bloom_filter")
    r = AsyncMock()
    r.execute_command = AsyncMock(side_effect=OSError("down"))
    svc = BloomFilterService(r, bloom_key="k")
    out = await svc.check_query("safe")
    assert out is False
    assert "bloom_exists_failed" in caplog.text
