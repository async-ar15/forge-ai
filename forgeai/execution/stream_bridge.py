"""Bridge asyncio execution pipelines to sync gRPC streaming handlers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TypeVar

T = TypeVar("T")
_LOG = logging.getLogger(__name__)


def sync_iterate_async_generator(
    factory: Callable[[], AsyncIterator[T]],
) -> Iterator[T]:
    """Drive ``factory()`` to completion on a dedicated event loop (per RPC)."""

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    agen = factory()
    try:
        while True:
            try:
                yield loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                break
    finally:
        aclose = getattr(agen, "aclose", None)
        if callable(aclose):
            try:
                loop.run_until_complete(aclose())
            except Exception:
                _LOG.debug("async_generator_aclose_failed", exc_info=True)
        loop.close()


__all__ = ["sync_iterate_async_generator"]
