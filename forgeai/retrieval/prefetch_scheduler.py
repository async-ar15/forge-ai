"""Thread-backed prefetch jobs with a hard concurrency ceiling."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable, Coroutine
from typing import Final

from forgeai.observability.metrics import RETRIEVAL_PREFETCH_ERRORS_TOTAL
from forgeai.retrieval.constants import PREFETCH_MAX_CONCURRENT

_LOG: Final[logging.Logger] = logging.getLogger(__name__)
# v1 uses daemon threads + semaphore for prefetch scheduling. In-flight prefetch
# tasks are not drained on shutdown — they are abandoned when the process exits.
# This is acceptable in v1 because prefetch failures never affect responses. v2
# should replace this with an asyncio TaskGroup with explicit cancellation and
# drain in the service lifespan handler.
_sem: Final[threading.BoundedSemaphore] = threading.BoundedSemaphore(
    PREFETCH_MAX_CONCURRENT,
)


def schedule_prefetch(
    coro_factory: Callable[[], Coroutine[None, None, None]],
) -> None:
    """Fire-and-forget prefetch; drops when ``PREFETCH_MAX_CONCURRENT`` busy."""

    acquired = _sem.acquire(blocking=False)
    if not acquired:
        _LOG.debug("prefetch_dropped reason=budget_exceeded")
        return

    def _runner() -> None:
        try:
            asyncio.run(coro_factory())
        except Exception:
            RETRIEVAL_PREFETCH_ERRORS_TOTAL.inc()
            _LOG.warning("prefetch_task_failed", exc_info=True)
        finally:
            _sem.release()

    threading.Thread(target=_runner, daemon=True).start()


__all__ = ["schedule_prefetch"]
