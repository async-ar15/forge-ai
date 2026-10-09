"""Process-wide vLLM engine shards keyed by ``(model_tier, precision)``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from forgeai.enums import ModelTier, Precision
from forgeai.execution.vllm_shard import VllmEngineShard

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

EngineKey = tuple[ModelTier, Precision]


class EnginePool:
    """Owns at most one live shard per key; ``ModelLoader`` decides eviction.

    Concurrent ``load`` calls for the same key must be serialized externally
    (``ModelLoader`` holds a lock). Loads for different keys may run in parallel.
    """

    __slots__ = ("_shards",)

    def __init__(self) -> None:
        self._shards: dict[EngineKey, VllmEngineShard] = {}

    async def load_engine(
        self,
        key: EngineKey,
        artifact_dir: Path,
        precision: Precision,
        *,
        model_version_id: str,
    ) -> VllmEngineShard:
        """Materialize a shard or return the existing one for ``key``."""

        existing = self._shards.get(key)
        if existing is not None:
            return existing
        shard = await VllmEngineShard.create(
            artifact_dir,
            precision,
            model_version_id=model_version_id,
        )
        self._shards[key] = shard
        return shard

    async def unload_engine(self, key: EngineKey) -> None:
        """Drop GPU memory for ``key`` if present."""

        shard = self._shards.pop(key, None)
        if shard is None:
            return
        await shard.shutdown()
        _LOG.info(
            "vllm_engine_unloaded tier=%s precision=%s",
            key[0].value,
            key[1].value,
        )

    def get_engine(self, key: EngineKey) -> VllmEngineShard | None:
        """Return the live shard when loaded."""

        return self._shards.get(key)

    async def total_queue_depth(self) -> int:
        """Sum queue depth across shards; ``0`` when empty or on per-shard errors.

        Uses a snapshot of values so concurrent ``load_engine`` / ``unload_engine``
        calls on the same event loop cannot trigger dict-size-during-iteration.
        Safe for cold start (no engines loaded). Never returns ``None``.
        """

        total = 0
        for shard in tuple(self._shards.values()):
            try:
                total += int(shard.queue_depth())
            except Exception:
                _LOG.debug("engine_pool_queue_depth_shard_failed", exc_info=True)
        return max(0, total)


__all__ = ["EngineKey", "EnginePool"]
