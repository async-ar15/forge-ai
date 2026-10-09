"""Registry-backed model resolution with LRU GPU cache (Section 2/5)."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Final

from forgeai.enums import Precision
from forgeai.execution.artifact_awq import require_awq_layout
from forgeai.execution.constants import ExecutionConfigKey
from forgeai.execution.engine_pool import EngineKey, EnginePool
from forgeai.execution.exceptions import PrecisionUnavailableError
from forgeai.execution.handles import ModelHandle
from forgeai.execution.model_loader_paths import local_artifact_directory
from forgeai.execution.registry_resolve import resolve_execution_model_sync
from forgeai.policy.bandit_actions import ActionSpec, action_spec_to_proto
from forgeai.proto import common_pb2, registry_service_pb2_grpc

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

_PROTO_PRECISION_TO_DOMAIN: Final[dict[int, Precision]] = {
    int(common_pb2.PRECISION_FP16): Precision.FP16,
    int(common_pb2.PRECISION_INT8): Precision.INT8,
    int(common_pb2.PRECISION_INT4): Precision.INT4,
}


def _serving_env() -> common_pb2.ServingEnvironment.ValueType:
    raw = os.environ.get(
        ExecutionConfigKey.SERVING_ENVIRONMENT.value,
        "production",
    ).lower()
    if raw == "staging":
        return common_pb2.SERVING_ENVIRONMENT_STAGING
    return common_pb2.SERVING_ENVIRONMENT_PRODUCTION


def _available_precisions(resp: object) -> tuple[Precision, ...]:
    seq = getattr(resp, "available_precisions_for_tier", [])
    out: list[Precision] = []
    for p in seq:
        dom = _PROTO_PRECISION_TO_DOMAIN.get(int(p))
        if dom is not None:
            out.append(dom)
    return tuple(out)


def _update_quant_memory_sync(
    stub: registry_service_pb2_grpc.RegistryServiceStub,
    profile_id: str,
    memory_mb: float,
) -> None:
    from forgeai.proto import registry_service_pb2

    req = registry_service_pb2.RegistryUpdateQuantProfileRequest(
        quant_profile_id=profile_id,
        model_memory_footprint_mb=int(round(memory_mb)),
    )
    stub.RegistryUpdateQuantProfile(req, timeout=10.0)


class ModelLoader:
    """Resolve ``ActionSpec`` to a loaded shard; LRU-evict when over budget.

    ``resolve_model`` awaits registry I/O on a thread pool and serialized cache
    mutations under ``asyncio.Lock`` — safe for concurrent RPCs targeting
    different keys; identical keys are serialized to avoid duplicate loads.
    """

    __slots__ = (
        "_cache",
        "_cache_dir",
        "_engine_pool",
        "_lock",
        "_max_loaded",
        "_registry",
    )

    def __init__(
        self,
        registry_stub: registry_service_pb2_grpc.RegistryServiceStub,
        cache_dir: str | Path,
        engine_pool: EnginePool,
    ) -> None:
        self._registry = registry_stub
        self._cache_dir = Path(cache_dir)
        self._engine_pool = engine_pool
        raw = os.environ.get(ExecutionConfigKey.MAX_LOADED_MODELS.value, "3")
        self._max_loaded = max(1, int(raw))
        self._cache: OrderedDict[EngineKey, ModelHandle] = OrderedDict()
        self._lock = asyncio.Lock()

    async def resolve_model(self, action: ActionSpec) -> ModelHandle:
        """Load or reuse a model for ``action``; never silently downgrades precision."""

        key: EngineKey = (action.model_tier, action.precision)
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            while len(self._cache) >= self._max_loaded:
                evict_key, handle = self._cache.popitem(last=False)
                await self._engine_pool.unload_engine(evict_key)
                _LOG.info(
                    "execution_model_evicted tier=%s precision=%s "
                    "reason=lru_cache_full version=%s",
                    evict_key[0].value,
                    evict_key[1].value,
                    handle.model_version_id,
                )
            handle = await self._load_new(key, action)
            self._cache[key] = handle
            self._cache.move_to_end(key)
            return handle

    async def _load_new(self, key: EngineKey, action: ActionSpec) -> ModelHandle:
        wire = action_spec_to_proto(action)
        resp = await asyncio.to_thread(
            resolve_execution_model_sync,
            self._registry,
            model_tier=wire.model_tier,
            execution_precision=wire.precision,
            serving_environment=_serving_env(),
        )
        if not resp.resolution_ok:
            raise PrecisionUnavailableError(
                model_tier=action.model_tier,
                requested_precision=action.precision,
                available_precisions=_available_precisions(resp),
            )
        artifact_path = local_artifact_directory(
            self._cache_dir,
            resp.model_version_id,
            resp.artifact_object_id,
        )
        if action.precision is Precision.INT4:
            await asyncio.to_thread(require_awq_layout, artifact_path)
        shard = await self._engine_pool.load_engine(
            key,
            artifact_path,
            action.precision,
            model_version_id=resp.model_version_id,
        )
        qconf = shard.quantization_config()
        qpid = (
            resp.linked_quant_profile_id
            if resp.HasField("linked_quant_profile_id")
            else None
        )
        if qpid and qconf.memory_mb > 0.0:
            await asyncio.to_thread(
                _update_quant_memory_sync,
                self._registry,
                qpid,
                qconf.memory_mb,
            )
        return ModelHandle(
            model_version_id=resp.model_version_id,
            artifact_path=artifact_path,
            quant_profile_id=qpid,
            quant_config=qconf,
            loaded_engine=shard,
        )


__all__ = ["ModelLoader"]
