"""``ModelLoader`` registry integration, LRU cache, and precision contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forgeai.enums import ModelTier, Precision
from forgeai.execution.exceptions import (
    PrecisionUnavailableError,
    QuantizationMismatchError,
)
from forgeai.execution.model_loader import ModelLoader
from forgeai.execution.quantization_config import (
    QuantizationConfig,
    QuantizationMethodKind,
)
from forgeai.policy.bandit_actions import ActionSpec
from forgeai.proto import common_pb2, registry_service_pb2, registry_service_pb2_grpc


def _action(tier: ModelTier, prec: Precision) -> ActionSpec:
    from forgeai.enums import OutputBudget, RetrievalMode

    return ActionSpec(
        model_tier=tier,
        precision=prec,
        retrieval_mode=RetrievalMode.OFF,
        output_budget=OutputBudget.SHORT,
    )


def _ok_resp(
    vid: str,
    *,
    artifact: str = "",
    qpid: str | None = None,
) -> registry_service_pb2.RegistryResolveExecutionModelResponse:
    r = registry_service_pb2.RegistryResolveExecutionModelResponse(
        resolution_ok=True,
        model_version_id=vid,
        artifact_object_id=artifact,
    )
    if qpid is not None:
        r.linked_quant_profile_id = qpid
    return r


def _fail_resp(
    *available: common_pb2.Precision.ValueType,
) -> registry_service_pb2.RegistryResolveExecutionModelResponse:
    r = registry_service_pb2.RegistryResolveExecutionModelResponse(
        resolution_ok=False,
        model_version_id="",
        artifact_object_id="",
    )
    for p in available:
        r.available_precisions_for_tier.append(p)
    return r


class _FakeShard:
    def quantization_config(self) -> QuantizationConfig:
        return QuantizationConfig(
            method=QuantizationMethodKind.NONE,
            bits=16,
            loaded_at_utc=datetime.now(UTC),
            memory_mb=128.0,
        )


@pytest.fixture()
def engine_pool() -> MagicMock:
    pool = MagicMock(spec=["load_engine", "unload_engine", "total_queue_depth"])

    async def _load(*_a: object, **_k: object) -> _FakeShard:
        return _FakeShard()

    pool.load_engine = AsyncMock(side_effect=_load)
    pool.unload_engine = AsyncMock()
    pool.total_queue_depth = AsyncMock(return_value=0)
    return pool


@pytest.fixture()
def registry_stub() -> registry_service_pb2_grpc.RegistryServiceStub:
    return MagicMock(spec=registry_service_pb2_grpc.RegistryServiceStub)


@pytest.mark.asyncio
async def test_resolve_model_cache_hit_skips_second_registry_call(
    tmp_path: Path,
    registry_stub: MagicMock,
    engine_pool: MagicMock,
) -> None:
    calls: list[int] = []

    def _resolve(*_a: object, **_k: object) -> object:
        calls.append(1)
        return _ok_resp("v1", artifact="w")

    with patch(
        "forgeai.execution.model_loader.resolve_execution_model_sync",
        side_effect=_resolve,
    ):
        loader = ModelLoader(registry_stub, tmp_path, engine_pool)
        a = _action(ModelTier.SMALL, Precision.FP16)
        (tmp_path / "v1" / "w").mkdir(parents=True)
        h1 = await loader.resolve_model(a)
        h2 = await loader.resolve_model(a)
    assert h1.model_version_id == h2.model_version_id
    assert len(calls) == 1
    assert engine_pool.load_engine.await_count == 1


@pytest.mark.asyncio
async def test_resolve_model_cache_miss_calls_registry_each_key(
    tmp_path: Path,
    registry_stub: MagicMock,
    engine_pool: MagicMock,
) -> None:
    def _resolve(*_a: object, **_k: object) -> object:
        return _ok_resp("v1", artifact="w")

    with patch(
        "forgeai.execution.model_loader.resolve_execution_model_sync",
        side_effect=_resolve,
    ):
        loader = ModelLoader(registry_stub, tmp_path, engine_pool)
        (tmp_path / "v1" / "w").mkdir(parents=True)
        await loader.resolve_model(_action(ModelTier.SMALL, Precision.FP16))
        await loader.resolve_model(_action(ModelTier.MEDIUM, Precision.FP16))
    assert engine_pool.load_engine.await_count == 2


@pytest.mark.asyncio
async def test_precision_unavailable_raises_no_silent_fallback(
    tmp_path: Path,
    registry_stub: MagicMock,
    engine_pool: MagicMock,
) -> None:
    with patch(
        "forgeai.execution.model_loader.resolve_execution_model_sync",
        return_value=_fail_resp(common_pb2.PRECISION_FP16),
    ):
        loader = ModelLoader(registry_stub, tmp_path, engine_pool)
        with pytest.raises(PrecisionUnavailableError) as ei:
            await loader.resolve_model(_action(ModelTier.SMALL, Precision.INT4))
    assert ei.value.requested_precision is Precision.INT4
    assert Precision.FP16 in ei.value.available_precisions


@pytest.mark.asyncio
async def test_lru_eviction_when_max_loaded_exceeded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    registry_stub: MagicMock,
    engine_pool: MagicMock,
) -> None:
    monkeypatch.setenv("MAX_LOADED_MODELS", "2")

    def _resolve(*_a: object, **_k: object) -> object:
        return _ok_resp("v1", artifact="w")

    with patch(
        "forgeai.execution.model_loader.resolve_execution_model_sync",
        side_effect=_resolve,
    ):
        loader = ModelLoader(registry_stub, tmp_path, engine_pool)
        (tmp_path / "v1" / "w").mkdir(parents=True)
        await loader.resolve_model(_action(ModelTier.SMALL, Precision.FP16))
        await loader.resolve_model(_action(ModelTier.MEDIUM, Precision.FP16))
        await loader.resolve_model(_action(ModelTier.LARGE, Precision.FP16))
    assert engine_pool.unload_engine.await_count >= 1


@pytest.mark.asyncio
async def test_evicted_model_reloads_on_next_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    registry_stub: MagicMock,
    engine_pool: MagicMock,
) -> None:
    monkeypatch.setenv("MAX_LOADED_MODELS", "1")

    def _resolve(*_a: object, **_k: object) -> object:
        return _ok_resp("v1", artifact="w")

    with patch(
        "forgeai.execution.model_loader.resolve_execution_model_sync",
        side_effect=_resolve,
    ):
        loader = ModelLoader(registry_stub, tmp_path, engine_pool)
        (tmp_path / "v1" / "w").mkdir(parents=True)
        await loader.resolve_model(_action(ModelTier.SMALL, Precision.FP16))
        await loader.resolve_model(_action(ModelTier.MEDIUM, Precision.FP16))
        before = engine_pool.load_engine.await_count
        await loader.resolve_model(_action(ModelTier.SMALL, Precision.FP16))
    assert engine_pool.load_engine.await_count > before


@pytest.mark.asyncio
async def test_int4_missing_awq_raises_quantization_mismatch(
    tmp_path: Path,
    registry_stub: MagicMock,
    engine_pool: MagicMock,
) -> None:
    with patch(
        "forgeai.execution.model_loader.resolve_execution_model_sync",
        return_value=_ok_resp("v4", artifact="w"),
    ):
        loader = ModelLoader(registry_stub, tmp_path, engine_pool)
        (tmp_path / "v4" / "w").mkdir(parents=True)
        with pytest.raises(QuantizationMismatchError):
            await loader.resolve_model(_action(ModelTier.SMALL, Precision.INT4))
