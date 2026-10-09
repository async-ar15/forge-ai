"""``ExecutionService.Execute`` streaming, billing, and failure surfaces."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from forgeai.execution.elephant_broker import (
    ElephantBrokerClient,
    GroundedPrompt,
    MemoryContext,
)
from forgeai.execution.grpc_servicer import ForgeExecutionServicer
from forgeai.execution.handles import ModelHandle
from forgeai.execution.metrics import EXECUTION_MODEL_LOAD_ERRORS_TOTAL
from forgeai.execution.quantization_config import (
    QuantizationConfig,
    QuantizationMethodKind,
)
from forgeai.proto import common_pb2, execution_service_pb2


def _req() -> execution_service_pb2.ExecutionExecuteRequest:
    spec = common_pb2.ActionSpec(
        model_tier=common_pb2.MODEL_TIER_SMALL,
        precision=common_pb2.PRECISION_FP16,
        retrieval_mode=common_pb2.RETRIEVAL_MODE_OFF,
        output_budget=common_pb2.OUTPUT_BUDGET_SHORT,
    )
    return execution_service_pb2.ExecutionExecuteRequest(
        correlation_request_id=str(uuid.uuid4()),
        tenant_id=str(uuid.uuid4()),
        routing_action_spec=spec,
        user_prompt_utf8="hello",
        max_output_token_limit=256,
    )


class _StreamShard:
    async def stream_text(
        self,
        _prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        request_id: str,
    ) -> AsyncIterator[str]:
        yield "hel"
        yield "lo"

    def quantization_config(self) -> QuantizationConfig:
        return QuantizationConfig(
            method=QuantizationMethodKind.NONE,
            bits=16,
            loaded_at_utc=datetime.now(UTC),
            memory_mb=64.0,
        )


def _handle() -> ModelHandle:
    return ModelHandle(
        model_version_id="mv",
        artifact_path=Path("/tmp/x"),
        quant_profile_id=None,
        quant_config=_StreamShard().quantization_config(),
        loaded_engine=_StreamShard(),
    )


def test_streaming_tokens_and_final_cost() -> None:
    loader = MagicMock()
    loader.resolve_model = AsyncMock(return_value=_handle())
    eb = ElephantBrokerClient()
    svc = ForgeExecutionServicer(loader, eb)
    chunks = list(svc.Execute(_req(), None))
    texts = [c.generated_text_delta_utf8 for c in chunks if not c.stream_finished]
    assert "".join(texts) == "hello"
    final = chunks[-1]
    assert final.stream_finished is True
    assert final.HasField("stream_final_cost_microdollars")
    assert final.stream_final_cost_microdollars.amount_microdollars >= 0


def test_model_load_failure_yields_error_chunk_not_exception() -> None:
    loader = MagicMock()
    loader.resolve_model = AsyncMock(side_effect=RuntimeError("boom"))
    eb = ElephantBrokerClient()
    svc = ForgeExecutionServicer(loader, eb)
    before = EXECUTION_MODEL_LOAD_ERRORS_TOTAL._value.get()
    chunks = list(svc.Execute(_req(), None))
    after = EXECUTION_MODEL_LOAD_ERRORS_TOTAL._value.get()
    assert after == before + 1
    assert len(chunks) == 1
    assert chunks[0].stream_finished is True
    assert "model_load_failed" in chunks[0].stream_error_message_utf8


def test_elephant_broker_call_order_memory_then_ground() -> None:
    order: list[str] = []
    eb = MagicMock()

    def _mem(tenant_id: str, session_id: str) -> MemoryContext:
        order.append("memory")
        return MemoryContext(snippets_utf8=("ctx",))

    def _ground(prompt: str, ctx: object) -> GroundedPrompt:
        order.append("ground")
        assert tuple(ctx) == ("ctx",)
        return GroundedPrompt(text_utf8=prompt + "!")

    eb.get_memory_context.side_effect = _mem
    eb.ground_prompt.side_effect = _ground

    loader = MagicMock()
    loader.resolve_model = AsyncMock(return_value=_handle())
    svc = ForgeExecutionServicer(loader, eb)
    list(svc.Execute(_req(), None))
    assert order == ["memory", "ground"]


def test_generation_failure_sets_fallback_hint() -> None:
    class _BadShard(_StreamShard):
        async def stream_text(
            self,
            *_a: object,
            **_k: object,
        ) -> AsyncIterator[str]:
            if False:
                yield ""
            raise RuntimeError("vllm stopped")

    h = ModelHandle(
        model_version_id="mv",
        artifact_path=__import__("pathlib").Path("/tmp/x"),
        quant_profile_id=None,
        quant_config=_BadShard().quantization_config(),
        loaded_engine=_BadShard(),
    )
    loader = MagicMock()
    loader.resolve_model = AsyncMock(return_value=h)
    svc = ForgeExecutionServicer(loader, ElephantBrokerClient())
    chunks = list(svc.Execute(_req(), None))
    assert len(chunks) == 1
    assert chunks[0].gateway_fallback_used_hint is True


def test_mid_stream_error_message_present() -> None:
    class _BadShard(_StreamShard):
        async def stream_text(
            self,
            *_a: object,
            **_k: object,
        ) -> AsyncIterator[str]:
            yield "x"
            raise RuntimeError("mid")

    h = ModelHandle(
        model_version_id="mv",
        artifact_path=__import__("pathlib").Path("/tmp/x"),
        quant_profile_id=None,
        quant_config=_BadShard().quantization_config(),
        loaded_engine=_BadShard(),
    )
    loader = MagicMock()
    loader.resolve_model = AsyncMock(return_value=h)
    svc = ForgeExecutionServicer(loader, ElephantBrokerClient())
    chunks = list(svc.Execute(_req(), None))
    assert any("generation_failed" in c.stream_error_message_utf8 for c in chunks)


def test_gpu_probe_mocked_nvml_returns_zero_without_crash() -> None:
    with patch("forgeai.execution.gpu_probe._sync_read_gpu_fraction", return_value=0.0):
        from forgeai.execution.gpu_probe import ProductionGpuLoadProbe

        async def _run() -> float:
            p = ProductionGpuLoadProbe()
            return await p.read_gpu_load()

        import asyncio

        assert asyncio.run(_run()) == 0.0
