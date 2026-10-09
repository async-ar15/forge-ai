"""Sync gRPC ``ExecutionService`` over an asyncio execution pipeline."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator
from typing import Final

from forgeai.execution.elephant_broker import ElephantBrokerClient
from forgeai.execution.execute_pipeline import async_execute_stream
from forgeai.execution.model_loader import ModelLoader
from forgeai.execution.stream_bridge import sync_iterate_async_generator
from forgeai.observability.metrics import (
    EXECUTION_REQUEST_DURATION_SECONDS,
    EXECUTION_TOKENS_PER_SECOND,
)
from forgeai.policy.bandit_actions import action_spec_from_proto
from forgeai.proto import execution_service_pb2, execution_service_pb2_grpc

_StreamChunk = execution_service_pb2.ExecutionExecuteServerStreamChunk


class ForgeExecutionServicer(execution_service_pb2_grpc.ExecutionServiceServicer):
    """Stream vLLM output; terminal chunks carry load or generation errors."""

    __slots__ = ("_elephant", "_loader")

    def __init__(
        self,
        model_loader: ModelLoader,
        elephant: ElephantBrokerClient,
    ) -> None:
        self._loader = model_loader
        self._elephant = elephant

    def Execute(
        self,
        request: execution_service_pb2.ExecutionExecuteRequest,
        _context: object,
    ) -> Iterator[execution_service_pb2.ExecutionExecuteServerStreamChunk]:
        """Unary→stream RPC; one fresh event loop for the full stream."""

        action = action_spec_from_proto(request.routing_action_spec)
        tier: Final[str] = action.model_tier.value
        prec: Final[str] = action.precision.value
        start = time.perf_counter()
        last_tokens = 0

        def _factory() -> AsyncIterator[_StreamChunk]:
            return async_execute_stream(
                request,
                model_loader=self._loader,
                elephant=self._elephant,
            )

        for chunk in sync_iterate_async_generator(_factory):
            last_tokens = chunk.cumulative_output_tokens_estimate
            yield chunk

        dt = time.perf_counter() - start
        EXECUTION_REQUEST_DURATION_SECONDS.labels(
            model_tier=tier,
            precision=prec,
        ).observe(dt)
        if last_tokens > 0 and dt > 0.0:
            EXECUTION_TOKENS_PER_SECOND.labels(
                model_tier=tier,
                precision=prec,
            ).set(float(last_tokens) / dt)


__all__ = ["ForgeExecutionServicer"]
