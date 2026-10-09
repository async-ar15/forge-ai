"""Async token stream for ``ExecutionService.Execute`` (Section 2)."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Final, Protocol, cast

from google.protobuf.timestamp_pb2 import Timestamp

from forgeai.execution.constants import cost_per_output_token_microdollars
from forgeai.execution.elephant_broker import ElephantBrokerClient
from forgeai.execution.exceptions import (
    PrecisionUnavailableError,
    QuantizationMismatchError,
)
from forgeai.execution.model_loader import ModelLoader
from forgeai.execution.quantization_config import QuantizationConfig
from forgeai.execution.sampling_params import build_sampling_params
from forgeai.observability.metrics import EXECUTION_MODEL_LOAD_ERRORS_TOTAL
from forgeai.policy.bandit_actions import action_spec_from_proto
from forgeai.proto import common_pb2, execution_service_pb2

_LOG: Final[logging.Logger] = logging.getLogger(__name__)


class _ExecutableShard(Protocol):
    """Minimal engine surface used by the execution pipeline."""

    def stream_text(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        request_id: str,
    ) -> AsyncIterator[str]: ...

    def quantization_config(self) -> QuantizationConfig: ...


def _now_ts() -> Timestamp:
    ts = Timestamp()
    ts.FromDatetime(datetime.now(UTC))
    return ts


def _error_chunk(
    msg: str,
    *,
    fallback_hint: bool,
) -> execution_service_pb2.ExecutionExecuteServerStreamChunk:
    chunk = execution_service_pb2.ExecutionExecuteServerStreamChunk(
        generated_text_delta_utf8="",
        stream_finished=True,
        cumulative_output_tokens_estimate=0,
        stream_error_message_utf8=msg,
        execution_end_timestamp_utc=_now_ts(),
    )
    if fallback_hint:
        chunk.gateway_fallback_used_hint = True
    return chunk


async def async_execute_stream(
    request: execution_service_pb2.ExecutionExecuteRequest,
    *,
    model_loader: ModelLoader,
    elephant: ElephantBrokerClient,
) -> AsyncIterator[execution_service_pb2.ExecutionExecuteServerStreamChunk]:
    """Yield stream chunks; load errors become a single terminal chunk.

    Safe to run multiple concurrent streams when ``model_loader`` lock rules are
    respected; each invocation uses a unique vLLM ``request_id``.
    """

    action = action_spec_from_proto(request.routing_action_spec)
    try:
        handle = await model_loader.resolve_model(action)
    except PrecisionUnavailableError as exc:
        yield _error_chunk(str(exc), fallback_hint=False)
        return
    except QuantizationMismatchError as exc:
        EXECUTION_MODEL_LOAD_ERRORS_TOTAL.inc()
        yield _error_chunk(str(exc), fallback_hint=False)
        return
    except Exception as exc:
        EXECUTION_MODEL_LOAD_ERRORS_TOTAL.inc()
        _LOG.exception("execution_model_load_failed")
        yield _error_chunk(f"model_load_failed:{exc!s}", fallback_hint=False)
        return

    session_key = (
        request.broker_session_id_utf8
        if request.HasField("broker_session_id_utf8")
        else request.correlation_request_id
    )
    mem = elephant.get_memory_context(request.tenant_id, session_key)
    grounded = elephant.ground_prompt(
        request.user_prompt_utf8,
        mem.snippets_utf8,
    )
    sp = build_sampling_params(
        action.output_budget,
        max_output_token_limit=request.max_output_token_limit,
    )
    shard = cast(_ExecutableShard, handle.loaded_engine)
    gen_id = f"exec-{request.correlation_request_id}-{uuid.uuid4()}"
    tokens = 0
    try:
        agen = shard.stream_text(
            grounded.text_utf8,
            max_tokens=sp.max_tokens,
            temperature=sp.temperature,
            request_id=gen_id,
        )
        async for delta in agen:
            tokens += max(1, len(delta) // 4)
            yield execution_service_pb2.ExecutionExecuteServerStreamChunk(
                generated_text_delta_utf8=delta,
                stream_finished=False,
                cumulative_output_tokens_estimate=tokens,
            )
    except Exception as exc:
        _LOG.exception("execution_generation_failed")
        yield _error_chunk(f"generation_failed:{exc!s}", fallback_hint=True)
        return

    rate = cost_per_output_token_microdollars(action.model_tier, action.precision)
    micro = int(tokens * rate)
    yield execution_service_pb2.ExecutionExecuteServerStreamChunk(
        generated_text_delta_utf8="",
        stream_finished=True,
        cumulative_output_tokens_estimate=tokens,
        stream_final_cost_microdollars=common_pb2.MoneyMicrodollars(
            amount_microdollars=micro,
        ),
        execution_end_timestamp_utc=_now_ts(),
    )


__all__ = ["async_execute_stream"]
