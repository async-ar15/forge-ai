"""RubricEval judge implementation for asynchronous offline labels."""

from __future__ import annotations

import json
import logging
import math
import uuid
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import mean, pstdev
from typing import Final, Protocol

from forgeai.eval.constants import (
    INTERNAL_JUDGE_SESSION_ID,
    INTERNAL_JUDGE_TENANT_ID,
    JUDGE_LOW_STD_THRESHOLD,
    JUDGE_MODEL_NAME,
    JUDGE_PARSE_FAILURE_RATE_THRESHOLD,
    JUDGE_PROMPT_VERSION,
    JUDGE_RELIABILITY_WINDOW_SIZE,
    RUBRIC_WEIGHT_COMPLETENESS,
    RUBRIC_WEIGHT_CONCISENESS,
    RUBRIC_WEIGHT_GROUNDEDNESS,
    RUBRIC_WEIGHT_RELEVANCE,
)
from forgeai.eval.prompts import render_judge_prompt
from forgeai.observability.metrics import (
    JUDGE_PARSE_FAILURE_RATE,
    JUDGE_SCORE_MEAN,
    JUDGE_SCORE_STD,
)
from forgeai.proto import common_pb2, execution_service_pb2

_LOG: Final[logging.Logger] = logging.getLogger(__name__)
_RUBRIC_DIMS: Final[tuple[str, ...]] = (
    "relevance",
    "groundedness",
    "completeness",
    "conciseness",
)


class _ExecutionStubLike(Protocol):
    def Execute(
        self,
        request: execution_service_pb2.ExecutionExecuteRequest,
        metadata: tuple[tuple[str, str], ...] = (),
    ) -> AsyncIterator[execution_service_pb2.ExecutionExecuteServerStreamChunk]: ...


@dataclass(frozen=True, slots=True)
class JudgeScores:
    relevance: float
    groundedness: float
    completeness: float
    conciseness: float
    composite: float
    raw_response: str
    judged_at: datetime
    judge_model: str
    parse_success: bool


def parse_judge_response(raw: str) -> JudgeScores | None:
    """Parse judge JSON output into normalized [0,1] score object."""

    payload = _extract_json(raw)
    if payload is None:
        _LOG.warning("judge_parse_failed raw=%s", raw[:500])
        return None
    try:
        scores = {k: _normalize_0_10(payload[k]) for k in _RUBRIC_DIMS}
    except Exception:
        _LOG.warning("judge_parse_failed raw=%s", raw[:500])
        return None
    return _scores_to_dataclass(scores=scores, raw=raw)


def _extract_json(raw: str) -> dict[str, object] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_0_10(value: object) -> float:
    if isinstance(value, (int, float)):
        x = float(value)
    elif isinstance(value, str):
        x = float(value.strip())
    else:
        raise TypeError(f"unsupported score value type: {type(value)!r}")
    return max(0.0, min(1.0, x / 10.0))


def _scores_to_dataclass(*, scores: dict[str, float], raw: str) -> JudgeScores:
    composite = (
        scores["relevance"] * RUBRIC_WEIGHT_RELEVANCE
        + scores["groundedness"] * RUBRIC_WEIGHT_GROUNDEDNESS
        + scores["completeness"] * RUBRIC_WEIGHT_COMPLETENESS
        + scores["conciseness"] * RUBRIC_WEIGHT_CONCISENESS
    )
    return JudgeScores(
        relevance=scores["relevance"],
        groundedness=scores["groundedness"],
        completeness=scores["completeness"],
        conciseness=scores["conciseness"],
        composite=float(composite),
        raw_response=raw,
        judged_at=datetime.now(UTC),
        judge_model=JUDGE_MODEL_NAME,
        parse_success=True,
    )


class RubricEvalJudge:
    """Judge response quality asynchronously via execution-service LLM calls."""

    __slots__ = ("_exec", "_parse_flags", "_scores", "_total_calls")

    def __init__(self, execution_stub: _ExecutionStubLike) -> None:
        self._exec = execution_stub
        self._scores: deque[JudgeScores] = deque(maxlen=JUDGE_RELIABILITY_WINDOW_SIZE)
        self._parse_flags: deque[bool] = deque(maxlen=JUDGE_RELIABILITY_WINDOW_SIZE)
        self._total_calls = 0

    async def score(
        self,
        *,
        query: str,
        response_text: str,
        retrieved_context: str,
    ) -> JudgeScores | None:
        """Score a response triple and emit reliability telemetry periodically."""

        _LOG.info(
            "judge_call prompt_version=%s model=%s",
            JUDGE_PROMPT_VERSION,
            JUDGE_MODEL_NAME,
        )
        raw = await self._call_judge_llm(query, response_text, retrieved_context)
        parsed = parse_judge_response(raw)
        self._record_reliability(parsed)
        return parsed

    async def _call_judge_llm(
        self,
        query: str,
        response_text: str,
        retrieved_context: str,
    ) -> str:
        prompt = render_judge_prompt(
            query=query,
            response_text=response_text,
            retrieved_context=retrieved_context,
        )
        request = _judge_execution_request(prompt)
        stream = self._exec.Execute(request, metadata=())
        return await _consume_stream_text(stream)

    def _record_reliability(self, parsed: JudgeScores | None) -> None:
        self._total_calls += 1
        ok = parsed is not None
        self._parse_flags.append(ok)
        if ok:
            assert parsed is not None
            self._scores.append(parsed)
        if self._total_calls % JUDGE_RELIABILITY_WINDOW_SIZE == 0:
            _emit_reliability_stats(self._scores, self._parse_flags)


def _judge_execution_request(
    prompt: str,
) -> execution_service_pb2.ExecutionExecuteRequest:
    """Build an execution request used exclusively for judge model scoring."""

    return execution_service_pb2.ExecutionExecuteRequest(
        correlation_request_id=str(uuid.uuid4()),
        tenant_id=INTERNAL_JUDGE_TENANT_ID,
        routing_action_spec=common_pb2.ActionSpec(
            model_tier=common_pb2.MODEL_TIER_MEDIUM,
            precision=common_pb2.PRECISION_FP16,
            retrieval_mode=common_pb2.RETRIEVAL_MODE_CACHE_ONLY,
            output_budget=common_pb2.OUTPUT_BUDGET_SHORT,
        ),
        user_prompt_utf8=prompt,
        max_output_token_limit=512,
        broker_session_id_utf8=INTERNAL_JUDGE_SESSION_ID,
    )


async def _consume_stream_text(
    stream: AsyncIterator[execution_service_pb2.ExecutionExecuteServerStreamChunk],
) -> str:
    parts: list[str] = []
    async for chunk in stream:
        if chunk.generated_text_delta_utf8:
            parts.append(chunk.generated_text_delta_utf8)
        if chunk.stream_finished:
            break
    return "".join(parts)


def _emit_reliability_stats(
    scores: deque[JudgeScores], parse_flags: deque[bool]
) -> None:
    dim_stats = _dimension_stats(scores)
    parse_failure_rate = _parse_failure_rate(parse_flags)
    JUDGE_PARSE_FAILURE_RATE.set(parse_failure_rate)
    for dim, stat in dim_stats.items():
        JUDGE_SCORE_MEAN.labels(dimension=dim).set(stat["mean"])
        JUDGE_SCORE_STD.labels(dimension=dim).set(stat["std"])
        if stat["std"] < JUDGE_LOW_STD_THRESHOLD:
            _LOG.warning(
                "judge may be collapsing to constant — possible reward hacking"
            )
    if parse_failure_rate > JUDGE_PARSE_FAILURE_RATE_THRESHOLD:
        _LOG.warning("judge parse failure rate exceeds 10% — check prompt template")
    _LOG.info(
        "judge_reliability dimensions=%s parse_failure_rate=%.4f",
        dim_stats,
        parse_failure_rate,
    )


def _dimension_stats(scores: deque[JudgeScores]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for dim in _RUBRIC_DIMS:
        vals = [float(getattr(s, dim)) for s in scores]
        out[dim] = _stat_record(vals)
    return out


def _stat_record(values: list[float]) -> dict[str, float]:
    if not values:
        nan = math.nan
        return {"mean": nan, "std": nan, "min": nan, "max": nan}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _parse_failure_rate(flags: deque[bool]) -> float:
    if not flags:
        return 0.0
    failures = sum(0 if ok else 1 for ok in flags)
    return float(failures / len(flags))


__all__ = ["JudgeScores", "RubricEvalJudge", "parse_judge_response"]
