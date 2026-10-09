from __future__ import annotations

import json

import pytest
from forgeai.eval.judge import RubricEvalJudge, parse_judge_response
from forgeai.proto import execution_service_pb2


class _Stub:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses

    def Execute(self, _request, metadata=()):
        raw = self._responses.pop(0)
        return _stream(raw)


async def _stream(raw: str):
    yield execution_service_pb2.ExecutionExecuteServerStreamChunk(
        generated_text_delta_utf8=raw,
        stream_finished=True,
        cumulative_output_tokens_estimate=1,
    )


@pytest.mark.asyncio
async def test_score_returns_judge_scores_composite_in_bounds() -> None:
    raw = json.dumps(
        {
            "relevance": 8,
            "groundedness": 9,
            "completeness": 7,
            "conciseness": 6,
            "justifications": {},
        }
    )
    judge = RubricEvalJudge(_Stub([raw]))
    scores = await judge.score(query="q", response_text="r", retrieved_context="c")
    assert scores is not None
    assert 0.0 <= scores.composite <= 1.0


def test_parse_failure_returns_none_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        parsed = parse_judge_response("not-json")
    assert parsed is None
    assert "judge_parse_failed" in caplog.text


def test_composite_weighted_sum_is_correct() -> None:
    raw = '{"relevance":10,"groundedness":10,"completeness":0,"conciseness":0}'
    parsed = parse_judge_response(raw)
    assert parsed is not None
    assert parsed.composite == pytest.approx(0.55)


@pytest.mark.asyncio
async def test_reliability_monitoring_fires_at_100_calls(
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = '{"relevance":8,"groundedness":8,"completeness":8,"conciseness":8}'
    judge = RubricEvalJudge(_Stub([payload] * 100))
    with caplog.at_level("INFO"):
        for _ in range(100):
            await judge.score(query="q", response_text="r", retrieved_context="c")
    assert "judge_reliability" in caplog.text


@pytest.mark.asyncio
async def test_low_std_warning_fires(caplog: pytest.LogCaptureFixture) -> None:
    payload = '{"relevance":5,"groundedness":5,"completeness":5,"conciseness":5}'
    judge = RubricEvalJudge(_Stub([payload] * 100))
    with caplog.at_level("WARNING"):
        for _ in range(100):
            await judge.score(query="q", response_text="r", retrieved_context="c")
    assert "judge may be collapsing to constant" in caplog.text


@pytest.mark.asyncio
async def test_high_parse_failure_warning_fires(
    caplog: pytest.LogCaptureFixture,
) -> None:
    good = '{"relevance":8,"groundedness":8,"completeness":8,"conciseness":8}'
    responses = [good] * 89 + ["bad-json"] * 11
    judge = RubricEvalJudge(_Stub(responses))
    with caplog.at_level("WARNING"):
        for _ in range(100):
            await judge.score(query="q", response_text="r", retrieved_context="c")
    assert "judge parse failure rate exceeds 10%" in caplog.text
