"""Tests for ``DecisionLogger`` durability, metrics, and invariant handling."""

from __future__ import annotations

import json
import logging
import random
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forgeai.enums import ExplorationType
from forgeai.policy.bandit import LinUCBBandit, RoutingAction
from forgeai.policy.bandit_actions import (
    ACTION_SPACE_SIZE,
    ALL_ACTIONS,
    canonical_action_key,
)
from forgeai.policy.constants import (
    DEFAULT_TENANT_POLICY,
    FeatureQueryType,
    FeatureTenantTier,
)
from forgeai.policy.decision_log_orm import build_orm_record
from forgeai.policy.decision_log_record import (
    R_COMPONENT_FIELD_NAMES,
    STATE_VECTOR_FIELD_NAMES,
    DecisionLog,
    decision_log_from_bandit_outcome,
)
from forgeai.policy.decision_logger import DecisionLogger
from forgeai.policy.decision_logger_io import JSONL_FALLBACK_REASON_KEY
from forgeai.policy.decision_logger_metrics import (
    DECISIONS_CONSISTENCY_VIOLATIONS_TOTAL,
    DECISIONS_FALLBACK_TOTAL,
    DECISIONS_LABEL_MISSING_TOTAL,
    DECISIONS_LOGGED_TOTAL,
    DecisionFallbackReason,
)
from forgeai.policy.features import FeatureVector
from forgeai.policy.reward import (
    RequestOutcome,
    compute_online_reward,
    reward_components_from_outcome,
)
from forgeai.registry.constants import LabelSource
from forgeai.registry.models.routing_decision import RoutingDecision


def _counter_value(metric: Any) -> float:
    return float(metric._value.get())


def _labeled_counter_value(metric: Any, **labels: str) -> float:
    return float(metric.labels(**labels)._value.get())


def _base_state_vector() -> dict[str, Any]:
    return {
        "query_len": 12,
        "token_budget": 2048,
        "query_type": "rag",
        "tenant_tier": "free",
        "latency_slo_ms": 2000,
        "queue_depth": 3,
        "gpu_load": 0.42,
        "cache_hit_prob": 0.11,
    }


def _full_scores() -> dict[str, float]:
    return {canonical_action_key(a): 0.0 for a in ALL_ACTIONS}


def _r_components() -> dict[str, float]:
    return {
        "latency": -0.1,
        "cost": -0.2,
        "slo_violation": 0.0,
        "fallback": 0.0,
        "retry": 0.0,
        "cache_hit": 0.05,
    }


def _make_decision(**overrides: Any) -> DecisionLog:
    rid = uuid.uuid4()
    tenant = uuid.uuid4()
    chosen = ALL_ACTIONS[0]
    ts = datetime.now(UTC)
    base = {
        "request_id": rid,
        "timestamp": ts,
        "policy_version": "pol-v1",
        "deployment_version": "dep-v1",
        "tenant_id": tenant,
        "state_vector": _base_state_vector(),
        "chosen_action": chosen,
        "greedy_action": chosen,
        "exploration_flag": False,
        "exploration_type": "none",
        "action_scores": _full_scores(),
        "final_latency_ms": 88,
        "final_cost_usd": 0.0012,
        "fallback_used": False,
        "retry_count": 0,
        "cache_hit": False,
        "slo_violation": False,
        "r_online": 0.5,
        "r_components": _r_components(),
    }
    base.update(overrides)
    return DecisionLog(
        request_id=base["request_id"],
        timestamp=base["timestamp"],
        policy_version=base["policy_version"],
        deployment_version=base["deployment_version"],
        tenant_id=base["tenant_id"],
        query_len=int(base["state_vector"]["query_len"]),
        token_budget=int(base["state_vector"]["token_budget"]),
        query_type=str(base["state_vector"]["query_type"]),
        tenant_tier=str(base["state_vector"]["tenant_tier"]),
        latency_slo_ms=int(base["state_vector"]["latency_slo_ms"]),
        queue_depth=int(base["state_vector"]["queue_depth"]),
        gpu_load=float(base["state_vector"]["gpu_load"]),
        cache_hit_prob=float(base["state_vector"]["cache_hit_prob"]),
        state_vector=dict(base["state_vector"]),
        chosen_action=base["chosen_action"],
        model_tier=base["chosen_action"].model_tier.value,
        precision=base["chosen_action"].precision.value,
        retrieval_mode=base["chosen_action"].retrieval_mode.value,
        output_budget=base["chosen_action"].output_budget.value,
        exploration_flag=base["exploration_flag"],
        exploration_type=base["exploration_type"],
        greedy_action=base["greedy_action"],
        action_scores=dict(base["action_scores"]),
        final_latency_ms=base["final_latency_ms"],
        final_cost_usd=base["final_cost_usd"],
        fallback_used=base["fallback_used"],
        retry_count=base["retry_count"],
        cache_hit=base["cache_hit"],
        slo_violation=base["slo_violation"],
        r_online=base["r_online"],
        r_components=dict(base["r_components"]),
    )


class _SessionCtx:
    def __init__(self, session: Any) -> None:
        self._session = session

    async def __aenter__(self) -> Any:
        return self._session

    async def __aexit__(self, *args: object) -> None:
        return None


def _session_factory(session: Any) -> Callable[[], _SessionCtx]:
    def _factory() -> _SessionCtx:
        return _SessionCtx(session)

    return _factory


def _log_session(captured: list[Any], commit_mock: AsyncMock) -> Any:
    s = MagicMock()
    s.add = lambda row: captured.append(row)
    s.commit = commit_mock
    return s


@pytest.fixture()
def fallback_path(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    p = tmp_path / "decisions.jsonl"
    monkeypatch.setenv("FORGEAI_DECISION_FALLBACK_PATH", str(p))
    return p


@pytest.mark.asyncio
async def test_log_decision_populates_orm_columns(
    fallback_path: Any,
) -> None:
    captured: list[Any] = []
    commit_ok = AsyncMock(return_value=None)
    session = _log_session(captured, commit_ok)
    factory = _session_factory(session)
    logger = DecisionLogger(factory, "dep-x")  # type: ignore[arg-type]
    decision = _make_decision()
    before = _counter_value(DECISIONS_LOGGED_TOTAL)

    await logger.log_decision(decision)

    assert _counter_value(DECISIONS_LOGGED_TOTAL) == before + 1
    assert len(captured) == 1
    rd = captured[0]
    assert isinstance(rd, RoutingDecision)
    expected = build_orm_record(decision)
    for col in RoutingDecision.__table__.columns:
        name = col.key
        phase2_only = {
            "q_offline",
            "judge_score",
            "groundedness_score",
            "label_source",
            "labeled_at",
        }
        if name in phase2_only:
            continue
        assert getattr(rd, name) == getattr(expected, name)


@pytest.mark.asyncio
async def test_log_decision_db_failure_writes_fallback_and_metrics(
    fallback_path: Any,
) -> None:
    captured: list[Any] = []
    commit_fail = AsyncMock(side_effect=RuntimeError("postgres down"))
    session = _log_session(captured, commit_fail)
    factory = _session_factory(session)
    logger = DecisionLogger(factory, "dep-x")  # type: ignore[arg-type]
    decision = _make_decision()
    fb_before = _labeled_counter_value(
        DECISIONS_FALLBACK_TOTAL,
        reason=DecisionFallbackReason.DB_ERROR,
    )
    log_before = _counter_value(DECISIONS_LOGGED_TOTAL)

    await logger.log_decision(decision)

    assert _counter_value(DECISIONS_LOGGED_TOTAL) == log_before
    assert (
        _labeled_counter_value(
            DECISIONS_FALLBACK_TOTAL,
            reason=DecisionFallbackReason.DB_ERROR,
        )
        == fb_before + 1
    )
    text = fallback_path.read_text(encoding="utf-8")
    line = json.loads(text.strip())
    assert line["request_id"] == str(decision.request_id)
    assert line[JSONL_FALLBACK_REASON_KEY] == DecisionFallbackReason.DB_ERROR


@pytest.mark.asyncio
async def test_log_decision_db_and_jsonl_both_fail_critical_stderr(
    caplog: pytest.LogCaptureFixture,
    fallback_path: Any,
) -> None:
    caplog.set_level(logging.CRITICAL)
    captured: list[Any] = []
    commit_fail = AsyncMock(side_effect=RuntimeError("db"))
    session = _log_session(captured, commit_fail)
    factory = _session_factory(session)
    logger = DecisionLogger(factory, "dep-x")  # type: ignore[arg-type]
    decision = _make_decision()

    with patch(
        "forgeai.policy.decision_logger_io.append_fallback_jsonl",
        side_effect=OSError("disk full"),
    ):
        await logger.log_decision(decision)

    crit_msgs = [r.message for r in caplog.records if r.levelno == logging.CRITICAL]
    assert any("routing_decision_db_commit_failed" in m for m in crit_msgs)
    assert any("decision_fallback_jsonl_failed" in m for m in crit_msgs)
    assert any("routing_decision_jsonl_unavailable" in m for m in crit_msgs)


@pytest.mark.asyncio
async def test_label_decision_updates_row(fallback_path: Any) -> None:
    rid = uuid.uuid4()
    row = RoutingDecision()
    row.request_id = rid
    row.labeled_at = None

    class LabelSess:
        def __init__(self) -> None:
            self.commit = AsyncMock(return_value=None)

        async def get(self, _model: Any, pk: Any) -> RoutingDecision | None:
            assert pk == rid
            return row

    ls = LabelSess()
    factory = _session_factory(ls)
    logger = DecisionLogger(factory, "dep-x")  # type: ignore[arg-type]

    await logger.label_decision(
        rid,
        q_offline=0.9,
        judge_score=0.8,
        groundedness_score=0.7,
        label_source=LabelSource.JUDGE,
    )

    assert row.q_offline == 0.9
    assert row.judge_score == 0.8
    assert row.groundedness_score == 0.7
    assert row.label_source == LabelSource.JUDGE.value
    assert row.labeled_at is not None
    ls.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_label_decision_missing_row_warning_and_metric(
    caplog: pytest.LogCaptureFixture,
    fallback_path: Any,
) -> None:
    caplog.set_level(logging.WARNING)
    rid = uuid.uuid4()

    class LabelSess:
        def __init__(self) -> None:
            self.commit = AsyncMock(return_value=None)

        async def get(self, _m: Any, _pk: Any) -> None:
            return None

    ls = LabelSess()
    factory = _session_factory(ls)
    logger = DecisionLogger(factory, "dep-x")  # type: ignore[arg-type]
    before = _counter_value(DECISIONS_LABEL_MISSING_TOTAL)

    await logger.label_decision(
        rid,
        q_offline=1.0,
        judge_score=1.0,
        groundedness_score=1.0,
        label_source=LabelSource.AUTO,
    )

    assert _counter_value(DECISIONS_LABEL_MISSING_TOTAL) == before + 1
    assert any("routing_decision_label_missing" in r.message for r in caplog.records)
    ls.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_label_decision_already_labeled_no_overwrite(
    caplog: pytest.LogCaptureFixture,
    fallback_path: Any,
) -> None:
    caplog.set_level(logging.WARNING)
    rid = uuid.uuid4()
    prior_ts = datetime(2024, 1, 1, tzinfo=UTC)
    row = RoutingDecision()
    row.request_id = rid
    row.labeled_at = prior_ts
    row.q_offline = 0.1
    row.judge_score = 0.2
    row.groundedness_score = 0.3
    row.label_source = LabelSource.HUMAN.value

    class LabelSess:
        def __init__(self) -> None:
            self.commit = AsyncMock(return_value=None)

        async def get(self, _m: Any, _pk: Any) -> RoutingDecision:
            return row

    ls = LabelSess()
    factory = _session_factory(ls)
    logger = DecisionLogger(factory, "dep-x")  # type: ignore[arg-type]

    await logger.label_decision(
        rid,
        q_offline=0.99,
        judge_score=0.99,
        groundedness_score=0.99,
        label_source=LabelSource.JUDGE,
        force=False,
    )

    assert row.q_offline == 0.1
    assert any("routing_decision_already_labeled" in r.message for r in caplog.records)
    ls.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_label_decision_force_overwrites(
    fallback_path: Any,
) -> None:
    rid = uuid.uuid4()
    row = RoutingDecision()
    row.request_id = rid
    row.labeled_at = datetime.now(UTC)
    row.q_offline = 0.1

    class LabelSess:
        def __init__(self) -> None:
            self.commit = AsyncMock(return_value=None)

        async def get(self, _m: Any, _pk: Any) -> RoutingDecision:
            return row

    ls = LabelSess()
    factory = _session_factory(ls)
    logger = DecisionLogger(factory, "dep-x")  # type: ignore[arg-type]

    await logger.label_decision(
        rid,
        q_offline=0.55,
        judge_score=0.66,
        groundedness_score=0.77,
        label_source=LabelSource.AUTO,
        force=True,
    )

    assert row.q_offline == 0.55
    ls.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_consistency_greedy_ne_chosen_still_commits(
    caplog: pytest.LogCaptureFixture,
    fallback_path: Any,
) -> None:
    caplog.set_level(logging.CRITICAL)
    chosen = ALL_ACTIONS[0]
    greedy = ALL_ACTIONS[1]
    decision = _make_decision(
        chosen_action=chosen,
        greedy_action=greedy,
        exploration_flag=False,
        model_tier=chosen.model_tier.value,
        precision=chosen.precision.value,
        retrieval_mode=chosen.retrieval_mode.value,
        output_budget=chosen.output_budget.value,
    )
    captured: list[Any] = []
    commit_ok = AsyncMock(return_value=None)
    session = _log_session(captured, commit_ok)
    factory = _session_factory(session)
    logger = DecisionLogger(factory, "dep-x")  # type: ignore[arg-type]
    viol_before = _counter_value(DECISIONS_CONSISTENCY_VIOLATIONS_TOTAL)

    await logger.log_decision(decision)

    assert _counter_value(DECISIONS_CONSISTENCY_VIOLATIONS_TOTAL) > viol_before
    viol = "routing_decision_consistency_violation"
    assert any(viol in r.message for r in caplog.records)
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_action_scores_wrong_count_tracks_violation(
    caplog: pytest.LogCaptureFixture,
    fallback_path: Any,
) -> None:
    caplog.set_level(logging.CRITICAL)
    scores = _full_scores()
    assert len(scores) == ACTION_SPACE_SIZE
    first_key = next(iter(scores))
    del scores[first_key]
    decision = _make_decision(action_scores=scores)
    captured: list[Any] = []
    commit_ok = AsyncMock(return_value=None)
    session = _log_session(captured, commit_ok)
    factory = _session_factory(session)
    logger = DecisionLogger(factory, "dep-x")  # type: ignore[arg-type]
    viol_before = _counter_value(DECISIONS_CONSISTENCY_VIOLATIONS_TOTAL)

    await logger.log_decision(decision)

    assert _counter_value(DECISIONS_CONSISTENCY_VIOLATIONS_TOTAL) > viol_before
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_r_components_wrong_keys_tracks_violation(
    caplog: pytest.LogCaptureFixture,
    fallback_path: Any,
) -> None:
    caplog.set_level(logging.CRITICAL)
    bad_rc = dict(_r_components())
    bad_rc["typo"] = bad_rc.pop("latency")
    decision = _make_decision(r_components=bad_rc)
    captured: list[Any] = []
    commit_ok = AsyncMock(return_value=None)
    session = _log_session(captured, commit_ok)
    factory = _session_factory(session)
    logger = DecisionLogger(factory, "dep-x")  # type: ignore[arg-type]
    viol_before = _counter_value(DECISIONS_CONSISTENCY_VIOLATIONS_TOTAL)

    await logger.log_decision(decision)

    assert _counter_value(DECISIONS_CONSISTENCY_VIOLATIONS_TOTAL) > viol_before
    assert len(captured) == 1


def test_decision_log_from_bandit_outcome_wires_vector_bandit_and_outcome() -> None:
    """Public factory: real ``FeatureVector``, LinUCB arm choice, ``RequestOutcome``."""

    features = FeatureVector(
        query_len=256,
        token_budget=1024,
        query_type=FeatureQueryType.CHAT,
        tenant_tier=FeatureTenantTier.ENTERPRISE,
        latency_slo_ms=500,
        queue_depth=2,
        gpu_load=0.33,
        cache_hit_prob=0.17,
    )
    state_vector = features.to_log_dict()
    assert frozenset(state_vector.keys()) == STATE_VECTOR_FIELD_NAMES

    bandit = LinUCBBandit(
        ridge_scale=1.0,
        ucb_alpha=0.5,
        epsilon=0.0,
        rng=random.Random(123),
    )
    bd = bandit.select_action(features)
    chosen: RoutingAction = bd.chosen_action
    greedy: RoutingAction = bd.greedy_action

    assert bd.exploration_flag is False
    assert chosen == greedy
    assert bd.exploration_type is ExplorationType.NONE

    outcome = RequestOutcome(
        final_latency_ms=120.0,
        final_cost_usd=0.004,
        slo_violated=False,
        fallback_used=False,
        retry_count=0,
        cache_hit=True,
        latency_slo_ms=float(features.latency_slo_ms),
    )
    policy = DEFAULT_TENANT_POLICY
    r_online = compute_online_reward(outcome, policy)
    r_components = reward_components_from_outcome(outcome, policy)
    assert frozenset(r_components.keys()) == R_COMPONENT_FIELD_NAMES
    assert sum(r_components.values()) == pytest.approx(r_online)

    rid = uuid.uuid4()
    tenant = uuid.uuid4()
    ts = datetime.now(UTC)
    log = decision_log_from_bandit_outcome(
        request_id=rid,
        timestamp=ts,
        policy_version=bandit.policy_version,
        deployment_version="deploy-e2e",
        tenant_id=tenant,
        state_vector=state_vector,
        chosen_action=chosen,
        greedy_action=greedy,
        exploration_flag=bd.exploration_flag,
        exploration_type_wire=bd.exploration_type.value,
        action_scores=bd.action_scores,
        final_latency_ms=int(outcome.final_latency_ms),
        final_cost_usd=outcome.final_cost_usd,
        fallback_used=outcome.fallback_used,
        retry_count=outcome.retry_count,
        cache_hit=outcome.cache_hit,
        slo_violation=outcome.slo_violated,
        r_online=r_online,
        r_components=r_components,
    )

    expected_keys = frozenset(canonical_action_key(a) for a in ALL_ACTIONS)
    assert frozenset(log.action_scores.keys()) == expected_keys
    assert len(log.action_scores) == ACTION_SPACE_SIZE
    assert log.greedy_action == log.chosen_action
    assert frozenset(log.r_components.keys()) == R_COMPONENT_FIELD_NAMES
    assert frozenset(log.state_vector.keys()) == STATE_VECTOR_FIELD_NAMES
    assert log.exploration_type == bd.exploration_type.value
    assert log.state_vector == dict(state_vector)
    assert log.r_online == pytest.approx(r_online)


def test_build_orm_record_maps_decision_to_row() -> None:
    decision = _make_decision()
    rd = build_orm_record(decision)
    assert rd.request_id == decision.request_id
    assert rd.policy_version == decision.policy_version
    assert rd.deployment_version == decision.deployment_version
    assert rd.tenant_id == decision.tenant_id
    assert rd.final_cost_usd == Decimal(str(decision.final_cost_usd))
    assert rd.greedy_action == {
        "model_tier": decision.greedy_action.model_tier.value,
        "precision": decision.greedy_action.precision.value,
        "retrieval_mode": decision.greedy_action.retrieval_mode.value,
        "output_budget": decision.greedy_action.output_budget.value,
    }


@pytest.mark.asyncio
async def test_internal_forced_empty_action_scores_no_consistency_violation(
    fallback_path: Any,
) -> None:
    """Internal forced decisions allow empty action_scores and persist JSONB {}."""

    chosen = ALL_ACTIONS[0]
    decision = _make_decision(
        state_vector={**_base_state_vector(), "tenant_tier": "internal"},
        exploration_flag=False,
        exploration_type="forced",
        action_scores={},
        chosen_action=chosen,
        greedy_action=chosen,
        model_tier=chosen.model_tier.value,
        precision=chosen.precision.value,
        retrieval_mode=chosen.retrieval_mode.value,
        output_budget=chosen.output_budget.value,
    )
    captured: list[Any] = []
    commit_ok = AsyncMock(return_value=None)
    session = _log_session(captured, commit_ok)
    logger = DecisionLogger(_session_factory(session), "dep-x")  # type: ignore[arg-type]
    before = _counter_value(DECISIONS_CONSISTENCY_VIOLATIONS_TOTAL)

    await logger.log_decision(decision)

    assert _counter_value(DECISIONS_CONSISTENCY_VIOLATIONS_TOTAL) == before
    assert len(captured) == 1
    rd = captured[0]
    assert rd.action_scores == {}
