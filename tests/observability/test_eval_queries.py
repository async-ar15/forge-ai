from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from forgeai.observability.eval_queries import (
    get_action_distribution,
    get_labeled_decisions,
    get_policy_performance,
    get_reward_by_action,
)
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@pytest.fixture
def sqlite_engine() -> Engine:
    eng = create_engine("sqlite+pysqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE routing_decisions (
                    request_id TEXT PRIMARY KEY,
                    "timestamp" TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    model_tier TEXT NOT NULL,
                    precision TEXT NOT NULL,
                    retrieval_mode TEXT NOT NULL,
                    output_budget TEXT NOT NULL,
                    tenant_tier TEXT NOT NULL,
                    exploration_flag INTEGER NOT NULL,
                    r_online REAL NOT NULL,
                    final_latency_ms INTEGER NOT NULL,
                    final_cost_usd REAL NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    slo_violation INTEGER NOT NULL,
                    q_offline REAL NULL,
                    judge_score REAL NULL,
                    groundedness_score REAL NULL,
                    label_source TEXT NULL,
                    labeled_at TEXT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE VIEW routing_decisions_exploitative AS
                SELECT *
                FROM routing_decisions
                WHERE exploration_flag = FALSE
                  AND tenant_tier != 'internal'
                """
            )
        )
        base = datetime.now(UTC)
        rows = [
            {
                "request_id": "a",
                "timestamp": (base - timedelta(minutes=5)).isoformat(),
                "policy_version": "v1",
                "model_tier": "small",
                "precision": "int8",
                "retrieval_mode": "cache_only",
                "output_budget": "short",
                "tenant_tier": "pro",
                "exploration_flag": 0,
                "r_online": 10.0,
                "final_latency_ms": 100,
                "final_cost_usd": 0.01,
                "cache_hit": 1,
                "slo_violation": 0,
                "q_offline": 0.8,
                "judge_score": 0.8,
                "groundedness_score": 0.9,
                "label_source": "judge",
                "labeled_at": base.isoformat(),
            },
            {
                "request_id": "b",
                "timestamp": (base - timedelta(minutes=4)).isoformat(),
                "policy_version": "v1",
                "model_tier": "large",
                "precision": "fp16",
                "retrieval_mode": "full",
                "output_budget": "long",
                "tenant_tier": "enterprise",
                "exploration_flag": 1,
                "r_online": -50.0,
                "final_latency_ms": 900,
                "final_cost_usd": 2.0,
                "cache_hit": 0,
                "slo_violation": 1,
                "q_offline": 0.2,
                "judge_score": 0.1,
                "groundedness_score": 0.1,
                "label_source": "judge",
                "labeled_at": base.isoformat(),
            },
            {
                "request_id": "c",
                "timestamp": (base - timedelta(minutes=3)).isoformat(),
                "policy_version": "v1",
                "model_tier": "medium",
                "precision": "fp16",
                "retrieval_mode": "off",
                "output_budget": "medium",
                "tenant_tier": "free",
                "exploration_flag": 0,
                "r_online": 5.0,
                "final_latency_ms": 200,
                "final_cost_usd": 0.2,
                "cache_hit": 0,
                "slo_violation": 0,
                "q_offline": None,
                "judge_score": None,
                "groundedness_score": None,
                "label_source": None,
                "labeled_at": None,
            },
            {
                "request_id": "d",
                "timestamp": (base - timedelta(minutes=2)).isoformat(),
                "policy_version": "v1",
                "model_tier": "small",
                "precision": "int8",
                "retrieval_mode": "cache_only",
                "output_budget": "short",
                "tenant_tier": "internal",
                "exploration_flag": 0,
                "r_online": 999.0,
                "final_latency_ms": 1,
                "final_cost_usd": 0.0,
                "cache_hit": 1,
                "slo_violation": 0,
                "q_offline": 1.0,
                "judge_score": 1.0,
                "groundedness_score": 1.0,
                "label_source": "judge",
                "labeled_at": base.isoformat(),
            },
        ]
        for row in rows:
            conn.execute(
                text(
                    """
                    INSERT INTO routing_decisions (
                        request_id, "timestamp", policy_version, model_tier, precision,
                        retrieval_mode, output_budget, tenant_tier,
                        exploration_flag, r_online,
                        final_latency_ms, final_cost_usd, cache_hit, slo_violation,
                        q_offline, judge_score, groundedness_score,
                        label_source, labeled_at
                    )
                    VALUES (
                        :request_id, :timestamp, :policy_version,
                        :model_tier, :precision,
                        :retrieval_mode, :output_budget, :tenant_tier,
                        :exploration_flag, :r_online,
                        :final_latency_ms, :final_cost_usd, :cache_hit, :slo_violation,
                        :q_offline, :judge_score, :groundedness_score,
                        :label_source, :labeled_at
                    )
                    """
                ),
                row,
            )
    return eng


def test_every_query_applies_exploration_filter(sqlite_engine: Engine) -> None:
    now = datetime.now(UTC)
    start = now - timedelta(hours=1)
    end = now + timedelta(hours=1)
    perf = get_policy_performance(sqlite_engine, "v1", start, end)
    dist = get_action_distribution(sqlite_engine, start, end)
    rewards = get_reward_by_action(sqlite_engine, start, end)
    labeled = get_labeled_decisions(sqlite_engine, start, end)
    assert float(perf["avg_r_online"].iloc[0]) == pytest.approx(7.5)
    assert int(dist["decision_count"].sum()) == 2
    assert len(rewards) == 2
    assert len(labeled) == 1


def test_get_policy_performance_returns_expected_aggregates(
    sqlite_engine: Engine,
) -> None:
    now = datetime.now(UTC)
    frame = get_policy_performance(
        sqlite_engine,
        "v1",
        now - timedelta(hours=1),
        now + timedelta(hours=1),
    )
    row = frame.iloc[0]
    assert float(row["avg_r_online"]) == pytest.approx(7.5)
    assert float(row["cost_per_request_usd"]) == pytest.approx(0.105)
    assert float(row["cache_hit_rate"]) == pytest.approx(0.5)
    assert float(row["slo_violation_rate"]) == pytest.approx(0.0)


def test_eval_view_excludes_exploratory_rows(sqlite_engine: Engine) -> None:
    with sqlite_engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM routing_decisions_exploitative"),
        ).scalar_one()
        assert int(count) == 2


def test_all_eval_queries_exclude_internal_tier(sqlite_engine: Engine) -> None:
    now = datetime.now(UTC)
    start = now - timedelta(hours=1)
    end = now + timedelta(hours=1)
    dist = get_action_distribution(sqlite_engine, start, end)
    rewards = get_reward_by_action(sqlite_engine, start, end)
    labeled = get_labeled_decisions(sqlite_engine, start, end)
    assert int(dist["decision_count"].sum()) == 2
    assert all("internal" not in str(v) for v in rewards["action_key"].tolist())
    assert len(labeled) == 1
