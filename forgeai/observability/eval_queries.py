"""Offline evaluation queries with mandatory exploitative-only filtering."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

_VIEW_NAME = "routing_decisions_exploitative"


def _ts(value: datetime) -> str:
    return value.isoformat()


def get_policy_performance(
    engine: Engine,
    policy_version: str,
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    """Aggregate policy metrics for exploitative decisions in the time window."""

    sql = text(
        f"""
        SELECT
            r_online,
            final_latency_ms,
            final_cost_usd,
            cache_hit,
            slo_violation
        FROM {_VIEW_NAME}
        WHERE policy_version = :policy_version
          AND tenant_tier != 'internal'
          AND "timestamp" >= :start_time
          AND "timestamp" < :end_time
        """
    )
    frame = pd.read_sql_query(
        sql=sql,
        con=engine,
        params={
            "policy_version": policy_version,
            "start_time": _ts(start_time),
            "end_time": _ts(end_time),
        },
    )
    if frame.empty:
        return pd.DataFrame(
            [
                {
                    "avg_r_online": None,
                    "p50_latency_ms": None,
                    "p95_latency_ms": None,
                    "cost_per_request_usd": None,
                    "cache_hit_rate": None,
                    "slo_violation_rate": None,
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "avg_r_online": float(frame["r_online"].astype(float).mean()),
                "p50_latency_ms": float(
                    frame["final_latency_ms"].astype(float).quantile(0.50)
                ),
                "p95_latency_ms": float(
                    frame["final_latency_ms"].astype(float).quantile(0.95)
                ),
                "cost_per_request_usd": float(
                    frame["final_cost_usd"].astype(float).mean()
                ),
                "cache_hit_rate": float(frame["cache_hit"].astype(int).mean()),
                "slo_violation_rate": float(frame["slo_violation"].astype(int).mean()),
            }
        ]
    )


def get_action_distribution(
    engine: Engine,
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    """Count exploitative decisions by action dimensions."""

    sql = text(
        f"""
        SELECT
            model_tier,
            precision,
            retrieval_mode,
            COUNT(*) AS decision_count
        FROM {_VIEW_NAME}
        WHERE "timestamp" >= :start_time
          AND tenant_tier != 'internal'
          AND "timestamp" < :end_time
        GROUP BY model_tier, precision, retrieval_mode
        ORDER BY decision_count DESC, model_tier, precision, retrieval_mode
        """
    )
    return pd.read_sql_query(
        sql=sql,
        con=engine,
        params={"start_time": _ts(start_time), "end_time": _ts(end_time)},
    )


def get_reward_by_action(
    engine: Engine,
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    """Average online reward by action key over exploitative decisions."""

    sql = text(
        f"""
        SELECT
            model_tier || '|' || precision || '|' || retrieval_mode || '|'
                || output_budget AS action_key,
            AVG(r_online) AS avg_r_online
        FROM {_VIEW_NAME}
        WHERE "timestamp" >= :start_time
          AND tenant_tier != 'internal'
          AND "timestamp" < :end_time
        GROUP BY action_key
        ORDER BY avg_r_online DESC, action_key
        """
    )
    return pd.read_sql_query(
        sql=sql,
        con=engine,
        params={"start_time": _ts(start_time), "end_time": _ts(end_time)},
    )


def get_labeled_decisions(
    engine: Engine,
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    """Exploitative decisions with offline labels, ordered by labeling time."""

    sql = text(
        f"""
        SELECT
            request_id,
            "timestamp",
            policy_version,
            model_tier,
            precision,
            retrieval_mode,
            output_budget,
            r_online,
            q_offline,
            judge_score,
            groundedness_score,
            label_source,
            labeled_at
        FROM {_VIEW_NAME}
        WHERE "timestamp" >= :start_time
          AND tenant_tier != 'internal'
          AND "timestamp" < :end_time
          AND q_offline IS NOT NULL
        ORDER BY labeled_at ASC
        """
    )
    return pd.read_sql_query(
        sql=sql,
        con=engine,
        params={"start_time": _ts(start_time), "end_time": _ts(end_time)},
    )


def get_retraining_rows(engine: Engine) -> pd.DataFrame:
    """Return all exploitative rows eligible for policy retraining."""

    sql = text(
        f"""
        SELECT
            state_vector,
            model_tier,
            precision,
            retrieval_mode,
            output_budget,
            r_online,
            q_offline
        FROM {_VIEW_NAME}
        WHERE q_offline IS NOT NULL
          AND tenant_tier != 'internal'
        ORDER BY "timestamp" ASC
        """
    )
    return pd.read_sql_query(sql=sql, con=engine)


def get_recent_policy_versions(engine: Engine, limit: int = 2) -> list[str]:
    """Return most recent policy versions from exploitative view only."""

    sql = text(
        f"""
        SELECT policy_version, MAX("timestamp") AS max_ts
        FROM {_VIEW_NAME}
        WHERE tenant_tier != 'internal'
        GROUP BY policy_version
        ORDER BY max_ts DESC
        LIMIT :limit
        """
    )
    frame = pd.read_sql_query(sql=sql, con=engine, params={"limit": int(limit)})
    return [str(v) for v in frame["policy_version"].tolist()]


__all__ = [
    "get_action_distribution",
    "get_labeled_decisions",
    "get_policy_performance",
    "get_recent_policy_versions",
    "get_retraining_rows",
    "get_reward_by_action",
]
