"""Create routing_decisions — bandit log, audit trail, and offline eval dataset.

Implements every Section 4 column with PostgreSQL types: ``FLOAT`` columns use
``DOUBLE PRECISION`` via :class:`sqlalchemy.Float`; ``NUMERIC`` uses
``Numeric(24, 12)`` for ``final_cost_usd``. JSONB columns use non-null defaults
for required structured fields.

Append-only contract (repository extension beyond Section 4 wording): rows are
inserted once for the synchronous path; the only follow-up write is the async
offline label path. ``labeled_at`` is NULL until ``q_offline`` (and related label
columns) are written, making that second write explicit — there is no
``updated_at`` implying general mutability.

CHECK constraints encode enum-like VARCHAR domains. ``precision`` is quoted in SQL
because it is a reserved keyword. ``exploration_type`` and ``label_source`` allow
NULL per Section 4 (``exploration_type`` may be null when not exploratory;
``label_source`` filled asynchronously).

Indexes: foreign key on ``tenant_id``, requested columns, and a partial index on
``timestamp`` only where ``exploration_flag = FALSE`` for eval queries (filter
exploration first; tenant/time use separate indexes).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from forgeai.registry.constants import (
    ExplorationType,
    GlobalPrecision,
    LabelSource,
    ModelTier,
    OutputBudget,
    QueryType,
    RetrievalMode,
    TenantTier,
)
from forgeai.registry.migration_checks import enum_in, enum_in_or_null
from sqlalchemy.dialects import postgresql

revision = "009_create_routing_decisions"
down_revision = "008_create_tenant_policies"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ``routing_decisions`` with constraints and indexes."""

    op.create_table(
        "routing_decisions",
        sa.Column(
            "request_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("policy_version", sa.String(length=255), nullable=False),
        sa.Column("deployment_version", sa.String(length=255), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("query_len", sa.Integer(), nullable=False),
        sa.Column("token_budget", sa.Integer(), nullable=False),
        sa.Column("query_type", sa.String(length=32), nullable=False),
        sa.Column("tenant_tier", sa.String(length=32), nullable=False),
        sa.Column("latency_slo_ms", sa.Integer(), nullable=False),
        sa.Column("queue_depth", sa.Integer(), nullable=False),
        sa.Column("gpu_load", sa.Float(), nullable=False),
        sa.Column("cache_hit_prob", sa.Float(), nullable=False),
        sa.Column(
            "state_vector",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("model_tier", sa.String(length=16), nullable=False),
        sa.Column("precision", sa.String(length=16), nullable=False),
        sa.Column("retrieval_mode", sa.String(length=32), nullable=False),
        sa.Column("output_budget", sa.String(length=16), nullable=False),
        sa.Column("exploration_flag", sa.Boolean(), nullable=False),
        sa.Column("exploration_type", sa.String(length=32), nullable=True),
        sa.Column(
            "greedy_action",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "action_scores",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("final_latency_ms", sa.Integer(), nullable=False),
        sa.Column("final_cost_usd", sa.Numeric(24, 12), nullable=False),
        sa.Column("fallback_used", sa.Boolean(), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("cache_hit", sa.Boolean(), nullable=False),
        sa.Column("slo_violation", sa.Boolean(), nullable=False),
        sa.Column("r_online", sa.Float(), nullable=False),
        sa.Column(
            "r_components",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("q_offline", sa.Float(), nullable=True),
        sa.Column("judge_score", sa.Float(), nullable=True),
        sa.Column("groundedness_score", sa.Float(), nullable=True),
        sa.Column("label_source", sa.String(length=32), nullable=True),
        sa.Column("labeled_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.tenant_id"],
            name="fk_routing_decisions_tenant_id_tenants",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("request_id", name="pk_routing_decisions"),
        enum_in("query_type", QueryType, "ck_routing_decisions_query_type"),
        enum_in("tenant_tier", TenantTier, "ck_routing_decisions_tenant_tier"),
        enum_in("model_tier", ModelTier, "ck_routing_decisions_model_tier"),
        enum_in('"precision"', GlobalPrecision, "ck_routing_decisions_precision"),
        enum_in("retrieval_mode", RetrievalMode, "ck_routing_decisions_retrieval_mode"),
        enum_in("output_budget", OutputBudget, "ck_routing_decisions_output_budget"),
        enum_in_or_null(
            "exploration_type",
            ExplorationType,
            "ck_routing_decisions_exploration_type",
        ),
        enum_in_or_null(
            "label_source",
            LabelSource,
            "ck_routing_decisions_label_source",
        ),
    )
    op.create_index(
        "ix_routing_decisions_timestamp",
        "routing_decisions",
        ["timestamp"],
    )
    op.create_index(
        "ix_routing_decisions_tenant_id",
        "routing_decisions",
        ["tenant_id"],
    )
    op.create_index(
        "ix_routing_decisions_exploration_flag",
        "routing_decisions",
        ["exploration_flag"],
    )
    op.create_index(
        "ix_routing_decisions_policy_version",
        "routing_decisions",
        ["policy_version"],
    )
    op.create_index(
        "ix_routing_decisions_eval_non_explore",
        "routing_decisions",
        ["timestamp"],
        postgresql_where=sa.text("exploration_flag = false"),
    )


def downgrade() -> None:
    """Drop ``routing_decisions`` indexes and table."""

    op.drop_index(
        "ix_routing_decisions_eval_non_explore",
        table_name="routing_decisions",
        postgresql_where=sa.text("exploration_flag = false"),
    )
    op.drop_index("ix_routing_decisions_policy_version", table_name="routing_decisions")
    op.drop_index(
        "ix_routing_decisions_exploration_flag",
        table_name="routing_decisions",
    )
    op.drop_index("ix_routing_decisions_tenant_id", table_name="routing_decisions")
    op.drop_index("ix_routing_decisions_timestamp", table_name="routing_decisions")
    op.drop_table("routing_decisions")
