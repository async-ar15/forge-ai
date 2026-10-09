"""Create datasets anchor and eval_runs tables.

Section 4 ``eval_runs`` lists ``dataset_id`` as a UUID foreign key to an eval
dataset, but no parent table is defined in the document. This file first creates
a minimal ``datasets`` table (PK + audit timestamps) so the FK is enforceable
(**deviation**: parent table invented; name ``datasets`` matches logical role).

Then creates ``eval_runs`` exactly per Section 4 with CHECK on ``eval_type``,
FK to ``model_versions`` (ON DELETE CASCADE) and FK to ``datasets`` (ON DELETE
RESTRICT). Adds ``updated_at`` per repository audit standard (deviation from
Section 4, which lists ``created_at`` only).

Alembic applies this revision **before** ``005_create_quant_profiles`` because
``quant_profiles.eval_run_id`` must reference existing ``eval_runs`` rows.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from forgeai.registry.constants import EvalType
from forgeai.registry.migration_checks import enum_in
from sqlalchemy.dialects import postgresql

revision = "006_create_eval_runs"
down_revision = "003_create_model_versions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ``datasets`` then ``eval_runs``."""

    op.create_table(
        "datasets",
        sa.Column(
            "dataset_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("dataset_id", name="pk_datasets"),
    )

    op.create_table(
        "eval_runs",
        sa.Column(
            "eval_run_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("model_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("eval_type", sa.String(length=32), nullable=False),
        sa.Column("judge_model", sa.String(length=255), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column(
            "score_breakdown",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.dataset_id"],
            name="fk_eval_runs_dataset_id_datasets",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.version_id"],
            name="fk_eval_runs_model_version_id_model_versions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("eval_run_id", name="pk_eval_runs"),
        enum_in("eval_type", EvalType, "ck_eval_runs_eval_type"),
    )
    op.create_index(
        "ix_eval_runs_model_version_id",
        "eval_runs",
        ["model_version_id"],
    )
    op.create_index("ix_eval_runs_dataset_id", "eval_runs", ["dataset_id"])


def downgrade() -> None:
    """Drop ``eval_runs`` then ``datasets``."""

    op.drop_index("ix_eval_runs_dataset_id", table_name="eval_runs")
    op.drop_index("ix_eval_runs_model_version_id", table_name="eval_runs")
    op.drop_table("eval_runs")
    op.drop_table("datasets")
