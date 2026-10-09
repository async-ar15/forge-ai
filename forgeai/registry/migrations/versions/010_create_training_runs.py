"""Create training_runs and enforce FK from model_versions.training_run_id.

``model_versions.training_run_id`` was nullable without a referent; this
migration adds the anchor table and the foreign key with ON DELETE SET NULL so
version rows stay valid if a training run row is removed administratively.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from forgeai.registry.constants import TrainingRunStatus
from forgeai.registry.migration_checks import enum_in
from sqlalchemy.dialects import postgresql

revision = "010_create_training_runs"
down_revision = "009_create_routing_decisions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ``training_runs`` and attach FK on ``model_versions``."""

    op.create_table(
        "training_runs",
        sa.Column(
            "training_run_id",
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
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("training_run_id", name="pk_training_runs"),
        enum_in("status", TrainingRunStatus, "ck_training_runs_status"),
    )
    op.create_foreign_key(
        "fk_model_versions_training_run_id_training_runs",
        "model_versions",
        "training_runs",
        ["training_run_id"],
        ["training_run_id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    """Drop FK and ``training_runs``."""

    op.drop_constraint(
        "fk_model_versions_training_run_id_training_runs",
        "model_versions",
        type_="foreignkey",
    )
    op.drop_table("training_runs")
