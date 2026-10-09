"""Expand training_runs metadata and add deterministic checkpoints table."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from forgeai.registry.constants import TrainingRunStatus
from sqlalchemy.dialects import postgresql

revision = "014_create_training_runs"
down_revision = "013_update_eval_view"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add orchestrator columns and checkpoint metadata tables."""

    op.create_table(
        "checkpoints",
        sa.Column(
            "checkpoint_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("training_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("s3_path", sa.Text(), nullable=False),
        sa.Column("sha256_hash", sa.String(length=64), nullable=False),
        sa.Column("loss_at_step", sa.Float(), nullable=False),
        sa.Column(
            "verified",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("checkpoint_id", name="pk_checkpoints"),
        sa.ForeignKeyConstraint(
            ["training_run_id"],
            ["training_runs.training_run_id"],
            name="fk_checkpoints_training_run_id_training_runs",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_checkpoints_training_run_id",
        "checkpoints",
        ["training_run_id"],
    )
    _expand_training_runs_table()


def _expand_training_runs_table() -> None:
    op.add_column(
        "training_runs",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "training_runs",
        sa.Column("base_model_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "training_runs",
        sa.Column("config_snapshot", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "training_runs",
        sa.Column(
            "current_epoch",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "training_runs",
        sa.Column(
            "current_loss",
            sa.Float(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "training_runs",
        sa.Column(
            "best_checkpoint_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.add_column(
        "training_runs",
        sa.Column(
            "submitted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.add_column(
        "training_runs",
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "training_runs",
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "training_runs",
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.add_column("training_runs", sa.Column("seed", sa.Integer(), nullable=True))
    op.alter_column(
        "training_runs",
        "status",
        existing_type=sa.String(length=64),
        nullable=False,
    )
    op.execute("UPDATE training_runs SET job_id = training_run_id WHERE job_id IS NULL")
    op.execute(
        """
        UPDATE training_runs
        SET base_model_id = (
          SELECT model_id FROM model_versions
          WHERE model_versions.training_run_id = training_runs.training_run_id
          ORDER BY created_at ASC
          LIMIT 1
        )
        WHERE base_model_id IS NULL
        """
    )
    op.execute(
        "UPDATE training_runs "
        "SET config_snapshot = '{}'::jsonb "
        "WHERE config_snapshot IS NULL"
    )
    op.execute("UPDATE training_runs SET seed = 0 WHERE seed IS NULL")
    op.alter_column("training_runs", "job_id", nullable=False)
    op.alter_column("training_runs", "base_model_id", nullable=False)
    op.alter_column("training_runs", "config_snapshot", nullable=False)
    op.alter_column("training_runs", "seed", nullable=False)
    op.create_unique_constraint(
        "uq_training_runs_job_id",
        "training_runs",
        ["job_id"],
    )
    op.create_foreign_key(
        "fk_training_runs_base_model_id_models",
        "training_runs",
        "models",
        ["base_model_id"],
        ["model_id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_training_runs_best_checkpoint_id_checkpoints",
        "training_runs",
        "checkpoints",
        ["best_checkpoint_id"],
        ["checkpoint_id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_training_runs_job_id", "training_runs", ["job_id"])
    op.create_index(
        "ix_training_runs_base_model_id",
        "training_runs",
        ["base_model_id"],
    )
    op.create_index(
        "ix_training_runs_best_checkpoint_id",
        "training_runs",
        ["best_checkpoint_id"],
    )
    values = ", ".join(repr(m.value) for m in TrainingRunStatus)
    op.drop_constraint("ck_training_runs_status", "training_runs", type_="check")
    op.create_check_constraint(
        "ck_training_runs_status",
        "training_runs",
        f"status IN ({values})",
    )


def downgrade() -> None:
    """Revert training orchestration schema expansion."""

    values = ", ".join(repr(m.value) for m in TrainingRunStatus)
    op.drop_constraint("ck_training_runs_status", "training_runs", type_="check")
    op.create_check_constraint(
        "ck_training_runs_status",
        "training_runs",
        f"status IN ({values})",
    )
    op.drop_index("ix_training_runs_best_checkpoint_id", table_name="training_runs")
    op.drop_index("ix_training_runs_base_model_id", table_name="training_runs")
    op.drop_index("ix_training_runs_job_id", table_name="training_runs")
    op.drop_constraint(
        "fk_training_runs_best_checkpoint_id_checkpoints",
        "training_runs",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_training_runs_base_model_id_models",
        "training_runs",
        type_="foreignkey",
    )
    op.drop_constraint("uq_training_runs_job_id", "training_runs", type_="unique")
    op.drop_column("training_runs", "seed")
    op.drop_column("training_runs", "error_message")
    op.drop_column("training_runs", "completed_at")
    op.drop_column("training_runs", "started_at")
    op.drop_column("training_runs", "submitted_at")
    op.drop_column("training_runs", "best_checkpoint_id")
    op.drop_column("training_runs", "current_loss")
    op.drop_column("training_runs", "current_epoch")
    op.drop_column("training_runs", "config_snapshot")
    op.drop_column("training_runs", "base_model_id")
    op.drop_column("training_runs", "job_id")
    op.drop_index("ix_checkpoints_training_run_id", table_name="checkpoints")
    op.drop_table("checkpoints")
