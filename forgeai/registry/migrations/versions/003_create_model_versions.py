"""Create the model_versions table.

Section 4 defines all columns including optional ``quant_profile_id`` and
``training_run_id``. ``artifact_id`` and ``model_id`` foreign keys are created here.
``quant_profile_id`` is created **without** a foreign key because
``quant_profiles`` does not exist yet (circular dependency); migration
``005_create_quant_profiles`` adds the FK. The foreign key for
``training_run_id`` → ``training_runs.training_run_id`` is added in
``010_create_training_runs`` after the anchor table exists.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "003_create_model_versions"
down_revision = "004_create_artifacts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ``model_versions`` with FKs to ``models`` and ``artifacts``."""

    op.create_table(
        "model_versions",
        sa.Column(
            "version_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("model_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_tag", sa.String(length=255), nullable=False),
        sa.Column("artifact_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("quant_profile_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("training_run_id", postgresql.UUID(as_uuid=True), nullable=True),
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
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["artifact_id"],
            ["artifacts.artifact_id"],
            name="fk_model_versions_artifact_id_artifacts",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["models.model_id"],
            name="fk_model_versions_model_id_models",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("version_id", name="pk_model_versions"),
    )
    op.create_index("ix_model_versions_model_id", "model_versions", ["model_id"])
    op.create_index("ix_model_versions_artifact_id", "model_versions", ["artifact_id"])
    op.create_index(
        "ix_model_versions_quant_profile_id",
        "model_versions",
        ["quant_profile_id"],
    )
    op.create_index(
        "ix_model_versions_training_run_id",
        "model_versions",
        ["training_run_id"],
    )


def downgrade() -> None:
    """Drop ``model_versions`` and its indexes."""

    op.drop_index("ix_model_versions_training_run_id", table_name="model_versions")
    op.drop_index("ix_model_versions_quant_profile_id", table_name="model_versions")
    op.drop_index("ix_model_versions_artifact_id", table_name="model_versions")
    op.drop_index("ix_model_versions_model_id", table_name="model_versions")
    op.drop_table("model_versions")
