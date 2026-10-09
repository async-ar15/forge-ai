"""Create the deployments table.

Implements Section 4 columns and CHECK constraints for ``environment`` and
``status``. Foreign key to ``model_versions`` uses ON DELETE CASCADE so serving
rows disappear when a version is purged.

Adds ``created_at`` and ``updated_at`` for repository audit consistency
(deviation: Section 4 specifies ``deployed_at`` / ``retired_at`` only).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from forgeai.registry.constants import DeploymentEnvironment, DeploymentStatus
from forgeai.registry.migration_checks import enum_in
from sqlalchemy.dialects import postgresql

revision = "007_create_deployments"
down_revision = "005_create_quant_profiles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ``deployments``."""

    op.create_table(
        "deployments",
        sa.Column(
            "deployment_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("model_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("environment", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("replicas", sa.Integer(), nullable=False),
        sa.Column(
            "deployed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
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
            ["model_version_id"],
            ["model_versions.version_id"],
            name="fk_deployments_model_version_id_model_versions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("deployment_id", name="pk_deployments"),
        enum_in("environment", DeploymentEnvironment, "ck_deployments_environment"),
        enum_in("status", DeploymentStatus, "ck_deployments_status"),
    )
    op.create_index(
        "ix_deployments_model_version_id",
        "deployments",
        ["model_version_id"],
    )


def downgrade() -> None:
    """Drop ``deployments``."""

    op.drop_index("ix_deployments_model_version_id", table_name="deployments")
    op.drop_table("deployments")
