"""Create quant_profiles and link model_versions.quant_profile_id.

``quant_profiles`` references ``model_versions`` and ``eval_runs`` as defined in
Section 4. CHECK constraints enforce ``method`` and ``global_precision`` enums.

Adds ``updated_at`` (deviation: Section 4 lists ``created_at`` only).

After creating the table, adds the deferred foreign key from
``model_versions.quant_profile_id`` → ``quant_profiles.profile_id`` with
ON DELETE SET NULL to match the nullable optional link in Section 4.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from forgeai.registry.constants import GlobalPrecision, QuantMethod
from forgeai.registry.migration_checks import enum_in
from sqlalchemy.dialects import postgresql

revision = "005_create_quant_profiles"
down_revision = "006_create_eval_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ``quant_profiles`` and the deferred FK on ``model_versions``."""

    op.create_table(
        "quant_profiles",
        sa.Column(
            "profile_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("model_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("method", sa.String(length=32), nullable=False),
        sa.Column("global_precision", sa.String(length=16), nullable=False),
        sa.Column(
            "layer_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("accuracy_delta", sa.Float(), nullable=False),
        sa.Column("latency_improvement", sa.Float(), nullable=False),
        sa.Column("memory_mb", sa.Integer(), nullable=False),
        sa.Column("eval_run_id", postgresql.UUID(as_uuid=True), nullable=False),
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
            ["eval_run_id"],
            ["eval_runs.eval_run_id"],
            name="fk_quant_profiles_eval_run_id_eval_runs",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.version_id"],
            name="fk_quant_profiles_model_version_id_model_versions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("profile_id", name="pk_quant_profiles"),
        enum_in("method", QuantMethod, "ck_quant_profiles_method"),
        enum_in(
            "global_precision",
            GlobalPrecision,
            "ck_quant_profiles_global_precision",
        ),
    )
    op.create_index(
        "ix_quant_profiles_model_version_id",
        "quant_profiles",
        ["model_version_id"],
    )
    op.create_index(
        "ix_quant_profiles_eval_run_id",
        "quant_profiles",
        ["eval_run_id"],
    )

    op.create_foreign_key(
        "fk_model_versions_quant_profile_id_quant_profiles",
        "model_versions",
        "quant_profiles",
        ["quant_profile_id"],
        ["profile_id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    """Remove FK to quant_profiles, then drop ``quant_profiles``."""

    op.drop_constraint(
        "fk_model_versions_quant_profile_id_quant_profiles",
        "model_versions",
        type_="foreignkey",
    )
    op.drop_index("ix_quant_profiles_eval_run_id", table_name="quant_profiles")
    op.drop_index("ix_quant_profiles_model_version_id", table_name="quant_profiles")
    op.drop_table("quant_profiles")
