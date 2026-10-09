"""Create the models table for base LLM metadata.

Implements Section 4 ``models`` columns exactly. Adds ``updated_at`` with timezone
default ``now()`` per repository-wide audit requirements (not listed in Section 4;
explicit deviation).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from forgeai.registry.constants import ModelProvider
from forgeai.registry.migration_checks import enum_in
from sqlalchemy.dialects import postgresql

revision = "002_create_models"
down_revision = "001_create_tenants"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ``models`` with provider CHECK constraint."""

    op.create_table(
        "models",
        sa.Column(
            "model_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("base_architecture", sa.String(length=128), nullable=False),
        sa.Column("parameter_count", sa.BigInteger(), nullable=False),
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
        sa.PrimaryKeyConstraint("model_id", name="pk_models"),
        enum_in("provider", ModelProvider, "ck_models_provider"),
    )


def downgrade() -> None:
    """Drop ``models``."""

    op.drop_table("models")
